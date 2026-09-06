# -*- coding: utf-8 -*-
"""
kmfig.digitize — 单页/单图的数字化编排：
  面板检测 → 轴标定（矢量优先，OCR 兜底）→ 曲线提取（矢量优先，光栅兜底）
  → 阶梯函数重建 → 端点分类（OS/PFS）→ 臂标签 → QA 指标 → 风险表。
"""
import re
import numpy as np
import cv2

from .panels import detect_panels
from .calibrate import calibrate_vector, calibrate_ocr, parse_num
from .vector_extract import candidate_curve_colors, extract_curve_vector, label_arms_by_text, _label_by_legend_swatches
from . import raster_extract

ENDPOINT_PATTERNS = [
    ("PFS", re.compile(r"progression[\s\-–]*free|\bPFS\b|\bIRRC\b", re.I)),
    ("OS", re.compile(r"overall\s*survival|\bOS\b", re.I)),
    ("EFS", re.compile(r"event[\s\-–]*free|\bEFS\b", re.I)),
    ("DFS", re.compile(r"disease[\s\-–]*free|\bDFS\b|recurrence[\s\-–]*free|\bRFS\b", re.I)),
    ("DOR", re.compile(r"duration\s*of\s*response|\bDOR\b|\bDoR\b", re.I)),
    ("TTD", re.compile(r"time\s*to\s*(deterioration|treatment\s*discontinuation)|\bTTD\b", re.I)),
    ("SURV", re.compile(r"surviv|probability|proportion", re.I)),
]


def classify_endpoint(texts):
    """按优先级匹配端点关键词。"""
    joined = " ".join(texts)
    for name, pat in ENDPOINT_PATTERNS:
        if pat.search(joined):
            return name
    return "UNKNOWN"


def panel_context_texts(page, panel):
    """面板的纵轴标题（竖排文字）、面板上方标题、横轴标题。"""
    x0, y0, x1, y1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    w, h = x1 - x0, y1 - y0
    ytitle, top, xtitle = [], [], []
    for sp in page.spans:
        cx, cy = (sp["x0"] + sp["x1"]) / 2, (sp["y0"] + sp["y1"]) / 2
        t = sp["text"].strip()
        if not t:
            continue
        # 竖排且在竖轴左侧
        if sp["vertical"] and x0 - 0.35 * w <= cx < x0 and y0 - 0.2 * h <= cy <= y1 + 0.2 * h:
            ytitle.append(t)
        # 面板上方（标题/图例区）
        elif not sp["vertical"] and x0 - 0.1 * w <= cx <= x1 + 0.1 * w and y0 - 0.45 * h <= cy < y0:
            top.append(t)
        # 横轴标题：轴下方 0.06~0.22 h
        elif not sp["vertical"] and x0 <= cx <= x1 and y1 + 0.06 * h <= cy <= y1 + 0.25 * h:
            if parse_num(t) is None:
                xtitle.append(t)
    return dict(ytitle=ytitle, top=top, xtitle=xtitle)


def _time_unit_from_texts(texts):
    j = " ".join(texts).lower()
    if "week" in j:
        return "weeks"
    if "day" in j:
        return "days"
    if "year" in j:
        return "years"
    return "months"


def points_to_step(points_px, xcal, ycal):
    """
    像素折线 → 数据坐标阶梯函数。
    返回 dict(steps=[(t,S)] 顶点序列, t0, S0, monotone_violation, raw=[(t,S)])
    S 统一为 0~1；steps 已做单调化（累计最小值），raw 为未单调化的原始顶点。
    """
    pts = list(points_px)
    if len(pts) < 2:
        return None
    if pts[0][0] > pts[-1][0]:
        pts = pts[::-1]
    scale = 100.0 if ycal.unit == "percent" else 1.0
    data = [(xcal.value(x), ycal.value(y) / scale) for x, y in pts]
    clean = [data[0]]
    for p in data[1:]:
        if abs(p[0] - clean[-1][0]) > 1e-9 or abs(p[1] - clean[-1][1]) > 1e-9:
            clean.append(p)
    # 单调性检查（KM 只能下降）：记录最大上升幅度，再做累计最小值单调化
    viol = 0.0
    mono = []
    cur = None
    for t, s in clean:
        if cur is not None and s > cur + 1e-6:
            viol = max(viol, s - cur)
            s = cur
        cur = s if cur is None else min(cur, s)
        mono.append((t, cur))
    # 时间也应非递减：小幅回退（绘图误差）钳制
    out = []
    tmax = -1e18
    for t, s in mono:
        if t < tmax:
            t = tmax
        tmax = t
        out.append((t, s))
    return dict(steps=out, raw=clean, t0=out[0][0], S0=out[0][1], monotone_violation=viol)


def step_eval(steps, t_grid):
    """阶梯函数取值：t 处取最近一个 t_i<=t 的 S（右连续）。"""
    ts = np.array([p[0] for p in steps])
    ss = np.array([p[1] for p in steps])
    order = np.argsort(ts, kind="stable")
    ts, ss = ts[order], ss[order]
    idx = np.searchsorted(ts, t_grid, side="right") - 1
    out = np.full(len(t_grid), np.nan)
    ok = idx >= 0
    out[ok] = ss[idx[ok]]
    return out


def median_survival(steps, tol=0.0):
    """曲线首次跌到 ≤0.5+tol 的时间（tol=数字化 1 像素对应生存率的量化误差）；未到达返回 None。"""
    for t, s in steps:
        if s <= 0.5 + tol:
            return float(t)
    return None


def extract_risk_table(page, panel, xcal, arm_colors):
    """
    横轴下方的“No. at risk”数字：按行聚类，列位置用 xcal 换算为时间。
    返回 [dict(color, label, times, counts)]，颜色对不上时 color=None。
    """
    x0, y0, x1, y1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    h = y1 - y0
    rows = {}
    for sp in page.spans:
        t = sp["text"].strip()
        v = parse_num(t)
        if v is None or not float(v).is_integer():
            continue
        cx, cy = (sp["x0"] + sp["x1"]) / 2, (sp["y0"] + sp["y1"]) / 2
        if not (x0 - 0.05 * (x1 - x0) <= cx <= x1 + 0.05 * (x1 - x0)):
            continue
        if not (y1 + 0.20 * h <= cy <= y1 + 1.1 * h):   # 跳过刻度标签行（紧贴轴下方）
            continue
        key = round(cy / max(4.0, 0.03 * h))
        rows.setdefault(key, []).append(dict(cx=cx, cy=cy, val=int(v), color=sp["color"]))
    out = []
    for key in sorted(rows):
        cells = sorted(rows[key], key=lambda c: c["cx"])
        if len(cells) < 3:
            continue
        times = [round(xcal.value(c["cx"]), 2) for c in cells]
        # 行颜色：取多数
        cols = [tuple(round(v, 2) for v in c["color"]) for c in cells if c["color"] is not None]
        color = max(set(cols), key=cols.count) if cols else None
        matched = None
        for ac in arm_colors:
            if ac is not None and color is not None and all(abs(a - b) <= 0.1 for a, b in zip(ac, color)):
                matched = ac
                break
        out.append(dict(color=matched, row_color=color, times=times, counts=[c["val"] for c in cells],
                        y_px=float(np.mean([c["cy"] for c in cells]))))
    return out


def _fullpage_ocr_texts(page):
    """整页 OCR 文本（无文字层的光栅图用），结果缓存在 page 对象上。"""
    if getattr(page, "_ocr_texts", None) is not None:
        return page._ocr_texts
    from .calibrate import _ocr_region
    res = _ocr_region(page.img, (0, 0, page.img.shape[1], page.img.shape[0]), upscale=1.5)
    page._ocr_texts = [r["text"] for r in res if r.get("text")]
    page._ocr_boxes = [r for r in res if r.get("text")]
    return page._ocr_texts


def _label_arms_raster(page, panel, arm_colors):
    """
    光栅图臂标签：全图 OCR，取每个文本框内暗像素的主色，若与某臂颜色相近
    且文本非纯数字，则计为该臂的候选标签；取出现次数最多者。
    """
    from .calibrate import parse_num
    texts = _fullpage_ocr_texts(page)
    if not texts:
        return {}
    import numpy as _np
    import cv2 as _cv2
    cand = {c: [] for c in arm_colors}
    for r in getattr(page, "_ocr_boxes", []):
        t = r["text"].strip()
        if len(t) < 2 or parse_num(t) is not None:
            continue
        # 文本框（略外扩）内暗像素主色
        pad = 2
        x0b, y0b = int(max(0, r["x0"] - pad)), int(max(0, r["y0"] - pad))
        x1b, y1b = int(min(page.img.shape[1], r["x1"] + pad)), int(min(page.img.shape[0], r["y1"] + pad))
        patch = page.img[y0b:y1b, x0b:x1b]
        if patch.size == 0:
            continue
        gray = _cv2.cvtColor(patch, _cv2.COLOR_BGR2GRAY)
        dark = patch[gray < 190]
        if len(dark) < 8:
            continue
        bgr = dark.mean(axis=0) / 255.0
        rgb = (float(bgr[2]), float(bgr[1]), float(bgr[0]))
        for c in arm_colors:
            if c is None:
                continue
            # 容差较宽：文字有抗锯齿，颜色比曲线本体浅
            if all(abs(a - b) <= 0.22 for a, b in zip(rgb, c)):
                cand[c].append(t)
                break
    out = {}
    for c, ts in cand.items():
        if ts:
            # 短候选（≤2 字符，多为图例色块被 OCR 误读）在有长候选时丢弃
            if any(len(t) >= 3 for t in ts):
                ts = [t for t in ts if len(t) >= 3]
            from collections import Counter
            out[c] = sorted(Counter(ts).items(), key=lambda kv: (-kv[1] * len(kv[0]),))[0][0]
    return out


def _annotation_median_pairs(page, panel):
    """
    面板区域内的注释文本里提取 (“组名”, 中位月数) 对：
    形如 “Atezolizumab plus chemotherapy group: median overall survival 18·6 months”，
    也支持组名行与中位数行上下相邻的布局。矢量图用文字层；光栅图用整页 OCR。
    返回 [(value, label)]。
    """
    med_pat = re.compile(r"median[^0-9\n]{0,40}?(\d+(?:[·.，,]\d+)?)\s*(?:months?|mo|月)", re.I)
    label_pat = re.compile(r"^([\w][\w \-/+()]{1,60}?)[::]\s*$")
    pairs = []
    if page.has_vector:
        lines = {}
        for sp in page.spans:
            t = sp["text"].strip()
            if not t:
                continue
            cy = round((sp["y0"] + sp["y1"]) / 2 / 8)
            lines.setdefault(cy, []).append((sp["x0"], t))
        rows = [" ".join(t for x, t in sorted(lines[cy])) for cy in sorted(lines)]
    else:
        texts = _fullpage_ocr_texts(page)
        rows = [t for t in texts if t]
    last_labels = []   # [(行距, label)]
    for row in rows:
        lm = label_pat.match(row.strip())
        if lm and not any(ch.isdigit() for ch in lm.group(1)):
            last_labels = [(0, lm.group(1))] + last_labels[:3]
            continue
        for m in med_pat.finditer(row):
            try:
                v = float(m.group(1).replace("·", ".").replace("，", ".").replace(",", "."))
            except ValueError:
                continue
            # 同行带冒号组名
            m2 = re.search(r"([\w][\w \-/+()]{1,60}?)[::]\s*median", row)
            if m2:
                pairs.append((v, m2.group(1)))
            elif last_labels:
                pairs.append((v, last_labels[0][1]))
    return pairs


def dedup_arms(arms, xcal, ycal, px_tol):
    """去除几乎完全重合的重复臂（同一曲线被两个相近色相各提取一次）。"""
    if len(arms) <= 1:
        return arms
    t_lo = max(min(a["steps"][0][0] for a in arms), 0.0)
    t_hi = min(max(a["steps"][-1][0] for a in arms), max(xcal.values))
    if t_hi <= t_lo:
        return arms
    grid = np.linspace(t_lo, t_hi, 200)
    traces = [step_eval(a["steps"], grid) for a in arms]
    keep = []
    for i, a in enumerate(arms):
        dup = False
        for j in range(len(arms)):
            if i == j:
                continue
            ti, tj = traces[i], traces[j]
            ok = ~np.isnan(ti) & ~np.isnan(tj)
            if ok.sum() < 50:
                continue
            if np.mean(np.abs(ti[ok] - tj[ok]) <= 3 * px_tol) >= 0.9:
                other = arms[j]
                if (a["t_end"] - a["steps"][0][0], a["n_steps"]) <= \
                        (other["t_end"] - other["steps"][0][0], other["n_steps"]):
                    dup = True
                    break
        if not dup:
            keep.append(a)
    return keep


def drop_band_edge_arms(arms, xcal, ycal, px_tol):
    """
    剔除“带边伪臂”：轨迹大部分（≥65%）贴着另一条臂（距离 ≤3px 容差）且像素数更少——
    多为残差提取梯队拾取的置信带边缘，而非真实曲线。
    """
    if len(arms) < 2:
        return arms
    t_lo = max(min(a["steps"][0][0] for a in arms), 0.0)
    t_hi = min(max(a["steps"][-1][0] for a in arms), max(xcal.values))
    if t_hi <= t_lo:
        return arms
    grid = np.linspace(t_lo, t_hi, 200)
    traces = {id(a): step_eval(a["steps"], grid) for a in arms}
    keep = []
    for i, a in enumerate(arms):
        npx_a = a.get("n_px")
        drop = False
        others = [b for j, b in enumerate(arms) if j != i]
        ta = traces[id(a)]
        if others:
            stack = np.vstack([traces[id(b)] for b in others])
            ok = ~np.isnan(ta) & ~np.all(np.isnan(stack), axis=0)
            close_any = np.zeros(ok.sum(), bool)
            vals = ta[ok]
            k = 0
            for c in range(stack.shape[1]):
                if np.isnan(ta[c]):
                    continue
                col = stack[:, c]
                col = col[~np.isnan(col)]
                if len(col):
                    close_any[k] = np.min(np.abs(col - ta[c])) <= 3 * px_tol
                k += 1
            if close_any.mean() >= 0.65 and (npx_a is not None and
                                             any(b.get("n_px") is not None and npx_a < b.get("n_px") for b in others)):
                drop = True

        if not drop:
            keep.append(a)
    return keep


def digitize_panel(page, panel, min_curve_len_frac=0.5):
    """对一个面板做完整数字化。返回 dict。"""
    W = panel["x1"] - panel["x0"]
    H = panel["y1"] - panel["y0"]
    res = dict(panel=panel, mode=None, calib={}, arms=[], qa={}, context={})
    # ---- 端点分类（光栅图无文字层时用整页 OCR）----
    ctx = panel_context_texts(page, panel) if page.has_vector else dict(ytitle=[], top=[], xtitle=[])
    res["context"] = ctx
    res["endpoint"] = classify_endpoint(ctx["ytitle"] + ctx["top"])
    if res["endpoint"] == "UNKNOWN" and not page.has_vector:
        ocr_texts = _fullpage_ocr_texts(page)
        res["endpoint"] = classify_endpoint(ocr_texts)
        res["context"]["ocr_sample"] = ocr_texts[:40]
    res["time_unit"] = _time_unit_from_texts(ctx["xtitle"])

    # ---- 轴标定 ----
    xcal = ycal = None
    if page.has_vector:
        xcal, ycal, info = calibrate_vector(page, panel)
        res["calib"]["vector_ticks"] = dict(nx=len(info["x_ticks"]), ny=len(info["y_ticks"]))
    if xcal is None or ycal is None:
        xo, yo, info2 = calibrate_ocr(page, panel)
        xcal = xcal or xo
        ycal = ycal or yo
        res["calib"]["ocr_ticks"] = dict(nx=len(info2["x_ticks"]), ny=len(info2["y_ticks"]))
    if xcal is None or ycal is None:
        res["qa"]["error"] = "calibration_failed"
        res["calib"]["x"] = xcal.to_dict() if xcal else None
        res["calib"]["y"] = ycal.to_dict() if ycal else None
        return res
    res["calib"]["x"] = xcal.to_dict()
    res["calib"]["y"] = ycal.to_dict()
    # 轴原点与拟合的一致性（x=0 应落在竖轴上；y=1.0/100 应落在竖轴顶端附近）
    x_at_0 = xcal.pixel(0.0)
    y_at_top = ycal.pixel(100.0 if ycal.unit == "percent" else 1.0)
    res["calib"]["origin_px"] = dict(x0=float(x_at_0), y_top=float(y_at_top),
                                     dx_from_axis=float(x_at_0 - panel["axis_y_x"]),
                                     dy_from_top=float(y_at_top - panel["y0"]))
    res["calib"]["origin_ok"] = bool(abs(x_at_0 - panel["axis_y_x"]) < 0.03 * W and
                                     abs(y_at_top - panel["y0"]) < 0.06 * H)

    # ---- 曲线提取：矢量优先 ----
    arms = []
    if page.has_vector:
        cands = candidate_curve_colors(page, panel)
        for color, total_len, n_paths in cands:
            cur = extract_curve_vector(page, panel, color)
            if cur is None:
                continue
            xs = [p[0] for p in cur["points"]]
            span = max(xs) - min(xs)
            if span < min_curve_len_frac * 0.3 * W:      # 太短：图例样条、标注线
                continue
            st = points_to_step(cur["points"], xcal, ycal)
            if st is None:
                continue
            arms.append(dict(color=color, mode="vector", steps=st["steps"], t0=st["t0"], S0=st["S0"],
                             monotone_violation=st["monotone_violation"],
                             censor_times=[round(xcal.value(x), 3) for x in cur["censors"]],
                             n_censors=cur["n_censors"], x_span_frac=span / W,
                             n_dashed_segs=cur["n_dashed_segs"]))
        # 真正的 KM 曲线应从左端开始且起点在顶部附近；不满足的臂保留但打标记
        tmax = max(xcal.values)
        for a in arms:
            a["start_ok"] = bool(a["t0"] <= 0.05 * tmax + 1e-9 and a["S0"] >= 0.85)
        # 若同色出现多条（如 CI 边界被当成曲线），只保留每种颜色里 start_ok 且最长的一条
        by_color = {}
        for a in arms:
            k = a["color"]
            if k not in by_color or (a["start_ok"], a["x_span_frac"]) > (by_color[k]["start_ok"], by_color[k]["x_span_frac"]):
                by_color[k] = a
        arms = list(by_color.values())
        res["mode"] = "vector" if arms else None
    # ---- 光栅兜底 ----
    if not arms:
        origin = (x_at_0, y_at_top)
        r_arms = []
        for s_min in (0.45, 0.30):   # 先排除浅色 CI 带；不行再放宽
            r_arms = raster_extract.extract_curves_raster(page.img, panel, s_min=s_min, origin_px=origin)
            if len(r_arms) >= 2:
                break
        # 残余色相再提取：排除已提取臂的色相后重新聚类（被大色相峰压制的细曲线）
        if len(r_arms) < 2:
            ex = []
            for a in r_arms:
                c = a["color"]
                bgr = cv2.cvtColor(np.uint8([[[c[2] * 255, c[1] * 255, c[0] * 255]]]), cv2.COLOR_BGR2HSV)
                ex.append(int(bgr[0, 0, 0]))
            extra = raster_extract.extract_curves_raster(
                page.img, panel, s_min=0.30, origin_px=origin,
                apply_decorations=False, exclude_hues=ex)
            for a in extra:
                a["extraction_pass"] = "residual"
            r_arms = r_arms + extra
        # 点线（dotted）兜底：更宽连通域桥接 + 更低饱和度
        if len(r_arms) < 2:
            ex = []
            for a in r_arms:
                c = a["color"]
                bgr = cv2.cvtColor(np.uint8([[[c[2] * 255, c[1] * 255, c[0] * 255]]]), cv2.COLOR_BGR2HSV)
                ex.append(int(bgr[0, 0, 0]))
            dot = raster_extract.extract_curves_raster(
                page.img, panel, s_min=0.15, origin_px=origin, dilate=4,
                min_w_frac=0.006, min_h_frac=0.006, apply_decorations=False, exclude_hues=ex)
            for a in dot:
                a.setdefault("extraction_pass", "dotted")
            # 单色（近黑）臂兜底
            if len(dot) + len(r_arms) < 2:
                dot += raster_extract.extract_monochrome_arm(page.img, panel, origin_px=origin)
            r_arms = r_arms + dot
        for a in r_arms:
            st = points_to_step(a["points"], xcal, ycal)
            if st is None:
                continue
            tmax = max(xcal.values)
            # KM 先验：光栅臂必须从 (0, 1) 出发（起点异常的是删失标记等碎片）
            if st["t0"] > 0.05 * tmax + 1e-9 or st["S0"] < 0.85:
                continue
            arms.append(dict(color=a["color"], mode="raster", steps=st["steps"], t0=st["t0"], S0=st["S0"],
                             monotone_violation=st["monotone_violation"], censor_times=[], n_censors=None,
                             x_span_frac=a["x_span_frac"], n_dashed_segs=0, start_ok=None,
                             overlap_frac=a.get("overlap_frac"), coverage=round(a.get("coverage", 1.0), 3),
                             extraction_pass=a.get("extraction_pass", "main")))
        px_tol_dd = abs(ycal.fit["a"]) / (100.0 if ycal.unit == "percent" else 1.0)
        arms = drop_band_edge_arms(arms, xcal, ycal, px_tol_dd)
        res["mode"] = "raster" if arms else "none"

    # ---- 臂标签 ----（优先级：同色文字 > 注释中位数匹配 > 图例线样+黑字）
    labels = label_arms_by_text(page, panel, [a["color"] for a in arms]) if page.has_vector else \
        _label_arms_raster(page, panel, [a["color"] for a in arms])
    # 1 像素对应生存率的量化误差（用于中位数等里程碑量的容差判定）
    px_tol = abs(ycal.fit["a"]) / (100.0 if ycal.unit == "percent" else 1.0)
    for i, a in enumerate(arms):
        a["label"] = labels.get(a["color"])
        a["arm_id"] = f"arm{i + 1}"
        a["median"] = median_survival(a["steps"], tol=px_tol)
        a["t_end"] = a["steps"][-1][0]
        a["S_end"] = a["steps"][-1][1]
        a["n_steps"] = len(a["steps"])
    # 回退1：注释文本 “<组名>: median ... X months” 与数字化中位数互相匹配
    need = [a for a in arms if not a.get("label") and a.get("median")]
    if need:
        pairs = _annotation_median_pairs(page, panel)
        used = set()
        for a in sorted(need, key=lambda x: x["median"]):
            cands = [(abs(v - a["median"]) / v, lab) for v, lab in pairs
                     if v > 0 and lab not in used and abs(v - a["median"]) / v <= 0.15]
            if cands:
                d, lab = min(cands)
                a["label"] = lab.strip().rstrip(":：").strip()
                used.add(lab)
    # 回退2：黑色图例文字 + 彩色线样
    need = [a for a in arms if not a.get("label")]
    if need and page.has_vector:
        sw_labels = _label_by_legend_swatches(page, panel, [a["color"] for a in need],
                                              (panel["x0"] - 0.6 * (panel["x1"] - panel["x0"]),
                                               panel["y0"] - 0.45 * (panel["y1"] - panel["y0"]),
                                               panel["x1"] + 0.6 * (panel["x1"] - panel["x0"]),
                                               panel["y1"] + 1.2 * (panel["y1"] - panel["y0"])))
        for a in need:
            lab = sw_labels.get(a["color"])
            if lab and not any(k in lab.lower() for k in ("median", " ci", "p<", "p=", "hr ")):
                a["label"] = lab
    px_tol_dd = abs(ycal.fit["a"]) / (100.0 if ycal.unit == "percent" else 1.0)
    arms = dedup_arms(arms, xcal, ycal, px_tol_dd)
    res["arms"] = arms

    # ---- 风险表 ----
    if page.has_vector and arms:
        res["at_risk"] = extract_risk_table(page, panel, xcal, [a["color"] for a in arms])
    else:
        res["at_risk"] = []

    # ---- QA ----
    qa = res["qa"]
    qa["n_arms"] = len(arms)
    qa["start_ok"] = all(abs(a["t0"]) <= 0.02 * max(xcal.values) and a["S0"] >= 0.95 for a in arms) if arms else False
    qa["monotone_ok"] = all(a["monotone_violation"] <= 0.02 for a in arms) if arms else False
    # 标定成功与否以刻度拟合质量为准；origin 偏移（x=0 不在竖轴上等）仅作提示，
    # 因为部分图的坐标轴有留白（x 轴左端 < 0）
    qa["calib_ok"] = bool(xcal.ok and ycal.ok)
    qa["labels_ok"] = all(a["label"] for a in arms) if arms else False
    qa["pass"] = bool(arms) and qa["start_ok"] and qa["monotone_ok"] and qa["calib_ok"]
    return res


def digitize_page(page, **kw):
    """整页：检测面板并逐个数字化。"""
    panels = detect_panels(page.img)
    results = []
    for i, p in enumerate(panels):
        r = digitize_panel(page, p, **kw)
        r["panel_index"] = i
        results.append(r)
    return results
