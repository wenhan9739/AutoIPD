# -*- coding: utf-8 -*-
"""Axis calibration: vector tick labels / geometric tick-marks + OCR / RANSAC fitting."""

import re
import numpy as np
import cv2

NUM_RE = re.compile(r"^-?\d+(?:[.,·]\d+)?%?$")

def parse_num(s):
    """'0·5' / '1,5' / '12' / '50%' / '100-' -> float；解析失败返回 None。
    先去掉 OCR 常带入的尾部刻度符号。"""
    s = s.strip().replace("%", "").strip()
    while s and s[-1] in "-–—_=|·.,":
        s = s[:-1].strip()
    if not NUM_RE.match(s):
        return None
    s = s.replace("·", ".").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _ransac_line(pix, vals, tol, rng_cap=500):
    """两点RANSAC找初始直线：返回 (a, b) 或 None。"""
    n = len(pix)
    idx = np.arange(n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > rng_cap:
        rs = np.random.default_rng(0)
        pick = rs.choice(len(pairs), rng_cap, replace=False)
        pairs = [pairs[k] for k in pick]
    best, best_in = None, -1
    for i, j in pairs:
        dx = pix[j] - pix[i]
        if abs(dx) < 1e-9:
            continue
        a = (vals[j] - vals[i]) / dx
        b = vals[i] - a * pix[i]
        inl = int(np.sum(np.abs(vals - (a * pix + b)) <= tol))
        if inl > best_in:
            best_in, best = inl, (a, b)
    return best


def fit_linear(pix, vals, max_iter=4, tol_frac=0.02):
    """y = a*x + b 的稳健拟合（RANSAC 初值 + 迭代剔除离群点），
    返回 dict(a,b,n,rmse,ok) 或 None。容差为纯相对值。"""
    pix = np.asarray(pix, float)
    vals = np.asarray(vals, float)
    if len(pix) < 2:
        return None
    rng = max(abs(vals.max() - vals.min()), 1e-9)
    tol = max(tol_frac * rng, 1e-6)
    ab = _ransac_line(pix, vals, tol)
    if ab is None:
        return None
    a, b = ab
    keep = np.abs(vals - (a * pix + b)) <= tol
    for _ in range(max_iter):
        if keep.sum() < 2:
            return None
        a, b = np.polyfit(pix[keep], vals[keep], 1)
        if abs(a) < 1e-9:
            return None
        res = np.abs(vals - (a * pix + b))
        new_keep = res <= tol
        if (new_keep == keep).all():
            break
        keep = new_keep
    rmse = float(np.sqrt(np.mean((vals[keep] - (a * pix[keep] + b)) ** 2)))
    return dict(a=float(a), b=float(b), n=int(keep.sum()), rmse=rmse, ok=bool(keep.sum() >= 3),
                kept_vals=vals[keep].tolist(), kept_px=pix[keep].tolist())


def spacing_consistency(vals_sorted, frac_tol=0.12):
    """数值刻度应近似等距；返回 (是否一致, 中位步长)。"""
    if len(vals_sorted) < 3:
        return True, None
    d = np.diff(np.asarray(vals_sorted, float))
    med = np.median(d)
    if med <= 0:
        return False, None
    return bool(np.all(d >= med * (1 - frac_tol)) and np.all(d <= med * (1 + frac_tol))), float(med)


class AxisCalib:
    """单轴标定结果：value = a*pixel + b。"""

    def __init__(self, fit, values, unit="months", source="vector"):
        self.fit = fit
        self.values = list(values)
        self.unit = unit
        self.source = source

    @property
    def ok(self):
        return self.fit is not None and self.fit["ok"]

    def value(self, px):
        return self.fit["a"] * px + self.fit["b"]

    def pixel(self, val):
        return (val - self.fit["b"]) / self.fit["a"]

    def to_dict(self):
        if self.fit is None:
            return dict(ok=False, n=0, source=self.source)
        return dict(ok=self.fit["ok"], n=self.fit["n"], rmse=round(self.fit["rmse"], 3),
                    a=self.fit["a"], b=self.fit["b"], source=self.source,
                    values=[round(v, 3) for v in self.values], unit=self.unit)


def _x_tick_words(words, panel, page_h):
    """横轴刻度词：位于横轴下方第一行（同一 y 聚类）内的数字词。"""
    ax_y = panel["axis_x_y"]
    pw = panel["x1"] - panel["x0"]
    band = max(8.0, 0.045 * page_h)
    cands = [w for w in words
             if panel["x0"] - 0.015 * pw <= (w["x0"] + w["x1"]) / 2 <= panel["x1"] + 0.06 * pw
             and ax_y + 1 <= (w["y0"] + w["y1"]) / 2 <= ax_y + band]
    nums = []
    for w in cands:
        v = parse_num(w["text"])
        if v is not None:
            nums.append(dict(px=(w["x0"] + w["x1"]) / 2.0, py=(w["y0"] + w["y1"]) / 2.0, val=v, text=w["text"]))
    # 只保留最靠近轴的一行（风险表数字在更下方）
    if nums:
        rows = {}
        for t in nums:
            rows.setdefault(round(t["py"] / max(3.0, 0.4 * (cands[0]["y1"] - cands[0]["y0"]))), []).append(t)
        first = min(rows, key=lambda k: np.mean([t["py"] for t in rows[k]]))
        nums = rows[first]
    return nums


def _y_tick_words(words, panel):
    """纵轴刻度词：位于竖轴左侧、沿竖轴高度分布的数字词（取最近一列）。"""
    ax_x = panel["axis_y_x"]
    pw = panel["x1"] - panel["x0"]
    ph = panel["y1"] - panel["y0"]
    cands = [w for w in words
             if panel["y0"] - 0.03 * ph <= (w["y0"] + w["y1"]) / 2 <= panel["y1"] + 0.03 * ph
             and ax_x - 0.10 * pw <= (w["x0"] + w["x1"]) / 2 <= ax_x - 1.0]
    nums = []
    for w in cands:
        v = parse_num(w["text"])
        if v is not None:
            nums.append(dict(px=(w["y0"] + w["y1"]) / 2.0, x1=w["x1"], val=v, text=w["text"]))
    # 只保留右边缘最靠近轴的一列（标签右对齐于轴）
    if len(nums) >= 3:
        xr = np.array([t["x1"] for t in nums])
        ref = np.median(xr)
        nums = [t for t in nums if abs(t["x1"] - ref) < 0.04 * pw]
    return nums


def calibrate_vector(page, panel):
    """用 PDF 文字层做两轴标定。返回 (x_calib, y_calib, info)。"""
    info = {}
    xt = _x_tick_words(page.words, panel, page.shape[0])
    yt = _y_tick_words(page.words, panel)
    fx = fit_linear([t["px"] for t in xt], [t["val"] for t in xt]) if len(xt) >= 3 else None
    fy = fit_linear([t["px"] for t in yt], [t["val"] for t in yt]) if len(yt) >= 3 else None
    okx = spacing_consistency(sorted(t["val"] for t in xt))[0] if len(xt) >= 3 else False
    oky = spacing_consistency(sorted(t["val"] for t in yt))[0] if len(yt) >= 3 else False
    if fx is not None and not okx:
        fx = None
    if fy is not None and not oky:
        fy = None
    ymax = max([t["val"] for t in yt], default=100.0)
    yunit = "fraction" if ymax <= 1.5 else "percent"
    x_cal = AxisCalib(fx, sorted(t["val"] for t in xt), "months", "vector") if fx else None
    y_cal = AxisCalib(fy, sorted(t["val"] for t in yt), yunit, "vector") if fy else None
    info["x_ticks"] = xt
    info["y_ticks"] = yt
    return x_cal, y_cal, info


# ---------------- OCR 路径（位图/扫描件） ----------------

_ocr_engine = None

def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_region(img, box, upscale=2.0):
    """裁剪区域并 OCR，返回 [dict(px, py, x0, y0, x1, y1, text, conf)]（原图坐标系）。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    crop = img[y0:y1, x0:x1]
    if upscale != 1.0:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    res, _ = _get_ocr()(crop)
    out = []
    if res:
        for box_, text, conf in res:
            xs = [p[0] for p in box_]; ys = [p[1] for p in box_]
            out.append(dict(px=(min(xs) + max(xs)) / 2 / upscale + x0,
                            py=(min(ys) + max(ys)) / 2 / upscale + y0,
                            x0=min(xs) / upscale + x0, y0=min(ys) / upscale + y0,
                            x1=max(xs) / upscale + x0, y1=max(ys) / upscale + y0,
                            text=text, conf=float(conf)))
    return out


def _dark_mask(img):
    from .panels import dark_lowsat_mask
    return dark_lowsat_mask(img) > 0


def _find_tick_marks(mask, axis_y=None, axis_x=None):
    """
    检测刻度短线：
      x 轴刻度：axis_y 下方 2~10px 的短竖笔的列位置；
      y 轴刻度：axis_x 左侧 2~10px 的短横笔的行位置。
    返回像素位置列表（已合并相邻）。
    """
    if axis_y is not None:
        band = mask[int(axis_y) + 2: int(axis_y) + 11, :]
        counts = band.sum(axis=0)
    else:
        band = mask[:, max(0, int(axis_x) - 10): int(axis_x) - 1]
        counts = band.sum(axis=1)
    if counts.max() < 4:
        return []
    thresh = 5   # 9px 宽的条带里至少 5px 为墨迹才算一条刻度线（排除文字残迹）
    idx = np.where(counts >= thresh)[0]
    marks = []
    for i in idx:
        if marks and i - marks[-1][-1] <= 2:
            marks[-1].append(i)
        else:
            marks.append([i])
    return [float(np.mean(g)) for g in marks]


def _ocr_ticks_at(page, panel, axis):
    """
    两阶段刻度识别：先找刻度线位置，再对每个刻度开小窗 OCR。
    axis='x'：横轴；axis='y'：纵轴。返回 [(pixel_pos, value)]。
    """
    H, W = page.img.shape[:2]
    mask = _dark_mask(page.img)
    ax_y, ax_x = panel["axis_x_y"], panel["axis_y_x"]
    pw, ph = panel["x1"] - panel["x0"], panel["y1"] - panel["y0"]
    charh = max(8.0, 0.035 * ph)
    out = []
    if axis == "x":
        marks = _find_tick_marks(mask, axis_y=ax_y)
        marks = [m for m in marks if panel["x0"] - 0.02 * pw <= m <= panel["x1"] + 0.02 * pw]
        gaps = np.diff(sorted(marks)) if len(marks) >= 2 else np.array([pw / 8])
        w = float(np.clip(0.9 * np.median(gaps), 20.0, 0.12 * pw))
        for m in marks:
            h = max(30.0, 3.2 * charh)
            res = _ocr_region(page.img, (m - w / 2, ax_y + 2, m + w / 2, ax_y + 2 + h))
            res = [r for r in res if parse_num(r["text"]) is not None]
            if res:
                res.sort(key=lambda r: -r["conf"])
                out.append(dict(px=m, val=parse_num(res[0]["text"]), text=res[0]["text"], from_marks=True))
    else:
        marks = _find_tick_marks(mask, axis_x=ax_x)
        marks = [m for m in marks if panel["y0"] - 0.03 * ph <= m <= panel["y1"] + 0.03 * ph]
        gaps = np.diff(sorted(marks)) if len(marks) >= 2 else np.array([ph / 8])
        h = float(np.clip(0.6 * np.median(gaps), 20.0, 2.8 * charh))
        w = max(16.0, 0.07 * pw)
        for m in marks:
            res = _ocr_region(page.img, (ax_x - w, m - h / 2, ax_x - 2, m + h / 2))
            res = [r for r in res if parse_num(r["text"]) is not None]
            if res:
                res.sort(key=lambda r: -r["conf"])
                out.append(dict(px=m, val=parse_num(res[0]["text"]), text=res[0]["text"], from_marks=True))
    return out


def _ocr_band_split(page, panel, axis):
    """兜底：无刻度线时，整条 OCR，再把纯数字串按已知“等差数列”拆分。
    更稳妥的做法是对横条做列投影分割后再逐块 OCR。"""
    H, W = page.img.shape[:2]
    ax_y, ax_x = panel["axis_x_y"], panel["axis_y_x"]
    pw, ph = panel["x1"] - panel["x0"], panel["y1"] - panel["y0"]
    charh = max(8.0, 0.035 * ph)
    out = []
    if axis == "x":
        band = page.img[int(ax_y) + 2: int(ax_y + 2.2 * charh) + 1,
                        max(0, int(panel["x0"] - 0.03 * pw)): int(panel["x1"] + 0.03 * pw)]
        ox = panel["x0"] - 0.03 * pw
    else:
        band = page.img[max(0, int(panel["y0"] - 0.03 * ph)): int(panel["y1"] + 0.03 * ph) + 1,
                        max(0, int(ax_x - 0.10 * pw)): int(ax_x) - 1]
        ox = ax_x - 0.10 * pw
        oy = panel["y0"] - 0.03 * ph
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw = (gray < 160).astype(np.uint8)
    if axis == "x":
        proj = bw.sum(axis=0)
        oy = ax_y + 2
        vert = True
    else:
        proj = bw.sum(axis=1)
        vert = False
    th = 1 if proj.max() <= 1 else max(1, int(0.15 * proj.max()))
    idx = np.where(proj > th)[0]
    groups = []
    for i in idx:
        if groups and i - groups[-1][-1] <= max(2, 0.004 * len(proj)):
            groups[-1].append(i)
        else:
            groups.append([i])
    for g in groups:
        a, b = g[0], g[-1]
        if vert:
            box = (ox + a - 2, oy, ox + b + 3, oy + band.shape[0])
        else:
            box = (ox, oy + a - 2, ox + band.shape[1], oy + b + 3)
        res = _ocr_region(page.img, box)
        res = [r for r in res if parse_num(r["text"]) is not None]
        if res:
            r = max(res, key=lambda r: r["conf"])
            out.append(dict(px=(r["px"] if not vert else (a + b) / 2 + ox), val=parse_num(r["text"]),
                            text=r["text"]))
            if vert:
                out[-1]["px"] = (a + b) / 2 + ox
            else:
                out[-1]["px"] = (a + b) / 2 + oy
    return out


def _ocr_band_numeric(page, panel, axis):
    """兜底：整条 OCR（3 倍放大），只保留能解析为数字的框。"""
    ax_y, ax_x = panel["axis_x_y"], panel["axis_y_x"]
    pw, ph = panel["x1"] - panel["x0"], panel["y1"] - panel["y0"]
    charh = max(8.0, 0.035 * ph)
    if axis == "x":
        box = (panel["x0"] - 0.03 * pw, ax_y + 2, panel["x1"] + 0.04 * pw, ax_y + 2 + 2.4 * charh)
    else:
        box = (ax_x - 0.11 * pw, panel["y0"] - 0.05 * ph, ax_x - 2, panel["y1"] + 1.2 * charh)
    res = _ocr_region(page.img, box, upscale=3.0)
    out = []
    for r in res:
        v = parse_num(r["text"])
        if v is None:
            continue
        out.append(dict(px=r["px"] if axis == "x" else r["py"], val=v, text=r["text"]))
    return out


def _fit_ticks(ticks):
    px = [t["px"] for t in ticks]
    vals = [t["val"] for t in ticks]
    if len(px) < 3:
        return None
    f = fit_linear(px, vals)
    if f is None:
        return None
    # 用剔除离群点后的保留刻度做等距/单调检验（个别 OCR 错值已被剔除）
    kv = f.get("kept_vals", vals)
    kpx = f.get("kept_px", px)
    srt = sorted(zip(kpx, kv))
    ok_spacing = spacing_consistency(sorted(kv))[0]
    rng = max(max(kv) - min(kv), 1e-9)
    # 值随像素位置单调（y 轴向上增大 / x 轴向右增大）
    mono = all((srt[i + 1][1] >= srt[i][1]) for i in range(len(srt) - 1)) or \
           all((srt[i + 1][1] <= srt[i][1]) for i in range(len(srt) - 1))
    # 等距检验不过时，若拟合残差极小且单调一致仍接受
    f["ok"] = bool(f["ok"] and (ok_spacing or (f["rmse"] <= 0.02 * rng and mono and f["n"] >= 3)))
    return f


def _best_ticks(*cands):
    """多种途径得到的刻度集合里，取能成功拟合且点数最多的；刻度线定位的集合加优先权
    （标记位置比 OCR 文本框中心更精确）。"""
    best, best_score = [], None
    for i, t in enumerate(cands):
        f = _fit_ticks(t)
        if f is None:
            continue
        from_marks = 1.5 if (t and t[0].get("from_marks")) else 0.0
        score = f["n"] + from_marks
        if best_score is None or score > best_score:
            best, best_score = t, score
    return best, (_fit_ticks(best) if best else None)


def calibrate_ocr(page, panel):
    """刻度线定位 + 小窗 OCR 为主；整条数字 OCR、投影分割 OCR 为兜底，取拟合最好的。"""
    xt, fx = _best_ticks(_ocr_ticks_at(page, panel, "x"), _ocr_band_numeric(page, panel, "x"))
    if fx is None:
        xt, fx = _best_ticks(_ocr_band_split(page, panel, "x"))
    yt, fy = _best_ticks(_ocr_ticks_at(page, panel, "y"), _ocr_band_numeric(page, panel, "y"))
    if fy is None:
        yt, fy = _best_ticks(_ocr_band_split(page, panel, "y"))
    ymax = max([t["val"] for t in yt], default=100.0)
    yunit = "fraction" if ymax <= 1.5 else "percent"
    x_cal = AxisCalib(fx, sorted(t["val"] for t in xt), "months", "ocr") if fx else None
    y_cal = AxisCalib(fy, sorted(t["val"] for t in yt), yunit, "ocr") if fy else None
    return x_cal, y_cal, dict(x_ticks=xt, y_ticks=yt)

def ocr_risk_rows(page, panel, xcal, max_rows=6):
    """
    OCR risk-table rows below the x-axis (multi-arm robust):
      1) OCR 条带内所有整数单元格；
      2) 每个单元格采样文字主色 → 色相族（彩色=各臂，灰黑=无色）；
      3) 按 (色相族, y 行聚类) 分组，每组为一条臂的风险行；
      4) 丢弃随时间递增的行（x轴刻度标签混入）。
    返回 [dict(times, counts, y)]，times 已由 xcal 换算为数据单位。
    """
    import cv2 as _cv2
    img = page.img
    H, W = img.shape[:2]
    ax_y, ax_x = panel["axis_x_y"], panel["axis_y_x"]
    pw = panel["x1"] - panel["x0"]
    box = (ax_x - 0.05 * pw, ax_y + 4, ax_x + 1.10 * pw, ax_y + 0.18 * H)
    res = _ocr_region(img, box, upscale=2.0)
    cells = []
    for r in res:
        t = r["text"].strip()
        m = re.match(r"^(\d{1,4})[(（]?", t)
        if not m:
            continue
        xi0, xi1 = int(r["x0"]), int(r["x1"])
        yi0, yi1 = int(r["y0"]), int(r["y1"])
        patch = img[max(0, yi0):yi1 + 1, max(0, xi0):xi1 + 1]
        family, huef = -1, None
        if patch.size:
            hsv = _cv2.cvtColor(patch, _cv2.COLOR_BGR2HSV)
            sat = hsv[..., 1].astype(float) / 255
            sel = sat >= 0.40
            if sel.sum() >= 3:
                huef = float(np.median(hsv[..., 0][sel]))
                family = int(huef // 20)
        cells.append(dict(x0=r["x0"], x1=r["x1"], cy=(r["y0"] + r["y1"]) / 2,
                          val=int(m.group(1)), family=family, hue=huef))
    if len(cells) < 4:
        return []

    # 分组：(色相族, y行) —— 同族内按 y 聚类
    ytol = max(6.0, 0.015 * H)
    groups = []
    for c in cells:
        placed = False
        for g in groups:
            if g["family"] == c["family"] and abs(c["cy"] - g["cy"]) <= ytol:
                g["cells"].append(c)
                g["cy"] = (g["cy"] * (len(g["cells"]) - 1) + c["cy"]) / len(g["cells"])
                placed = True
                break
        if not placed:
            groups.append(dict(family=c["family"], cy=c["cy"], cells=[c]))

    out = []
    for g in groups:
        cells_g = sorted(g["cells"], key=lambda c: c["x0"])
        if len(cells_g) < 3:
            continue
        vals = [c["val"] for c in cells_g]
        if vals[-1] > vals[0]:   # 递增行 = 刻度标签
            continue
        times = [round(xcal.value((c["x0"] + c["x1"]) / 2), 2) for c in cells_g]
        out.append(dict(times=times, counts=vals,
                        y=float(np.mean([c["cy"] for c in cells_g]))))
        if len(out) >= max_rows:
            break
    return out
