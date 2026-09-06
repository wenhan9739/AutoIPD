# -*- coding: utf-8 -*-
"""
kmfig.vector_extract — 从 PDF 矢量路径中精确提取 KM 曲线。

原理：KM 曲线是“彩色描边”的折线路径，删失标记是同色的短小线段，
置信区间是虚线（dashes）。把路径按颜色分组 → 链式重组连续折线 →
最长链即曲线，其余短链是删失标记。全部在像素坐标系下进行。
"""
import numpy as np


def _color_close(c1, c2, tol=0.08):
    if c1 is None or c2 is None:
        return c1 is c2
    return all(abs(a - b) <= tol for a, b in zip(c1, c2))


def _is_grayish(c):
    if c is None:
        return True
    return (max(c) - min(c)) < 0.08


def _segs_bbox_len(segs):
    xs0 = [s[0] for s in segs]; xs1 = [s[2] for s in segs]
    ys0 = [s[1] for s in segs]; ys1 = [s[3] for s in segs]
    L = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in segs)
    return (min(xs0 + xs1), min(ys0 + ys1), max(xs0 + xs1), max(ys0 + ys1)), L


def candidate_curve_colors(page, panel, max_colors=6):
    """在面板内按描边颜色统计折线总长度，返回 [(color, total_len, n_paths)]，灰色/白色除外。"""
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    pad = 0.02 * max(px1 - px0, py1 - py0)
    stat = {}
    for dr in page.drawings:
        if dr["color"] is None:
            continue
        r = dr["rect"]
        if r is None:
            continue
        # 与面板相交（略放宽）
        if r[2] < px0 - pad or r[0] > px1 + pad or r[3] < py0 - pad or r[1] > py1 + pad:
            continue
        c = tuple(round(v, 3) for v in dr["color"])
        if _is_grayish(c) or min(c) > 0.92:   # 灰/黑/白：坐标轴、网格、文字
            continue
        L = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in dr["segs"])
        if L <= 0:
            continue
        key = c
        d = stat.setdefault(key, [0.0, 0])
        d[0] += L
        d[1] += 1
    out = sorted(((c, v[0], v[1]) for c, v in stat.items()), key=lambda t: -t[1])
    return out[:max_colors]


def _collect_segments(page, panel, color):
    """
    收集面板内该颜色的全部直线段，返回 (solid_segs, dashed_segs)。
    实线段按“所属路径总长度”降序排列——链式重组从长路径（曲线本体）开始，
    避免从删失标记出发误把曲线切成两段。
    """
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    pad = 0.03 * max(px1 - px0, py1 - py0)
    solid_groups, dashed = [], []
    for dr in page.drawings:
        if dr["color"] is None or not _color_close(tuple(round(v, 3) for v in dr["color"]), color):
            continue
        grp = []
        for s in dr["segs"]:
            mx, my = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
            if px0 - pad <= mx <= px1 + pad and py0 - pad <= my <= py1 + pad:
                (dashed if dr["dashed"] else grp).append(s)
        if grp:
            L = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in grp)
            solid_groups.append((L, grp))
    solid_groups.sort(key=lambda t: -t[0])
    solid = [s for _, grp in solid_groups for s in grp]
    return solid, dashed


def _remove_spurs(pts, tol=0.75):
    """去掉“走出去又原路返回”的毛刺顶点（删失标记并入链时产生）。"""
    changed = True
    pts = list(pts)
    while changed and len(pts) >= 3:
        changed = False
        out = [pts[0]]
        k = 1
        while k < len(pts):
            if k + 1 < len(pts) and np.hypot(pts[k + 1][0] - out[-1][0], pts[k + 1][1] - out[-1][1]) <= tol:
                # out[-1] -> pts[k] -> pts[k+1]≈out[-1]：跳过毛刺
                k += 2
                changed = True
                continue
            out.append(pts[k])
            k += 1
        pts = out
    return pts


def _trim_terminal_ticks(pts, top_y, max_tick_px):
    """
    修剪链两端并入的删失标记：
      末端：最后一段竖直向上（像素 y 减小，即生存率“上升”）且长度不超过删失标记高度 → 删掉；
      起点：第一段竖直、且起点高于绘图区顶端（100% 线之上）→ 删掉。
    """
    pts = list(pts)
    while len(pts) >= 2 and np.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]) < 1e-6:
        pts.pop()
    while len(pts) >= 2:
        a, b = pts[-2], pts[-1]
        if abs(b[0] - a[0]) < 0.75 and (a[1] - b[1]) > 0.5 and (a[1] - b[1]) <= max_tick_px:
            pts.pop()
        else:
            break
    while len(pts) >= 2:
        a, b = pts[0], pts[1]
        if abs(b[0] - a[0]) < 0.75 and a[1] < top_y - 1.0 and (b[1] - a[1]) <= max_tick_px:
            pts.pop(0)
        else:
            break
    return pts


def _merge_fragments(main, others, panel, join_tol, mono_tol):
    """
    把与主链端点相接（且不在 x 上重叠）的碎链拼到主链上，修复被删失标记切断的曲线。
    join_tol：端点距离容差（断口常见于删失标记处）；mono_tol：拼接处允许的“向上”偏移，
    拼接后仍需满足 KM 单调性（生存率不升，即像素 y 不减小超过 mono_tol）。
    """
    main_pts = list(main["pts"])
    if main_pts[0][0] > main_pts[-1][0]:
        main_pts = main_pts[::-1]
    pool = [c for c in others]
    merged_any = True
    while merged_any and pool:
        merged_any = False
        left, right = main_pts[0], main_pts[-1]
        for c in list(pool):
            pts = list(c["pts"])
            if pts[0][0] > pts[-1][0]:
                pts = pts[::-1]
            # 接在右端：碎片起点接近主链右端，且拼接处生存率不上升
            if np.hypot(pts[0][0] - right[0], pts[0][1] - right[1]) <= join_tol \
                    and pts[0][1] >= right[1] - mono_tol \
                    and pts[-1][0] >= right[0] - join_tol:
                main_pts = main_pts + pts[1:]
                pool.remove(c); merged_any = True; break
            # 接在左端
            if np.hypot(pts[-1][0] - left[0], pts[-1][1] - left[1]) <= join_tol \
                    and pts[-1][1] <= left[1] + mono_tol \
                    and pts[0][0] <= left[0] + join_tol:
                main_pts = pts[:-1] + main_pts
                pool.remove(c); merged_any = True; break
    return main_pts, pool


def extract_curve_vector(page, panel, color):
    """
    提取指定颜色的一条 KM 曲线。
    返回 dict(points=[(x, y)]（沿曲线顺序的像素点列）,
              censors=[x_px...]（删失标记位置）, n_censors, length,
              n_chains, n_dashed_segs, n_merged)
    选链规则：链在面板内的长度最长者为曲线主链，端点相接的碎链拼回主链；
    同色短链视为删失标记；虚线段不参与曲线选择（多为置信区间）。
    """
    solid, dashed = _collect_segments(page, panel, color)
    if not solid and not dashed:
        return None
    chains = _chain_segments(solid) if solid else []
    if not chains:
        chains = _chain_segments(dashed)   # 曲线本身是虚线的少见情形
        if not chains:
            return None
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    W, H = px1 - px0, py1 - py0
    for c in chains:
        c["inside"] = _inside_len(c["pts"], panel)
        xs = [p[0] for p in c["pts"]]
        c["xspan"] = max(xs) - min(xs)
    cands = [c for c in chains if c["inside"] >= 0.5 * c["length"] and c["xspan"] >= 0.03 * W]
    # 排除贴着面板边框走的链（图框、坐标轴盒），防止被当成曲线
    def _border_frac(c):
        on_border = 0.0
        for a, b in zip(c["pts"][:-1], c["pts"][1:]):
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            near = (abs(mx - px0) <= 4 or abs(mx - px1) <= 4 or abs(my - py0) <= 4 or abs(my - py1) <= 4)
            if near:
                on_border += np.hypot(b[0] - a[0], b[1] - a[1])
        return on_border / max(c["length"], 1e-9)
    cands = [c for c in cands if _border_frac(c) < 0.7]
    if not cands:
        return None
    cands.sort(key=lambda c: -c["inside"])
    main = cands[0]
    join_tol = max(8.0, 0.025 * W)
    main_pts, leftover = _merge_fragments(main, cands[1:], panel, join_tol, mono_tol=0.04 * H)
    main_pts = _remove_spurs(main_pts)
    main_pts = _trim_terminal_ticks(main_pts, top_y=py0, max_tick_px=0.08 * H + 4)
    n_merged = len(cands) - 1 - len(leftover)
    # 删失标记：短链（长度小于面板高度的 6%，点数少）
    censor_xs = []
    for c in chains:
        if c is main or c in cands[1:] and c not in leftover:
            continue
        if c["length"] < 0.06 * H + 4 and len(c["pts"]) <= 6:
            censor_xs.append(float(np.mean([p[0] for p in c["pts"]])))
    censor_xs.sort()
    length = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(main_pts[:-1], main_pts[1:]))
    return dict(points=main_pts, censors=censor_xs, n_censors=len(censor_xs),
                length=length, n_chains=len(chains), n_dashed_segs=len(dashed), n_merged=n_merged)


def _inside_len(pts, panel):
    """折线在面板严格范围内的长度（用于给链打分，排除面板外的表格横线等）。"""
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    L = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if px0 - 1 <= mx <= px1 + 1 and py0 - 1 <= my <= py1 + 1:
            L += np.hypot(b[0] - a[0], b[1] - a[1])
    return L


def _chain_segments(segs, eps=1.0):
    """
    把线段按端点相接链成折线；返回 [dict(pts=[(x,y)...], length)]，按长度降序。
    分叉点（例如删失标记恰好画在曲线顶点上）优先选择“远端仍有后继”的线段，避免走进死胡同。
    """
    # 去掉零长线段与完全重复的线段
    seen = set()
    clean = []
    for s in segs:
        if np.hypot(s[2] - s[0], s[3] - s[1]) < 1e-6:
            continue
        k = (round(s[0], 2), round(s[1], 2), round(s[2], 2), round(s[3], 2))
        k2 = (k[2], k[3], k[0], k[1])
        if k in seen or k2 in seen:
            continue
        seen.add(k)
        clean.append(s)
    segs = clean

    def key(p):
        return (round(p[0] / eps), round(p[1] / eps))

    adj = {}
    for i, s in enumerate(segs):
        adj.setdefault(key((s[0], s[1])), []).append(i)
        adj.setdefault(key((s[2], s[3])), []).append(i)

    def far_end(j, p):
        s = segs[j]
        return (s[2], s[3]) if np.hypot(s[0] - p[0], s[1] - p[1]) <= np.hypot(s[2] - p[0], s[3] - p[1]) else (s[0], s[1])

    def degree(p):
        return len(adj.get(key(p), []))

    used = set()
    chains = []
    for i in range(len(segs)):
        if i in used:
            continue
        used.add(i)
        s = segs[i]
        line = [(s[0], s[1]), (s[2], s[3])]
        for direction in (0, 1):
            while True:
                p = line[-1] if direction == 0 else line[0]
                cands = [j for j in adj.get(key(p), []) if j not in used]
                if not cands:
                    break
                # 优先：远端还有后继的线段（度>=2）；其次：更长的线段
                def score(j):
                    fe = far_end(j, p)
                    return (1 if degree(fe) >= 2 else 0, np.hypot(segs[j][2] - segs[j][0], segs[j][3] - segs[j][1]))
                j = max(cands, key=score)
                used.add(j)
                other = far_end(j, p)
                if direction == 0:
                    line.append(other)
                else:
                    line.insert(0, other)
        length = sum(np.hypot(line[k + 1][0] - line[k][0], line[k + 1][1] - line[k][1])
                     for k in range(len(line) - 1))
        chains.append(dict(pts=line, length=length))
    chains.sort(key=lambda c: -c["length"])
    return chains


def _group_colored_spans(spans, color, region, tol=0.10):
    """
    收集区域内与曲线同色的非数字文字 span，并把上下相邻、左对齐的 span 合并成一条标签
    （如风险表里的 "Pembrolizumab" / "+ chemo" 两行）。返回文本列表。
    """
    x0, y0, x1, y1 = region
    items = []
    for sp in spans:
        if sp["color"] is None or not _color_close(sp["color"], color, tol=tol) or sp["vertical"]:
            continue
        t = sp["text"].strip()
        if len(t) < 2 or parse_num_safe(t) is not None:
            continue
        cx, cy = (sp["x0"] + sp["x1"]) / 2, (sp["y0"] + sp["y1"]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            items.append(dict(x0=sp["x0"], y0=sp["y0"], x1=sp["x1"], y1=sp["y1"], text=t, size=sp["size"] or (sp["y1"] - sp["y0"])))
    items.sort(key=lambda s: (round(s["x0"] / 4), s["y0"]))
    groups = []
    for it in items:
        if groups:
            g = groups[-1]
            same_col = abs(it["x0"] - g["x0"]) <= 0.6 * it["size"]
            below = 0 <= it["y0"] - g["y1"] <= 0.9 * it["size"]
            if same_col and below:
                g["text"] = (g["text"] + " " + it["text"]).replace("  ", " ")
                g["y1"] = it["y1"]
                continue
        groups.append(dict(it))
    return [g["text"] for g in groups]


def label_arms_by_text(page, panel, colors):
    """
    用与曲线同色的文字推测各臂的标签（图例/风险表行名通常与曲线同色）。
    若图例文字是黑色（Lancet 风格：彩色线样 + 黑色文字），则回退到
    “同色图例线样 + 右侧最近文字”匹配。返回 {color: label}。
    """
    from collections import Counter
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    w, h = px1 - px0, py1 - py0
    region = (px0 - 0.6 * w, py0 - 0.45 * h, px1 + 0.6 * w, py1 + 1.2 * h)
    out = {}
    for color in colors:
        if color is None:
            continue
        texts = _group_colored_spans(page.spans, color, region)
        if texts:
            cnt = Counter(texts)
            best = sorted(cnt.items(), key=lambda kv: (-kv[1], -len(kv[0])))[0][0]
            out[color] = best
    return out


def _label_by_legend_swatches(page, panel, colors, region):
    """
    找每个颜色的“图例线样”（水平、长度 8~120px 的同色线段，且其端点附近
    没有其他同色线段相连——曲线平台是连通的，线样是孤立的），
    取其右侧最近的文字 span 作为标签。
    """
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    out = {}
    for color in colors:
        cand_segs = []
        for dr in page.drawings:
            if dr["color"] is None or not _color_close(tuple(round(v, 3) for v in dr["color"]), color, 0.06):
                continue
            for s in dr["segs"]:
                x0_, y0_, x1_, y1_ = s
                if abs(y1_ - y0_) > 1.5:
                    continue
                L = abs(x1_ - x0_)
                if not (8 <= L <= 120):
                    continue
                mx, my = (x0_ + x1_) / 2, (y0_ + y1_) / 2
                if not (region[0] <= mx <= region[2] and region[1] <= my <= region[3]):
                    continue
                cand_segs.append((x0_, y0_, x1_, y1_))
        # 隔离性：端点 12px 内没有其他同色线段端点
        def isolated(seg):
            x0_, y0_, x1_, y1_ = seg
            for other in cand_segs:
                if other is seg:
                    continue
                for ex, ey in ((x0_, y0_), (x1_, y1_)):
                    for ox, oy in ((other[0], other[1]), (other[2], other[3])):
                        if abs(ex - ox) < 12 and abs(ey - oy) < 12:
                            return False
            return True
        swatches = [( (s[0]+s[2])/2, (s[1]+s[3])/2 ) for s in cand_segs if isolated(s)]
        best = None
        for sx, sy in swatches:
            # 同一行的候选文字（数字占比高的统计注释文本跳过）
            line = []
            for sp in page.spans:
                t = sp["text"].strip()
                if len(t) < 2 or parse_num_safe(t) is not None:
                    continue
                if sum(ch.isdigit() for ch in t) > 0.3 * len(t):
                    continue
                cx, cy = (sp["x0"] + sp["x1"]) / 2, (sp["y0"] + sp["y1"]) / 2
                if abs(cy - sy) > 0.02 * (px1 - px0) + 14:
                    continue
                if not (sx - 6 <= cx <= sx + 0.6 * (px1 - px0)):
                    continue
                line.append((sp["x0"], sp["x1"], t))
            if not line:
                continue
            line.sort()
            # 拼接同一行连续文本（间隔 ≤ 60px）成完整标签
            label = line[0][2]
            last_x1 = line[0][1]
            for x0_, x1_, t in line[1:]:
                if x0_ - last_x1 <= 60:
                    label += " " + t
                    last_x1 = x1_
                else:
                    break
            d = line[0][0] - sx
            if best is None or d < best[0]:
                best = (d, label)
        if best:
            out[color] = best[1]
    return out


def parse_num_safe(s):
    from .calibrate import parse_num
    return parse_num(s)
