# -*- coding: utf-8 -*-
"""
kmfig.raster_extract — 位图 KM 曲线提取（矢量不可用时的兜底）。

流程：绘图区裁剪 → HSV 饱和像素 → 色相峰值分臂 → 同色连通域并集（剔除过小碎片）
→ 逐列“跟踪”曲线（每列选离上一列最近、不向上跳的像素段）→ 删失标记短暂下探回填
→ 阶梯化 → 像素折线。利用 KM 先验：曲线从 (t=0, S=1) 出发、单调不升。
"""
import numpy as np
import cv2


def _hue_clusters(hues, weights=None, min_sep=14, rel_min=0.03, min_count=40):
    """
    色相直方图（0~180，环形）的峰值：平滑后取局部极大值，
    保留高度 ≥ rel_min×最高峰 且总像素数 ≥ min_count 的峰，按高度贪心去除间距 <min_sep 的峰。
    """
    hist, _ = np.histogram(hues, bins=180, range=(0, 180), weights=weights)
    if hist.sum() <= 0:
        return []
    # 环形平滑
    pad = 6
    ext = np.concatenate([hist[-pad:], hist, hist[:pad]]).astype(np.float32)
    sm = cv2.GaussianBlur(ext.reshape(1, -1), (0, 0), 2.5).ravel()[pad:-pad]
    mx = sm.max()
    if mx <= 0:
        return []
    cands = []
    for h in range(180):
        l, r = sm[(h - 1) % 180], sm[(h + 1) % 180]
        if sm[h] >= l and sm[h] >= r and sm[h] >= rel_min * mx:
            # 峰附近 ±6 bin 的像素总数
            idx = [(h + d) % 180 for d in range(-6, 7)]
            if hist[idx].sum() >= min_count:
                cands.append((sm[h], h))
    cands.sort(reverse=True)
    peaks = []
    for _, h in cands:
        if all(min(abs(h - p), 180 - abs(h - p)) >= min_sep for p in peaks):
            peaks.append(int(h))
    return peaks


def _estimate_thickness(mask, max_frac_h=0.05):
    """线宽估计：各采样列中像素段高度的中位数。"""
    heights = []
    h, w = mask.shape
    for x in range(0, w, max(1, w // 200)):
        for top, bot in _runs_in_column(mask[:, x]):
            heights.append(bot - top + 1)
    if not heights:
        return 2.0
    t = float(np.median(heights))
    return max(1.5, min(t, max_frac_h * h))


def _union_components(mask, min_w, min_h, thickness, dilate=2):
    """
    保留宽度≥min_w 或高度≥min_h 的连通域，剔除“文字状”连通域
    （高度 ≥4 倍线宽、填充率高、宽度有限——图例/标注文字），返回并集掩膜。
    """
    m = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8)) if dilate > 1 else mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(mask)
    W = mask.shape[1]
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < min_w and bh < min_h:
            continue
        comp = ((lab == i) & (mask > 0))
        a = int(comp.sum())
        fill = a / max(1.0, float(bw * bh))
        textlike = (bh >= 4 * thickness) and (fill >= 0.28) and (bw < 0.35 * W)
        if textlike:
            continue
        keep |= comp.astype(np.uint8)
    return keep


def _runs_in_column(col):
    """列内连续像素段 [(top, bottom)]。"""
    idx = np.where(col > 0)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(r[0]), int(r[-1])) for r in np.split(idx, splits)]


def _track(mask, start_y, max_jump, thickness):
    """
    逐列跟踪：返回每列的曲线 y（NaN 表示该列无像素/丢失）。
    规则：候选段中心离上一列 y 最近者优先；禁止向上移动（KM 生存率不会上升）；
          向下跳变超过 max_jump（丢失时最多放宽到 1.5 倍）视为丢失。
          竖直落差段（高度大）取其底部为新水平。
    """
    h, w = mask.shape
    ys = np.full(w, np.nan)
    y_prev = float(start_y)
    lost = 0
    for x in range(w):
        runs = _runs_in_column(mask[:, x])
        if not runs:
            lost += 1
            continue
        best, best_cost = None, None
        jump_limit = max_jump * (1 + 0.05 * min(lost, 10))
        for top, bot in runs:
            c = (top + bot) / 2.0
            tall = (bot - top + 1) > 2.5 * thickness
            # 竖直落差：曲线到达底部；否则线段中心
            y_c = bot - thickness / 2.0 if tall and top <= y_prev + thickness else c
            dy = y_c - y_prev
            if dy < -2.0 * thickness:      # KM 不允许向上（向上即删失标记/文字干扰）
                continue
            if abs(dy) > jump_limit:
                continue
            cost = abs(dy)
            if best_cost is None or cost < best_cost:
                best, best_cost = y_c, cost
        if best is None:
            lost += 1
            continue
        ys[x] = best
        y_prev = best
        lost = 0
    return ys


def _fill_and_fix_ticks(ys, thickness, max_tick_w):
    """
    前向填充缺失列（仅限内部缺口；在最后一个真实像素列截断，避免把平台无限外推）；
    把“短暂下探又回到原水平”的删失标记（宽度≤max_tick_w）拉回原水平；
    然后强制单调不升（像素 y 不减小）。
    """
    w = len(ys)
    valid_idx = np.where(~np.isnan(ys))[0]
    if len(valid_idx) < 2:
        return ys
    last_valid = int(valid_idx[-1])
    f = ys.copy()
    last = np.nan
    for x in range(last_valid + 1):        # 只填充到最后一个真实像素列
        if np.isnan(f[x]):
            f[x] = last
        else:
            last = f[x]
    f[last_valid + 1:] = np.nan            # 截断外推
    x0 = int(valid_idx[0])
    # 删失标记回填
    k = x0 + 1
    while k < last_valid:
        if f[k] > f[k - 1] + thickness:            # 突然下探
            base = f[k - 1]
            j = k
            while j < last_valid and f[j] > base + thickness and j - k <= max_tick_w:
                j += 1
            if j < last_valid and abs(f[j] - base) <= thickness:   # 回到原水平：是标记，不是台阶
                f[k:j] = base
                k = j
                continue
        k += 1
    # 单调化
    cur = f[x0]
    for x in range(x0, last_valid + 1):
        if f[x] < cur - 0.5:       # 向上（像素 y 变小）—— 钳制
            f[x] = cur
        cur = f[x]
    return f


def _steps_from_trace(ys, y_tol):
    """列轨迹 → 阶梯顶点像素折线（水平段+竖直段）。"""
    valid = ~np.isnan(ys)
    if valid.sum() < 2:
        return None
    xs = np.where(valid)[0]
    pts = [(float(xs[0]), float(ys[xs[0]]))]
    cur_y = ys[xs[0]]
    for x in xs[1:]:
        y = ys[x]
        if abs(y - cur_y) > y_tol:
            pts.append((float(x), float(cur_y)))
            pts.append((float(x), float(y)))
            cur_y = y
    pts.append((float(xs[-1]), float(cur_y)))
    return pts


def extract_curves_raster(img_bgr, panel, s_min=0.30, v_min=0.20, inset=3, origin_px=None,
                          dilate=2, min_w_frac=0.02, min_h_frac=0.02, apply_decorations=True,
                          exclude_hues=()):
    """
    返回 [dict(color=(r,g,b), points=[(x,y)像素折线], x_span_frac, thickness, ...)]，按 x 跨度降序。
    origin_px=(x_of_t0, y_of_S1)：若给出，曲线起点补到该原点（KM 先验）。
    dilate/min_w_frac/min_h_frac：放宽后可提取点线（dotted）曲线；
    exclude_hues：跳过与已有臂色相相近的峰（用于分轮次提取）。
    """
    x0, y0, x1, y1 = [int(round(panel[k])) for k in ("x0", "y0", "x1", "y1")]
    crop = img_bgr[y0 + inset:y1 - inset, x0 + inset:x1 - inset]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    Hh = hsv[..., 0].astype(np.int32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0
    sat = (S >= s_min) & (V >= v_min)
    if sat.sum() < 50:
        return []
    peaks = _hue_clusters(Hh[sat])
    W, Hgt = x1 - x0, y1 - y0
    arms = []
    for pk in peaks:
        if any(min(abs(pk - eh), 180 - abs(pk - eh)) <= 10 for eh in exclude_hues):
            continue
        dh = np.minimum(np.abs(Hh - pk), 180 - np.abs(Hh - pk))
        m = (sat & (dh <= 12)).astype(np.uint8)
        if m.sum() < 30:
            continue
        thickness = _estimate_thickness(m)
        m = _union_components(m, min_w=min_w_frac * W, min_h=min_h_frac * Hgt, thickness=thickness,
                              dilate=dilate)
        if m.sum() < 30:
            continue
        start_y = (origin_px[1] - (y0 + inset)) if origin_px else 0.0
        ys_raw = _track(m, start_y=start_y, max_jump=0.35 * Hgt, thickness=thickness)
        ys = _fill_and_fix_ticks(ys_raw, thickness, max_tick_w=max(3, int(0.012 * W)))
        pts = _steps_from_trace(ys, y_tol=max(1.0, 0.6 * thickness))
        if pts is None:
            continue
        valid = np.where(~np.isnan(ys_raw))[0]
        coverage = len(valid) / max(1, (valid[-1] - valid[0] + 1)) if len(valid) else 0.0
        pts = [(x + x0 + inset, y + y0 + inset) for x, y in pts]
        span = pts[-1][0] - pts[0][0]
        if span < 0.15 * W:
            continue
        # KM 先验：曲线从 (0, 1) 出发；若起点在左上角附近则补上原点
        if origin_px is not None:
            ox, oy = origin_px
            if pts[0][0] - ox <= 0.12 * W and pts[0][1] - oy <= 0.12 * Hgt:
                first_y = pts[0][1]
                pts = [(ox, oy), (pts[0][0], oy)] + ([(pts[0][0], first_y)] if abs(first_y - oy) > 0.6 * thickness else []) + pts[1:]
        mean_bgr = crop[m > 0].mean(axis=0) / 255.0
        color = (round(float(mean_bgr[2]), 3), round(float(mean_bgr[1]), 3), round(float(mean_bgr[0]), 3))
        arms.append(dict(color=color, points=pts, x_span_frac=span / W, thickness=thickness,
                         coverage=coverage, ys_raw=ys_raw, ys=ys, n_px=int(m.sum())))
    if apply_decorations:
        arms = _drop_decorations(arms)
    _annotate_overlap(arms)
    for a in arms:
        a.pop("ys_raw", None); a.pop("ys", None)
    arms.sort(key=lambda a: -a["x_span_frac"])
    return arms


def _drop_decorations(arms):
    """
    剔除“装饰臂”：像素覆盖率低（稀疏，如删失标记/离散点）且其真实像素列 ≥70% 落在
    另一臂轨迹的 2 倍线宽范围内。两臂都稠密时都保留。
    """
    keep = []
    for i, a in enumerate(arms):
        drop = False
        if a["coverage"] < 0.5:
            for j, b in enumerate(arms):
                if i == j:
                    continue
                cols = np.where(~np.isnan(a["ys_raw"]) & ~np.isnan(b["ys"]))[0]
                if len(cols) < 10:
                    continue
                close = np.abs(a["ys_raw"][cols] - b["ys"][cols]) <= 2.0 * max(a["thickness"], b["thickness"])
                if close.mean() >= 0.7 and b["n_px"] > a["n_px"]:
                    drop = True
                    break
        if not drop:
            keep.append(a)
    return keep


def _annotate_overlap(arms):
    """每臂与其他臂轨迹距离 ≤2 倍线宽的列占比（重叠遮挡会带来 ~半线宽的偏差，需在 QA 中提示）。"""
    for i, a in enumerate(arms):
        worst = 0.0
        for j, b in enumerate(arms):
            if i == j:
                continue
            cols = np.where(~np.isnan(a["ys"]) & ~np.isnan(b["ys"]))[0]
            if len(cols) < 5:
                continue
            frac = float((np.abs(a["ys"][cols] - b["ys"][cols]) <= 2.0 * max(a["thickness"], b["thickness"])).mean())
            worst = max(worst, frac)
        a["overlap_frac"] = round(worst, 3)


def extract_monochrome_arm(img_bgr, panel, origin_px=None, inset=3):
    """
    黑白 KM 图的黑色实线臂：取绘图区内近黑像素，剔除“字符状”连通域
    （中小高度、中小宽度——图内注释文字/数字），保留曲线平台、竖直落差与刻度线，
    再逐列跟踪（禁止向上跳，天然避开位于曲线上方的参考虚线与图例）。
    返回与 extract_curves_raster 同构的 list。
    """
    from .panels import dark_lowsat_mask
    x0, y0, x1, y1 = [int(round(panel[k])) for k in ("x0", "y0", "x1", "y1")]
    crop = img_bgr[y0 + inset:y1 - inset, x0 + inset:x1 - inset]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0
    m = ((V < 0.34) & (S < 0.35)).astype(np.uint8)
    if m.sum() < 100:
        return []
    W, Hgt = x1 - x0, y1 - y0
    charh = max(10.0, 0.035 * Hgt)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # 字符状：高度约一个字高、宽度有限、填充率不低 → 注释文字/数字
        if (0.5 * charh <= bh <= 2.2 * charh) and bw <= 3 * charh and area / max(1, bw * bh) >= 0.18:
            continue
        keep |= ((lab == i) & (m > 0)).astype(np.uint8)
    if keep.sum() < 100:
        return []
    thickness = _estimate_thickness(keep)
    start_y = (origin_px[1] - (y0 + inset)) if origin_px else 0.0
    ys_raw = _track(keep, start_y=start_y, max_jump=0.35 * Hgt, thickness=thickness)
    ys = _fill_and_fix_ticks(ys_raw, thickness, max_tick_w=max(3, int(0.012 * W)))
    pts = _steps_from_trace(ys, y_tol=max(1.0, 0.6 * thickness))
    if pts is None:
        return []
    valid = np.where(~np.isnan(ys_raw))[0]
    coverage = len(valid) / max(1, valid[-1] - valid[0] + 1)
    pts = [(x + x0 + inset, y + y0 + inset) for x, y in pts]
    span = pts[-1][0] - pts[0][0]
    if span < 0.4 * W:
        return []
    if origin_px is not None:
        ox, oy = origin_px
        if pts[0][0] - ox <= 0.12 * W and pts[0][1] - oy <= 0.12 * Hgt:
            first_y = pts[0][1]
            pts = [(ox, oy), (pts[0][0], oy)] + ([(pts[0][0], first_y)] if abs(first_y - oy) > 0.6 * thickness else []) + pts[1:]
    return [dict(color=(0.08, 0.08, 0.08), points=pts, x_span_frac=span / W, thickness=thickness,
                 coverage=coverage, ys_raw=ys_raw, ys=ys, n_px=int(keep.sum()))]
