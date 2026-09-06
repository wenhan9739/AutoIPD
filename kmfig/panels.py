# -*- coding: utf-8 -*-
"""
kmfig.panels — 在渲染后的页面/图像光栅上检测 KM 图的坐标轴系统（面板）。

思路：坐标轴是“深色、低饱和”的长直线；曲线是彩色的、网格线是浅色的，
先用 HSV 把深灰/黑像素分离出来，再用形态学开运算提取长水平线/长竖线，
最后把“竖轴底端 ≈ 横轴左端”的 L 形配对成一个面板的绘图区。
所有坐标均为该光栅图像的像素坐标。
"""
import numpy as np
import cv2


def dark_lowsat_mask(img_bgr, v_max=0.55, s_max=0.35):
    """深色且低饱和的像素（坐标轴、黑色文字），排除彩色曲线与浅色网格线。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    return ((v < v_max) & (s < s_max)).astype(np.uint8) * 255


def _runs_1d(arr_bool):
    """返回布尔一维数组中连续 True 段的 (start, end_exclusive) 列表。"""
    a = np.concatenate([[0], arr_bool.astype(np.int8), [0]])
    d = np.diff(a)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


def find_lines(mask, min_len_frac=0.06, axis=0):
    """
    axis=0: 找水平线，返回 [(y, x0, x1)]；axis=1: 找竖线，返回 [(x, y0, y1)]。
    先用开运算保留“长度≥min_len”的线段，再逐行/列取 run。
    相邻行的同一条线（线宽>1px）合并成一条。
    """
    h, w = mask.shape
    if axis == 0:
        k_len = max(15, int(w * min_len_frac * 0.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    else:
        k_len = max(15, int(h * min_len_frac * 0.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    lines = []
    if axis == 0:
        min_len = w * min_len_frac
        for y in range(h):
            row = opened[y] > 0
            if row.sum() < min_len:
                continue
            for s, e in _runs_1d(row):
                if e - s >= min_len:
                    lines.append([y, s, e - 1])
    else:
        min_len = h * min_len_frac
        for x in range(w):
            col = opened[:, x] > 0
            if col.sum() < min_len:
                continue
            for s, e in _runs_1d(col):
                if e - s >= min_len:
                    lines.append([x, s, e - 1])
    # 合并相邻（线宽>1px）的同一条线：位置差≤2px 且区间重叠≥80%
    lines.sort(key=lambda l: (l[1], l[0]))
    merged = []
    for ln in lines:
        hit = None
        for m in merged:
            if abs(ln[0] - m[0]) <= 2.5:
                ov = min(m[2], ln[2]) - max(m[1], ln[1])
                L = min(m[2] - m[1], ln[2] - ln[1]) + 1
                if ov >= 0.8 * L:
                    hit = m
                    break
        if hit is not None:
            hit[0] = (hit[0] + ln[0]) / 2.0
            hit[1] = min(hit[1], ln[1])
            hit[2] = max(hit[2], ln[2])
        else:
            merged.append([float(ln[0]), int(ln[1]), int(ln[2])])
    return merged


def _extend_along(nonwhite, pos, start, step, axis, max_gap=3):
    """
    沿 axis 方向从 start 开始按 step(+1/-1) 扫描，只要 pos 行/列（±1）上仍有非白像素就继续，
    允许 ≤max_gap 的小间隙。返回最后一个非白像素的位置。
    axis=0: 沿 x 扫描行 y=pos；axis=1: 沿 y 扫描列 x=pos。
    """
    H, W = nonwhite.shape
    p = int(round(pos))
    cur = int(start)
    last = cur
    gap = 0
    while True:
        nxt = cur + step
        if nxt < 0 or (axis == 0 and nxt >= W) or (axis == 1 and nxt >= H):
            break
        if axis == 0:
            hit = nonwhite[max(0, p - 1):p + 2, nxt].any()
        else:
            hit = nonwhite[nxt, max(0, p - 1):p + 2].any()
        if hit:
            last = nxt
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        cur = nxt
    return last


def detect_panels(img_bgr, min_w_frac=0.08, min_h_frac=0.06, debug=False):
    """
    检测所有 L 形坐标轴系统，返回面板列表：
      dict(x0, y0, x1, y1,  # 绘图区（竖轴 x → 横轴右端，竖轴顶端 → 横轴 y）
           axis_x_y,        # 横轴的 y 像素
           axis_y_x)        # 竖轴的 x 像素
    """
    H, W = img_bgr.shape[:2]
    mask = dark_lowsat_mask(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    nonwhite = gray < 225
    hlines = find_lines(mask, min_len_frac=min_w_frac, axis=0)   # (y, x0, x1)
    vlines = find_lines(mask, min_len_frac=min_h_frac, axis=1)   # (x, y0, y1)
    tol = max(4, int(0.004 * max(H, W)))
    panels = []
    for vx, vy0, vy1 in vlines:
        for hy, hx0, hx1 in hlines:
            # 竖轴底端与横轴在同一高度，且竖轴位于横轴左端附近
            if abs(vy1 - hy) > tol * 2:
                continue
            if not (hx0 - tol * 3 <= vx <= hx0 + tol * 3):
                continue
            # 轴线可能被贴轴的彩色曲线“打断”，沿非白像素延伸到真实端点
            x1 = _extend_along(nonwhite, hy, hx1, +1, axis=0)
            y0 = _extend_along(nonwhite, vx, vy0, -1, axis=1)
            w = x1 - vx
            h = hy - y0
            if w < W * min_w_frac or h < H * min_h_frac:
                continue
            panels.append(dict(x0=float(vx), y0=float(y0), x1=float(x1), y1=float(hy),
                               axis_x_y=float(hy), axis_y_x=float(vx)))
    def _area(p):
        return (p['x1'] - p['x0']) * (p['y1'] - p['y0'])

    def _inter(p, q):
        ix = min(p['x1'], q['x1']) - max(p['x0'], q['x0'])
        iy = min(p['y1'], q['y1']) - max(p['y0'], q['y0'])
        return ix * iy if (ix > 0 and iy > 0) else 0.0

    # 1) 近似重复（同一面板被多条线重复配对）：重叠≥85% 且面积接近，保留一个
    panels.sort(key=lambda p: -_area(p))
    uniq = []
    for p in panels:
        dup = False
        for q in uniq:
            inter = _inter(p, q)
            if inter / min(_area(p), _area(q)) > 0.85 and _area(p) / _area(q) > 0.8:
                dup = True
                break
        if not dup:
            uniq.append(p)
    # 2) 容器（如整幅图的外框）：内部包含 ≥1 个其他面板且自身面积明显更大 → 丢弃
    kept = []
    for p in uniq:
        inside = [q for q in uniq if q is not p and _inter(p, q) / _area(q) > 0.9 and _area(q) < 0.6 * _area(p)]
        if inside:
            continue
        kept.append(p)
    # 阅读顺序排序（先上后下，再左右）
    kept.sort(key=lambda p: (round(p['y0'] / (H * 0.05)), p['x0']))
    if debug:
        return kept, dict(mask=mask, hlines=hlines, vlines=vlines)
    return kept
