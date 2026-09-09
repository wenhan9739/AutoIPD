# -*- coding: utf-8 -*-
"""Raster curve extraction v2: component threading + risk-table consistency scoring."""

import numpy as np
import cv2


# ---------------------------------------------------------------- 基础工具

def _runs_in_column(col):
    idx = np.where(col > 0)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(r[0]), int(r[-1])) for r in np.split(idx, splits)]


def _estimate_thickness(mask, max_frac_h=0.05):
    heights = []
    h, w = mask.shape
    for x in range(0, w, max(1, w // 200)):
        for top, bot in _runs_in_column(mask[:, x]):
            heights.append(bot - top + 1)
    if not heights:
        return 2.0
    return max(1.5, min(float(np.median(heights)), max_frac_h * h))


def _hue_clusters(hues, weights=None, min_sep=14, rel_min=0.02, min_count=30):
    """Local maxima of the circular hue histogram (0-180, smoothed)."""
    hist, _ = np.histogram(hues, bins=180, range=(0, 180), weights=weights)
    if hist.sum() <= 0:
        return []
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
            idx = [(h + d) % 180 for d in range(-6, 7)]
            if hist[idx].sum() >= min_count:
                cands.append((sm[h], h))
    cands.sort(reverse=True)
    peaks = []
    for _, h in cands:
        if all(min(abs(h - p), 180 - abs(h - p)) >= min_sep for p in peaks):
            peaks.append(int(h))
    return peaks


def _track(mask, start_y, max_jump, thickness):
    """Column-by-column tracking: nearest-to-previous, strictly no upward movement."""
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
            y_c = bot - thickness / 2.0 if tall and top <= y_prev + thickness else c
            dy = y_c - y_prev
            if dy < -2.0 * thickness:
                continue
            if abs(dy) > jump_limit:
                continue
            if best_cost is None or abs(dy) < best_cost:
                best, best_cost = y_c, abs(dy)
        if best is None:
            lost += 1
            continue
        ys[x] = best
        y_prev = best
        lost = 0
    return ys


def _fill_and_fix_ticks(ys, thickness, max_tick_w):
    """Forward-fill internal gaps; back-fill censor-mark dips; enforce monotonicity; truncate extrapolation."""
    w = len(ys)
    valid_idx = np.where(~np.isnan(ys))[0]
    if len(valid_idx) < 2:
        return ys
    last_valid = int(valid_idx[-1])
    f = ys.copy()
    last = np.nan
    for x in range(last_valid + 1):
        if np.isnan(f[x]):
            f[x] = last
        else:
            last = f[x]
    f[last_valid + 1:] = np.nan
    x0 = int(valid_idx[0])
    k = x0 + 1
    while k < last_valid:
        if f[k] > f[k - 1] + thickness:
            base = f[k - 1]
            j = k
            while j < last_valid and f[j] > base + thickness and j - k <= max_tick_w:
                j += 1
            if j < last_valid and abs(f[j] - base) <= thickness:
                f[k:j] = base
                k = j
                continue
        k += 1
    cur = f[x0]
    for x in range(x0, last_valid + 1):
        if f[x] < cur - 0.5:
            f[x] = cur
        cur = f[x]
    return f


def _steps_from_trace(ys, y_tol):
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


# ---------------------------------------------------------------- v2 核心

def _hue_mask(hsv_h, sat, val, peak, dh_tol=12):
    dh = np.minimum(np.abs(hsv_h - peak), 180 - np.abs(hsv_h - peak))
    return (sat >= 0) & (val >= 0) & (dh <= dh_tol)  # 占位：由调用方与饱和度合成


def _components(mask, min_w, min_h, top=None, drop_textlike=True):
    m = cv2.dilate(mask, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < min_w and bh < min_h:
            continue
        comp = ((lab == i) & (mask > 0)).astype(np.uint8)
        if comp.sum() < 8:
            continue
        # 文字状：高度在文字范围（≥8px）且宽度有限——点/线碎片（bh<8）不算文字
        if drop_textlike and 8 <= bh <= 40 and bw <= 60 and area / max(1.0, float(bw * bh)) >= 0.18:
            continue
        comps.append(dict(mask=comp, x=int(x), y=int(y), w=int(bw), h=int(bh),
                          area=int(comp.sum())))
    comps.sort(key=lambda c: -c["w"])
    if top:
        comps = comps[:top]
    return comps


def _chain_components(comps, thickness, H, W, origin_y):
    """Track each component and chain into threads at junctions. Returns list of thread dicts."""
    threads = []
    for c in comps:
        ys = _track(c["mask"], start_y=origin_y, max_jump=0.35 * H, thickness=thickness)
        ys = _fill_and_fix_ticks(ys, thickness, max(3, int(0.012 * W)))
        valid = np.where(~np.isnan(ys))[0]
        if len(valid) < 5:
            c["valid"] = False
            continue
        c["ys"] = ys
        c["x0"] = int(valid[0]); c["x1"] = int(valid[-1])
        c["y0"] = float(ys[c["x0"]]); c["y1"] = float(ys[c["x1"]])
        c["valid"] = True
    comps = [c for c in comps if c.get("valid")]
    remaining = set(range(len(comps)))
    threads = []
    while remaining:
        cur_i = min(remaining, key=lambda i: comps[i]["x0"])
        remaining.discard(cur_i)
        thread = [comps[cur_i]]
        cur = comps[cur_i]
        while True:
            best_i, best_cost = None, None
            for ci in remaining:
                c = comps[ci]
                gap = c["x0"] - cur["x1"]
                if -0.06 * W <= gap <= 0.22 * W:
                    dy = abs(c["y0"] - cur["y1"])
                    if dy <= 0.22 * H:
                        cost = gap + dy * 0.5
                        if best_cost is None or cost < best_cost:
                            best_i, best_cost = ci, cost
            if best_i is None:
                break
            remaining.discard(best_i)
            thread.append(comps[best_i])
            cur = comps[best_i]
        merged = np.full(W, np.nan)
        for c in thread:
            merged[c["x0"]:c["x1"] + 1] = c["ys"][c["x0"]:c["x1"] + 1]
        valid = np.where(~np.isnan(merged))[0]
        span = int(valid[-1] - valid[0] + 1) if len(valid) else 0
        if span < 0.20 * W:
            continue
        valid = np.where(~np.isnan(merged))[0]
        if len(valid) < 5:
            continue
        threads.append(dict(comps=thread, merged=merged,
                            x0=int(valid[0]), x1=int(valid[-1]), span=span,
                            thickness=thickness))
    return threads


def _verify_drops(merged, mask_2d, thickness, W):
    """后验落差验证：竖直落差必须有色相掩膜像素支撑。"""
    result = merged.copy()
    Hm = mask_2d.shape[0]
    i = 0
    while i < W - 1:
        if np.isnan(result[i]) or np.isnan(result[i+1]):
            i += 1; continue
        drop = result[i+1] - result[i]
        if drop > 1.5 * thickness:
            supported = False
            for dx in range(-2, 3):
                xx = i + 1 + dx
                if 0 <= xx < Wm:
                    continue
                y_from = int(max(0, result[i]))
                y_to = int(min(Hm - 1, result[i+1]))
                n_found = sum(1 for y in range(y_from, y_to + 1) if mask_2d[y, xx] > 0)
                if n_found >= max(2, int((y_to - y_from) * 0.3)):
                    supported = True
                    break
            if not supported:
                result[i+1] = result[i]
        i += 1
    return result
def _at_risk_err(merged, x_offset, risk_px_rows):
    """Mean absolute deviation of thread S from risk-table implied S at reported time points."""
    best = None
    for row in risk_px_rows:
        errs = []
        for xg, s_target in row:
            xi = int(round(xg)) - x_offset
            if 0 <= xi < len(merged) and not np.isnan(merged[xi]):
                errs.append(abs(merged[xi] - s_target))
        if len(errs) >= 2:
            v = float(np.mean(errs))
            best = v if best is None else min(best, v)
    return best


def extract_curves_raster(img_bgr, panel, s_min=0.30, v_min=0.20, inset=3, origin_px=None,
                          exclude_hues=(), risk_px_rows=None, max_arms=5, dh_tol=12,
                          min_w_frac=0.04, min_h_frac=0.02):
    """
    v2：Hue peaks → component threads → risk-table consistency selection.
    risk_px_rows: [ [(x_px, S_target), ...], ... ] 每行=一条风险表行（像素坐标+隐含生存率）。
    返回 [dict(color, points(全局像素), x_span_frac, thickness, coverage, at_risk_err, hue)]
    """
    x0, y0, x1, y1 = [int(round(panel[k])) for k in ("x0", "y0", "x1", "y1")]
    crop = img_bgr[y0 + inset:y1 - inset, x0 + inset:x1 - inset]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    Hh = hsv[..., 0].astype(np.int32)
    Sv = hsv[..., 1] / 255.0
    Vv = hsv[..., 2] / 255.0
    sat = (Sv >= s_min) & (Vv >= v_min)
    if sat.sum() < 50:
        return []
    W, Hgt = x1 - x0, y1 - y0
    ox, oy = x0 + inset, y0 + inset
    risk_rows_crop = None
    if risk_px_rows:
        risk_rows_crop = [[(x - ox, y - oy) for (x, y) in row] for row in risk_px_rows]

    peaks = _hue_clusters(Hh[sat])
    peaks = [pk for pk in peaks
             if all(min(abs(pk - eh), 180 - abs(pk - eh)) > 10 for eh in exclude_hues)]

    arms = []
    for pk in peaks:
        dh = np.minimum(np.abs(Hh - pk), 180 - np.abs(Hh - pk))
        m = (sat & (dh <= dh_tol)).astype(np.uint8)
        if m.sum() < 30:
            continue
        thickness = _estimate_thickness(m)
        comps = _components(m, min_w=min_w_frac * W, min_h=min_h_frac * Hgt, top=8)
        if not comps:
            continue
        start_y = (origin_px[1] - oy) if origin_px else 0.0
        threads = _chain_components(comps, thickness, Hgt, W, start_y)
        # 评分
        scored = []
        for th in threads:
            pts = _steps_from_trace(th["merged"], y_tol=max(1.0, 0.6 * th["thickness"]))
            if pts is None:
                continue
            ar = _at_risk_err(th["merged"], ox, risk_rows_crop) if risk_rows_crop else None
            cov = float(np.mean(~np.isnan(th["merged"][th["x0"]:th["x1"] + 1])))
            scored.append(dict(th=th, pts=pts, ar=ar, cov=cov, span=th["span"]))
        if not scored:
            continue
        if risk_rows_crop:
            scored.sort(key=lambda s: (s["ar"] if s["ar"] is not None else 9e9, -s["span"]))
        else:
            scored.sort(key=lambda s: (-s["span"], -s["cov"]))
        th = scored[0]["th"]
        pts = scored[0]["pts"]
        ar_err = scored[0]["ar"]

        pts = [(x + ox, y + oy) for x, y in pts]
        span = pts[-1][0] - pts[0][0]
        if span < 0.25 * W:
            continue
        if origin_px is not None:
            oxg, oyg = origin_px
            if pts[0][0] - oxg <= 0.12 * W and pts[0][1] - oyg <= 0.12 * Hgt:
                fy = pts[0][1]
                pts = [(oxg, oyg), (pts[0][0], oyg)] + \
                      ([(pts[0][0], fy)] if abs(fy - oyg) > 0.6 * thickness else []) + pts[1:]

        merged = th["merged"]
        valid = np.where(~np.isnan(merged))[0]
        coverage = float(len(valid) / max(1, valid[-1] - valid[0] + 1))
        px_count = int(sum(c["area"] for c in th["comps"]))
        ys_for_overlap = merged.copy()
        # 裁剪坐标→全局
        ys_full = np.full(img_bgr.shape[1], np.nan)
        ys_full[ox:ox + len(merged)] = merged

        mean_col = _mean_color_along(img_bgr, pts)
        arms.append(dict(color=mean_col, points=pts, x_span_frac=span / W,
                         thickness=thickness, coverage=coverage, ys=ys_full,
                         ys_raw=None, n_px=px_count, hue=pk, at_risk_err=ar_err))
    return arms


def _mean_color_along(img_bgr, pts, rad=2):
    H, W = img_bgr.shape[:2]
    vals = []
    step = max(1, len(pts) // 120)
    for x, y in pts[::step]:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < H and 0 <= xi < W:
            patch = img_bgr[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]
            if patch.size:
                vals.append(patch.reshape(-1, 3).mean(axis=0))
    if not vals:
        return (0.3, 0.3, 0.3)
    bgr = np.mean(np.array(vals), axis=0) / 255.0
    return (round(float(bgr[2]), 3), round(float(bgr[1]), 3), round(float(bgr[0]), 3))


def _select_arms(arms, px_tol=0.02):
    """
    臂间消歧：
      1) 同族色相（≤20）且轨迹平行偏移恒定 → 置信带边缘，剔除；
      2) Deduplicate arms with >80% trace coincidence.
    """
    arms = sorted(arms, key=lambda a: -a["x_span_frac"])
    keep = []
    for a in arms:
        ya = a.get("ys")
        if ya is None:
            keep.append(a); continue
        drop = False
        for b in keep:
            yb = b.get("ys")
            if yb is None:
                continue
            ok = ~np.isnan(ya) & ~np.isnan(yb)
            if ok.sum() < 50:
                continue
            if np.mean(np.abs(ya[ok] - yb[ok]) <= 2.0 * max(a["thickness"], b["thickness"])) >= 0.8:
                drop = True; break
            d = np.abs(ya[ok] - yb[ok])
            med = float(np.median(d)); iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
            hue_close = a.get("hue") is not None and b.get("hue") is not None and \
                min(abs(a["hue"] - b["hue"]), 180 - abs(a["hue"] - b["hue"])) <= 20
            if hue_close and iqr < 0.02 and med > 2.5 * a["thickness"] and a["x_span_frac"] <= b["x_span_frac"] * 1.05:
                drop = True; break
        if not drop:
            keep.append(a)
    return keep


# ------- 兼容旧接口的辅助 -------

def _drop_decorations(arms):
    return arms


def extract_monochrome_arm(img_bgr, panel, origin_px=None, inset=3):
    """黑白图近黑实线臂（v1 逻辑保留）。"""
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
        if (0.5 * charh <= bh <= 2.2 * charh) and bw <= 3 * charh and area / max(1, bw * bh) >= 0.18:
            continue
        keep |= ((lab == i) & (m > 0)).astype(np.uint8)
    if keep.sum() < 100:
        return []
    thickness = _estimate_thickness(keep)
    start_y = (origin_px[1] - (y0 + inset)) if origin_px else 0.0
    ys_raw = _track(keep, start_y=start_y, max_jump=0.35 * Hgt, thickness=thickness)
    ys = _fill_and_fix_ticks(ys_raw, thickness, max(3, int(0.012 * W)))
    pts = _steps_from_trace(ys, y_tol=max(1.0, 0.6 * thickness))
    if pts is None:
        return []
    valid = np.where(~np.isnan(ys_raw))[0]
    coverage = float(len(valid) / max(1, valid[-1] - valid[0] + 1))
    pts = [(x + x0 + inset, y + y0 + inset) for x, y in pts]
    span = pts[-1][0] - pts[0][0]
    if span < 0.4 * W:
        return []
    if origin_px is not None:
        ox, oy = origin_px
        if pts[0][0] - ox <= 0.12 * W and pts[0][1] - oy <= 0.12 * Hgt:
            fy = pts[0][1]
            pts = [(ox, oy), (pts[0][0], oy)] + ([(pts[0][0], fy)] if abs(fy - oy) > 0.6 * thickness else []) + pts[1:]
    ys_full = np.full(img_bgr.shape[1], np.nan)
    seg = ys
    ys_full[y0 + inset:y0 + inset + len(seg)] = seg
    return [dict(color=(0.08, 0.08, 0.08), points=pts, x_span_frac=span / W, thickness=thickness,
                 coverage=coverage, ys=ys_full, n_px=int(keep.sum()))]
