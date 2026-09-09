# -*- coding: utf-8 -*-
"""Quantitative verification: trace precision, data integrity, CSV consistency."""

import os
import sys
import json
import csv
import glob
import traceback

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmfig.render import load_pdf_page, load_image   # noqa: E402

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
RESULTS = os.path.join(ROOT, "results")
VERIFY = os.path.join(ROOT, "work", "verify")
DPI = 300


def hue_mask_for(crop, color_rgb):
    """裁剪图内某臂颜色的像素掩膜；近黑色用暗色低饱和掩膜。"""
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    Hh = hsv[..., 0].astype(np.int32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0
    if color_rgb is not None and max(color_rgb) - min(color_rgb) < 0.16 and max(color_rgb) < 0.30:
        return (V < 0.36) & (S < 0.35)
    bgr = np.uint8([[[color_rgb[2] * 255, color_rgb[1] * 255, color_rgb[0] * 255]]])
    hue = int(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 0])
    dh = np.minimum(np.abs(Hh - hue), 180 - np.abs(Hh - hue))
    return dh <= 18


def corridor_for(steps_px, shape):
    """轨迹走廊：折线加粗 + 膨胀。"""
    corr = np.zeros(shape, np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in steps_px], np.int32)
    if len(pts) >= 2:
        cv2.polylines(corr, [pts.reshape(-1, 1, 2)], False, 1, 3)
    corr = cv2.dilate(corr, np.ones((5, 5), np.uint8))
    return corr > 0


def check_arm_fit(img, panel, calib, arm, y_unit):
    """返回 (precision, recall, crop_vis) 或 (None, None, None)=无法评估。
    precision：沿轨迹每 3px 采样一点，到最近“原曲线像素”距离 ≤2.5px 的比例；
    recall：原曲线（与轨迹相交的大连通域）像素被走廊覆盖的比例。"""
    px0, py0, px1, py1 = [float(panel[k]) for k in ("x0", "y0", "x1", "y1")]
    ax, bx = calib["x"]["a"], calib["x"]["b"]
    ay, by = calib["y"]["a"], calib["y"]["b"]
    scale = 100.0 if y_unit == "percent" else 1.0
    steps_px = [((t - bx) / ax, (s * scale - by) / ay) for t, s in arm["steps"]]
    ix0 = max(0, int(px0) - 2)
    iy0 = max(0, int(py0) - 2)
    ix1 = min(img.shape[1], int(px1) + 3)
    iy1 = min(img.shape[0], int(py1) + 3)
    crop = img[iy0:iy1, ix0:ix1]
    if crop.size == 0:
        return None, None, None
    hm_loose = hue_mask_for(crop, arm.get("color"))   # S>=0.15：含浅色/点线
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hm_strict = (hsv[..., 1] / 255.0 >= 0.32) & (hsv[..., 2] / 255.0 >= 0.15)
    if hm_loose is None or hm_loose.sum() < 30:
        return None, None, None
    steps_c = [(x - ix0, y - iy0) for x, y in steps_px]
    corr = corridor_for(steps_c, hm_loose.shape)
    # precision：密集采样轨迹量距离；严格/宽松两个掩膜取优
    def prec_on(mask):
        dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        vals = []
        for (xa, ya), (xb, yb) in zip(steps_c[:-1], steps_c[1:]):
            L = float(np.hypot(xb - xa, yb - ya))
            nseg = max(1, int(L / 3.0))
            for k in range(nseg + 1):
                x = xa + (xb - xa) * k / nseg
                y = ya + (yb - ya) * k / nseg
                if 0 <= int(y) < dt.shape[0] and 0 <= int(x) < dt.shape[1]:
                    vals.append(dt[int(round(y)), int(round(x))])
        return float(np.mean([v <= 2.5 for v in vals])) if vals else None
    p_strict = prec_on(hm_strict & (hm_loose > 0))
    p_loose = prec_on(hm_loose)
    precision = max(v for v in (p_strict, p_loose) if v is not None)
    hm = hm_strict if hm_strict.sum() >= 30 else hm_loose
    inter = corr & hm
    # recall：只在与走廊相交的大连通域上算（曲线本体），排除图例/风险表同色像素
    n, lab, stats, _ = cv2.connectedComponentsWithStats((hm > 0).astype(np.uint8), connectivity=8)
    W = hm.shape[1]
    recall_num, recall_den = 0, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 0.25 * W:
            continue
        comp = (lab == i)
        recall_den += int(comp.sum())
        recall_num += int((comp & corr).sum())
    recall = recall_num / max(1, recall_den)
    # 可视化：原曲线红、走廊绿、交集黄
    vis = crop.copy()
    vis[hm & ~corr] = (0, 0, 255)
    vis[corr & ~hm] = (0, 255, 0)
    vis[inter] = (0, 255, 255)
    return precision, recall, vis


def main():
    jsons = sorted(glob.glob(os.path.join(RESULTS, "*", "*__panel*.json")))
    print(f"panels: {len(jsons)}")
    os.makedirs(VERIFY, exist_ok=True)
    page_cache = {}
    rows = []

    def get_page(source, page_no):
        key = (source, page_no)
        if key not in page_cache:
            try:
                if page_no is None or str(page_no) == "None" or page_no == "":
                    page_cache[key] = ("img", load_image(source))
                else:
                    page_cache[key] = ("pdf", load_pdf_page(source, int(page_no), dpi=DPI))
            except Exception as e:
                print(f"  [render-error] {source} p{page_no}: {e}")
                page_cache[key] = None
        return page_cache[key]

    for jp in jsons:
        d = json.load(open(jp, encoding="utf-8"))
        trial = os.path.basename(os.path.dirname(jp))
        base = os.path.splitext(os.path.basename(jp))[0]
        gp = get_page(d.get("source"), d.get("page_no"))
        if gp is None:
            continue
        kind, page = gp
        panel = d["panel"]
        calib = d["calib"]
        if not calib.get("x") or not calib.get("y"):
            continue
        for k, arm in enumerate(d["arms"]):
            name = f"{base}__{arm.get('arm_id', 'arm' + str(k + 1))}"
            # 数据完整性
            steps = arm["steps"]
            integrity = "ok"
            ts = [s[0] for s in steps]
            ss = [s[1] for s in steps]
            if any(b < a - 1e-9 for a, b in zip(ts[:-1], ts[1:])):
                integrity = "time_backwards"
            if min(ss) < -0.005 or max(ss) > 1.01:
                integrity = "surv_out_of_range"
            if abs(steps[0][0]) > 0.05 * max(d["calib"]["x"].get("values") or [10]) or steps[0][1] < 0.9:
                integrity = integrity if integrity != "ok" else "start_offset"
            # CSV 与 JSON 一致性
            csv_path = os.path.join(RESULTS, trial, name + ".csv")
            csv_ok = ""
            if os.path.isfile(csv_path):
                with open(csv_path, encoding="utf-8") as f:
                    rr = list(csv.DictReader(f))
                if len(rr) != len(steps):
                    csv_ok = "row_mismatch"
                else:
                    try:
                        t0c = float(rr[0]["time"]); s0c = float(rr[0]["surv"])
                        if abs(t0c - steps[0][0]) > 5e-4 or abs(s0c - steps[0][1]) > 5e-4:
                            csv_ok = "value_mismatch"
                    except Exception:
                        csv_ok = "parse_error"
            else:
                csv_ok = "missing"
            # 贴合度
            try:
                prec, rec, vis = check_arm_fit(page.img, panel, calib, arm, calib["y"]["unit"])
            except Exception:
                prec, rec, vis = None, None, None
            if vis is not None:
                cv2.imencode(".png", vis)[1].tofile(os.path.join(VERIFY, name + ".png"))
            rows.append(dict(
                trial=trial, panel_file=os.path.basename(jp), arm=arm.get("arm_id"),
                mode=arm.get("mode"), endpoint=d.get("endpoint"),
                label=arm.get("label") or "",
                median=round(arm["median"], 3) if arm.get("median") else "",
                integrity=integrity, csv_ok=csv_ok,
                precision=round(prec, 3) if prec is not None else "",
                recall=round(rec, 3) if rec is not None else "",
                fit_flag="" if prec is None else ("low_precision" if prec < 0.75 else ""),
                qa_pass=d["qa"].get("pass"),
                vis=os.path.join(VERIFY, name + ".png")))

    out = os.path.join(RESULTS, "verification.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    integ_bad = [r for r in rows if r["integrity"] != "ok"]
    csv_bad = [r for r in rows if r["csv_ok"] not in ("ok", "")]
    evald = [r for r in rows if r["precision"] != ""]
    lowp = [r for r in evald if float(r["precision"]) < 0.80]
    lowr = [r for r in evald if r["recall"] != "" and float(r["recall"]) < 0.75]
    print(f"arms: {n}")
    print(f"  integrity bad: {len(integ_bad)}")
    print(f"  csv bad: {len(csv_bad)}")
    print(f"  fit evaluated: {len(evald)} | precision<0.80: {len(lowp)} | recall<0.75: {len(lowr)}")
    for r in lowp[:15]:
        print("    LOW-P", r["trial"], r["panel_file"][:40], r["arm"], "prec", r["precision"], "rec", r["recall"])
    for r in lowr[:15]:
        print("    LOW-R", r["trial"], r["panel_file"][:40], r["arm"], "prec", r["precision"], "rec", r["recall"])
    print("written:", out)


if __name__ == "__main__":
    main()
