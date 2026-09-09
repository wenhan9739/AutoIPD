# -*- coding: utf-8 -*-
"""Batch digitization driver: manifest -> per-panel curves + QA + overlays."""

import os
import sys
import json
import csv
import math
import time
import traceback

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmfig.render import load_pdf_page, load_image           # noqa: E402
from kmfig.digitize import digitize_page, step_eval          # noqa: E402
import validate                                              # noqa: E402

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
WORK = os.path.join(ROOT, "work")
RESULTS = os.path.join(ROOT, "results")
DPI = 300
MILESTONES = [6, 12, 18, 24, 36, 48, 60]
OVERLAY_COLORS = [(255, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 0), (128, 0, 255), (0, 128, 255)]


def rgb_hex(c):
    if c is None:
        return ""
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def write_arm_csv(path, arm, calib_x, calib_y, y_unit):
    """阶梯顶点 CSV：time, surv, surv_pct, x_px_rel_origin, y_px_rel_origin。"""
    ax, bx = calib_x["a"], calib_x["b"]
    ay, by = calib_y["a"], calib_y["b"]
    scale = 100.0 if y_unit == "percent" else 1.0
    x_org = (0.0 - bx) / ax
    y_org = (1.0 * scale - by) / ay
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "surv", "surv_pct", "x_px_rel_origin", "y_px_rel_origin"])
        for t, s in arm["steps"]:
            xp = (t - bx) / ax - x_org
            yp = (s * scale - by) / ay - y_org
            w.writerow([f"{t:.4f}", f"{s:.5f}", f"{s * 100:.3f}", f"{xp:.2f}", f"{yp:.2f}"])


def draw_overlay(page, results, out_path):
    vis = page.img.copy()
    for r in results:
        p = r["panel"]
        cv2.rectangle(vis, (int(p["x0"]), int(p["y0"])), (int(p["x1"]), int(p["y1"])), (0, 200, 0), 2)
        cal = r.get("calib", {})
        if "x" in cal and cal["x"] and "y" in cal and cal["y"]:
            ax, bx = cal["x"]["a"], cal["x"]["b"]
            ay, by = cal["y"]["a"], cal["y"]["b"]
            scale = 100.0 if cal["y"].get("unit") == "percent" else 1.0
            ox, oy = (0 - bx) / ax, (scale - by) / ay
            cv2.drawMarker(vis, (int(ox), int(oy)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            for k, a in enumerate(r["arms"]):
                col = OVERLAY_COLORS[k % len(OVERLAY_COLORS)]
                pts = [(int(round((t - bx) / ax)), int(round((s * scale - by) / ay))) for t, s in a["steps"]]
                for q1, q2 in zip(pts[:-1], pts[1:]):
                    cv2.line(vis, q1, q2, col, 2)
                label = f"{a['arm_id']} {a.get('label') or ''} med={a['median']:.1f}" if a.get("median") else f"{a['arm_id']} {a.get('label') or ''}"
                cv2.putText(vis, label, (int(p["x0"]) + 6, int(p["y0"]) + 18 + 18 * k),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        tag = f"{r.get('endpoint')} [{r.get('mode')}] {'PASS' if r['qa'].get('pass') else 'CHECK'}"
        cv2.putText(vis, tag, (int(p["x0"]) + 6, int(p["y1"]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if not r["qa"].get("pass") else (0, 128, 0), 2, cv2.LINE_AA)
    cv2.imencode(".png", vis)[1].tofile(out_path)


def office_images(entry):
    """docx/pptx：候选页上的图片文件列表（image/chart 类型）。"""
    docdir = entry["docdir"]
    cl_file = [f for f in os.listdir(docdir) if f.endswith("_content_list.json")][0]
    cl = json.load(open(os.path.join(docdir, cl_file), encoding="utf-8"))
    imgs = [it["img_path"] for it in cl
            if it.get("type") in ("image", "chart") and it.get("page_idx") == entry["page"]]
    return [os.path.join(docdir, p) for p in imgs]


def process_entry(entry, reported, summary_rows, log):
    trial, stem, page_no = entry["trial"], entry["stem"], entry["page"]
    out_dir = os.path.join(RESULTS, safe_name(trial))
    ov_dir = os.path.join(RESULTS, "overlays", safe_name(trial))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ov_dir, exist_ok=True)

    jobs = []
    if entry["kind"] == "pdf" and entry["source"] and entry["source"].lower().endswith(".pdf"):
        jobs.append(("pdf", entry["source"], page_no, f"{safe_name(stem)}__p{page_no:03d}"))
    else:
        for k, ip in enumerate(office_images(entry)):
            jobs.append(("img", ip, None, f"{safe_name(stem)}__p{page_no:03d}__img{k}"))

    n_panels_total = 0
    for kind, path, pno, base in jobs:
        try:
            page = load_pdf_page(path, pno, dpi=DPI) if kind == "pdf" else load_image(path)
            results = digitize_page(page)
        except Exception as e:
            log.write(f"[ERROR] {trial} {base}: {e}\n{traceback.format_exc()}\n")
            continue
        results = [r for r in results if r.get("arms")]
        if not results:
            continue
        # 面板端点 UNKNOWN 时继承页级（caption）端点
        page_eps = entry.get("endpoints") or []
        for r in results:
            r["endpoint_page"] = list(page_eps)
            if r.get("endpoint") in (None, "UNKNOWN") and len(page_eps) == 1:
                r["endpoint"] = page_eps[0]
        n_panels_total += len(results)
        draw_overlay(page, results, os.path.join(ov_dir, base + ".png"))
        for r in results:
            pi = r["panel_index"]
            pbase = f"{base}__panel{pi}"
            # JSON（去掉大数组前先保留 steps）
            jr = dict(r)
            jr["source"] = path
            jr["page_no"] = pno
            jr["dpi"] = DPI if kind == "pdf" else None
            jr["scale"] = page.zoom
            with open(os.path.join(out_dir, pbase + ".json"), "w", encoding="utf-8") as f:
                json.dump(jr, f, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
            cal = r["calib"]
            for a in r["arms"]:
                csv_path = os.path.join(out_dir, f"{pbase}__{a['arm_id']}.csv")
                write_arm_csv(csv_path, a, cal["x"], cal["y"], cal["y"]["unit"])
                S_ms = step_eval(a["steps"], np.array(MILESTONES, float))
                rep_v, rep_d = (None, None)
                if a.get("median"):
                    rep_v, rep_d = validate.match_median(a["median"], reported)
                row = dict(
                    trial=trial, source=os.path.basename(path), page=pno if pno is not None else "",
                    panel=pi, endpoint=r.get("endpoint"), endpoint_page="+".join(r.get("endpoint_page") or []),
                    y_title=" ".join(r["context"].get("ytitle", []))[:60],
                    arm_id=a["arm_id"], label=a.get("label") or "", color=rgb_hex(a["color"]),
                    mode=a["mode"], time_unit=r.get("time_unit"),
                    t0=round(a["t0"], 3), S0=round(a["S0"], 4),
                    median=round(a["median"], 2) if a.get("median") else "",
                    reported_median_match=rep_v if rep_v is not None else "",
                    median_rel_diff=round(rep_d, 3) if rep_d is not None else "",
                    t_end=round(a["t_end"], 2), S_end=round(a["S_end"], 4), n_steps=a["n_steps"],
                    n_censors=a.get("n_censors") if a.get("n_censors") is not None else "",
                    n_at_risk_rows=len(r.get("at_risk", [])),
                    **{f"S{m}": (round(float(v), 4) if not math.isnan(v) else "") for m, v in zip(MILESTONES, S_ms)},
                    qa_pass=r["qa"].get("pass"), start_ok=r["qa"].get("start_ok"),
                    monotone_ok=r["qa"].get("monotone_ok"), calib_ok=r["qa"].get("calib_ok"),
                    calib_src=f"x:{cal['x']['source']}/y:{cal['y']['source']}",
                    csv=os.path.relpath(csv_path, RESULTS),
                    overlay=os.path.relpath(os.path.join(ov_dir, base + ".png"), RESULTS))
                summary_rows.append(row)
    return n_panels_total


def main(only_trials=None):
    manifest = json.load(open(os.path.join(WORK, "manifest.json"), encoding="utf-8"))
    if only_trials:
        manifest = [m for m in manifest if m["trial"] in only_trials]
    os.makedirs(RESULTS, exist_ok=True)
    summary_rows = []
    reported_cache = {}
    t0 = time.time()
    with open(os.path.join(RESULTS, "digitize_log.txt"), "a", encoding="utf-8") as log:
        log.write(f"==== run {time.strftime('%F %T')} entries={len(manifest)}\n")
        for i, e in enumerate(manifest):
            if e["trial"] not in reported_cache:
                reported_cache[e["trial"]] = validate.reported_medians(e["trial"])
            n = process_entry(e, reported_cache[e["trial"]], summary_rows, log)
            print(f"[{i + 1}/{len(manifest)}] {e['trial']} / {e['stem'][:25]} p{e['page']}: {n} KM panel(s)  ({time.time() - t0:.0f}s)")
    # 汇总
    if summary_rows:
        cols = list(summary_rows[0].keys())
        with open(os.path.join(RESULTS, "curves_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary_rows)
    write_report(summary_rows, reported_cache)
    print(f"DONE: {len(summary_rows)} arms from {len(set((r['trial'], r['source'], r['page'], r['panel']) for r in summary_rows))} panels; "
          f"{sum(1 for r in summary_rows if r['qa_pass'])} arm-rows pass QA")


def write_report(rows, reported_cache):
    lines = ["# KM 曲线数字化 QA 报告", "", f"生成时间：{time.strftime('%F %T')}", ""]
    trials = sorted(set(r["trial"] for r in rows))
    lines.append(f"共 {len(trials)} 个试验、{len(rows)} 条曲线（臂）。")
    lines.append("")
    lines.append("| 试验 | 来源 | 页 | 面板 | 端点 | 臂 | 标签 | 模式 | 中位(数字化) | 中位(文中匹配) | 相对差 | QA |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['trial']} | {r['source'][:28]} | {r['page']} | {r['panel']} | {r['endpoint']} | {r['arm_id']} | "
                     f"{str(r['label'])[:28]} | {r['mode']} | {r['median']} | {r['reported_median_match']} | {r['median_rel_diff']} | "
                     f"{'✅' if r['qa_pass'] else '⚠️'} |")
    lines.append("")
    lines.append("## 需人工复核的条目（QA 未通过或中位数与文中不匹配）")
    for r in rows:
        bad = (not r["qa_pass"]) or (r["median"] != "" and r["reported_median_match"] == "")
        if bad:
            lines.append(f"- {r['trial']} / {r['source']} p{r['page']} panel{r['panel']} {r['arm_id']} ({r['endpoint']}, {r['mode']}): "
                         f"start_ok={r['start_ok']} monotone_ok={r['monotone_ok']} calib_ok={r['calib_ok']} median={r['median']} → overlay: {r['overlay']}")
    lines.append("")
    lines.append("## 文中提取到的中位生存时间（供比对）")
    for t in trials:
        vals = reported_cache.get(t, [])
        uniq = sorted(set((v, str(ep)) for v, ep, s in vals))
        lines.append(f"- **{t}**: " + ", ".join(f"{v}{'(' + ep + ')' if ep != 'None' else ''}" for v, ep in uniq[:40]))
    with open(os.path.join(RESULTS, "QA_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    only = sys.argv[1:] or None
    main(only)
