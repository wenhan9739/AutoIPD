# -*- coding: utf-8 -*-
"""Per-panel visual QA: original vs Py vs R three-way overlay images."""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmfig.render import load_pdf_page, load_image
from ipd_reconstruct import km_from_ipd

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
IPD = os.path.join(ROOT, "results", "ipd")
DST = os.path.join(ROOT, "肉眼质检_IPD重建全部")

ARM_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def km_eval(km_t, km_s, tq):
    idx = np.searchsorted(km_t, tq, side="right") - 1
    return np.where(idx >= 0, np.array(km_s)[np.clip(idx, 0, None)], 1.0)


def median_km(km_t, km_s):
    arr = [(t, s) for t, s in zip(km_t, km_s) if s <= 0.5]
    return arr[0][0] if arr else None


def read_ipd_key(path, key):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    ecol = "event" if "event" in df.columns else "status"
    sub = df[df.key == key] if "key" in df.columns else df
    if len(sub) == 0:
        return None
    return sub["time"].astype(float).tolist(), sub[ecol].astype(int).tolist()


def main():
    os.makedirs(DST, exist_ok=True)
    master = pd.read_csv(os.path.join(ROOT, "results", "curves_master.csv"))
    kv = pd.read_csv(os.path.join(IPD, "key_values.csv"))
    cox = pd.read_csv(os.path.join(IPD, "r_out", "r_cox.csv"))
    r_all_path = os.path.join(IPD, "r_out", "r_ipd_all.csv")
    r_all = pd.read_csv(r_all_path) if os.path.isfile(r_all_path) else pd.DataFrame()

    master_idx = {(r["面板"], r["臂ID"]): r for _, r in master.iterrows()}
    panels = sorted(kv.panel.unique())
    summary = []
    for pi, base in enumerate(panels):
        kv_p = kv[kv.panel == base]
        if kv_p.empty:
            continue
        trial = kv_p.iloc[0]["trial"]
        endpoint = kv_p.iloc[0]["endpoint"]
        d_json = os.path.join(ROOT, "results", trial, base + ".json")
        d = json.load(open(d_json, encoding="utf-8"))
        panel = d["panel"]
        src = d.get("source")
        pageno = d.get("page_no")

        # 人群与干预标注
        mrows = [master_idx[(base, a)] for a in kv_p.arm if (base, a) in master_idx]
        population = mrows[0]["人群"] if mrows else ""
        pop_short = str(population).split("｜")[0][:80]
        int_txt = "  |  ".join(
            f"{m['臂ID']}({m['角色']}): {m['干预']}" for m in mrows)

        fig = plt.figure(figsize=(14, 5.6))
        ax = fig.add_axes([0.055, 0.16, 0.50, 0.78])
        for ai, (_, kr) in enumerate(kv_p.iterrows()):
            arm = kr["arm"]
            color = ARM_COLORS[ai % len(ARM_COLORS)]
            orig_csv = os.path.join(ROOT, "results", trial, f"{base}__{arm}.csv")
            if not os.path.isfile(orig_csv):
                continue
            od = pd.read_csv(orig_csv)
            ax.step(od.time, od.surv, where="post", lw=2.8, color=color, alpha=0.9,
                    label=f"{arm} original")
            py_ipd = read_ipd_key(os.path.join(IPD, f"{base}__{arm}.ipd.csv"), f"{base}__{arm}")
            r_ipd = read_ipd_key(r_all_path, f"{base}__{arm}")
            m_o = median_km(od.time.values, od.surv.values)
            m_p = m_r = None
            if py_ipd:
                kt, ks = km_from_ipd(*py_ipd)
                m_p = median_km(kt, ks)
                ax.step(kt, ks, where="post", lw=1.1, ls="--", color="crimson", label=f"{arm} Py重建")
            if r_ipd:
                kt2, ks2 = km_from_ipd(*r_ipd)
                m_r = median_km(kt2, ks2)
                ax.step(kt2, ks2, where="post", lw=1.1, ls=":", color="green", label=f"{arm} R重建")
            ann = f"{arm}: 原={m_o and round(m_o,2)}  Py={m_p and round(m_p,2)}  R={m_r and round(m_r,2)}"
            ax.text(0.02, 0.06 - 0.05 * ai, ann, transform=ax.transAxes, fontsize=8,
                    color=color, va="top")
        ax.axhline(0.5, color="gray", lw=0.6, ls="-.")
        ax.set_title(f"{trial}  /  {base.replace('__', ' / ')}  /  {endpoint}", fontsize=10)
        ax.set_xlabel("Time (months)"); ax.set_ylabel("Survival probability")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.text(0.99, 0.99, "蓝实线=原图数字化  红虚线=Python重建  绿点线=R重建",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, alpha=0.8)

        # 右侧：发表图裁剪
        try:
            z = (d.get("dpi", 300) or 300) / 150.0
            if str(src).endswith(".pdf") and pageno is not None:
                page = load_pdf_page(src, int(pageno), dpi=150)
                H, W = page.img.shape[:2]
                x0 = max(0, int(panel["x0"] / z) - 15); x1 = min(W, int(panel["x1"] / z) + 15)
                y0 = max(0, int(panel["y0"] / z) - 15); y1 = min(H, int(panel["y1"] / z) + 15)
                crop = page.img[y0:y1, x0:x1]
            else:
                page = load_image(src)
                s = page.zoom
                H, W = page.img.shape[:2]
                x0 = max(0, int(panel["x0"] / s * (300/150)) - 15)
                y0 = max(0, int(panel["y0"] / s * (300/150)) - 15)
                x1 = min(W, int(panel["x1"] / s * (300/150)) + 15)
                y1 = min(H, int(panel["y1"] / s * (300/150)) + 15)
                crop = page.img[y0:y1, x0:x1]
            ax2 = fig.add_axes([0.60, 0.16, 0.385, 0.78])
            import cv2
            ax2.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ax2.set_title("published figure (crop)", fontsize=9)
            ax2.axis("off")
        except Exception as e:
            print(f"  [crop-skip] {base}: {e}")

        # 底部标注：人群 + 干预
        fig.text(0.055, 0.015, f"人群: {pop_short}", fontsize=8.5)
        fig.text(0.055, 0.002, f"干预: {int_txt[:190]}", fontsize=8)
        out_png = os.path.join(DST, f"{trial}__{base}.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)

        summary.append(dict(试验=trial, 面板=base, 端点=endpoint, 人群=pop_short,
                            图=os.path.abspath(out_png),
                            臂数=len(kv_p),
                           cox_HR=(cox[cox.panel == base].HR.iloc[0] if (len(cox) and (cox.panel == base).any()) else "")))
        print(f"[{pi+1}/{len(panels)}] {base}")

    pd.DataFrame(summary).to_csv(os.path.join(DST, "质检面板清单.csv"),
                                 index=False, encoding="utf-8-sig")
    md = ["# 全部重建面板 肉眼质检（原图 / Python重建 / R重建）", "",
          "每张图：**蓝实线**=原图数字化曲线，**红虚线**=Python重建KM，**绿点线**=R(IPDfromKM)重建KM；",
          "右侧为发表图裁剪；图内黄字/彩字为各臂三方中位数；底部为该面板的人群与干预标注。", "",
          "质检要点：① 红虚线与绿点线应彼此重合（两实现一致性）；",
          "② 彩色实线应与右侧发表图中的对应曲线形状一致（重建忠实性）；",
          "③ 各臂中位数与右侧发表图标注对照。", "",
          "| # | 面板 | 端点 | 臂数 | Cox HR(重建) |", "|---|---|---|---|---|"]
    for i, r in enumerate(summary):
        md.append(f"| {i+1} | {r['试验']} {r['面板']} | {r['端点']} | {r['臂数']} | {r['cox_HR']} |")
    open(os.path.join(DST, "README.md"), "w", encoding="utf-8").write("\n".join(md))
    print("DONE:", len(summary), "panel QA images →", DST)


if __name__ == "__main__":
    main()
