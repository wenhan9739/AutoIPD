# -*- coding: utf-8 -*-
"""Visual QA package generator for selected curves."""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmfig.render import load_pdf_page
from ipd_reconstruct import km_from_ipd

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
IPD = os.path.join(ROOT, "results", "ipd")
DST = os.path.join(ROOT, "肉眼质检_IPD重建")


def km_eval(km_t, km_s, tq):
    idx = np.searchsorted(km_t, tq, side="right") - 1
    return np.where(idx >= 0, np.array(km_s)[np.clip(idx, 0, None)], 1.0)


def median_km(km_t, km_s):
    for t, s in zip(km_t, km_s):
        if s <= 0.5:
            return float(t)
    return None


def read_ipd(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    ecol = "event" if "event" in df.columns else "status"
    return df["time"].astype(float).tolist(), df[ecol].astype(int).tolist()


def make_curve(name, trial, base, arm, published=""):
    ipd_py = read_ipd(os.path.join(IPD, f"{base}__{arm}.ipd.csv"))
    ipd_r = read_ipd(os.path.join(IPD, "r_out", "r_ipd", f"{base}__{arm}.ipd.csv"))
    orig_csv = os.path.join(ROOT, "results", trial, f"{base}__{arm}.csv")
    orig = pd.read_csv(orig_csv)
    jp = os.path.join(ROOT, "results", trial, f"{base}.json")
    d = json.load(open(jp, encoding="utf-8"))
    panel = d["panel"]
    calib = d["calib"]

    ot, osv = orig.time.values, orig.surv.values
    kt_py, ks_py = km_from_ipd(*ipd_py)
    kt_r, ks_r = km_from_ipd(*ipd_r)
    grid = np.linspace(0, max(max(kt_py), max(kt_r), ot[-1]), 400)
    py_r_diff = float(np.max(np.abs(km_eval(kt_py, ks_py, grid) - km_eval(kt_r, ks_r, grid))))
    rmse_orig_py = float(np.sqrt(np.mean((km_eval(kt_py, ks_py, ot) - osv) ** 2)))

    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0.06, 0.12, 0.52, 0.82])
    ax.step(ot, osv, where="post", lw=2.6, color="#1f77b4", label="original (digitized)")
    ax.step(kt_py, ks_py, where="post", lw=1.2, ls="--", color="crimson", label="Python reconstructed")
    ax.step(kt_r, ks_r, where="post", lw=1.2, ls=":", color="green", label="R (IPDfromKM) reconstructed")
    m_o, m_p, m_r = median_km(ot, osv), median_km(kt_py, ks_py), median_km(kt_r, ks_r)
    ax.axhline(0.5, color="gray", lw=0.6, ls="-.")
    txt = f"median  original={m_o and round(m_o,2)}  py={m_p and round(m_p,2)}  r={m_r and round(m_r,2)}\n"
    txt += f"Py vs R max|ΔS|={py_r_diff:.4f}   Py vs orig RMSE={rmse_orig_py:.4f}"
    if published:
        txt += f"\npublished median: {published}"
    ax.set_title(f"{trial} / {base.replace('__', ' / ')} / {arm}", fontsize=10)
    ax.set_xlabel("Time (months)"); ax.set_ylabel("Survival probability")
    ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=8, loc="upper right")
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round", fc="lemonchiffon", alpha=0.9))

    # 右：发表图裁剪（原PDF 150dpi 渲染）
    try:
        if d.get("page_no") is not None and str(d.get("source", "")).endswith(".pdf"):
            page = load_pdf_page(d["source"], int(d["page_no"]), dpi=150)
            z = d.get("dpi", 300) / 150.0
            H, W = page.img.shape[:2]
            x0 = max(0, int(panel["x0"] / z) - 20); x1 = min(W, int(panel["x1"] / z) + 20)
            y0 = max(0, int(panel["y0"] / z) - 20); y1 = min(H, int(panel["y1"] / z) + 20)
            crop = page.img[y0:y1, x0:x1]
            ax2 = fig.add_axes([0.62, 0.08, 0.36, 0.88])
            ax2.imshow(cv2cvtColor(crop))
            ax2.set_title("published figure (crop)", fontsize=9)
            ax2.axis("off")
    except Exception as e:
        print(f"  [crop-skip] {name}: {e}")
    fig.savefig(os.path.join(DST, name + ".png"), dpi=120)
    plt.close(fig)
    return dict(名称=name, 试验=trial, 臂=arm,
                中位_原图数字化=(m_o and round(m_o, 3)), 中位_python=(m_p and round(m_p, 3)),
                中位_r=(m_r and round(m_r, 3)), 发表中位=published,
                Py_vs_R最大差=round(py_r_diff, 5), Py_vs_原RMSE=round(rmse_orig_py, 5),
                图=os.path.abspath(os.path.join(DST, name + ".png")))


def cv2cvtColor(bgr):
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    os.makedirs(DST, exist_ok=True)
    picks = [
        ("01_K189_ITT_OS_实验组", "KEYNOTE-189", "KEYNOTE-189__p003__panel0", "arm1", "22.0"),
        ("02_K189_ITT_OS_对照组", "KEYNOTE-189", "KEYNOTE-189__p003__panel0", "arm2", "10.6"),
        ("03_K189_ITT_PFS_实验组", "KEYNOTE-189", "KEYNOTE-189__p003__panel4", "arm1", "9.0"),
        ("04_AK105302_PFS_ITT_实验组", "AK105-302", "AK105-302__p006__panel0", "arm1", "7.2"),
        ("05_AK105302_PFS_ITT_对照组", "AK105-302", "AK105-302__p006__panel0", "arm2", "4.2"),
        ("06_POSEIDON_OS_多臂", "POSEIDON", "POSEIDON__p004__panel1", "arm2", "12.6 (D+CT)"),
        ("07_MYSTIC_OS_多臂", "MYSTIC", "MYSTIC_1__p007__panel0", "arm1", "21.9 (Nivo+Ipi)"),
        ("08_KEYNOTE407_OS", "KEYNOTE-407", "KEYNOTE-407__p003__panel0", "arm1", "15.0 (pembro+CT)"),
        ("09_CHOICE01_PFS", "CHOICE-01", "CHOICE-01__p007__panel0", "arm1", "8.4 (toripalimab)"),
        ("10_CM227_OS_多臂", "CheckMate_227_Part_1", "CheckMate_227_Part_1__p005__panel0", "arm2", "12.7 (chemo)"),
        ("11_RATIONALE304_OS", "RATIONALE-304", "RATIONALE-304__p007__panel0", "arm1", "21.4 (tislelizumab)"),
        ("12_K189_DOR_亚组", "KEYNOTE-189", "KEYNOTE-189__p004__panel0", "arm1", "—"),
    ]
    rows = []
    for name, trial, base, arm, pub in picks:
        try:
            rows.append(make_curve(name, trial, base, arm, pub))
            print("done", name)
        except Exception as e:
            import traceback
            print("ERROR", name, e)
            traceback.print_exc()
    pd.DataFrame(rows).to_csv(os.path.join(DST, "三方对比数值表.csv"),
                              index=False, encoding="utf-8-sig")
    md = ["# IPD 重建 肉眼质检（Python / R / 原图）", "",
          "每张图左侧：蓝实线=原图数字化曲线，红虚线=Python重建KM，绿点线=R重建KM；",
          "黄框为三方中位数与Py/R曲线最大差。右侧为发表图裁剪。", "",
          "| # | 试验 | 臂 | 中位(原) | 中位(PY) | 中位(R) | 发表中位 | Py vs R 最大差 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["名称"].split("_")[0], r["试验"], r["臂"], r["中位_原图数字化"],
            r["中位_python"], r["中位_r"], r["发表中位"], r["Py_vs_R最大差"]))
    open(os.path.join(DST, "README.md"), "w", encoding="utf-8").write("\n".join(md))
    print("package →", DST)


if __name__ == "__main__":
    main()
