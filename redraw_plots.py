# -*- coding: utf-8 -*-
"""redraw_plots — 用 R cox 值重绘 重建曲线 vs 原曲线 对比图。"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipd_reconstruct import km_from_ipd, median_from_km

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
OUT = os.path.join(ROOT, "results", "ipd")


def main():
    cox = pd.read_csv(os.path.join(OUT, "r_out", "r_cox.csv"))
    kv = pd.read_csv(os.path.join(OUT, "key_values.csv"))
    for ipdf in sorted(glob.glob(os.path.join(OUT, "*__arm*.ipd.csv"))):
        base = os.path.basename(ipdf).replace(".ipd.csv", "")
        panel = base.rsplit("__", 1)[0]
        arm = base.rsplit("__", 1)[1]
        df = pd.read_csv(ipdf)
        orig_csv = glob.glob(os.path.join(ROOT, "results", panel.split("__")[0], f"{panel}__{arm}.csv"))
        cox_row = cox[cox.panel == panel]
        if not orig_csv:
            continue
        od = pd.read_csv(orig_csv[0])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.step(od.time, od.surv, where="post", lw=2.2, label=f"{arm} original (digitized)")
        km_t, km_s = km_from_ipd(df.time.tolist(), df.event.tolist())
        med = median_from_km(km_t, km_s)
        ax.step(km_t, km_s, where="post", lw=1.0, ls="--", color="crimson",
                label=f"{arm} reconstructed (median={med and round(med, 2)})")
        if len(cox_row):
            c = cox_row.iloc[0]
            ax.text(0.02, 0.02, f"reconstructed Cox HR={c.HR:.2f} ({c.lo:.2f}-{c.hi:.2f}), p={c.p:.2g}",
                    transform=ax.transAxes, fontsize=8)
        ax.set_title(panel.replace("__", " / "))
        ax.set_xlabel("Time"); ax.set_ylabel("Survival probability")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", panel + ".png"), dpi=130)
        plt.close(fig)
    print("redraw done:", len(glob.glob(os.path.join(OUT, 'plots', '*.png'))))


if __name__ == "__main__":
    main()
