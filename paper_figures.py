# -*- coding: utf-8 -*-
"""Publication-quality figure generation (300 dpi, serif fonts)."""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"D:\work\IPDR"
FIG = os.path.join(ROOT, "results", "paper_figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})
COL = {"orig": "#1f77b4", "py": "#d62728", "r": "#2ca02c", "gray": "#555555"}

def median_km_local(times, events):
    xs, ys = km_steps(times, events)
    for x, y in zip(xs, ys):
        if y <= 0.5:
            return x
    return None


kv = pd.read_csv(os.path.join(ROOT, r"results\ipd\key_values.csv"))
rep = pd.read_csv(os.path.join(ROOT, r"results\ipd\compare_report.csv"))
ver = pd.read_csv(os.path.join(ROOT, r"results\verification.csv"))
master = pd.read_csv(os.path.join(ROOT, r"results\curves_master.csv"))
r_all = pd.read_csv(os.path.join(ROOT, r"results\ipd\r_out\r_ipd_all.csv"))


def km_steps(times, events):
    order = np.argsort(times, kind="mergesort")
    t, e = np.asarray(times, float)[order], np.asarray(events, int)[order]
    n = len(t); surv = 1.0; i = 0
    xs, ys = [0.0], [1.0]
    while i < n:
        tt = t[i]; j = i
        while j < n and t[j] == tt:
            j += 1
        d = int(e[i:j].sum())
        if d > 0:
            surv *= (1 - d / (n - i)); xs.append(float(tt)); ys.append(surv)
        i = j
    return xs, ys


# ---------------- Fig 1: pipeline schematic ----------------
fig, ax = plt.subplots(figsize=(7.0, 2.6))
ax.axis("off")
stages = [
    ("Published\nRCT documents\n(PDF/docx/pptx)", "#eef2f7"),
    ("MinerU parsing\nlayout + chart detection\n+ text/figure extraction", "#e4ecf5"),
    ("Candidate-page\nselection\n(caption/keywords)", "#dae5f2"),
    ("Curve digitization\npanel detection | axis calibration\nvector/raster extraction", "#cfddef"),
    ("IPD reconstruction\nGuyot 2012 — dual\nimplementation (Py + R)", "#c5d5ec"),
    ("Quality control\n4 layers + visual\nverification", "#bbcee8"),
]
n = len(stages)
bw, bh, gap = 1.42, 1.1, 0.28
for i, (txt, c) in enumerate(stages):
    x = i * (bw + gap)
    box = FancyBboxPatch((x, 0.35), bw, bh, boxstyle="round,pad=0.05",
                         fc=c, ec="#3a5a80", lw=1.0)
    ax.add_patch(box)
    ax.text(x + bw / 2, 0.35 + bh / 2, txt, ha="center", va="center", fontsize=7.4)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + bw + 0.03, 0.9), (x + bw + gap - 0.03, 0.9),
                                     arrowstyle="-|>", mutation_scale=12, color="#3a5a80", lw=1.1))
labels = ["corpus", "parsing", "selection", "digitization", "reconstruction", "QC"]
for i, lb in enumerate(labels):
    x = i * (bw + gap)
    ax.text(x + bw / 2, 0.12, f"Stage {i+1}\n({lb})", ha="center", va="center",
            fontsize=7.2, style="italic", color="#3a5a80")
ax.set_xlim(-0.15, n * (bw + gap)); ax.set_ylim(0, 1.7)
fig.savefig(os.path.join(FIG, "fig1_pipeline.png"), bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 2: three-way KM overlay example ----------------
d = json.load(open(os.path.join(ROOT, r"results\KEYNOTE-189\KEYNOTE-189__p003__panel4.json"), encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ai, arm in enumerate(["arm1", "arm2"]):
    ax = axes[ai]
    od = pd.read_csv(os.path.join(ROOT, "results", "KEYNOTE-189", f"KEYNOTE-189__p003__panel4__{arm}.csv"))
    ax.step(od.time, od.surv, where="post", lw=2.0, color=COL["orig"], label="Original (digitized)")
    py = pd.read_csv(os.path.join(ROOT, f"results\\ipd\\KEYNOTE-189__p003__panel4__{arm}.ipd.csv"))
    kx, ky = km_steps(py.time.values, py.event.values)
    ax.step(kx, ky, where="post", lw=1.3, ls="--", color=COL["py"], label="Python reconstruction")
    rsub = r_all[r_all.key == f"KEYNOTE-189__p003__panel4__{arm}"]
    rx, ry = km_steps(rsub.time.values, rsub.status.values)
    ax.step(rx, ry, where="post", lw=1.3, ls=":", color=COL["r"], label="R (IPDfromKM) reconstruction")
    ax.set_xlabel("Time since randomization (months)")
    if ai == 0:
        ax.set_ylabel("Progression-free survival")
    ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=6.5, loc="upper right")
fig.suptitle("KEYNOTE-189, ITT population — original vs. reconstructed KM curves", fontsize=9, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG, "fig2_km_overlay.png"), bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 3: trace precision + median agreement ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
p = ver["precision"].dropna().astype(float)
axes[0].hist(p, bins=np.linspace(0.4, 1.0, 31), color="#4a7ebb", edgecolor="white", lw=0.4)
axes[0].axvline(p.median(), color="crimson", lw=1.2)
axes[0].text(p.median() - 0.015, axes[0].get_ylim()[1] * 0.92,
             f"median = {p.median():.3f}", rotation=90, va="top", ha="right",
             fontsize=7.5, color="crimson")
axes[0].set_xlabel("Trace precision"); axes[0].set_ylabel("Number of curves")
axes[0].set_title("(a) Digitization trace precision", fontsize=9)

md = pd.to_numeric(rep["median_diff"], errors="coerce")
mr = pd.to_numeric(rep["median_recon"], errors="coerce")
mo = []
for _, r in rep.iterrows():
    oc = os.path.join(ROOT, "results", r["trial"], f"{r['panel']}__{r['arm']}.csv")
    if os.path.isfile(oc):
        od = pd.read_csv(oc)
        arr = [(t, s) for t, s in zip(od.time, od.surv) if s <= 0.5]
        mo.append(arr[0][0] if arr else np.nan)
    else:
        mo.append(np.nan)
mo = np.array(mo, float)
ok = ~np.isnan(mr.values) & ~np.isnan(mo)
axes[1].scatter(mo[ok], mr.values[ok], s=14, alpha=0.65, color="#4a7ebb", edgecolors="none")
lim = [0, max(np.nanmax(mo), np.nanmax(mr.values)) * 1.05]
axes[1].plot(lim, lim, color="crimson", lw=0.9, label="identity")
axes[1].set_xlim(lim); axes[1].set_ylim(lim)
axes[1].set_xlabel("Median of original digitized curve (months)")
axes[1].set_ylabel("Median of reconstructed KM (months)")
axes[1].set_title("(b) Median agreement (reconstructed vs. original)", fontsize=9)
axes[1].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_accuracy.png"), bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 4: Py vs R agreement ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
n = len(rep)
ident = (rep.py_r_identical == "YES").sum()
axes[0].bar(["Row-identical\nIPD tables", "Micro-differences in\ncensor placement"],
            [ident, n - ident], color=["#2ca02c", "#f0ad4e"], width=0.55, edgecolor="white")
for i, v in enumerate([ident, n - ident]):
    axes[0].text(i, v + 2, str(v), ha="center", fontsize=8.5)
axes[0].set_ylabel("Number of arms"); axes[0].set_ylim(0, n * 0.8 + 10)
axes[0].set_title(f"(a) Python vs. R (IPDfromKM), n = {n}", fontsize=9)

med_py, med_r = [], []
for _, r in rep.iterrows():
    pyf = os.path.join(ROOT, f"results\\ipd\\{r['panel']}__{r['arm']}.ipd.csv")
    if not os.path.isfile(pyf):
        continue
    sub = r_all[r_all.key == f"{r['panel']}__{r['arm']}"]
    if len(sub) == 0:
        continue
    a = pd.read_csv(pyf); b = sub
    ma = median_km_local(a.time.values, a.event.values)
    mb = median_km_local(b.time.values, b.status.values)
    if ma is not None and mb is not None:
        med_py.append(ma); med_r.append(mb)
med_py, med_r = np.array(med_py), np.array(med_r)
axes[1].scatter(med_py, med_r, s=14, alpha=0.65, color="#2ca02c", edgecolors="none")
lim = [0, max(med_py.max(), med_r.max()) * 1.05]
axes[1].plot(lim, lim, color="crimson", lw=0.9)
axes[1].set_xlim(lim); axes[1].set_ylim(lim)
axes[1].set_xlabel("Median, Python reconstruction (months)")
axes[1].set_ylabel("Median, R reconstruction (months)")
axes[1].set_title(f"(b) Median agreement, n = {len(med_py)}", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_py_vs_r.png"), bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 5: corpus overview ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
ep = master["端点"].value_counts()
main_ep = {k: v for k, v in ep.items() if k in ("OS", "PFS", "DOR")}
other = sum(v for k, v in ep.items() if k not in main_ep)
axes[0].bar(list(main_ep.keys()) + ["other"], list(main_ep.values()) + [other],
            color=["#4a7ebb", "#6aa7c9", "#8fc0d6", "#c9d8e6"], width=0.6, edgecolor="white")
for i, (k, v) in enumerate(list(main_ep.items()) + [("other", other)]):
    axes[0].text(i, v + 3, str(v), ha="center", fontsize=8)
axes[0].set_ylabel("Number of curves"); axes[0].set_ylim(0, max(main_ep.values()) * 1.2)
axes[0].set_title("(a) Endpoint distribution (n = 477)", fontsize=9)

per_trial = master[master["重建IPD"].astype(str).str.len() > 0].groupby("试验").size().sort_values(ascending=False)
axes[1].barh(range(len(per_trial)), per_trial.values, color="#4a7ebb", height=0.7)
axes[1].set_yticks(range(len(per_trial)))
axes[1].set_yticklabels(per_trial.index, fontsize=6)
axes[1].invert_yaxis()
axes[1].set_xlabel("Reconstructed arms")
axes[1].set_title("(b) Reconstructed arms per trial", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_corpus.png"), bbox_inches="tight")
plt.close(fig)

print("figures done:", os.listdir(FIG))
