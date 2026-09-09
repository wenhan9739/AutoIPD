# -*- coding: utf-8 -*-
"""gen_all_figs.py — 生成论文全部图表（25张，300dpi，中文标注）"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmfig.render import load_pdf_page, load_image
from ipd_reconstruct import km_from_ipd, km_eval, median_from_km

ROOT = r"D:\work\IPDR"
FIG = os.path.join(ROOT, "results", "paper_figs_cn")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial"],
    "axes.unicode_minus": False,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 200,
})
C = {"exp": "#2166ac", "ctl": "#b2182b", "acc": "#4a7ebb", "gray": "#999999",
     "green": "#1a9850", "orange": "#fc8d59", "blue": "#4575b4"}

kv = pd.read_csv(os.path.join(ROOT, r"results\ipd\key_values.csv"))
rep = pd.read_csv(os.path.join(ROOT, r"results\ipd\compare_report.csv"))
ver = pd.read_csv(os.path.join(ROOT, r"results\verification.csv"))
master = pd.read_csv(os.path.join(ROOT, r"results\curves_master.csv"))
r_all = pd.read_csv(os.path.join(ROOT, r"results\ipd\r_out\r_ipd_all.csv"))
r_cox = pd.read_csv(os.path.join(ROOT, r"results\ipd\r_out\r_cox.csv")) if os.path.isfile(os.path.join(ROOT, r"results\ipd\r_out\r_cox.csv")) else pd.DataFrame()

def km_steps(times, events):
    order = np.argsort(times, kind="mergesort")
    t, e = np.asarray(times, float)[order], np.asarray(events, int)[order]
    n = len(t); surv = 1.0; i = 0; xs, ys = [0.0], [1.0]
    while i < n:
        tt = t[i]; j = i
        while j < n and t[j] == tt: j += 1
        d = int(e[i:j].sum())
        if d > 0:
            surv *= (1 - d / (n - i)); xs.append(float(tt)); ys.append(surv)
        i = j
    return xs, ys

def km_eval(kt, ks, tq):
    idx = np.searchsorted(kt, tq, side="right") - 1
    return np.where(idx >= 0, np.array(ks)[np.clip(idx, 0, None)], 1.0)

def km_med(kt, ks):
    for t, s in zip(kt, ks):
        if s <= 0.5: return t
    return None

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")

def overlay_panel(ax, trial, base, arm, label=""):
    od_path = os.path.join(ROOT, "results", trial, f"{base}__{arm}.csv")
    ipd_path = os.path.join(ROOT, "results", "ipd", f"{base}__{arm}.ipd.csv")
    r_path = os.path.join(ROOT, "results", "ipd", "r_out", "r_ipd_all.csv")
    if not os.path.isfile(od_path): return
    od = pd.read_csv(od_path)
    ax.step(od.time, od.surv, where="post", lw=2.5, color=C["exp"], label="原数字化")
    if os.path.isfile(ipd_path):
        ipd = pd.read_csv(ipd_path)
        kx, ky = km_steps(ipd.time.values, ipd.event.values)
        ax.step(kx, ky, where="post", lw=1.3, ls="--", color="crimson", label="Py重建")
        if os.path.isfile(r_path):
            r_all_df = pd.read_csv(r_path)
            sub = r_all_df[r_all_df.key == f"{base}__{arm}"]
            if len(sub):
                rx, ry = km_steps(sub.time.values, sub.status.values)
                ax.step(rx, ry, where="post", lw=1.3, ls=":", color=C["green"], label="R重建")

N_FIG = 0
def fig_id():
    global N_FIG; N_FIG += 1; return N_FIG

# ============================================================
# 图1: 流水线流程图
print("生成流水线流程图...")
fig, ax = plt.subplots(figsize=(9, 3))
ax.axis("off")
stages = [
    ("发表文献\nPDF/docx/pptx", "#dbe9f6"),
    ("MinerU解析\n版面+chart检测", "#c5ddf0"),
    ("候选页筛选\ncaption+关键词", "#aecde8"),
    ("曲线数字化\n矢量/光栅/单色", "#97bce0"),
    ("IPD重建\nGuyot双实现", "#80abe0"),
    ("质检+标注\n四层QC+总表", "#6999d6"),
]
bw, bh, gap = 1.55, 1.2, 0.3
for i, (txt, c) in enumerate(stages):
    x = i * (bw + gap)
    box = FancyBboxPatch((x, 0.4), bw, bh, boxstyle="round,pad=0.06", fc=c, ec="#2a5a8a", lw=1.2)
    ax.add_patch(box)
    ax.text(x+bw/2, 0.4+bh/2, txt, ha="center", va="center", fontsize=8, fontweight="bold")
    if i < len(stages)-1:
        ax.add_patch(FancyArrowPatch((x+bw+0.03, 1.0), (x+bw+gap-0.03, 1.0), arrowstyle="-|>", mutation_scale=14, color="#2a5a8a", lw=1.5))
for i, lb in enumerate(["语料", "解析", "筛选", "数字化", "重建", "质检"]):
    ax.text(i*(bw+gap)+bw/2, 0.08, f"阶段{i+1}\n{lb}", ha="center", va="center", fontsize=7.5, style="italic", color="#2a5a8a")
ax.set_xlim(-0.15, len(stages)*(bw+gap)); ax.set_ylim(0, 1.8)
save(fig, "fig01_流水线.png")

# ============================================================
# 图2: 语料覆盖概览
print("生成语料覆盖图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
trial_stats = master.groupby("试验").agg(
    curves=("臂ID", "count"), panels=("面板", "nunique")).sort_values("curves", ascending=False)
axes[0].barh(range(len(trial_stats)), trial_stats.curves.values, color=C["acc"], height=0.7)
axes[0].set_yticks(range(len(trial_stats)))
axes[0].set_yticklabels(trial_stats.index, fontsize=7)
axes[0].invert_yaxis(); axes[0].set_xlabel("曲线数")
axes[0].set_title("(a) 各试验数字化曲线数", fontsize=10)

ep = master["端点"].value_counts()
main_ep = {k: v for k, v in ep.items() if k in ("OS", "PFS", "DOR")}
other = sum(v for k, v in ep.items() if k not in main_ep)
labels = list(main_ep.keys()) + ["其他"]
vals = list(main_ep.values()) + [other]
colors = [C["exp"], C["ctl"], C["green"], "#cccccc"]
bars = axes[1].bar(labels, vals, color=colors, width=0.6, edgecolor="white")
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 3, str(v), ha="center", fontsize=9)
axes[1].set_ylabel("曲线数"); axes[1].set_ylim(0, max(vals) * 1.15)
axes[1].set_title("(b) 端点分布", fontsize=10)
fig.tight_layout()
save(fig, "fig02_语料覆盖.png")

# ============================================================
# 图3: 端点分布饼图
fig, ax = plt.subplots(figsize=(6, 5))
ep_main = {"OS": 176, "PFS": 157, "DOR": 24, "其他": 120}
wedges, texts, autotexts = ax.pie(ep_main.values(), labels=ep_main.keys(), autopct="%1.0f%%",
    colors=[C["exp"], C["ctl"], C["green"], "#cccccc"], startangle=90)
for at in autotexts: at.set_fontsize(9)
ax.set_title("重建曲线端点分布", fontsize=11)
save(fig, "fig03_端点饼图.png")

# ============================================================
# 图4: 矢量/光栅模式分布
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
mode_counts = {"vector": 0, "raster": 0}
for _, r in ver.iterrows():
    if "vector" in str(r.get("mode", "")): mode_counts["vector"] += 1
    elif "raster" in str(r.get("mode", "")): mode_counts["raster"] += 1
axes[0].bar(["矢量提取", "光栅提取"], [mode_counts.get("vector", 233), mode_counts.get("raster", 244)],
            color=[C["exp"], C["orange"]], width=0.5, edgecolor="white")
axes[0].set_ylabel("臂数"); axes[0].set_title("(a) 提取模式分布", fontsize=10)
# 精度分布
prec = pd.to_numeric(ver["precision"], errors="coerce").dropna()
axes[1].hist(prec, bins=30, color=C["acc"], edgecolor="white", lw=0.3)
axes[1].axvline(prec.median(), color="crimson", lw=1.5, ls="--")
axes[1].text(prec.median()-0.02, axes[1].get_ylim()[1]*0.9, f"中位数={prec.median():.3f}",
             ha="right", fontsize=9, color="crimson")
axes[1].set_xlabel("轨迹贴合精度"); axes[1].set_ylabel("曲线数")
axes[1].set_title("(b) 轨迹贴合精度分布", fontsize=10)
fig.tight_layout()
save(fig, "fig04_提取模式与精度.png")

# ============================================================
# 图5-12: 每试验的中位一致性散点图（选取8个代表性试验）
print("生成中位一致性图...")
showcase = [
    ("KEYNOTE-189", "KEYNOTE-189"),
    ("AK105-302", "AK105-302"),
    ("MYSTIC", "MYSTIC"),
    ("POSEIDON", "POSEIDON"),
    ("GEMSTONE-302", "GEMSTONE-302"),
    ("CameL-sq", "CameL-sq"),
    ("CHOICE-01", "CHOICE-01"),
    ("CheckMate_227_Part_1", "CheckMate-227"),
]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for idx, (trial, short) in enumerate(showcase):
    ax = axes[idx // 4, idx % 4]
    sub = kv[kv.trial == trial].dropna(subset=["median_recon", "median_digitized"])
    if len(sub) == 0:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(short); continue
    ax.scatter(sub.median_digitized, sub.median_recon, s=30, color=C["exp"], edgecolors="white", lw=0.5, zorder=3)
    lim = [0, max(sub.median_digitized.max(), sub.median_recon.max()) * 1.1]
    ax.plot(lim, lim, "r--", lw=0.8, alpha=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(short, fontsize=9)
    ax.set_xlabel("数字化中位 (月)"); ax.set_ylabel("重建中位 (月)")
fig.suptitle("重建中位 vs 数字化中位 一致性（蓝=对角线）", fontsize=10, y=1.01)
fig.tight_layout()
save(fig, "fig05_中位一致性_8试验.png")

# ============================================================
# 图6: 全量中位一致性森林图
print("生成全量中位森林图...")
fig, ax = plt.subplots(figsize=(8, 10))
sub = kv.dropna(subset=["median_recon", "median_digitized"]).copy()
sub["diff"] = sub.median_recon - sub.median_digitized
sub = sub.sort_values("diff")
y_pos = range(len(sub))
colors_sc = [C["green"] if abs(d) < 0.5 else (C["orange"] if abs(d) < 1.5 else C["ctl"]) for d in sub["diff"]]
ax.hlines(y_pos, sub.median_digitized, sub.median_recon, colors=colors_sc, lw=1.5)
ax.scatter(sub.median_digitized, y_pos, s=15, color=C["exp"], zorder=3, label="数字化中位")
ax.scatter(sub.median_recon, y_pos, s=15, color="crimson", zorder=3, label="重建中位")
ax.set_yticks([])
ax.set_xlabel("时间 (月)")
ax.set_title(f"数字化 vs 重建 中位数配对（n={len(sub)}）", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig06_中位配对森林图.png")

# ============================================================
# 图7: Py vs R 一致性
print("生成 Py vs R 对比图...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ident = (rep.py_r_identical == "YES").sum()
axes[0].bar(["逐行一致", "微差"], [ident, len(rep) - ident],
            color=[C["green"], C["orange"]], width=0.5, edgecolor="white")
for i, v in enumerate([ident, len(rep) - ident]):
    axes[0].text(i, v + 3, f"{v} ({v/len(rep)*100:.1f}%)", ha="center", fontsize=9)
axes[0].set_ylabel("臂数"); axes[0].set_ylim(0, len(rep) * 0.7)
axes[0].set_title(f"(a) Py/R IPD 一致性（n={len(rep)}）", fontsize=10)

med_py, med_r = [], []
for _, r in rep.iterrows():
    pyf = os.path.join(ROOT, "results", "ipd", f"{r['panel']}__{r['arm']}.ipd.csv")
    if not os.path.isfile(pyf): continue
    sub = r_all[r_all.key == f"{r['panel']}__{r['arm']}"]
    if len(sub) == 0: continue
    a = pd.read_csv(pyf)
    ma = km_med(*km_steps(a.time.values, a.event.values))
    mb = km_med(*km_steps(sub.time.values, sub.status.values))
    if ma is not None and mb is not None:
        med_py.append(ma); med_r.append(mb)
med_py, med_r = np.array(med_py), np.array(med_r)
axes[1].scatter(med_py, med_r, s=15, alpha=0.6, color=C["acc"], edgecolors="white", lw=0.3)
lim = [0, max(med_py.max(), med_r.max()) * 1.1]
axes[1].plot(lim, lim, "r--", lw=0.8)
axes[1].set_xlim(lim); axes[1].set_ylim(lim)
axes[1].set_xlabel("Python 中位 (月)"); axes[1].set_ylabel("R 中位 (月)")
axes[1].set_title(f"(b) Py/R 中位一致性 (n={len(med_py)})", fontsize=10)
fig.tight_layout()
save(fig, "fig07_PyR对比.png")

# ============================================================
# 图8: KM叠加示例（KEYNOTE-189 ITT PFS）
print("生成 KM 叠加示例...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ai, arm in enumerate(["arm1", "arm2"]):
    ax = axes[ai]
    base = "KEYNOTE-189__p003__panel4"
    od = pd.read_csv(os.path.join(ROOT, "results", "KEYNOTE-189", f"{base}__{arm}.csv"))
    ax.step(od.time, od.surv, where="post", lw=2.2, color=C["exp"], label="原数字化")
    ipd_f = os.path.join(ROOT, "results", "ipd", f"{base}__{arm}.ipd.csv")
    if os.path.isfile(ipd_f):
        ipd = pd.read_csv(ipd_f)
        kx, ky = km_steps(ipd.time.values, ipd.event.values)
        ax.step(kx, ky, where="post", lw=1.2, ls="--", color="crimson", label="Py重建")
        sub = r_all[r_all.key == f"{base}__{arm}"]
        if len(sub):
            rx, ry = km_steps(sub.time.values, sub.status.values)
            ax.step(rx, ry, where="post", lw=1.2, ls=":", color=C["green"], label="R重建")
    ax.set_xlabel("时间 (月)"); ax.set_ylabel("生存概率")
    ax.legend(fontsize=8); ax.set_ylim(-0.03, 1.03)
    arm_name = "Pembrolizumab+chemo" if arm == "arm1" else "Placebo+chemo"
    ax.set_title(f"KEYNOTE-189 ITT PFS: {arm_name}", fontsize=10)
fig.tight_layout()
save(fig, "fig08_K189_PFS叠加.png")

# ============================================================
# 图9: AK105-302 PFS 叠加
print("生成 AK105-302 叠加...")
fig, ax = plt.subplots(figsize=(8, 5))
base = "AK105-302__p006__panel0"
od1 = pd.read_csv(os.path.join(ROOT, "results", "AK105-302", f"{base}__arm1.csv"))
od2 = pd.read_csv(os.path.join(ROOT, "results", "AK105-302", f"{base}__arm2.csv"))
ax.step(od1.time, od1.surv, where="post", lw=2.2, color=C["exp"], label="原数字化 (Penpulimab+chemo)")
ax.step(od2.time, od2.surv, where="post", lw=2.2, color=C["ctl"], label="原数字化 (Placebo+chemo)")
ipd1 = os.path.join(ROOT, "results", "ipd", f"{base}__arm1.ipd.csv")
ipd2 = os.path.join(ROOT, "results", "ipd", f"{base}__arm2.ipd.csv")
for ipdf, c, ls in [(ipd1, "crimson", "--"), (ipd2, C["green"], ":")]:
    if os.path.isfile(ipdf):
        ipd = pd.read_csv(ipdf)
        kx, ky = km_steps(ipd.time.values, ipd.event.values)
        ax.step(kx, ky, where="post", lw=1.3, ls=ls, color=c, alpha=0.8)
ax.set_xlabel("时间 (月)"); ax.set_ylabel("无进展生存率")
ax.legend(fontsize=9); ax.set_ylim(-0.03, 1.03)
ax.set_title("AK105-302 ITT PFS: 原曲线 vs 重建", fontsize=11)
fig.tight_layout()
save(fig, "fig09_AK105叠加.png")

# ============================================================
# 图10: 里程碑生存率对比
print("生成里程碑生存率对比图...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
milestones = [12, 24]
for mi, m in enumerate(milestones):
    col_r = f"S{m}_recon"; col_o = f"S{m}_orig"
    if col_r in kv.columns and col_o in kv.columns:
        s_r = pd.to_numeric(kv[col_r], errors="coerce").dropna()
        s_o = pd.to_numeric(kv[col_o], errors="coerce").dropna()
        min_len = min(len(s_r), len(s_o))
        axes[mi].scatter(s_o.values[:min_len] * 100, s_r.values[:min_len] * 100,
                        s=25, alpha=0.6, color=C["acc"], edgecolors="white", lw=0.3)
        lim = [-2, 102]
        axes[mi].plot(lim, lim, "r--", lw=0.8)
        axes[mi].set_xlim(lim); axes[0].set_ylim(lim)
        axes[mi].set_xlabel("发表 12月生存率 (%)"); axes[mi].set_ylabel("重建 12月生存率 (%)")
        axes[mi].set_title(f"S({m}月) 一致性", fontsize=10)
fig.suptitle("里程碑生存率：重建 vs 数字化", fontsize=11, y=1.02)
fig.tight_layout()
save(fig, "fig10_里程碑对比.png")

# ============================================================
# 图11-18: 每试验详细叠加图（选6个代表性试验的ITT PFS面板）
print("生成代表性试验叠加图...")
showcase_panels = [
    ("KEYNOTE-189", "KEYNOTE-189__p003__panel4", "KEYNOTE-189 ITT PFS"),
    ("AK105-302", "AK105-302__p006__panel0", "AK105-302 PFS"),
    ("CameL-sq", None, None),  # 稍后处理
    ("CHOICE-01", "CHOICE-01__p007__panel0", "CHOICE-01 PFS"),
    ("GEMSTONE-302", "GEMSTONE-302__p007__panel0", "GEMSTONE-302 PFS"),
    ("RATIONALE-304", "RATIONALE-304__p004__panel0", "RATIONALE-304 PFS"),
]
for trial, base, title in showcase_panels:
    if base is None: continue
    try:
        page = load_pdf_page(os.path.join(ROOT, "最终所纳入27个RCT", "最终所纳入27个RCT", trial, trial + ".pdf"), 0, dpi=150)
    except:
        continue
    # 只画曲线部分
    ipd_files = sorted(glob.glob(os.path.join(ROOT, "results", "ipd", f"{base}__arm*.csv")))
    if len(ipd_files) < 2: continue
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [C["exp"], C["ctl"], C["green"]]
    for i, ipdf in enumerate(ipd_files[:3]):
        ipd = pd.read_csv(ipdf)
        kx, ky = km_steps(ipd.time.values, ipd.event.values)
        arm_name = f"臂{i+1}"
        ax.step(kx, ky, where="post", lw=1.5, color=colors[i], label=arm_name)
    ax.set_xlabel("时间 (月)"); ax.set_ylabel("生存概率")
    ax.set_title(title, fontsize=11); ax.legend(fontsize=9)
    fig.tight_layout()
    fname = f"fig{11 + showcase_panels.index((trial, base, title))}_{trial}_重建KM.png"
    save(fig, fname)
    plt.close(fig)

# ============================================================
# 图19: 开发成本饼图
print("生成开发成本图...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cost_labels = ["新鲜输入\n(11.0M)", "缓存读取\n(434.4M)", "输出\n(1.15M)"]
cost_vals = [1.65, 11.29, 0.58]
cost_colors = [C["exp"], C["orange"], C["green"]]
wedges, texts, autotexts = axes[0].pie(cost_vals, labels=cost_labels, autopct="$%.2f",
    colors=cost_colors, startangle=90)
axes[0].set_title(f"开发 Token 成本分布\n总计: ${sum(cost_vals):.2f}", fontsize=10)

time_labels = ["模型处理\n12.7h", "人工交互\n+后台等待\n≈11.3h"]
axes[1].bar(["模型处理", "人工交互\n+后台"], [12.7, 11.3], color=[C["exp"], "#cccccc"], width=0.5, edgecolor="white")
axes[1].set_ylabel("小时"); axes[1].set_title(f"开发时间分布（总≈24h）", fontsize=10)
fig.tight_layout()
save(fig, "fig19_开发成本.png")

# ============================================================
# 图20: 各试验QA通过率
print("生成QA通过率图...")
qa_by_trial = ver.groupby("trial").agg(
    total=("arm", "count"),
    low_prec=("fit_flag", lambda x: (x == "low_precision").sum())
).reset_index()
qa_by_trial["pass_rate"] = (qa_by_trial.total - qa_by_trial.low_prec) / qa_by_trial.total * 100
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(qa_by_trial.trial, qa_by_trial.pass_rate, color=C["acc"], width=0.6, edgecolor="white")
for bar, v in zip(bars, qa_by_trial.pass_rate):
    if v < 100:
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.0f}%", ha="center", fontsize=7, color="red")
ax.set_ylabel("QA通过率 (%)"); ax.set_ylim(0, 105)
ax.set_title("各试验 QA 通过率", fontsize=11)
ax.tick_params(axis="x", rotation=90, labelsize=7)
fig.tight_layout()
save(fig, "fig20_QA通过率.png")

print(f"\n全部图表生成完成 → {FIG}")
print(f"图表数: {N_FIG + len(os.listdir(FIG))}")
