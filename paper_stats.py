# -*- coding: utf-8 -*-
"""Aggregate statistics for the manuscript."""

import os, json
import numpy as np
import pandas as pd

ROOT = r"D:\work\IPDR"
S = {}

kv = pd.read_csv(os.path.join(ROOT, r"results\ipd\key_values.csv"))
rep = pd.read_csv(os.path.join(ROOT, r"results\ipd\compare_report.csv"))
ver = pd.read_csv(os.path.join(ROOT, r"results\verification.csv"))
master = pd.read_csv(os.path.join(ROOT, r"results\curves_master.csv"))
rsum = pd.read_csv(os.path.join(ROOT, r"results\ipd\r_out\r_summary.csv"))

# 语料
S["trials"] = int(master["试验"].nunique())
S["curves_total"] = int(len(master))
S["panels_total"] = int(master["面板"].nunique())
S["endpoint"] = master["端点"].value_counts().to_dict()
S["role"] = master["角色"].value_counts().to_dict()
S["pop_annotated"] = int(master["人群"].astype(str).str.strip().ne("").sum())

# 数字化 precision
p = ver["precision"].dropna().astype(float)
S["precision_n"] = int(len(p))
S["precision_median"] = round(float(p.median()), 4)
S["precision_ge09"] = int((p >= 0.9).sum())
S["precision_lt075"] = int((p < 0.75).sum())
S["integrity_bad"] = int((ver["integrity"] != "ok").sum())

# 重建
S["recon_panels"] = int(kv.panel.nunique())
S["recon_arms"] = int(len(kv))
ident = (rep.py_r_identical == "YES").sum()
both = (rep.py_r_identical != "").sum()
S["pyr_both"] = int(both)
S["pyr_identical"] = int(ident)
S["pyr_rmse_max"] = None

# 重建KM vs 原曲线
rmse = pd.to_numeric(rep.curve_rmse, errors="coerce").dropna()
S["curve_rmse_median"] = round(float(rmse.median()), 4)
S["curve_rmse_max"] = round(float(rmse.max()), 4)
md = pd.to_numeric(rep.median_diff, errors="coerce").dropna()
S["median_diff_n"] = int(len(md))
S["median_diff_abs_median"] = round(float(md.abs().median()), 4)
S["median_diff_gt05"] = int((md.abs() > 0.5).sum())
re_ = pd.to_numeric(rep.risk_max_abs_err, errors="coerce").dropna()
S["risk_err_median"] = round(float(re_.median()), 2)

# 发表中位比对（对 verification 中位数用 validate 匹配报告值）
import sys as _sys
_sys.path.insert(0, ".")
from validate import reported_medians, match_median
cache = {}
n_m = n_match = n_le5 = 0
for _, r in ver.iterrows():
    if r["median"] == "" or pd.isna(r["median"]):
        continue
    t = r["trial"]
    if t not in cache:
        cache[t] = reported_medians(t)
    n_m += 1
    v, d = match_median(float(r["median"]), cache[t])
    if v is not None:
        n_match += 1
        if d <= 0.05:
            n_le5 += 1
S["pub_median_total"] = n_m
S["pub_median_matched_n"] = n_match
S["pub_median_le5"] = n_le5

# Cox 样例
cox = pd.read_csv(os.path.join(ROOT, r"results\ipd\r_out\r_cox.csv"))
S["cox_pairs"] = int(len(cox))

print(json.dumps(S, ensure_ascii=False, indent=1, default=str))
json.dump(S, open(os.path.join(ROOT, r"work\paper_stats.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
