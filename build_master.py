# -*- coding: utf-8 -*-
"""
build_master — 汇总 人群/干预标注 + IPD重建结果 → results/curves_master.csv
每臂一行：试验、人群、面板、端点、臂、干预(标准名)、角色、重建统计、来源文件。
"""
import os
import sys
import json
import glob

import pandas as pd

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
RESULTS = os.path.join(ROOT, "results")


def main():
    cur = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trial_arms.json"), encoding="utf-8"))
    ann = pd.read_csv(os.path.join(RESULTS, "panels_annotated.csv"))
    kv_path = os.path.join(RESULTS, "ipd", "key_values.csv")
    kv = pd.read_csv(kv_path) if os.path.isfile(kv_path) else pd.DataFrame()
    rep_path = os.path.join(RESULTS, "ipd", "compare_report.csv")
    rep = pd.read_csv(rep_path) if os.path.isfile(rep_path) else pd.DataFrame()

    rep_idx = {(r["panel"], r["arm"]): r for _, r in rep.iterrows()} if len(rep) else {}
    kv_idx = {(r["panel"], r["arm"]): r for _, r in kv.iterrows()} if len(kv) else {}

    rows = []
    for _, a in ann.iterrows():
        trial = a["trial"]
        base = a["panel_file"]
        arm = a["arm_id"]
        cur_trial = cur.get(trial)
        intervention, role, match_how = str(a.get("intervention") or ""), a.get("intervention_role") or "unclear", "extracted"
        if cur_trial:
            arms_cur = cur_trial["arms"]
            label = str(a.get("arm_label") or "").lower()
            hit = None
            if label:
                scored = []
                for ac in arms_cur:
                    # 只用特异性关键词（≥6字符的药名）做标签匹配，
                    # 泛化词(ct/chemo)会把对照臂错误匹配到实验臂
                    kws = [k for k in ac.get("keywords", []) if len(k) >= 6]
                    s = sum(len(k) for k in kws if k.lower() in label)
                    not_ok = any(n.lower() in label for n in ac.get("not", []))
                    if s > 0 and not not_ok:
                        scored.append((s, ac))
                if scored:
                    scored.sort(key=lambda x: -x[0])
                    hit = scored[0][1]
                    match_how = "by-label"
            if hit is None:
                exp = [ac for ac in arms_cur if ac["role"] == "experimental"]
                ctl = [ac for ac in arms_cur if ac["role"] == "control"]
                order = exp + ctl
                try:
                    idx = int(str(arm).replace("arm", "")) - 1
                except ValueError:
                    idx = 0
                hit = order[idx] if idx < len(order) else (ctl[0] if ctl else (order[0] if order else None))
                match_how = "by-order"
            if hit:
                intervention, role = hit["name"], hit["role"]
        population = cur_trial.get("population", "") if cur_trial else ""
        cap = ""
        if pd.notna(a.get("population")) and str(a["population"]).strip():
            cap = str(a["population"])
            population = f"{population}｜面板图注: {cap[:60]}" if population else cap

        krow = kv_idx.get((base, arm))
        rrow = rep_idx.get((base, arm))
        ipd_file = os.path.join(RESULTS, "ipd", f"{base}__{arm}.ipd.csv")
        rows.append(dict(
            试验=trial, 来源=os.path.basename(str(a["source"])), 页=a["page"], 面板=base,
            端点=a["endpoint"],
            人群=population, 干预=intervention, 角色=role, 标注依据=match_how,
            臂ID=arm, 臂原始标签=str(a.get("arm_label") or ""),
            重建IPD=ipd_file if os.path.isfile(ipd_file) else "",
            n_IPD=(int(rrow["n_ipd"]) if rrow is not None and pd.notna(rrow.get("n_ipd")) else ""),
            中位_重建=(float(rrow["median_recon"]) if rrow is not None and pd.notna(rrow.get("median_recon")) else ""),
            中位_数字化=(float(krow["median_digitized"]) if krow is not None and pd.notna(krow.get("median_digitized")) else ""),
            重建vs原曲线RMSE=(float(rrow["curve_rmse"]) if rrow is not None and pd.notna(rrow.get("curve_rmse")) else ""),
            Py与R曲线一致=(str(rrow["py_r_identical"]) if rrow is not None else ""),
            风险表来源=a.get("risk_source"), 风险表列数=a.get("n_risk_cols"),
            QA通过=a.get("qa_pass"),
            面板JSON=os.path.join(trial, base + ".json"),
            数字化CSV=os.path.join(trial, f"{base}__{arm}.csv"),
        ))
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "curves_master.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    rec = (df["重建IPD"] != "").sum()
    print("master rows:", len(df), "| trials:", df["试验"].nunique(), "| reconstructed:", rec)
    print("角色分布:", dict(df["角色"].value_counts()))
    print("端点分布:", dict(df["端点"].value_counts()))
    print("人群已标注:", (df["人群"] != "").sum())
    print("written:", out)


if __name__ == "__main__":
    main()
