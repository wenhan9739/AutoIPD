# -*- coding: utf-8 -*-
"""Batch IPD reconstruction driver (Python Guyot) + survival analysis + comparison."""

import os
import sys
import json
import glob
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmfig.guyot import preprocess, getIPD

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "ipd")
MILESTONES = [12.0, 24.0, 36.0, 48.0, 60.0]


def km_from_ipd(times, events):
    """从伪IPD直接计算KM阶梯（与survfit一致）。返回 (km_t, km_s)。"""
    order = np.argsort(times, kind="mergesort")
    t, e = np.asarray(times, float)[order], np.asarray(events, int)[order]
    n_all = len(t)
    km_t, km_s = [0.0], [1.0]
    surv = 1.0
    i = 0
    while i < n_all:
        tt = t[i]
        j = i
        while j < n_all and t[j] == tt:
            j += 1
        n_risk = n_all - i          # 该时点在险人数（尚未退出观察者）
        d = int(e[i:j].sum())
        if d > 0:
            surv *= (1 - d / n_risk)
            km_t.append(float(tt)); km_s.append(surv)
        i = j
    return km_t, km_s


def km_eval(km_t, km_s, t):
    """阶梯取值（右连续）。"""
    idx = np.searchsorted(km_t, t, side="right") - 1
    out = np.where(idx >= 0, np.array(km_s)[np.clip(idx, 0, None)], 1.0)
    return out


def median_from_km(km_t, km_s, tol=0.0):
    for t, s in zip(km_t, km_s):
        if s <= 0.5 + tol:
            return float(t)
    return None


def match_risk_rows(d):
    """把 at_risk 行匹配到臂：先按颜色，再按 y 位置顺序。"""
    arms = d["arms"]
    rows = d.get("at_risk") or []
    if not rows:
        return {}
    def close(c1, c2, tol=0.1):
        return c1 is not None and c2 is not None and all(abs(a - b) <= tol for a, b in zip(c1, c2))
    assigned = {}
    used = set()
    for a in arms:
        for k, row in enumerate(rows):
            if k in used:
                continue
            if close(row.get("color"), a.get("color")):
                assigned[id(a)] = row
                used.add(k)
                break
    rest_rows = [k for k in range(len(rows)) if k not in used]
    rest_arms = [a for a in arms if id(a) not in assigned]
    rest_rows.sort(key=lambda k: rows[k].get("y_px", 0))
    for a, k in zip(rest_arms, rest_rows):
        assigned[id(a)] = rows[k]
        used.add(k)
    return assigned


def cox_two_arm(ipd0, ipd1):
    """两臂Cox（arm1=1 为实验组 vs arm0=0）。返回 HR, ci_low, ci_high, p。"""
    df = pd.DataFrame(
        {"time": [p[0] for p in ipd0 + ipd1],
         "event": [p[1] for p in ipd0 + ipd1],
         "arm": [0] * len(ipd0) + [1] * len(ipd1)})
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    s = cph.summary.loc["arm"]
    return float(np.exp(s["coef"])), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), float(s["p"])


def digitize_panel_file(jp):
    d = json.load(open(jp, encoding="utf-8"))
    if not d["qa"].get("pass"):
        return None
    if len(d.get("at_risk") or []) < 2 or len(d.get("arms") or []) < 2:
        return None
    return d


def main(limit=None):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "r_input"), exist_ok=True)
    long_rows, risk_long_rows = [], []
    jsons = [j for j in sorted(glob.glob(os.path.join(RESULTS, "*", "*__panel*.json")))
             if "risk_ocr" not in j]
    key_rows = []
    done = 0
    for jp in jsons:
        if limit and done >= limit:
            break
        d = digitize_panel_file(jp)
        if d is None:
            continue
        trial = os.path.basename(os.path.dirname(jp))
        base = os.path.splitext(os.path.basename(jp))[0]
        risk_map = match_risk_rows(d)
        arm_ipds = {}
        ok_arms = []
        r_export = []
        for a in d["arms"]:
            row = risk_map.get(id(a))
            if row is None:
                continue
            trisk = [float(t) for t in row["times"]]
            nrisk = [float(c) for c in row["counts"]]
            # 风险表首列必在 t=0；OCR/标定抖动产生的小正数归零
            if trisk and 0 < trisk[0] <= 0.03 * max(trisk[-1], 1.0):
                trisk[0] = 0.0
            if len(trisk) < 2 or trisk[0] > 0:
                continue
            r_export.append(dict(arm=a["arm_id"], key=f"{base}__{a['arm_id']}",
                                 times=[t for t, _ in a["steps"]],
                                 survs=[s for _, s in a["steps"]],
                                 trisk=trisk, nrisk=nrisk))
            try:
                prep = preprocess([t for t, _ in a["steps"]], [s for _, s in a["steps"]],
                                  trisk=trisk, nrisk=nrisk, maxy=1)
                est = getIPD(prep, arm_id=1)
            except Exception as e:
                print(f"  [guyot-error] {trial} {base} {a['arm_id']}: {e}")
                continue
            arm_ipds[a["arm_id"]] = est
            ok_arms.append((a, row, est))
        if len(ok_arms) < 2:
            continue
        # 导出 R(IPDfromKM) 输入（追加到长表，R v2 一次性读取）
        long_rows.append(pd.DataFrame([
            dict(key=re_["key"], time=t, surv=s)
            for re_ in r_export for t, s in zip(re_["times"], re_["survs"])]))
        risk_long_rows.append(pd.DataFrame([
            dict(key=re_["key"], trisk=tr, nrisk=nr)
            for re_ in r_export for tr, nr in zip(re_["trisk"], re_["nrisk"])]))
        # IPD CSV + 图 + 关键值
        fig, ax = plt.subplots(figsize=(7, 5))
        cox_arms = []
        for ai, (a, row, est) in enumerate(ok_arms):
            ipd = est["IPD"]
            tcol = [p[0] for p in ipd]; ecol = [p[1] for p in ipd]
            pd.DataFrame({"time": tcol, "event": ecol}).to_csv(
                os.path.join(OUT, f"{base}__{a['arm_id']}.ipd.csv"), index=False)
            km_t, km_s = km_from_ipd(tcol, ecol)
            med_r = median_from_km(km_t, km_s)
            orig_t = [p[0] for p in a["steps"]]
            orig_s = [p[1] for p in a["steps"]]
            # 里程碑生存率（重建 vs 数字化）
            mile = {}
            for m in MILESTONES:
                s_re = km_eval(km_t, km_s, [m])[0]
                s_di = km_eval(orig_t, orig_s, [m])[0]
                mile[m] = (float(s_re), float(s_di))
            # 风险数误差：重建IPD在报告时点的风险数 vs 报告值
            tarr = np.array(tcol)
            risk_err = []
            for t_rep, n_rep in zip(row["times"], row["counts"]):
                n_re = int((tarr >= t_rep - 1e-9).sum())
                risk_err.append(abs(n_re - n_rep))
            key_rows.append(dict(
                trial=trial, panel=base, arm=a["arm_id"], label=a.get("label") or "",
                endpoint=d.get("endpoint"),
                n_ipd=len(ipd), n_events=sum(ecol), n_censored=len(ecol) - sum(ecol),
                median_recon=round(med_r, 3) if med_r else "",
                median_digitized=round(a["median"], 3) if a.get("median") else "",
                rmse_recon_vs_orig=est["rmse"],
                risk_max_abs_err=max(risk_err) if risk_err else "",
                **{f"S{int(m)}_recon": round(mile[m][0], 4) for m in MILESTONES},
                **{f"S{int(m)}_orig": round(mile[m][1], 4) for m in MILESTONES},
            ))
            ok_arms[ai] = (a, row, est, km_t, km_s)
            # 存为 (time,event) 对列表；直接存 (tcol,ecol) 元组会把两支列表
            # 当成两条观测传给 Cox（n=2 的假拟合）
            cox_arms.append(list(zip(tcol, ecol)))
            ax.step(orig_t, orig_s, where="post", lw=2.2, label=f"{a['arm_id']} original")
            ax.step(km_t, km_s, where="post", lw=1.0, ls="--", label=f"{a['arm_id']} reconstructed")
        # Cox（前两臂）
        if len(cox_arms) == 2:
            try:
                hr, lo, hi, p = cox_two_arm(cox_arms[0], cox_arms[1])
                for r in key_rows[-2:]:
                    r["cox_HR"], r["cox_CI"], r["cox_p"] = round(hr, 4), f"[{lo:.3f},{hi:.3f}]", round(p, 5)
                ax.text(0.02, 0.02, f"reconstructed Cox HR={hr:.2f} ({lo:.2f}-{hi:.2f}), p={p:.1g}",
                        transform=ax.transAxes, fontsize=8)
            except Exception as e:
                print(f"  [cox-error] {base}: {e}")
        ax.set_title(f"{trial} {base}")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plots", base + ".png"), dpi=130)
        plt.close(fig)
        done += 1
        print(f"[{done}] {trial} {base}: {len(ok_arms)} arms reconstructed")
    # 汇总
    df = pd.DataFrame(key_rows)
    df.to_csv(os.path.join(OUT, "key_values.csv"), index=False, encoding="utf-8-sig")
    # 供 R ipdfromkm_batch.R (v2) 使用的单次读取长表
    if long_rows:
        pd.concat(long_rows).to_csv(os.path.join(OUT, "r_input", "allarms_long.csv"), index=False)
    if risk_long_rows:
        pd.concat(risk_long_rows).to_csv(os.path.join(OUT, "r_input", "risks_long.csv"), index=False)
    print("DONE:", len(key_rows), "arms |", done, "panels →", OUT)


if __name__ == "__main__":
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    main(limit)
