# -*- coding: utf-8 -*-
"""
compare_ipd — Python(Guyot移植) vs R(IPDfromKM) vs 原始数字化曲线 三方对比。

  1. Python IPD 与 R IPD 逐行一致性（人数、事件数、每行 time/status 差异）
  2. 重建 IPD 的 KM 曲线 vs 原数字化曲线：RMSE、中位差、里程碑生存率差
  3. 重建风险数 vs 报告风险数
输出：results/ipd/compare_report.csv + 控制台汇总
"""
import os
import glob
import numpy as np
import pandas as pd

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
OUT = os.path.join(ROOT, "results", "ipd")


def km_from_ipd(times, events):
    order = np.argsort(times, kind="mergesort")
    t, e = np.asarray(times, float)[order], np.asarray(events, int)[order]
    n_all = len(t)
    km_t, km_s = [0.0], [1.0]
    surv, i = 1.0, 0
    while i < n_all:
        tt = t[i]; j = i
        while j < n_all and t[j] == tt:
            j += 1
        d = int(e[i:j].sum())
        if d > 0:
            surv *= (1 - d / (n_all - i))
            km_t.append(float(tt)); km_s.append(surv)
        i = j
    return km_t, km_s


def km_eval(km_t, km_s, tq):
    idx = np.searchsorted(km_t, tq, side="right") - 1
    return np.where(idx >= 0, np.array(km_s)[np.clip(idx, 0, None)], 1.0)


def median_km(km_t, km_s):
    arr = [(t, s) for t, s in zip(km_t, km_s) if s <= 0.5]
    return arr[0][0] if arr else None


def main():
    kv = pd.read_csv(os.path.join(OUT, "key_values.csv"))
    rsum = pd.read_csv(os.path.join(OUT, "r_out", "r_summary.csv"))
    cox = pd.read_csv(os.path.join(OUT, "r_out", "r_cox.csv")) if os.path.isfile(os.path.join(OUT, "r_out", "r_cox.csv")) else None
    # R summary 的 key 已去掉 trial 前缀；kv.panel 形如 "TRIAL__pXXX__panelN"
    rsum["panel_match"] = rsum["key"].apply(lambda k: k.split("__")[0])
    out_rows = []
    n_py, n_r, n_both, n_identical = 0, 0, 0, 0
    for _, row in kv.iterrows():
        base, arm = row["panel"], row["arm"]
        key = base.split("__", 1)[1]           # pXXX__panelN
        py_csv = os.path.join(OUT, f"{base}__{arm}.ipd.csv")
        def read_ipd(path):
            if not os.path.isfile(path):
                return None
            df = pd.read_csv(path)
            tcol = "time" if "time" in df.columns else df.columns[0]
            ecol = "event" if "event" in df.columns else ("status" if "status" in df.columns else df.columns[1])
            return list(zip(df[tcol].astype(float), df[ecol].astype(int)))
        def read_ipd_key(path, key):
            if not os.path.isfile(path):
                return None
            df = pd.read_csv(path)
            ecol = "event" if "event" in df.columns else "status"
            sub = df[df.key == key] if "key" in df.columns else df
            return list(zip(sub["time"].astype(float), sub[ecol].astype(int)))
        py_ipd = read_ipd(py_csv)
        r_ipd = read_ipd_key(os.path.join(OUT, "r_out", "r_ipd_all.csv"), f"{base}__{arm}")
        n_py += py_ipd is not None
        n_r += r_ipd is not None
        if py_ipd and r_ipd:
            n_both += 1
            a = sorted(py_ipd); b = sorted(r_ipd)
            identical = len(a) == len(b) and all(abs(x[0] - y[0]) < 1e-6 and x[1] == y[1] for x, y in zip(a, b))
            n_identical += identical
            row_diff = sum(1 for x, y in zip(a, b) if abs(x[0] - y[0]) > 1e-6 or x[1] != y[1]) if len(a) == len(b) else len(a) + len(b)
        else:
            identical, row_diff = None, None
        # 重建KM vs 原始数字化
        km_t, km_s = km_from_ipd(*zip(*py_ipd)) if py_ipd else ([], [])
        orig_csv = glob.glob(os.path.join(ROOT, "results", row["trial"], f"{base}__{arm}.csv"))
        rmse_curve = med_diff = ""
        if orig_csv:
            od = pd.read_csv(orig_csv[0])
            ot, osv = od.time.values, od.surv.values
            es = km_eval(km_t, km_s, ot)
            rmse_curve = float(np.sqrt(np.mean((es - osv) ** 2)))
            m_r, m_o = median_km(km_t, km_s), median_km(ot, osv) or (None if osv[-1] > 0.5 else ot[-1])
            med_diff = (m_r - m_o) if (m_r is not None and m_o is not None) else ""
        hr_val = ""
        if cox is not None:
            cc = cox[cox.panel == base]
            if len(cc):
                hr_val = f"{cc.iloc[0].HR:.3f} [{cc.iloc[0].lo:.3f}-{cc.iloc[0].hi:.3f}], p={cc.iloc[0].p:.2g}"
        out_rows.append(dict(
            trial=row["trial"], panel=base, arm=arm, endpoint=row["endpoint"],
            n_ipd=row["n_ipd"], n_events=row["n_events"],
            r_status=("OK" if r_ipd else "MISSING"),
            py_r_identical=("YES" if identical else "NO") if identical is not None else "",
            py_r_row_diff=row_diff if row_diff is not None else "",
            median_recon=row["median_recon"], median_digitized=row["median_digitized"],
            median_diff=(round(med_diff, 3) if med_diff != "" else ""),
            curve_rmse=round(rmse_curve, 4) if rmse_curve != "" else "",
            risk_max_abs_err=row["risk_max_abs_err"],
            cox_HR_R=hr_val,
        ))
    df = pd.DataFrame(out_rows)
    df.to_csv(os.path.join(OUT, "compare_report.csv"), index=False, encoding="utf-8-sig")
    ev = df[df["py_r_identical"] != ""]
    print(f"arms: py={n_py} r={n_r} both={n_both} | IPD identical: {n_identical}/{n_both}")
    if len(ev):
        print(f"重建KM vs 原曲线 RMSE: 中位数 {ev.curve_rmse.median():.4f} | 最大 {ev.curve_rmse.max():.4f}")
        md = pd.to_numeric(ev.median_diff, errors="coerce").dropna()
        print(f"重建 vs 数字化 中位差: 中位数 {md.abs().median():.3f} 月 | >0.5月者 {sum(md.abs()>0.5)}")
        re_ = pd.to_numeric(ev.risk_max_abs_err, errors="coerce").dropna()
        print(f"风险数最大绝对误差: 中位数 {re_.median():.0f} | >2者 {sum(re_>2)}")
    bad = df[df["py_r_identical"] == "NO"]
    print("Python≠R 的臂:", len(bad))
    if len(bad):
        print(bad[["trial", "panel", "arm", "n_ipd", "py_r_row_diff"]].to_string(index=False))
    print("written:", os.path.join(OUT, "compare_report.csv"))


if __name__ == "__main__":
    main()
