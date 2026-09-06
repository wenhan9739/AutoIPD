# -*- coding: utf-8 -*-
"""
kmfig.guyot — Guyot 2012 IPD 重建算法的 Python 实现。

忠实移植自 IPDfromKM 0.1.10 (CRAN) 的 preprocess.R 与 getIPD.R，
逐行对应其数值细节（包括 nhat 初值 n.risk[1]+1、R 的 round 半舍入、
删失时间在区间内均匀分布等），以便与 R 版结果直接对拍。

输入：
  times/survs : 数字化阶梯顶点（t 从 0 开始，s 为 0-1）
  trisk/nrisk : 风险表（报告时间点、对应风险数）；或 totalpts（初始人数）
输出：
  preprocess → dict(dat, lower, upper, t_risk, n_risk, endpts)   （下标已转 0 基）
  getIPD     → dict(IPD=[(time,status)], nhat, cen, d, KMhat, estsurv, rmse...)
"""
import numpy as np


def _rdiv(a, b):
    """R 语义的除法：0/0=NaN，x/0=±Inf（Python 会抛异常）。"""
    if b == 0:
        if a == 0:
            return float("nan")
        return float("inf") if a > 0 else float("-inf")
    return a / b


def _quantile7(x, q):
    """R quantile(type=7)，与 numpy percentile 'linear' 相同。"""
    return float(np.percentile(x, q * 100))


def preprocess(times, survs, trisk=None, nrisk=None, totalpts=None, maxy=1):
    T = [float(t) for t in times]
    S = [float(s) / maxy for s in survs]
    rows = sorted(zip(T, S), key=lambda r: r[0])
    T = [r[0] for r in rows]
    S = [r[1] for r in rows]

    # 剔除离群点（Tukey fences, k=0.5，作用于相邻读数差 |surv - lag|）
    span = [abs(S[i] - (S[i - 1] if i > 0 else S[0])) for i in range(len(S))]
    q1, q3 = _quantile7(span, 0.25), _quantile7(span, 0.75)
    iqr = q3 - q1
    outl = [(x <= q1 - 0.5 * iqr) or (x >= q3 + 0.5 * iqr) for x in span]
    keep = [not (outl[i] and (outl[i - 1] if i > 0 else False) and (outl[i + 1] if i + 1 < len(S) else True))
            for i in range(len(S))]
    T = [t for t, k in zip(T, keep) if k]
    S = [s for s, k in zip(S, keep) if k]

    # 生存率非增
    for i in range(1, len(S)):
        if S[i] > S[i - 1]:
            S[i] = S[i - 1]
    # 去重
    ded = []
    for t, s in zip(T, S):
        if not ded or (t, s) != ded[-1]:
            ded.append((t, s))
    T = [r[0] for r in ded]; S = [r[1] for r in ded]

    # 同一时间点最多保留两个读数（该时间的最大、最小生存率）
    from collections import OrderedDict
    groups = OrderedDict()
    for t, s in zip(T, S):
        groups.setdefault(t, []).append(s)
    newrows = []
    for t, ss in groups.items():
        ss_sorted = sorted(ss, reverse=True)
        take = ss_sorted[:1] if len(ss_sorted) == 1 else [ss_sorted[0], ss_sorted[-1]]
        for s in take:
            newrows.append((t, s))
    T = [r[0] for r in newrows]; S = [r[1] for r in newrows]

    # 过滤、排序，并确保起点 (0,1)
    filt = [(t, s) for t, s in zip(T, S) if 0 <= s <= 1 and t >= 0]
    filt.sort(key=lambda r: r[0])
    T = [r[0] for r in filt]; S = [r[1] for r in filt]
    if not T or T[0] != 0 or S[0] != 1:
        T = [0.0] + T; S = [1.0] + S

    total = len(T)

    # 风险表整理
    if nrisk is not None and trisk is not None and len(nrisk) > 0 and len(trisk) > 0:
        nrisk = [float(x) for x in nrisk]
        trisk = [float(x) for x in trisk]
        nint = len(nrisk)
        while nint > 1 and nrisk[nint - 1] == 0 and nrisk[nint - 2] == 0:
            nint -= 1
        nrisk = nrisk[:nint]; trisk = trisk[:nint]
    elif totalpts is not None:
        nrisk = [float(totalpts)]; trisk = [0.0]; nint = 1
    else:
        raise ValueError("需要提供 trisk/nrisk 或 totalpts")

    # 每个读数点归属区间（R locate_interval）
    def locate(x):
        interval = 0  # 0 基
        for i in range(1, nint):
            if x >= trisk[i]:
                interval = i
        return interval

    intervals = [locate(t) for t in T]

    # riskmat：每个区间的 lower/upper（0 基）与 t.risk/n.risk
    riskmat = []
    for iv in range(nint):
        idx = [k for k, ivv in enumerate(intervals) if ivv == iv]
        if not idx:
            continue
        riskmat.append(dict(lower=min(idx), upper=max(idx),
                            t_risk=trisk[iv], n_risk=nrisk[iv]))

    # endpts：曲线终点早于最后报告时间时，末尾存活人数
    endpts = None
    if riskmat and max(rm["t_risk"] for rm in riskmat) < max(trisk):
        endpts = min(nrisk)

    return dict(time=T, surv=S, total=total, riskmat=riskmat, endpts=endpts, trisk_all=trisk)


def _est_cen(ncen, i, lower, upper, TT):
    """区间 i 内 ncen 个删失在 [TT[lower], TT[upper]] 上均匀分布（R est_cen）。"""
    low, upp = lower[i], upper[i]
    k = upp - low + 1
    if k < 0:
        raise ValueError("Riskmat error: upper<lower")
    if k == 0:
        return [], (0 if ncen <= 0 else float(ncen))
    cen = [0.0] * k
    if ncen > 0:
        cen_t = [TT[low] + (j + 1) * (TT[upp] - TT[low]) / (ncen + 1) for j in range(int(ncen))]
        m = 0
        for j0 in range(low, upp + 1):
            nxt = TT[j0 + 1] if j0 + 1 < len(TT) else None
            if nxt is None:
                cen[m] = sum(1 for ct in cen_t if TT[j0] <= ct)
            else:
                cen[m] = sum(1 for ct in cen_t if TT[j0] <= ct < nxt)
            m += 1
        cen = [0.0 if (c is None or (isinstance(c, float) and np.isnan(c))) else c for c in cen]
    return cen, float(sum(cen))


def getIPD(prep, arm_id=1, tot_events=None, rng=None):
    """R getIPD 的逐行移植。返回 dict。"""
    TT = list(prep["time"]); SS = list(prep["surv"])
    total = prep["total"]
    rmat = prep["riskmat"]
    ninterval = len(rmat)
    lower = [rm["lower"] for rm in rmat]
    upper = [rm["upper"] for rm in rmat]
    t_risk = [rm["t_risk"] for rm in rmat]
    n_risk = [rm["n_risk"] for rm in rmat]

    # R round: 与 Python round 一样为半偶数舍入
    ncensor = [0.0] * ninterval
    lasti = [0] * ninterval
    cen = [0.0] * total
    nhat = [n_risk[0] + 1] * (total + 1)
    d = [0.0] * total
    KMhat = [1.0] * total

    if ninterval > 1:
        for i in range(ninterval - 1):
            ncensor[i] = round(n_risk[i] * SS[lower[i + 1]] / SS[lower[i]] - n_risk[i + 1])
            while ((nhat[lower[i + 1]] > n_risk[i + 1]) and (ncensor[i] < (n_risk[i] - n_risk[i + 1] + 1))) or \
                  ((nhat[lower[i + 1]] < n_risk[i + 1]) and (ncensor[i] > 0)):
                cen_v, ncen_v = _est_cen(ncensor[i], i, lower, upper, TT)
                cen[lower[i]:upper[i] + 1] = cen_v
                ncensor[i] = ncen_v
                nhat[lower[i]] = n_risk[i]
                las = lasti[i]
                for k in range(lower[i], upper[i] + 1):
                    if k == 0:
                        d[k] = 0.0; KMhat[k] = 1.0
                    else:
                        if KMhat[las] != 0:
                            d[k] = round(nhat[k] * (1 - _rdiv(SS[k], KMhat[las])))
                        else:
                            d[k] = 0
                        KMhat[k] = KMhat[las] * (1 - _rdiv(d[k], nhat[k]))
                    nhat[k + 1] = nhat[k] - d[k] - cen[k]
                    if d[k] != 0:
                        las = k
                    if nhat[k + 1] < 0:
                        nhat[k + 1] = 0.0
                ncensor[i] += (nhat[lower[i + 1]] - n_risk[i + 1])
            n_risk[i + 1] = nhat[lower[i + 1]]
            lasti[i + 1] = las

    # 最后一个区间
    if ninterval > 1:
        if tot_events is None:
            leftd = 0
        else:
            temp = sum(d[0:upper[ninterval - 1] + 1])
            leftd = max(0.0, tot_events - temp)
        mm = 0 if prep["endpts"] is None else prep["endpts"]
        mean_prev = float(np.mean(ncensor[0:ninterval - 1])) if ninterval - 1 > 0 else 0.0
        ncensor[ninterval - 1] = min(
            mean_prev * (TT[total - 1] - t_risk[ninterval - 1]) /
            (t_risk[ninterval - 1] - t_risk[ninterval - 2]),
            n_risk[ninterval - 1] - mm - leftd)
    else:
        if tot_events is None:
            ncensor[ninterval - 1] = 0
        else:
            ncensor[ninterval - 1] = n_risk[ninterval - 1] - tot_events

    cen_v, ncen_v = _est_cen(ncensor[ninterval - 1], ninterval - 1, lower, upper, TT)
    cen[lower[ninterval - 1]:upper[ninterval - 1] + 1] = cen_v
    ncensor[ninterval - 1] = ncen_v

    nhat[lower[ninterval - 1]] = n_risk[ninterval - 1]
    las = lasti[ninterval - 1]
    for k in range(lower[ninterval - 1], upper[ninterval - 1] + 1):
        if k == 0:
            d[k] = 0.0; KMhat[k] = 1.0
        else:
            if KMhat[las] != 0:
                d[k] = round(nhat[k] * (1 - _rdiv(SS[k], KMhat[las])))
            else:
                d[k] = 0
            KMhat[k] = KMhat[las] * (1 - _rdiv(d[k], nhat[k]))
        nhat[k + 1] = nhat[k] - d[k] - cen[k]
        if nhat[k + 1] < 0:
            nhat[k + 1] = 0.0
            cen[k] = nhat[k] - d[k]
        if d[k] != 0:
            las = k

    # 重建 IPD
    ipd = []  # (time, status)
    for i in range(total):
        if d[i] > 0:
            ipd += [(TT[i], 1)] * int(d[i])
        if cen[i] > 0:
            t = (TT[i] + TT[i + 1]) / 2.0 if i < total - 1 else TT[i]
            ipd += [(t, 0)] * int(cen[i])
    if nhat[total] > 0:
        ipd += [(TT[total - 1], 0)] * int(nhat[total])

    # 重构 KM 曲线在每个读数点的估计值（对应 R survfit+anypoint）
    ipd_t = np.array([p[0] for p in ipd], float)
    ipd_s = np.array([p[1] for p in ipd], int)
    order = np.argsort(ipd_t, kind="mergesort")
    ipd_t, ipd_s = ipd_t[order], ipd_s[order]
    km_t, km_s = [0.0], [1.0]
    at_risk, surv = len(ipd_t), 1.0
    i = 0
    while i < len(ipd_t):
        t = ipd_t[i]
        j = i
        while j < len(ipd_t) and ipd_t[j] == t:
            j += 1
        n_i = at_risk
        n_event = int(ipd_s[i:j].sum())
        if n_i > 0 and n_event > 0:
            surv *= (1 - n_event / n_i)
            km_t.append(t); km_s.append(surv)
        at_risk -= (j - i)
        i = j

    def anypoint(a):
        if a < km_t[0]:
            return 1.0
        if a > km_t[-1]:
            return km_s[-1]
        for idx in range(len(km_t)):
            if a == km_t[idx]:
                return km_s[idx]
            if a < km_t[idx]:
                return km_s[idx - 1]
        return km_s[-1]

    estsurv = [round(anypoint(t), 3) for t in TT]
    diffs = [e - s for e, s in zip(estsurv, SS)]
    n_d = len(diffs)
    rmse = round(float(np.sqrt(sum(x * x for x in diffs) / max(1, n_d - 1))), 3)
    mean_ae = round(sum(abs(x) for x in diffs) / max(1, n_d), 3)
    max_ae = round(max(abs(x) for x in diffs), 3) if diffs else 0.0

    return dict(
        arm_id=arm_id,
        IPD=[(float(t), int(s)) for t, s in ipd],
        TT=TT, SS=SS,
        risk=nhat[:total],
        cen=cen, d=d, KMhat=KMhat,
        ncensor=ncensor,
        estsurv=estsurv, rmse=rmse, mean_ae=mean_ae, max_ae=max_ae,
        n_at_end=int(nhat[total]),
    )
