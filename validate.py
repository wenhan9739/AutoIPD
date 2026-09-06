# -*- coding: utf-8 -*-
"""
validate — 从 MinerU markdown 中提取文中报告的中位生存时间，用于交叉校验数字化结果。
"""
import os
import re
import glob

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
MINERU_OUT = os.path.join(ROOT, "mineru_out")

NUM = r"(\d+(?:[·.，,]\d+)?)"
# “median 7·6 months” / “medians of 9.0 and 4.9 months” / “中位 … 个月”
MEDIAN_PAT = re.compile(
    r"median(?:s)?(?:\s+of)?[^.;()]{0,120}?" + NUM + r"\s*(?:months?|mo|月)", re.I)
# “7·6 months (95% CI …)” 前有 OS/PFS 关键词的上下文
CTX_OS = re.compile(r"overall\s+survival|\bOS\b", re.I)
CTX_PFS = re.compile(r"progression[\s\-–]*free\s+survival|\bPFS\b", re.I)


def _to_float(s):
    return float(s.replace("·", ".").replace("，", ".").replace(",", "."))


def reported_medians(trial):
    """
    扫描该试验所有 MinerU markdown，返回 [(value_months, endpoint_guess, snippet)]。
    endpoint_guess：上下文 ±300 字符里同时出现 OS/PFS 时给 'OS'/'PFS'，否则 None。
    """
    texts = []
    for md in glob.glob(os.path.join(MINERU_OUT, trial, "**", "*.md"), recursive=True):
        try:
            texts.append(open(md, encoding="utf-8").read())
        except Exception:
            pass
    out = []
    for text in texts:
        for m in MEDIAN_PAT.finditer(text):
            try:
                v = _to_float(m.group(1))
            except ValueError:
                continue
            if not (0.3 <= v <= 300):
                continue
            a, b = max(0, m.start() - 300), min(len(text), m.end() + 300)
            ctx = text[a:b].replace("\n", " ")
            ep = None
            is_os, is_pfs = bool(CTX_OS.search(ctx)), bool(CTX_PFS.search(ctx))
            if is_os and not is_pfs:
                ep = "OS"
            elif is_pfs and not is_os:
                ep = "PFS"
            out.append((v, ep, m.group(0)[:80]))
    return out


def match_median(digitized_m, reported, rel_tol=0.15):
    """与报告值求最接近的匹配；返回 (best_value, rel_diff) 或 (None, None)。"""
    cands = [(v, abs(v - digitized_m) / max(v, 1e-9)) for v, ep, snip in reported
             if v > 0.5]  # 排除明显不是中位生存的小值
    if not cands:
        return None, None
    v, d = min(cands, key=lambda t: t[1])
    if d <= rel_tol:
        return v, d
    return None, None
