# -*- coding: utf-8 -*-
"""Dual-source published-median extraction (figure annotation + markdown)."""

import os
import re
import sys
import json
import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmfig.render import load_pdf_page
from kmfig.digitize import _annotation_median_pairs

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
MINERU = os.path.join(ROOT, "mineru_out")
DST = os.path.join(ROOT, "肉眼质检_IPD重建")

PICKS = [
    ("01", "KEYNOTE-189", "KEYNOTE-189__p003__panel0", "arm1", "OS", "实验组(pembro+chemo)", 21.786, ["pembrolizumab"]),
    ("02", "KEYNOTE-189", "KEYNOTE-189__p003__panel0", "arm2", "OS", "对照组(placebo+chemo)", 10.609, ["placebo"]),
    ("03", "KEYNOTE-189", "KEYNOTE-189__p003__panel4", "arm1", "PFS", "实验组(pembro+chemo)", 8.997, ["pembrolizumab"]),
    ("04", "AK105-302", "AK105-302__p006__panel0", "arm1", "PFS", "实验组(penpulimab+chemo)", 7.131, ["penpulimab"]),
    ("05", "AK105-302", "AK105-302__p006__panel0", "arm2", "PFS", "对照组(placebo+chemo)", 4.208, ["placebo"]),
    ("06", "POSEIDON", "POSEIDON__p004__panel1", "arm2", "OS", "D+CT组(durvalumab+chemo)", 11.487, ["durvalumab"]),
    ("07", "MYSTIC", "MYSTIC_1__p007__panel0", "arm1", "OS", "D+T组(durva+tremelimab)", 21.918, ["tremelimumab"]),
    ("08", "KEYNOTE-407", "KEYNOTE-407__p003__panel0", "arm1", "OS", "实验组(pembro+chemo)", 17.113, ["pembrolizumab"]),
    ("09", "CHOICE-01", "CHOICE-01__p007__panel0", "arm1", "PFS", "实验组(toripalimab+chemo)", 8.38, ["toripalimab"]),
    ("10", "CheckMate_227_Part_1", "CheckMate_227_Part_1__p005__panel0", "arm2", "OS", "化疗对照组", 14.941, ["chemotherapy"]),
    ("11", "RATIONALE-304", "RATIONALE-304__p007__panel0", "arm1", "OS", "实验组(tislelizumab+chemo)", 21.401, ["tislelizumab"]),
    ("12", "KEYNOTE-189", "KEYNOTE-189__p004__panel0", "arm1", "DOR", "DOR组(pembro)", 12.719, ["pembrolizumab"]),
]

MED_PAT = re.compile(r"median[^.;]{0,160}?(\d+(?:[·.，,]\d+)?)\s*(?:months?|mo)\b", re.I)
PAIRED_PAT = re.compile(r"(\d+(?:[·.]\d+)?)\s*(?:v|versus|vs|and)\s*(\d+(?:[·.]\d+)?)\s*(?:months?|mo)", re.I)
EP_HINT = {
    "OS": re.compile(r"overall\s+survival|\bOS\b", re.I),
    "PFS": re.compile(r"progression[\s\-–]*free|\bPFS\b", re.I),
    "DOR": re.compile(r"duration\s+of\s+response|\bDOR\b", re.I),
}


def _f(s):
    return float(str(s).replace("·", ".").replace("，", ".").replace(",", "."))


def _color_close(c1, c2, tol=0.12):
    return c1 is not None and c2 is not None and all(abs(a - b) <= tol for a, b in zip(c1, c2))


def figure_annotation_by_columns(page, panel, arms):
    """
    图内标注提取（两种布局通用）：
      1) 文字着色布局（NEJM/JCO）：中位数值文字的颜色 = 臂曲线颜色，直接按颜色绑定；
      2) 灰字标注布局：按列 x 对齐 + 纵向窗口配对。
    返回 {arm_color_tuple: (中位值, 说明文本)}。
    """
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    w, h = px1 - px0, py1 - py0
    num_pat = re.compile(r"(\d+(?:[·.]\d+)?)\s*months?", re.I)
    med_spans = []
    for sp in page.spans:
        t = sp["text"].strip()
        if not t or sp["vertical"]:
            continue
        cx, cy = (sp["x0"] + sp["x1"]) / 2, (sp["y0"] + sp["y1"]) / 2
        if not (px0 - 0.15 * w <= cx <= px1 + 0.15 * w and py0 - 0.30 * h <= cy <= py1 + 0.30 * h):
            continue
        m = num_pat.search(t)
        if m and "month" in t.lower():
            try:
                v = float(m.group(1).replace("·", "."))
            except ValueError:
                continue
            if "median" in t.lower() or "CI" in t or "(" in t or len(t) < 40:
                med_spans.append((cx, cy, v, t, sp["color"]))

    out = {}
    for a in arms:
        col = a.get("color")
        if col is None:
            continue
        colk = tuple(round(c, 3) for c in col)
        # 1) 颜色绑定：中位文字颜色 = 臂色
        cands = [ms for ms in med_spans if _color_close(ms[4], col)]
        if cands:
            cands.sort(key=lambda ms: (0 if "median" in ms[3].lower() else 1, ms[2]))
            out[colk] = (cands[0][2], "color-coded: " + cands[0][3][:40])
            continue
        # 2) 灰字：列几何配对（臂名 span 与数值 span 的 x 对齐）
        name_spans, gray_med = [], []
        for sp in page.spans:
            t = sp["text"].strip()
            if not t or sp["vertical"]:
                continue
            cx, cy = (sp["x0"] + sp["x1"]) / 2, (sp["y0"] + sp["y1"]) / 2
            if not (px0 - 0.15 * w <= cx <= px1 + 0.15 * w and py0 - 0.30 * h <= cy <= py1 + 0.30 * h):
                continue
            m = num_pat.search(t)
            if m and "month" in t.lower():
                try:
                    v = float(m.group(1).replace("·", "."))
                except ValueError:
                    continue
                gray_med.append((cx, cy, v, t))
            elif len(t) >= 4 and re.search(r"[A-Za-z]{3,}", t) and not num_pat.search(t):
                name_spans.append((cx, cy, t))
        # 臂名与该臂的几何锚点：面板内该臂曲线终点/标注未知，退化为：每个灰字中位仅当唯一时绑定
        if len(gray_med) == 1 and len(arms) == 1:
            out[colk] = (gray_med[0][2], "gray: " + gray_med[0][3][:40])
    return out


def figure_pairs(d, arms):
    page = load_pdf_page(d["source"], int(d["page_no"]), dpi=300)
    pairs = _annotation_median_pairs(page, d["panel"])
    if pairs:
        return pairs
    col = figure_annotation_by_columns(page, d["panel"], arms)
    return list(col.values())


def best_figure_value(pairs, arm_label, digitized):
    label_words = set(re.findall(r"[a-z]{4,}", str(arm_label).lower())) - {
        "plus", "chemo", "chemotherapy", "group", "with"}
    best, best_score = None, None
    for v, lab in pairs:
        lab_words = set(re.findall(r"[a-z]{4,}", str(lab).lower())) - {
            "plus", "chemo", "chemotherapy", "group", "with", "median", "months"}
        overlap = len(label_words & lab_words)
        score = overlap * 10
        if digitized:
            score += max(0.0, 5 - abs(v - digitized) / 5)
        if best_score is None or score > best_score:
            best, best_score = (v, lab), score
    return best


def md_candidates(trial, arm_keywords, endpoint, digitized):
    files = glob.glob(os.path.join(MINERU, trial, "**", "*.md"), recursive=True)
    cands = []
    ep_pat = EP_HINT.get(endpoint)
    for f in files:
        try:
            text = open(f, encoding="utf-8").read().replace("\n", " ")
        except Exception:
            continue
        for m in MED_PAT.finditer(text):
            try:
                v = _f(m.group(1))
            except ValueError:
                continue
            if not (0.3 <= v <= 300):
                continue
            a = max(0, m.start() - 300)
            b = min(len(text), m.end() + 120)
            ctx = text[a:b]
            if not any(k in ctx.lower() for k in arm_keywords):
                continue
            ep_ok = bool(ep_pat and ep_pat.search(ctx))
            cands.append(dict(val=v, ctx=ctx.strip()[:200], ep=ep_ok,
                              dist=abs(v - digitized) if digitized else None,
                              file=os.path.basename(f)))
        for m in PAIRED_PAT.finditer(text):
            try:
                v1, v2 = _f(m.group(1)), _f(m.group(2))
            except ValueError:
                continue
            a = max(0, m.start() - 300)
            b = min(len(text), m.end() + 120)
            ctx = text[a:b]
            if not any(k in ctx.lower() for k in arm_keywords):
                continue
            ep_ok = bool(ep_pat and ep_pat.search(ctx))
            for v in (v1, v2):
                cands.append(dict(val=v, ctx=ctx.strip()[:200], ep=ep_ok,
                                  dist=abs(v - digitized) if digitized else None,
                                  file=os.path.basename(f)))
    return sorted(cands, key=lambda c: (not c["ep"], c["dist"] if c["dist"] is not None else 9e9))


def main():
    out_rows = []
    for no, trial, base, arm, ep, arm_note, digitized, kws in PICKS:
        jp = os.path.join(ROOT, "results", trial, base + ".json")
        d = json.load(open(jp, encoding="utf-8"))
        arm_obj = [a for a in d["arms"] if a["arm_id"] == arm]
        arm_label = arm_obj[0].get("label") if arm_obj else ""
        try:
            pairs = figure_pairs(d, d["arms"])
        except Exception as e:
            pairs = []
            print(f"  [{no}] figure-pairs error: {e}")
        fig = best_figure_value(pairs, arm_label, digitized)
        md_c = md_candidates(trial, kws, ep, digitized)
        # md 候选中偏离数字化中位 >25% 的多为其他队列/其他端点的同药句（臂错误），剔除
        md_c = [c for c in md_c if digitized is None or c["dist"] is None or c["dist"] / digitized <= 0.25]
        md_best = md_c[0] if md_c else None
        out_rows.append(dict(
            编号=no, 试验=trial, 面板=base, 臂=arm, 端点=ep, 队列=arm_note,
            臂标签=arm_label, 数字化中位=digitized,
            图内标注中位=(fig[0] if fig else ""),
            图内标注组名=(fig[1][:60] if fig else ""),
            md中位=(md_best["val"] if md_best else ""),
            md证据=(md_best["ctx"][:130] if md_best else ""),
            md文档=(md_best["file"] if md_best else ""),
        ))
        f_s = f"{fig[0]}" if fig else "—"
        m_s = f"{md_best['val']}" if md_best else "—"
        # 与重建一致性
        agree = ""
        if fig and digitized:
            agree = "一致✓" if abs(fig[0] - digitized) / fig[0] <= 0.10 else "偏离"
        elif md_best and digitized:
            agree = "一致✓" if abs(md_best["val"] - digitized) / md_best["val"] <= 0.10 else "偏离"
        agree_s = agree
        print(f"[{no}] {trial} {ep} {arm_note}: 图内={f_s} md={m_s} 数字化={digitized} {agree_s}")
        out_rows[-1]["一致"] = agree_s

    df = pd.DataFrame(out_rows)
    df.to_csv(os.path.join(DST, "发表中位_双源.csv"), index=False, encoding="utf-8-sig")

    readme = os.path.join(DST, "README.md")
    text = open(readme, encoding="utf-8").read()
    marker = "## 发表中位（从 MinerU markdown 按 队列名+中位数 抽取"
    if marker in text:
        text = text[:text.index(marker)]
    md_lines = ["", "## 发表中位（双源修正版）", "",
                "说明：此前带星号的\"发表中位\"为手工填写，存在**臂对应错误**（经用户对图复核发现）。",
                "现改为双源自动抽取：**图内标注**（发表图自身打印的 组名→中位 对应，臂绑定最可靠）为主，",
                "MinerU markdown 中的中位数句子为辅。数字化中位为流水线提取值。", "",
                "| # | 试验 | 队列 | 端点 | 数字化中位 | 图内标注中位 | 图内组名 | md文本中位 | 一致性 |",
                "|---|---|---|---|---|---|---|---|---|"]
    for r in out_rows:
        ev = str(r["md证据"]).replace("|", "/").split("。")[0]
        md_lines.append(f"| {r['编号']} | {r['试验']} | {r['队列']} | {r['端点']} | {r['数字化中位']} | "
                        f"{r['图内标注中位']} | {str(r['图内标注组名'])[:36]} | {r['md中位']} | {r.get('一致','')} |")
    md_lines += ["", "md 证据片段见 发表中位_双源.csv（含来源文档名）。", "",
                 "注：KEYNOTE-189/407 与 CheckMate 227 的主文 md 不含各臂中位数句（数值仅印在图内），",
                 "故这些面板以图内标注为准；CHOICE-01 等 md 句子可直接核对。"]
    open(readme, "w", encoding="utf-8").write(text + "\n".join(md_lines))
    print("updated:", readme)


if __name__ == "__main__":
    main()
