# -*- coding: utf-8 -*-
"""
annotate_panels — 为每个面板/臂标注 人群(population)、干预(intervention)、对照标记，
并给缺风险表的面板补 OCR 风险表 / totalpts 兜底。

输出：
  results/panels_annotated.csv   每臂一行（含人群/干预/角色/风险来源）
  results/risk_ocr/*.json        OCR 得到的风险表（times/counts），供建模读取
"""
import os
import re
import sys
import json
import glob
import traceback

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmfig.render import load_pdf_page, load_image
from kmfig.calibrate import calibrate_vector, calibrate_ocr, parse_num
from kmfig.panels import detect_panels

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
RESULTS = os.path.join(ROOT, "results")
RISK_OCR = os.path.join(RESULTS, "risk_ocr")

CONTROL_PAT = re.compile(
    r"placebo|chemotherapy\s+alone|chemo\s+alone|\bCT\b$|^CT\b|chemotherapy$|control|soctreated|standard\s+of\s+care", re.I)
# 干预(实验组)关键词不穷举：非对照即实验/联合
GENERIC = {"chemo", "chemotherapy", "plus", "group", "with", "arm", "ct", "and"}

POP_LEXICON = [
    ("ITT", r"intention[- ]to[- ]treat|\bITT\b"),
    ("ITT-WT", r"wild[- ]type|ITT[- ]WT"),
    ("PD-L1 TPS≥50%", r"PD-?L1\s*(TPS)?\s*≥?\s*50|TPS\s*≥\s*50"),
    ("PD-L1 TPS 1-49%", r"PD-?L1.*1[-––]?49|TPS.*1[-–49]*49%"),
    ("PD-L1 TPS≥1%", r"PD-?L1\s*(TPS)?\s*≥\s*1%|CPS\s*≥\s*1"),
    ("PD-L1 TPS<1%", r"PD-?L1\s*<\s*1|TPS\s*<\s*1"),
    ("PD-L1阳性", r"PD-?L1[- ]positive"),
    ("鳞癌", r"squamous"),
    ("非鳞癌", r"non[- ]?squamous"),
    ("EGFR/ALK阴性", r"EGFR.{0,20}ALK|without.{0,30}(EGFR|ALK)"),
    ("驱动基因阴性", r"no sensitising|driver[- ]negative"),
    ("既往治疗", r"previously treated|second[- ]line|refractory"),
    ("初治", r"treatment[- ]naive|first[- ]line|naive"),
    ("bTMB≥20", r"bTMB\s*≥?\s*20"),
]


def _pt(px, z=72.0 / 300.0):
    return px * z


def panel_letter(words, panel):
    px0, py0, px1, py1 = _pt(panel["x0"]), _pt(panel["y0"]), _pt(panel["x1"]), _pt(panel["y1"])
    w = px1 - px0
    h = py1 - py0
    hits = []
    for x0, y0, x1, y1, txt, *_ in words:
        if not re.fullmatch(r"[A-Ha-h]", txt.strip()):
            continue
        if (y1 - y0) > 0.10 * h:
            continue
        if px0 - 0.10 * w <= x0 <= px0 + 0.28 * w and py0 - 0.10 * h <= y0 <= py0 + 0.25 * h:
            hits.append(txt.strip().upper())
    return hits[0] if hits else None


def find_caption(words, panel):
    """找 Figure N./FIG N. 开头的图注行，拼接其后 500pt 内同块文字。"""
    z = 72.0 / 300.0
    px0, py0, px1, py1 = _pt(panel["x0"]), _pt(panel["y0"]), _pt(panel["x1"]), _pt(panel["y1"])
    cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
    cap_words = [wd for wd in words if re.match(r"^(Figure|FIG|Fig)[\s.:]*\d", wd[4])]
    best = None
    for wd in cap_words:
        d = abs((wd[1] + wd[3]) / 2 - cy) + abs((wd[0] + wd[2]) / 2 - cx) * 0.5
        if best is None or d < best[0]:
            best = (d, wd)
    if best is None:
        return ""
    _, wd = best
    cap_y = wd[1]
    # 同一图注块：y 在 cap_y .. cap_y+60pt，按 (y, x) 排序拼接
    blk = [w2 for w2 in words if cap_y - 2 <= w2[1] <= cap_y + 60 and w2[4] != wd[4] or w2 is wd]
    blk = sorted(set(blk) | {wd}, key=lambda w2: (round(w2[1] / 6), w2[0]))
    return " ".join(w2[4] for w2 in blk)[:600]


def population_from_caption(letter, caption):
    if not caption:
        return ""
    if letter:
        # "(A) ... (B) ..." 字母映射
        seg = re.split(r"\(([A-Ha-h])\)", caption)
        # seg: [前缀, 'A', 文本, 'B', 文本, ...]
        for i in range(1, len(seg) - 1, 2):
            if seg[i].upper() == letter.upper():
                nxt = seg[i + 1].strip(" ,.;:")
                nxt = re.split(r"\([A-Ha-h]\)", nxt)[0]
                return nxt.strip()[:120]
    return caption[:120]


def population_from_keywords(texts):
    joined = " ".join(texts)
    found = []
    for name, pat in POP_LEXICON:
        if re.search(pat, joined, re.I):
            found.append(name)
    return "; ".join(found[:3])


def classify_role(label):
    if not label:
        return "unclear"
    if re.search(CONTROL_PAT, label):
        return "control"
    return "experimental"


def ocr_risk_table(page, panel, xcal, max_rows=6):
    """
    OCR 横轴下方条带，聚类出风险表行/列。
    返回 [dict(times, counts, color)]（times 经 xcal 映射；color 为行主色或 None）。
    """
    from kmfig.calibrate import _ocr_region, parse_num
    H, W = page.img.shape[:2]
    ax_y, ax_x = panel["axis_x_y"], panel["axis_y_x"]
    pw, ph = panel["x1"] - panel["x0"], panel["y1"] - panel["y0"]
    box = (ax_x - 0.05 * pw, ax_y + 4, ax_x + 1.10 * pw, ax_y + 0.16 * H)
    res = _ocr_region(page.img, box, upscale=2.0)
    cells = []
    for r in res:
        t = r["text"].strip()
        # "283(7)" -> 283；纯数字保留
        m = re.match(r"^(\d{1,4})[(（]?", t)
        if not m:
            continue
        v = int(m.group(1))
        cells.append(dict(x0=r["x0"], x1=r["x1"], cy=r["py"], val=v))
    if len(cells) < 4:
        return []
    # 行聚类（按 cy）
    cells.sort(key=lambda c: c["cy"])
    rows = []
    for c in cells:
        if rows and c["cy"] - rows[-1][-1]["cy"] <= max(6.0, 0.02 * ph):
            rows[-1].append(c)
        else:
            rows.append([c])
    out = []
    for row in rows:
        row.sort(key=lambda c: c["x0"])
        if len(row) < 3:
            continue
        times = [round(xcal.value((c["x0"] + c["x1"]) / 2), 2) for c in row]
        counts = [c["val"] for c in row]
        # 颜色：取行内数字像素的主色（饱和者）
        color = None
        cols = []
        for c in row:
            patch = page.img[int(c["cy"]) - 4:int(c["cy"]) + 4, int(c["x0"]):int(c["x1"])]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            s = hsv[..., 1].astype(float) / 255
            m2 = s >= 0.4
            if m2.sum() >= 4:
                bgr = patch[m2].mean(axis=0) / 255.0
                cols.append((float(bgr[2]), float(bgr[1]), float(bgr[0])))
        if cols:
            color = tuple(round(v, 3) for v in np.mean(np.array(cols), axis=0))
        out.append(dict(times=times, counts=counts, color=color,
                        y=float(np.mean([c["cy"] for c in row]))))
        if len(out) >= max_rows:
            break
    return out


def main():
    os.makedirs(RISK_OCR, exist_ok=True)
    jsons = [j for j in sorted(glob.glob(os.path.join(RESULTS, "*", "*__panel*.json")))
             if "risk_ocr" not in j]
    rows = []
    page_cache = {}
    ocr_saved = 0

    for jp in jsons:
        d = json.load(open(jp, encoding="utf-8"))
        trial = os.path.basename(os.path.dirname(jp))
        base = os.path.splitext(os.path.basename(jp))[0]
        src = d.get("source")
        pageno = d.get("page_no")
        panel = d["panel"]
        ctx = d.get("context") or {}
        endpoint = d.get("endpoint")
        if d.get("endpoint") in (None, "UNKNOWN") and d.get("endpoint_page"):
            endpoint = d["endpoint_page"][0] if len(d["endpoint_page"]) == 1 else "+".join(d["endpoint_page"])

        words = None
        caption = ""
        letter = None
        pop_src = ""
        population = ""
        if d.get("page_no") is not None and str(src).endswith(".pdf"):
            key = ("words", src, pageno)
            if key not in page_cache:
                try:
                    import fitz
                    doc = fitz.open(src)
                    page_cache[key] = doc[pageno].get_text("words")
                except Exception:
                    page_cache[key] = []
            words = page_cache[key]
            letter = panel_letter(words or [], panel)
            caption = find_caption(words or [], panel)
            population = population_from_caption(letter, caption)
            pop_src = "caption"
        # 端点/标题关键词兜底
        kw_pop = population_from_keywords(
            (ctx.get("ytitle") or []) + (ctx.get("top") or []) + [caption])
        if kw_pop:
            population = (population + " | " + kw_pop).strip(" |") if population else kw_pop
            pop_src = pop_src + "+keywords" if pop_src else "keywords"

        # 风险表
        at_risk = d.get("at_risk") or []
        risk_src = "risk_table" if at_risk else ""
        ocr_rows = []
        if not at_risk:
            key = ("page", src, pageno)
            if key not in page_cache:
                try:
                    if str(src).endswith(".pdf"):
                        page_cache[key] = load_pdf_page(src, int(pageno), dpi=300)
                    else:
                        page_cache[key] = load_image(src)
                except Exception:
                    page_cache[key] = None
            pg = page_cache.get(key)
            if pg is not None:
                try:
                    xc, yc, _ = (calibrate_vector(pg, panel) if pg.has_vector else (None, None, {}))
                    if xc is None or yc is None:
                        xc, yc, _ = calibrate_ocr(pg, panel)
                    if xc is not None and yc is not None:
                        ocr_rows = ocr_risk_table(pg, panel, xc)
                except Exception:
                    ocr_rows = []
                if ocr_rows:
                    risk_src = "ocr"
                    with open(os.path.join(RISK_OCR, base + ".json"), "w", encoding="utf-8") as f:
                        json.dump(ocr_rows, f, ensure_ascii=False, indent=1)
                    ocr_saved += 1
        if at_risk:
            pass
        elif ocr_rows:
            d["at_risk"] = ocr_rows
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1, default=str)

        # 每臂一行
        risk_rows = at_risk or ocr_rows
        def close(c1, c2, tol=0.1):
            return c1 is not None and c2 is not None and all(abs(x - y) <= tol for x, y in zip(c1, c2))
        used = set()
        arm_risk = {}
        for a in d["arms"]:
            for k, rr in enumerate(risk_rows):
                if k in used:
                    continue
                if close(rr.get("color"), a.get("color")):
                    arm_risk[id(a)] = rr
                    used.add(k)
                    break
        rest_r = [k for k in range(len(risk_rows)) if k not in used]
        rest_a = [a for a in d["arms"] if id(a) not in arm_risk]
        rest_r.sort(key=lambda k: risk_rows[k].get("y", 0))
        for a, k in zip(rest_a, rest_r):
            arm_risk[id(a)] = risk_rows[k]
            used.add(k)

        for a in d["arms"]:
            rr = arm_risk.get(id(a))
            label = a.get("label") or ""
            role = classify_role(label)
            rows.append(dict(
                trial=trial, panel_file=base, source=os.path.basename(src or ""),
                page=pageno if pageno is not None else "",
                endpoint=endpoint, mode=d.get("mode"),
                panel_letter=letter or "",
                population=population, population_source=pop_src,
                arm_id=a["arm_id"], arm_label=label,
                intervention=label, intervention_role=role,
                median_digitized=a.get("median") or "",
                qa_pass=d["qa"].get("pass"),
                risk_source=(risk_src if (rr or at_risk or ocr_rows) else "none"),
                n_risk_cols=(len(rr["counts"]) if rr else (len(at_risk[0]["counts"]) if at_risk else 0)),
                risk_file=(os.path.join("risk_ocr", base + ".json") if (ocr_rows and not at_risk) else ""),
                ipd_done=os.path.isfile(os.path.join(RESULTS, "ipd", f"{base}__{a['arm_id']}.ipd.csv")),
            ))
    out = os.path.join(RESULTS, "panels_annotated.csv")
    pd_out = pd.DataFrame(rows)
    pd_out.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"arms: {len(pd_out)} | panels: {pd_out.panel_file.nunique()} | OCR risk saved: {ocr_saved}")
    print("risk_source 分布:", dict(pd_out.drop_duplicates('panel_file').risk_source.value_counts()))
    print("role 分布:", dict(pd_out.intervention_role.value_counts()))
    print("有 letter 的面板:", pd_out.drop_duplicates('panel_file').panel_letter.ne("").sum())
    print("written:", out)


if __name__ == "__main__":
    import pandas as pd
    main()
