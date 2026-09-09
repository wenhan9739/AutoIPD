# -*- coding: utf-8 -*-
"""Candidate-page selection: MinerU chart detection + keyword/vector-density matching."""

import os
import re
import json

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
MINERU_OUT = os.path.join(ROOT, "mineru_out")
SRC = os.path.join(ROOT, "最终所纳入27个RCT", "最终所纳入27个RCT")
WORK = os.path.join(ROOT, "work")

KM_PAT = re.compile(r"kaplan[\s\-–]*meier|survival\s+curve|km\s+curve", re.I)
ENDPOINT_PAT = {
    "PFS": re.compile(r"progression[\s\-–]*free\s+survival|\bPFS\b", re.I),
    "OS": re.compile(r"overall\s+survival|\bOS\b", re.I),
    "EFS": re.compile(r"event[\s\-–]*free\s+survival|\bEFS\b", re.I),
    "DFS": re.compile(r"disease[\s\-–]*free\s+survival|\bDFS\b|recurrence[\s\-–]*free", re.I),
    "DOR": re.compile(r"duration\s+of\s+response|\bDOR\b", re.I),
}
# 排除：纯安全性/剂量图
EXCLUDE_PAT = re.compile(r"adverse\s+event|\bae\b|toxicit|dose|intensity|exposure|pharmacok", re.I)


def find_doc_dirs(trial_dir):
    """trial 下每个已处理文档目录（含 auto/ 或 office/）。"""
    out = []
    if not os.path.isdir(trial_dir):
        return out
    for stem in sorted(os.listdir(trial_dir)):
        base = os.path.join(trial_dir, stem)
        for sub, kind in (("auto", "pdf"), ("office", "office")):
            d = os.path.join(base, sub)
            if os.path.isdir(d) and any(f.endswith("_content_list.json") for f in os.listdir(d)):
                out.append((base, d, kind, stem))
                break
    return out


def source_file(trial, stem, kind):
    """找到原始文件路径（pdf/docx/pptx）。"""
    d = os.path.join(SRC, trial)
    if not os.path.isdir(d):
        return None
    for f in os.listdir(d):
        s = os.path.splitext(f)[0]
        if s == stem:
            return os.path.join(d, f)
    return None


def page_items(content_list, page_idx):
    return [it for it in content_list if it.get("page_idx") == page_idx]


FIG_TYPES = ("image", "chart")


def _caps(it):
    return (it.get("img_caption") or []) + (it.get("img_footnote") or []) + \
           (it.get("chart_caption") or []) + (it.get("chart_footnote") or [])


def scan():
    manifest = []
    import fitz
    _draw_cache = {}
    def dense_vector(src, pg):
        """PDF 页是否有密集矢量图形（纯矢量 KM 图页面可能没有嵌入位图）。"""
        key = (src, pg)
        if key not in _draw_cache:
            try:
                doc = fitz.open(src)
                _draw_cache[key] = sum(len(d["items"]) for d in doc[pg].get_drawings()) > 300
            except Exception:
                _draw_cache[key] = False
        return _draw_cache[key]

    for trial in sorted(os.listdir(MINERU_OUT)):
        tdir = os.path.join(MINERU_OUT, trial)
        if not os.path.isdir(tdir):
            continue
        for base, docdir, kind, stem in find_doc_dirs(tdir):
            cl_file = [f for f in os.listdir(docdir) if f.endswith("_content_list.json")][0]
            cl = json.load(open(os.path.join(docdir, cl_file), encoding="utf-8"))
            src = source_file(trial, stem, kind)
            pages = sorted(set(it.get("page_idx", 0) for it in cl))
            for pg in pages:
                items = page_items(cl, pg)
                texts = [it.get("text", "") for it in items if it.get("type") in ("text", "caption") and it.get("text")]
                page_text = " ".join(texts)
                images = [it for it in items if it.get("type") in FIG_TYPES]
                charts = [it for it in items if it.get("type") == "chart"]
                # MinerU 已判定为 chart 的页面直接入选；其余需要图文或矢量证据
                if not charts and not images and not (src and src.lower().endswith(".pdf") and dense_vector(src, pg)):
                    continue
                cap_texts = []
                for it in images:
                    cap_texts += _caps(it)
                cap_text = " ".join(cap_texts)
                hit_km = bool(KM_PAT.search(page_text) or KM_PAT.search(cap_text))
                endpoints = [e for e, p in ENDPOINT_PAT.items() if p.search(page_text) or p.search(cap_text)]
                if not hit_km and not endpoints and not charts:
                    continue
                if EXCLUDE_PAT.search(cap_text) and not hit_km and not any(
                        ENDPOINT_PAT[e].search(cap_text) for e in endpoints):
                    continue
                # 端点优先以 caption 为准
                ep_cap = [e for e, p in ENDPOINT_PAT.items() if p.search(cap_text)]
                manifest.append(dict(
                    trial=trial, stem=stem, kind=kind, page=pg, source=src,
                    docdir=docdir, n_images=len(images), n_charts=len(charts),
                    endpoints=ep_cap or endpoints, hit_km=hit_km,
                    caption=cap_text[:300],
                    evidence=("chart" if charts else ("caption" if ep_cap else "page_text"))))
    os.makedirs(WORK, exist_ok=True)
    out = os.path.join(WORK, "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"manifest: {len(manifest)} candidate pages -> {out}")
    return manifest


if __name__ == "__main__":
    scan()
