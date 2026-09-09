# -*- coding: utf-8 -*-
"""流水线编排：MinerU API 解析结果 → 候选页筛选 → 数字化 → IPD重建 → 质检。

单任务隔离：所有模块级路径常量在导入后按 job 重写，
输出全部落在该任务的 result_dir 内，互不干扰。
"""
import os
import sys
import re
import glob
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
# webapp/ -> deploy/ -> repo root；pipeline 模块在 repo 根与 pipeline/ 下均可解析
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (_REPO, os.path.join(_REPO, "pipeline"), _HERE):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _safe_stem(name):
    stem = re.sub(r"\.pdf$", "", name, flags=re.I)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem)[:60] or "paper"


def _backfill_risk_ocr(results_dir):
    """光栅图风险表回填：面板 JSON 无 at_risk 时，OCR 横轴下方条带补齐。

    与 annotate_panels.ocr_risk_table 相同逻辑的最小子集（不依赖
    trial_arms.json 的人群/干预标注，适配任意上传的 PDF）。
    """
    import json
    from kmfig.render import load_pdf_page, load_image
    from kmfig.calibrate import calibrate_vector, calibrate_ocr
    from annotate_panels import ocr_risk_table

    page_cache = {}
    for jp in sorted(glob.glob(os.path.join(results_dir, "*", "*__panel*.json"))):
        if "risk_ocr" in jp:
            continue
        try:
            with open(jp, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("at_risk"):
            continue
        src, pageno, panel = d.get("source"), d.get("page_no"), d.get("panel")
        if not src or panel is None:
            continue
        key = (src, pageno)
        if key not in page_cache:
            try:
                page_cache[key] = (load_pdf_page(src, int(pageno), dpi=300)
                                   if str(src).lower().endswith(".pdf")
                                   else load_image(src))
            except Exception:
                page_cache[key] = None
        pg = page_cache.get(key)
        if pg is None:
            continue
        try:
            xc, yc, _ = (calibrate_vector(pg, panel) if pg.has_vector
                         else (None, None, {}))
            if xc is None or yc is None:
                xc, yc, _ = calibrate_ocr(pg, panel)
            if xc is None or yc is None:
                continue
            rows = ocr_risk_table(pg, panel, xc)
        except Exception:
            continue
        if rows:
            d["at_risk"] = rows
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1, default=str)


def run_full_pipeline(parsed_dir, result_dir, pdf_path):
    """从 MinerU 解析结果到 IPD + 对比图的完整流程。返回可下载产物列表。"""
    stem = _safe_stem(os.path.basename(pdf_path))

    # ---- 布局：mineru/<stem>/<stem>/auto/ + src/<stem>/<stem>.pdf + work/ + results/ ----
    # select_figures.find_doc_dirs 期望 <trial>/<stem>/auto 两级结构，
    # 网页任务里 trial 与 stem 同名（均为文件名 stem）。
    auto_dir = os.path.join(result_dir, "mineru", stem, stem, "auto")
    os.makedirs(auto_dir, exist_ok=True)
    for f in glob.glob(os.path.join(parsed_dir, "*")):
        bn = os.path.basename(f)
        if bn.endswith((".json", ".md")) or bn in ("images", "origin"):
            dst = os.path.join(auto_dir, bn)
            if os.path.isdir(f):
                shutil.copytree(f, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(f, dst)

    src_dir = os.path.join(result_dir, "src", stem)
    os.makedirs(src_dir, exist_ok=True)
    shutil.copy2(pdf_path, os.path.join(src_dir, stem + ".pdf"))

    work_dir = os.path.join(result_dir, "work")
    results_dir = os.path.join(result_dir, "results")
    os.makedirs(work_dir, exist_ok=True)

    # ---- 按 job 重写各模块的路径常量（模块常量在 import 时从 env 计算） ----
    import select_figures as sf
    import digitize_all as da
    import validate as vd
    import ipd_reconstruct as ir

    sf.MINERU_OUT = os.path.join(result_dir, "mineru")
    sf.ROOT = result_dir
    sf.WORK = work_dir
    sf.SRC = os.path.join(result_dir, "src")
    vd.MINERU_OUT = sf.MINERU_OUT
    da.WORK = work_dir
    da.RESULTS = results_dir
    ir.RESULTS = results_dir
    ir.OUT = os.path.join(results_dir, "ipd")

    # ---- 1) 候选页筛选 ----
    manifest = sf.scan()
    if not manifest:
        raise RuntimeError("未在该 PDF 中找到 KM 生存曲线候选页 "
                           "(no Kaplan-Meier candidate pages found)")

    # ---- 2) 数字化 ----
    da.main(only_trials=None)

    # ---- 2b) 风险表 OCR 回填（光栅图无文字层数字时） ----
    _backfill_risk_ocr(results_dir)

    # ---- 3) IPD 重建（Python Guyot）+ 对比图 ----
    ir.main()

    # ---- 4) 收集可下载产物 ----
    downloads = []

    def _add(dtype, f):
        downloads.append(dict(type=dtype, name=os.path.basename(f),
                              rel=os.path.relpath(f, result_dir).replace("\\", "/"),
                              path=f))

    for f in sorted(glob.glob(os.path.join(ir.OUT, "*.ipd.csv"))):
        _add("ipd", f)
    for f in sorted(glob.glob(os.path.join(ir.OUT, "plots", "*.png"))):
        _add("plot", f)
    for f in sorted(glob.glob(os.path.join(results_dir, "overlays", "**", "*.png"),
                              recursive=True)):
        _add("overlay", f)
    for name, sub in (("key_values.csv", "ipd"), ("curves_summary.csv", ""),
                      ("QA_report.md", "")):
        f = os.path.join(results_dir, sub, name)
        if os.path.isfile(f):
            _add("report", f)
    return downloads
