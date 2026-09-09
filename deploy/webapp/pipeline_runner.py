# -*- coding: utf-8 -*-
"""流水线编排：MinerU 解析结果 → 候选页筛选 → 数字化 → IPD重建 → 质检"""
import os
import sys
import json
import glob
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_full_pipeline(parsed_dir, result_dir):
    """
    从 MinerU 解析结果到 IPD + 质检完整流程。
    返回 dict 列表（每个文件的可下载产物）。
    """
    downloads = []

    # 复制 MinerU 解析结果到 result_dir 供后续处理
    work = os.path.join(result_dir, "work")
    os.makedirs(work, exist_ok=True)
    for f in glob.glob(os.path.join(parsed_dir, "*")):
        bn = os.path.basename(f)
        if bn.endswith((".json", ".md")) or bn == "images":
            dst = os.path.join(work, bn)
            if os.path.isdir(f):
                shutil.copytree(f, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(f, dst)

    # 候选页筛选
    from select_figures import scan as select_scan
    import select_figures as sf
    sf.MINERU_OUT = work
    sf.ROOT = result_dir
    manifest = sf.scan()

    # 数字化 + IPD重建
    from digitize_all import main as digitize_main
    digitize_main(only_trials=None)  # 使用 result_dir 下的 manifest

    # 收集可下载产物
    ipd_files = glob.glob(os.path.join(result_dir, "ipd", "*.ipd.csv"))
    plots = glob.glob(os.path.join(result_dir, "ipd", "plots", "*.png"))
    overlays = glob.glob(os.path.join(result_dir, "overlays", "**", "*.png"), recursive=True)

    for f in ipd_files:
        downloads.append(dict(type="ipd", name=os.path.basename(f), path=f))
    for f in plots:
        downloads.append(dict(type="plot", name=os.path.basename(f), path=f))
    for f in overlays:
        downloads.append(dict(type="overlay", name=os.path.basename(f), path=f))
    master = os.path.join(result_dir, "curves_master.csv")
    if os.path.isfile(master):
        downloads.append(dict(type="master", name="curves_master.csv", path=master))

    return downloads
