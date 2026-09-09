# -*- coding: utf-8 -*-
"""MinerU Cloud API 客户端：上传 PDF → 获取结构化解析结果"""
import os
import time
import requests

MINERU_API = "https://mineru.net/api/v4"
API_KEY = os.environ.get("MINERU_API_KEY", "")


def parse_pdf(pdf_path, output_dir, api_key=None):
    """
    调用 MinerU Cloud API 解析 PDF，返回解析结果目录。
    输出目录包含：content_list.json、origin.pdf、images/ 等。
    """
    key = api_key or API_KEY
    if not key:
        raise ValueError("MINERU_API_KEY 未设置")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # Step 1: 申请上传链接
    fname = os.path.basename(pdf_path)
    resp = requests.post(
        f"{MINERU_API}/file-urls/batch",
        headers=headers,
        json={"enable_formula": True, "enable_table": True, "files": [
            {"name": fname, "is_ocr": True, "data_id": fname}
        ]},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"MinerU API error: {data.get('msg')}")
    upload_url = data["data"]["file_urls"][0]
    batch_id = data["data"]["batch_id"]

    # Step 2: 上传 PDF（注意：不要带额外 header，否则破坏预签名）
    with open(pdf_path, "rb") as f:
        put = requests.put(upload_url, data=f, timeout=300)
    put.raise_for_status()

    # Step 3: 等待解析完成
    for _ in range(120):  # 最多等 10 分钟
        time.sleep(5)
        st = requests.get(f"{MINERU_API}/extract-results/batch/{batch_id}",
                          headers=headers, timeout=30)
        st.raise_for_status()
        sdata = st.json()
        results = sdata.get("data", {}).get("extract_result", [])
        if results and results[0].get("state") == "done":
            break
        if results and results[0].get("state") == "failed":
            raise RuntimeError(f"MinerU parse failed: {results[0].get('err_msg')}")

    # Step 4: 下载解析结果
    full_zip_url = results[0].get("full_zip_url")
    if full_zip_url:
        import io
        import zipfile
        zr = requests.get(full_zip_url, timeout=120)
        zr.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(zr.content)) as z:
            z.extractall(output_dir)

    # 找到 content_list.json 所在子目录
    for root, dirs, files in os.walk(output_dir):
        if "content_list.json" in files or any(f.endswith("_content_list.json") for f in files):
            return root
    # 找不到则返回 output_dir
    return output_dir
