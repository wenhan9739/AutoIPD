# -*- coding: utf-8 -*-
"""
AtuoIPDR Web Service — Upload PDF, get reconstructed IPD + comparison plots.
Flask app with background processing via threading.
"""
import os
import uuid
import shutil
import threading
import traceback
from datetime import datetime

from flask import (Flask, request, jsonify, render_template,
                   send_from_directory, redirect, url_for)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB
app.config["UPLOAD_FOLDER"] = os.environ.get("DATA_DIR", "/data") + "/uploads"
app.config["RESULT_FOLDER"] = os.environ.get("DATA_DIR", "/data") + "/results"
app.config["JSON_AS_ASCII"] = False

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

# 全局任务状态（内存存储；生产可换 Redis）
JOBS = {}


# ---------- Pipeline 执行（后台线程） ----------

def run_pipeline_thread(job_id, pdf_path, result_dir):
    """后台执行完整流水线：MinerU API → 筛选 → 数字化 → IPD重建 → 质检。"""
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["stage"] = "parsing"

        sys_path = os.path.dirname(os.path.abspath(__file__))
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)

        from mineru_client import parse_pdf
        from pipeline_runner import run_full_pipeline

        # Step 1: MinerU API 解析
        parsed_dir = parse_pdf(pdf_path, result_dir)
        JOBS[job_id]["stage"] = "digitizing"

        # Step 2: 数字化 + IPD 重建
        results = run_full_pipeline(parsed_dir, result_dir)
        JOBS[job_id]["stage"] = "done"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["results"] = results
        JOBS[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        import traceback
        traceback.print_exc()
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["traceback"] = traceback.format_exc()[:2000]


# ---------- 路由 ----------

@app.route("/health")
def health():
    return jsonify := __import__("flask").jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "请上传 PDF 文件"}), 400
    f = request.files["pdf"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "仅支持 PDF 格式"}), 400

    job_id = uuid.uuid4().hex[:12]
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}.pdf")
    result_dir = os.path.join(app.config["RESULT_FOLDER"], job_id)
    os.makedirs(result_dir, exist_ok=True)
    f.save(pdf_path)

    JOBS[job_id] = {
        "status": "queued", "stage": "queued",
        "filename": f.filename,
        "created": datetime.now().isoformat(),
        "result_dir": result_dir,
    }
    threading.Thread(target=run_pipeline_thread, args=(job_id, pdf_path, result_dir),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    info = JOBS.get(job_id)
    if not info:
        return jsonify({"error": "job not found"}), 404
    return jsonify(info)


@app.route("/results/<job_id>")
def results(job_id):
    info = JOBS.get(job_id)
    if not info or info["status"] != "completed":
        return redirect(url_for("index"))
    return render_template("results.html", job_id=job_id, info=info)


@app.route("/download/<job_id>/<filename>")
def download(job_id, filename):
    result_dir = os.path.join(app.config["RESULT_FOLDER"], job_id)
    safe = os.path.basename(filename)
    return send_from_directory(result_dir, safe)


# ---------- 每试验汇总 API ----------

@app.route("/api/jobs")
def list_jobs():
    return jsonify({k: {"status": v["status"], "filename": v.get("filename", ""),
                        "created": v.get("created", "")}
                    for k, v in JOBS.items()})
