#!/usr/bin/env python3
from __future__ import annotations

import os
import secrets
import threading
from datetime import datetime
from pathlib import Path

import markdown
from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory, url_for

from autosummary.pipeline import ensure_workspace_dirs, process_one_pdf
from autosummary.settings import Settings
from autosummary.text_utils import load_directions, sanitize_filename, unique_path


ROOT = Path(__file__).resolve().parent
RESULTS: dict[str, dict[str, str]] = {}
JOBS: dict[str, dict[str, object]] = {}
JOB_LOCK = threading.Lock()

app = Flask(__name__, template_folder=str(ROOT / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


def build_settings_from_form() -> Settings:
    api_key = request.form.get("api_key", "").strip() or os.getenv("HAPPYAPI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("请在页面填写 API Key，或在后端配置 HAPPYAPI_API_KEY。")
    base_url = request.form.get("base_url", "").strip() or os.getenv("HAPPYAPI_BASE_URL", "https://happyapi.org/v1").strip()
    model = request.form.get("model", "").strip() or os.getenv("HAPPYAPI_MODEL", "gpt-5.1-high").strip()
    sharer = request.form.get("sharer", "").strip() or os.getenv("SUMMARY_SHARER", "自动生成").strip()
    return Settings(
        root=ROOT,
        api_key=api_key,
        base_url=base_url,
        model=model,
        sharer=sharer,
        max_pages=int(os.getenv("SUMMARY_MAX_PAGES", "16")),
        scan_pages=int(os.getenv("SUMMARY_SCAN_PAGES", "30")),
        max_chars=int(os.getenv("SUMMARY_MAX_CHARS", "36000")),
        timeout=int(os.getenv("SUMMARY_TIMEOUT", "180")),
        retries=int(os.getenv("SUMMARY_RETRIES", "3")),
        use_vlm_rerank=os.getenv("SUMMARY_USE_VLM_RERANK", "1") in {"1", "true", "True", "yes"},
        vlm_top_k=int(os.getenv("SUMMARY_VLM_TOP_K", "3")),
        dry_run=False,
    )


def save_uploaded_pdf(upload) -> Path:
    if not upload or not upload.filename:
        raise ValueError("请先上传 PDF 文件。")
    if not upload.filename.lower().endswith(".pdf"):
        raise ValueError("仅支持 PDF 文件。")
    ensure_workspace_dirs(ROOT)
    pending_dir = ROOT / "待处理pdf"
    safe_name = sanitize_filename(Path(upload.filename).name)
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    target = unique_path(pending_dir / safe_name)
    upload.save(target)
    return target


@app.get("/")
def index():
    cache_items = sorted(RESULTS.values(), key=lambda x: x.get("created_at", ""), reverse=True)[:20]
    return render_template(
        "index.html",
        cache_items=cache_items,
        default_sharer=os.getenv("SUMMARY_SHARER", "自动生成").strip(),
        default_base_url=os.getenv("HAPPYAPI_BASE_URL", "https://happyapi.org/v1").strip(),
        default_model=os.getenv("HAPPYAPI_MODEL", "gpt-5.1-high").strip(),
    )


def _run_job(job_id: str, pdf_path: Path, settings: Settings) -> None:
    def progress(percent: int, message: str) -> None:
        with JOB_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["progress"] = percent
                JOBS[job_id]["message"] = message

    try:
        directions = load_directions(settings.root)
        result = process_one_pdf(pdf_path=pdf_path, settings=settings, direction_candidates=directions, progress_cb=progress)
        md_path = Path(result["markdown_path"])
        md_text = md_path.read_text(encoding="utf-8")
        html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        token = secrets.token_urlsafe(12)
        RESULTS[token] = {
            **result,
            "markdown_html": html_body,
            "markdown_text": md_text,
            "token": token,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with JOB_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["message"] = "完成"
            JOBS[job_id]["result_token"] = token
    except Exception as exc:  # noqa: BLE001
        with JOB_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["message"] = "处理失败"


@app.post("/start_job")
def start_job():
    try:
        upload = request.files.get("pdf")
        pdf_path = save_uploaded_pdf(upload)
        settings = build_settings_from_form()
        job_id = secrets.token_urlsafe(10)
        with JOB_LOCK:
            JOBS[job_id] = {"status": "running", "progress": 1, "message": "任务已创建", "error": None, "result_token": None}
        t = threading.Thread(target=_run_job, args=(job_id, pdf_path, settings), daemon=True)
        t.start()
        return jsonify({"ok": True, "job_id": job_id})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/job_status/<job_id>")
def job_status(job_id: str):
    with JOB_LOCK:
        data = JOBS.get(job_id)
        if not data:
            return jsonify({"ok": False, "error": "job not found"}), 404
        payload = dict(data)
    if payload.get("status") == "done" and payload.get("result_token"):
        payload["result_url"] = url_for("result_page", token=payload["result_token"])
    return jsonify({"ok": True, **payload})


@app.get("/result/<token>")
def result_page(token: str):
    data = RESULTS.get(token)
    if not data:
        abort(404)
    return render_template("result.html", data=data)


@app.get("/download/<token>")
def download_md(token: str):
    data = RESULTS.get(token)
    if not data:
        abort(404)
    return send_file(data["markdown_path"], as_attachment=True, download_name=data["markdown_name"])


@app.get("/image/<name>")
def image_file(name: str):
    return send_from_directory(ROOT / "image", name)


if __name__ == "__main__":
    ensure_workspace_dirs(ROOT)
    app.run(host="0.0.0.0", port=5050, debug=False)
