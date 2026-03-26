from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from .constants import REQUIRED_KEYS
from .figure_extractor import (
    detect_framework_candidates,
    fallback_framework_region,
    render_pdf_page_region_png,
    vlm_rerank_candidates,
)
from .llm_client import extract_structured_info, polish_content, repair_fields
from .settings import Settings
from .summary_writer import build_markdown
from .text_utils import (
    apply_quality_constraints,
    extract_pdf_text,
    fallback_title_from_filename,
    list_pending_pdfs,
    load_directions,
    next_image_name,
    normalize_year,
    quality_gaps,
    safe_string,
    sanitize_filename,
    unique_path,
)


def select_figure_candidate(pdf_path: Path, settings: Settings) -> tuple[int, float, float, tuple[float, float, float, float], str]:
    cands = detect_framework_candidates(pdf_path=pdf_path, scan_pages=settings.scan_pages, limit=max(8, settings.vlm_top_k))
    if not cands:
        return fallback_framework_region(pdf_path=pdf_path, scan_pages=settings.scan_pages)
    pick = 0
    reason = f"cv-top1;score={cands[0]['score']:.4f};line={str(cands[0]['line'])[:70]}"
    if settings.use_vlm_rerank:
        try:
            idx = vlm_rerank_candidates(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=settings.model,
                timeout=settings.timeout,
                retries=settings.retries,
                pdf_path=pdf_path,
                candidates=cands,
                top_k=settings.vlm_top_k,
            )
            if idx is not None:
                pick = idx
                reason = f"vlm-top{settings.vlm_top_k}-pick={idx+1};base-score={cands[idx]['score']:.4f};line={str(cands[idx]['line'])[:70]}"
            else:
                reason = "vlm-no-pick-fallback-cv;" + reason
        except Exception as exc:  # noqa: BLE001
            reason = f"vlm-error-fallback-cv:{exc};{reason}"
    chosen = cands[pick]
    return int(chosen["page"]), float(chosen["pw"]), float(chosen["ph"]), chosen["bbox"], reason


def ensure_workspace_dirs(root: Path) -> None:
    for d in [root / "待处理pdf", root / "paper", root / "image", root / "已处理pdf"]:
        d.mkdir(parents=True, exist_ok=True)


def process_one_pdf(
    pdf_path: Path,
    settings: Settings,
    direction_candidates: set[str],
    progress_cb: Callable[[int, str], None] | None = None,
) -> dict[str, str]:
    def report(percent: int, message: str) -> None:
        if progress_cb:
            progress_cb(max(0, min(100, percent)), message)

    root = settings.root
    paper_dir = root / "paper"
    image_dir = root / "image"
    done_dir = root / "已处理pdf"

    print(f"[INFO] Processing: {pdf_path.name}", flush=True)
    report(5, f"开始处理: {pdf_path.name}")
    text = extract_pdf_text(pdf_path, max_pages=settings.max_pages, max_chars=settings.max_chars)
    if not text:
        raise RuntimeError("No text extracted from PDF.")
    report(15, "PDF 文本提取完成")

    image_name = next_image_name(image_dir)
    image_path = image_dir / image_name

    page_num, pw, ph, bbox, reason = select_figure_candidate(pdf_path=pdf_path, settings=settings)
    if settings.dry_run:
        print(f"[DRY-RUN] Would render framework-like page: {page_num} ({reason}) -> {image_name}")
        print(f"[DRY-RUN] Page size: ({pw:.1f}, {ph:.1f}), bbox={bbox}")
        print("[DRY-RUN] Would call LLM to extract and polish summary content.")
        print(f"[DRY-RUN] Would write markdown into: {paper_dir}")
        print(f"[DRY-RUN] Would move PDF to: {done_dir / pdf_path.name}")
        return {"status": "dry_run", "pdf": str(pdf_path)}

    print(f"[INFO] Framework page candidate: page={page_num}, {reason}", flush=True)
    report(30, "框架图候选定位完成")
    render_pdf_page_region_png(pdf_path, page_num, image_path, page_width=pw, page_height=ph, bbox=bbox)
    report(40, "框架图截图完成")

    extracted: dict[str, Any] | None = None
    for cap in [len(text), min(len(text), 18000), min(len(text), 9000)]:
        try:
            print(f"[INFO] Extracting with text cap={cap} chars", flush=True)
            extracted = extract_structured_info(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=settings.model,
                timeout=settings.timeout,
                retries=settings.retries,
                pdf_name=pdf_path.name,
                pdf_text=text[:cap],
                direction_candidates=direction_candidates,
            )
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Extraction attempt failed with cap={cap}: {exc}", flush=True)
    if extracted is None:
        raise RuntimeError("LLM extraction failed after multiple truncation attempts.")
    report(60, "结构化信息抽取完成")

    polished = polish_content(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        timeout=settings.timeout,
        retries=settings.retries,
        extracted=extracted,
    )
    polished = apply_quality_constraints(polished)
    report(72, "摘要内容润色完成")
    gaps = quality_gaps(polished)
    if gaps:
        print(f"[INFO] Repairing low-quality fields: {gaps}", flush=True)
        report(80, "正在修复低质量字段")
        polished = repair_fields(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
            timeout=settings.timeout,
            retries=settings.retries,
            data=polished,
            fields=gaps,
        )
        polished = apply_quality_constraints(polished)
    report(88, "质量检查完成")

    for key in REQUIRED_KEYS:
        polished.setdefault(key, "未提及")

    direction = safe_string(polished.get("direction"), "PD")
    if direction not in direction_candidates:
        direction = "PD"
    venue = sanitize_filename(safe_string(polished.get("venue"), "arXiv"))
    year = normalize_year(polished.get("year"))
    title = sanitize_filename(safe_string(polished.get("title"), fallback_title_from_filename(pdf_path.name)))
    polished["direction"] = direction
    polished["venue"] = venue
    polished["year"] = year
    polished["title"] = title

    markdown = build_markdown(polished, image_filename=image_name, sharer=settings.sharer)
    md_name = sanitize_filename(f"{direction}-{venue}-{year}-{title}") + ".md"
    md_path = unique_path(paper_dir / md_name)
    md_path.write_text(markdown, encoding="utf-8")
    target_pdf = unique_path(done_dir / pdf_path.name)
    shutil.move(str(pdf_path), str(target_pdf))
    report(96, "文件写入完成，正在收尾")
    print(f"[OK] Markdown: {md_path.name}", flush=True)
    print(f"[OK] Image: {image_name}", flush=True)
    print(f"[OK] PDF moved: {target_pdf.name}", flush=True)
    report(100, "完成")
    return {
        "status": "ok",
        "markdown_path": str(md_path),
        "markdown_name": md_path.name,
        "image_path": str(image_path),
        "image_name": image_name,
        "processed_pdf_path": str(target_pdf),
        "processed_pdf_name": target_pdf.name,
    }


def run_pipeline(settings: Settings) -> int:
    root = settings.root
    ensure_workspace_dirs(root)
    pending_dir = root / "待处理pdf"

    pdfs = list_pending_pdfs(pending_dir)
    if not pdfs:
        print("[INFO] No pending PDFs found in 待处理pdf/.", flush=True)
        return 0
    directions = load_directions(root)
    print(f"[INFO] Found {len(pdfs)} PDF(s).", flush=True)
    success, failed = 0, 0
    for pdf in pdfs:
        try:
            process_one_pdf(pdf, settings=settings, direction_candidates=directions)
            success += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {pdf.name}: {exc}", flush=True)
    print(f"[SUMMARY] success={success}, failed={failed}", flush=True)
    return 0 if failed == 0 else 1
