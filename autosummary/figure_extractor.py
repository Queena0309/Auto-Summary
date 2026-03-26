from __future__ import annotations

import base64
import html
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter
from pypdf import PdfReader

from .constants import FIGURE_KEYWORDS
from .llm_client import call_chat, parse_choice_index


def render_pdf_page_png(pdf_path: Path, page_num: int, output_path: Path) -> None:
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm not found.")
    prefix = output_path.with_suffix("")
    cmd = ["pdftoppm", "-f", str(page_num), "-l", str(page_num), "-singlefile", "-png", str(pdf_path), str(prefix)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr.strip()}")
    if not output_path.exists():
        raise RuntimeError(f"failed to render page {page_num}")


def render_pdf_page_pil(pdf_path: Path, page_num: int) -> Image.Image:
    with tempfile.TemporaryDirectory(prefix="autosum_page_") as td:
        tmp = Path(td) / "page.png"
        render_pdf_page_png(pdf_path, page_num, tmp)
        with Image.open(tmp) as img:
            return img.copy()


def clamp_box(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 2, x0))
    y0 = max(0, min(h - 2, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return x0, y0, x1, y1


def crop_variants(base: tuple[int, int, int, int], w: int, h: int) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = base
    bw, bh = x1 - x0, y1 - y0
    candidates = [
        (x0, y0, x1, y1),
        (x0 + int(bw * 0.02), y0 + int(bh * 0.02), x1 - int(bw * 0.02), y1 - int(bh * 0.02)),
        (x0 + int(bw * 0.05), y0 + int(bh * 0.05), x1 - int(bw * 0.05), y1 - int(bh * 0.05)),
    ]
    return [clamp_box(c, w, h) for c in candidates]


def variant_score(img: Image.Image, box: tuple[int, int, int, int]) -> float:
    crop = img.crop(box).convert("L")
    edges = crop.filter(ImageFilter.FIND_EDGES)
    hist = edges.histogram()
    total = max(1, crop.size[0] * crop.size[1])
    edge_ratio = sum(hist[48:]) / total
    entropy = crop.entropy() / 8.0
    area_ratio = total / max(1, img.size[0] * img.size[1])
    area_penalty = 0.12 if area_ratio > 0.82 else (0.08 if area_ratio < 0.08 else 0.0)
    return edge_ratio * 2.2 + entropy * 0.7 - area_penalty


def render_pdf_page_region_png(
    pdf_path: Path,
    page_num: int,
    output_path: Path,
    page_width: float,
    page_height: float,
    bbox: tuple[float, float, float, float] | None,
) -> None:
    render_pdf_page_png(pdf_path, page_num, output_path)
    if not bbox:
        return
    x0, y0, x1, y1 = bbox
    with Image.open(output_path) as img:
        w_px, h_px = img.size
        sx = w_px / max(page_width, 1e-6)
        sy = h_px / max(page_height, 1e-6)
        base = clamp_box((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)), w_px, h_px)
        scored = [(variant_score(img, b), b) for b in crop_variants(base, w_px, h_px)]
        scored.sort(key=lambda x: x[0], reverse=True)
        print("[INFO] Crop variant scores: " + ", ".join(f"{s:.4f}" for s, _ in scored), flush=True)
        img.crop(scored[0][1]).save(output_path)


def get_bbox_layout_xml(pdf_path: Path, scan_pages: int) -> str:
    cmd = ["pdftotext", "-bbox-layout", "-f", "1", "-l", str(max(scan_pages, 1)), str(pdf_path), "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext -bbox-layout failed: {result.stderr.strip()}")
    return result.stdout


def _caption_score(line_text: str) -> int:
    txt = line_text.lower()
    score = 0
    starts_like_caption = bool(re.search(r"^\s*(figure|fig\.|图)\s*\d+\s*[:.]", txt))
    if "figure" in txt or re.search(r"\bfig\.", txt):
        score += 6
    if "图" in line_text:
        score += 3
    for kw, weight in FIGURE_KEYWORDS.items():
        if kw.lower() in txt:
            score += weight
    if any(k in txt for k in ["framework", "overview", "architecture", "pipeline", "method"]):
        score += 8
    if any(k in txt for k in ["系统框架", "总体框架", "方法框架", "框架图"]):
        score += 8
    if any(k in txt for k in ["example", "case study", "question", "ablation", "dataset", "distribution", "performance", "result", "accuracy"]):
        score -= 5
    if any(k in txt for k in ["示例", "案例", "消融", "数据集", "分布", "结果", "准确率"]):
        score -= 5
    if starts_like_caption or re.search(r"^\s*(figure|fig\.)\s*\d+", txt):
        score += 6
    elif "figure" in txt or "fig." in txt:
        score -= 8
    return score


def _candidate_boxes_from_caption(page_w: float, page_h: float, lx0: float, ly0: float, lx1: float, ly1: float) -> list[tuple[float, float, float, float]]:
    center_x = (lx0 + lx1) / 2.0
    boxes: list[tuple[float, float, float, float]] = []
    if ly0 > page_h * 0.20:
        top, bottom = page_h * 0.08, max(page_h * 0.18, ly0 - page_h * 0.012)
        boxes += [(page_w * 0.05, top, page_w * 0.95, bottom), (page_w * 0.05, top, page_w * 0.52, bottom), (page_w * 0.48, top, page_w * 0.95, bottom), (page_w * 0.12, top, page_w * 0.88, bottom)]
    if ly1 < page_h * 0.82:
        top, bottom = min(page_h * 0.86, ly1 + page_h * 0.015), page_h * 0.94
        boxes += [(page_w * 0.05, top, page_w * 0.95, bottom), (page_w * 0.10, top, page_w * 0.90, bottom)]
    if center_x < page_w * 0.5:
        boxes.append((page_w * 0.05, page_h * 0.10, page_w * 0.55, min(page_h * 0.90, ly0)))
    else:
        boxes.append((page_w * 0.45, page_h * 0.10, page_w * 0.95, min(page_h * 0.90, ly0)))
    return boxes


def _clamp_box_pdf(box: tuple[float, float, float, float], pw: float, ph: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    x0 = max(0.0, min(pw - 1.0, x0))
    y0 = max(0.0, min(ph - 1.0, y0))
    x1 = max(x0 + 1.0, min(pw, x1))
    y1 = max(y0 + 1.0, min(ph, y1))
    return x0, y0, x1, y1


def _pdf_box_to_px(box: tuple[float, float, float, float], pw: float, ph: float, iw: int, ih: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return clamp_box((int(x0 * iw / max(pw, 1e-6)), int(y0 * ih / max(ph, 1e-6)), int(x1 * iw / max(pw, 1e-6)), int(y1 * ih / max(ph, 1e-6))), iw, ih)


def _px_box_to_pdf(px_box: tuple[int, int, int, int], pw: float, ph: float, iw: int, ih: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = px_box
    return _clamp_box_pdf((x0 * pw / max(iw, 1), y0 * ph / max(ih, 1), x1 * pw / max(iw, 1), y1 * ph / max(ih, 1)), pw, ph)


def _entropy_from_gray(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) / 8.0


def cv_diagram_score(pil_img: Image.Image, px_box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = px_box
    arr = np.array(pil_img.crop((x0, y0, x1, y1)).convert("RGB"))
    if arr.size == 0:
        return -1e9
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    total = float(max(1, h * w))
    edges = cv2.Canny(gray, 60, 160)
    edge_ratio = float(np.count_nonzero(edges)) / total
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=max(20, int(min(h, w) * 0.10)), maxLineGap=8)
    line_count = 0 if lines is None else int(len(lines))
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7)
    contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rect_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < max(120, total * 0.0003):
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if 4 <= len(approx) <= 6:
            rect_count += 1
    text_thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    t_contours, _ = cv2.findContours(text_thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    line_like = 0
    for c in t_contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw >= int(0.42 * w) and ch <= max(5, int(0.055 * h)):
            line_like += 1
    area_ratio = total / float(max(1, pil_img.size[0] * pil_img.size[1]))
    area_penalty = 0.14 if area_ratio > 0.85 else (0.10 if area_ratio < 0.07 else 0.0)
    return edge_ratio * 2.4 + min(line_count, 120) * 0.004 + min(rect_count, 80) * 0.008 + _entropy_from_gray(gray) * 0.65 - area_penalty - min(line_like, 12) * 0.035


def refine_graphic_region(pil_img: Image.Image, px_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = px_box
    crop = np.array(pil_img.crop((x0, y0, x1, y1)).convert("L"))
    h, w = crop.shape[:2]
    if h < 20 or w < 20:
        return px_box
    edges = cv2.Canny(crop, 60, 170)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []
    total = h * w
    for i in range(1, num):
        sx, sy, sw, sh, area = stats[i]
        if area < max(80, int(total * 0.003)):
            continue
        if sw >= int(0.45 * w) and sh <= max(5, int(0.06 * h)):
            continue
        boxes.append((sx, sy, sx + sw, sy + sh, area))
    if not boxes:
        return px_box
    boxes.sort(key=lambda t: t[4], reverse=True)
    keep = boxes[: min(6, len(boxes))]
    rx0, ry0 = min(b[0] for b in keep), min(b[1] for b in keep)
    rx1, ry1 = max(b[2] for b in keep), max(b[3] for b in keep)
    pad_x, pad_y = int((rx1 - rx0) * 0.12), int((ry1 - ry0) * 0.13)
    rx0, ry0, rx1, ry1 = max(0, rx0 - pad_x), max(0, ry0 - pad_y), min(w, rx1 + pad_x), min(h, ry1 + pad_y)
    fx0, fy0, fx1, fy1 = clamp_box((x0 + rx0, y0 + ry0, x0 + rx1, y0 + ry1), pil_img.size[0], pil_img.size[1])
    extra_x, extra_y = int((fx1 - fx0) * 0.10), int((fy1 - fy0) * 0.11)
    return clamp_box((fx0 - extra_x, fy0 - extra_y, fx1 + extra_x, fy1 + extra_y), pil_img.size[0], pil_img.size[1])


def detect_framework_candidates(pdf_path: Path, scan_pages: int, limit: int = 8) -> list[dict[str, Any]]:
    xml_text = get_bbox_layout_xml(pdf_path, scan_pages=scan_pages)
    page_re = re.compile(r'<page width="([^"]+)" height="([^"]+)">(.*?)</page>', re.S)
    line_re = re.compile(r'<line xMin="([^"]+)" yMin="([^"]+)" xMax="([^"]+)" yMax="([^"]+)">(.*?)</line>', re.S)
    word_re = re.compile(r"<word [^>]*>(.*?)</word>", re.S)

    captions: list[dict[str, Any]] = []
    for page_idx, pm in enumerate(page_re.finditer(xml_text), start=1):
        pw, ph, body = float(pm.group(1)), float(pm.group(2)), pm.group(3)
        for lm in line_re.finditer(body):
            words = [html.unescape(w).strip() for w in word_re.findall(lm.group(5))]
            words = [w for w in words if w]
            if not words:
                continue
            text = " ".join(words)
            sc = _caption_score(text)
            if sc <= 0:
                continue
            captions.append({"score": sc, "page": page_idx, "page_w": pw, "page_h": ph, "line": text, "lx0": float(lm.group(1)), "ly0": float(lm.group(2)), "lx1": float(lm.group(3)), "ly1": float(lm.group(4))})

    if not captions:
        return []
    captions.sort(key=lambda x: x["score"], reverse=True)
    top_caps = captions[: min(len(captions), 14)]
    rendered: dict[int, Image.Image] = {}
    items: list[dict[str, Any]] = []
    for cap in top_caps:
        page, pw, ph = int(cap["page"]), float(cap["page_w"]), float(cap["page_h"])
        if page not in rendered:
            rendered[page] = render_pdf_page_pil(pdf_path, page)
        img = rendered[page]
        iw, ih = img.size
        for rb in _candidate_boxes_from_caption(pw, ph, float(cap["lx0"]), float(cap["ly0"]), float(cap["lx1"]), float(cap["ly1"])):
            pb = _clamp_box_pdf(rb, pw, ph)
            px = _pdf_box_to_px(pb, pw, ph, iw, ih)
            px = refine_graphic_region(img, px)
            pb = _px_box_to_pdf(px, pw, ph, iw, ih)
            score = cv_diagram_score(img, px) + float(cap["score"]) * 0.09 + (0.10 if page <= 3 else 0.0)
            items.append({"score": score, "cap_score": cap["score"], "line": cap["line"], "page": page, "pw": pw, "ph": ph, "bbox": pb})
    items.sort(key=lambda x: x["score"], reverse=True)
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for it in items:
        x0, y0, x1, y1 = it["bbox"]
        key = (int(it["page"]), int(x0 // 8), int(y0 // 8), int(x1 // 8), int(y1 // 8))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
        if len(dedup) >= limit:
            break
    return dedup


def _pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def vlm_rerank_candidates(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    retries: int,
    pdf_path: Path,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> int | None:
    if not candidates:
        return None
    k = max(1, min(top_k, len(candidates)))
    picked = candidates[:k]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "从候选图中选最像方法框架图/模型架构图/流程图的一张。"
                "优先模块框+箭头+整体流程，不要结果图/分布图/示例。只输出一个数字(1..%d)。" % k
            ),
        }
    ]
    for idx, c in enumerate(picked, start=1):
        page_img = render_pdf_page_pil(pdf_path, int(c["page"]))
        px = _pdf_box_to_px(c["bbox"], float(c["pw"]), float(c["ph"]), page_img.size[0], page_img.size[1])
        crop = page_img.crop(px)
        content.append({"type": "text", "text": f"候选{idx}: page={c['page']}, caption={str(c['line'])[:70]}"})
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(crop)}})
    text = call_chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        retries=retries,
        messages=[{"role": "user", "content": content}],
    )
    return parse_choice_index(text, k)


def fallback_framework_region(pdf_path: Path, scan_pages: int) -> tuple[int, float, float, tuple[float, float, float, float], str]:
    reader = PdfReader(str(pdf_path))
    max_idx = min(len(reader.pages), max(scan_pages, 1))
    best_score, best_page = -1, 1
    for idx in range(max_idx):
        raw = (reader.pages[idx].extract_text() or "").lower()
        score = 0
        for kw, weight in FIGURE_KEYWORDS.items():
            score += raw.count(kw.lower()) * weight
        if re.search(r"figure\s*\d+", raw):
            score += 3
        if re.search(r"fig\.\s*\d+", raw):
            score += 3
        if "introduction" in raw or "related work" in raw:
            score -= 1
        if "framework" in raw and ("figure" in raw or "fig." in raw):
            score += 6
        if score > best_score:
            best_score, best_page = score, idx + 1
    page = reader.pages[best_page - 1]
    pw, ph = float(page.mediabox.width), float(page.mediabox.height)
    bbox = (pw * 0.10, ph * 0.15, pw * 0.90, ph * 0.82)
    return best_page, pw, ph, bbox, "fallback"

