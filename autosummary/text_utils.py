from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .constants import CORE_KEYWORDS, DIRECTION_FALLBACK, LIST_FIELD_LIMITS


def load_directions(root: Path) -> set[str]:
    rd_path = root.parent / "Social-AI-Group" / "Research Direction.md"
    if not rd_path.exists():
        return set(DIRECTION_FALLBACK)
    pattern = re.compile(r"^###\s*([A-Za-z0-9_-]+)\s*:")
    found: set[str] = set()
    for line in rd_path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            found.add(m.group(1).strip())
    return found or set(DIRECTION_FALLBACK)


def list_pending_pdfs(pending_dir: Path) -> list[Path]:
    files = [p for p in pending_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(files, key=lambda p: p.name.lower())


def extract_pdf_text(pdf_path: Path, max_pages: int, max_chars: int) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    total = 0
    for page_idx, page in enumerate(reader.pages):
        if page_idx >= max_pages or total >= max_chars:
            break
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        remain = max_chars - total
        text = text[:remain]
        chunks.append(f"[Page {page_idx + 1}] {text}")
        total += len(text)
    return "\n".join(chunks).strip()


def next_image_name(image_dir: Path) -> str:
    day = dt.datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf"^{day}(\d{{2}})\.png$")
    max_id = 0
    for p in image_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return f"{day}{(max_id + 1):02d}.png"


def clean_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found.")
    return json.loads(text[start : end + 1])


def safe_string(value: Any, default: str = "未提及") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [safe_string(x, "").strip() for x in value if safe_string(x, "").strip()]
    if isinstance(value, str):
        return [x.strip(" -\t") for x in value.splitlines() if x.strip()]
    return [safe_string(value)]


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|]", "-", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text or "Untitled"


def normalize_year(value: Any) -> str:
    txt = safe_string(value, "unknown")
    m = re.search(r"(19|20)\d{2}", txt)
    return m.group(0) if m else txt


def fallback_title_from_filename(pdf_name: str) -> str:
    stem = Path(pdf_name).stem
    return stem.replace("_", " ").strip() or "Untitled Paper"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _normalize_item_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("。.;；,，")


def _dedup_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        k = re.sub(r"[\W_]+", "", item.lower())
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def _core_score(item: str) -> int:
    score = 0
    low = item.lower()
    for kw in CORE_KEYWORDS:
        if kw.lower() in low:
            score += 2
    n = len(item)
    if 12 <= n <= 48:
        score += 2
    if n > 80:
        score -= 1
    return score


def trim_list_items(items: list[str], min_n: int, max_n: int) -> list[str]:
    cleaned = [_normalize_item_text(x) for x in items if _normalize_item_text(x)]
    deduped = _dedup_items(cleaned)
    if not deduped:
        return []
    ranked = sorted(deduped, key=lambda x: (_core_score(x), -len(x)), reverse=True)
    selected = ranked[:max_n]
    if len(selected) < min_n:
        for x in ranked[max_n:]:
            if x not in selected:
                selected.append(x)
            if len(selected) >= min_n:
                break
    return selected[:max_n]


def apply_quality_constraints(data: dict[str, Any]) -> dict[str, Any]:
    for field, (min_n, max_n) in LIST_FIELD_LIMITS.items():
        data[field] = trim_list_items(ensure_list(data.get(field)), min_n=min_n, max_n=max_n)
    for field in ["one_sentence_summary", "one_sentence_contrib"]:
        value = re.sub(r"\s+", " ", safe_string(data.get(field))).strip()
        if len(value) > 120:
            value = value[:120].rstrip("，,;；。.") + "。"
        data[field] = value
    return data


def quality_gaps(data: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field, (min_n, _max_n) in LIST_FIELD_LIMITS.items():
        if len(ensure_list(data.get(field))) < min_n:
            missing.append(field)
    for field in ["one_sentence_summary", "one_sentence_contrib", "innovation_example", "workflow_description"]:
        if safe_string(data.get(field)) == "未提及":
            missing.append(field)
    return missing

