from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    root: Path
    api_key: str
    base_url: str
    model: str
    sharer: str
    max_pages: int
    scan_pages: int
    max_chars: int
    timeout: int
    retries: int
    use_vlm_rerank: bool
    vlm_top_k: int
    dry_run: bool


def parse_settings() -> Settings:
    parser = argparse.ArgumentParser(description="Auto summarize PDFs into markdown.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--api-key", default=os.getenv("HAPPYAPI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("HAPPYAPI_BASE_URL", "https://happyapi.org/v1"))
    parser.add_argument("--model", default=os.getenv("HAPPYAPI_MODEL", "gpt-5.1-high"))
    parser.add_argument("--sharer", default=os.getenv("SUMMARY_SHARER", "自动生成"))
    parser.add_argument("--max-pages", type=int, default=16)
    parser.add_argument("--scan-pages", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=36000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--use-vlm-rerank", action="store_true")
    parser.add_argument("--vlm-top-k", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return Settings(
        root=args.root.resolve(),
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        sharer=args.sharer,
        max_pages=args.max_pages,
        scan_pages=args.scan_pages,
        max_chars=args.max_chars,
        timeout=args.timeout,
        retries=args.retries,
        use_vlm_rerank=args.use_vlm_rerank,
        vlm_top_k=args.vlm_top_k,
        dry_run=args.dry_run,
    )

