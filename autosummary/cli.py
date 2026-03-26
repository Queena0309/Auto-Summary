from __future__ import annotations

from .pipeline import run_pipeline
from .settings import parse_settings


def main() -> int:
    settings = parse_settings()
    return run_pipeline(settings)

