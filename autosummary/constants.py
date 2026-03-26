from __future__ import annotations

DIRECTION_FALLBACK = {"DWM", "Data", "Memory", "Planing", "PD", "Recommend"}

REQUIRED_KEYS = [
    "direction",
    "venue",
    "year",
    "title",
    "paper_url",
    "code_open_source",
    "code_url",
    "one_sentence_summary",
    "one_sentence_contrib",
    "innovation_example",
    "workflow_description",
    "challenges",
    "impressive_points",
    "inspirations",
    "idea_analysis",
    "novelty",
    "hotspot",
    "other_points",
    "relations",
    "future_work",
]

LIST_FIELDS = {"challenges", "impressive_points", "inspirations", "other_points", "relations", "future_work"}

LIST_FIELD_LIMITS: dict[str, tuple[int, int]] = {
    "challenges": (2, 4),
    "impressive_points": (2, 4),
    "inspirations": (2, 4),
    "other_points": (1, 3),
    "relations": (1, 3),
    "future_work": (2, 4),
}

CORE_KEYWORDS = {
    "问题",
    "挑战",
    "不足",
    "方法",
    "框架",
    "机制",
    "策略",
    "实验",
    "结果",
    "提升",
    "误差",
    "局限",
    "未来",
    "对比",
    "贡献",
}

FIGURE_KEYWORDS = {
    "framework": 6,
    "overview": 5,
    "architecture": 5,
    "pipeline": 4,
    "method": 3,
    "approach": 3,
    "model": 2,
    "系统框架": 6,
    "总体框架": 6,
    "方法框架": 5,
    "框架": 4,
    "整体流程": 4,
    "方法": 2,
}

