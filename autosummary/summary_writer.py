from __future__ import annotations

from typing import Any

from .text_utils import ensure_list, normalize_year, safe_string


def build_markdown(data: dict[str, Any], image_filename: str, sharer: str) -> str:
    direction = safe_string(data.get("direction"), "PD")
    venue = safe_string(data.get("venue"), "arXiv")
    year = normalize_year(data.get("year"))
    title = safe_string(data.get("title"), "Untitled Paper")
    header_title = f"{direction}-{venue}-{year}-{title}"

    paper_url = safe_string(data.get("paper_url"), "未提及")
    code_open = safe_string(data.get("code_open_source"), "未提及")
    code_url = safe_string(data.get("code_url"), "未提及")
    if code_url == "未提及":
        code_line = f"*代码是否开源：{code_open}*"
    else:
        code_line = f"*代码是否开源：{code_open} {code_url}*"

    one_sum = safe_string(data.get("one_sentence_summary"))
    one_contrib = safe_string(data.get("one_sentence_contrib"))
    example = safe_string(data.get("innovation_example"))
    workflow = safe_string(data.get("workflow_description"))
    challenges = ensure_list(data.get("challenges"))
    impressive = ensure_list(data.get("impressive_points"))
    inspirations = ensure_list(data.get("inspirations"))
    idea_analysis = safe_string(data.get("idea_analysis"))
    novelty = safe_string(data.get("novelty"))
    hotspot = safe_string(data.get("hotspot"))
    other_points = ensure_list(data.get("other_points"))
    relations = ensure_list(data.get("relations"))
    future_work = ensure_list(data.get("future_work"))

    def list_block(items: list[str]) -> str:
        if not items:
            return "> 未提及"
        return "\n".join(f"> {i}. {x}" for i, x in enumerate(items, 1))

    lines = [
        f"# {header_title}",
        f"*论文下载地址：{paper_url}*",
        "",
        code_line,
        "",
        f"*分享人：{sharer}*",
        "",
        "## 一句话总结内容",
        f"> {one_sum}",
        "",
        "## 一句话总结创新贡献",
        f"> {one_contrib}",
        "",
        "## 举一个例子说明这篇文章的创新点",
        f"> {example}",
        "",
        "## 框架图",
        f"![framework](../image/{image_filename})",
        "",
        "**框架工作流描述**：",
        f"> {workflow}",
        "",
        "## 本文挑战及已有工作不足",
        list_block(challenges),
        "",
        "## 印象最深刻的点",
        list_block(impressive),
        "",
        "## 对我们的启发",
        list_block(inspirations),
        "",
        "## Idea是否好想",
        f"> {idea_analysis}",
        "",
        "## 是否有开创性",
        f"> {novelty}",
        "",
        "## 是否属于热点",
        f"> {hotspot}",
        "",
        "## 其他需要补充的点（可选）",
        list_block(other_points),
        "",
        "## 与其他论文的关联（可选）",
        list_block(relations),
        "",
        "## 还有哪些不足的地方（未来工作）",
        list_block(future_work),
        "",
    ]
    return "\n".join(lines)

