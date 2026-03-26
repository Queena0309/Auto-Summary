from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from .constants import LIST_FIELDS, REQUIRED_KEYS
from .text_utils import clean_json_block


def call_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    retries: int,
) -> str:
    if not api_key:
        raise RuntimeError("API key is empty. Set --api-key or HAPPYAPI_API_KEY.")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                sleep_s = min(2 ** (attempt - 1), 8)
                print(f"[WARN] API call failed ({attempt}/{retries}): {exc}. Retrying in {sleep_s}s...", flush=True)
                time.sleep(sleep_s)
    raise RuntimeError(f"API call failed after {retries} attempts: {last_error}")


def extract_structured_info(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    retries: int,
    pdf_name: str,
    pdf_text: str,
    direction_candidates: set[str],
) -> dict[str, Any]:
    direction_str = ", ".join(sorted(direction_candidates))
    user_prompt = f"""
你将阅读论文文本片段，输出一个JSON对象（不要输出其它内容）。

要求：
1) 所有字段都必须有值，没有就填"未提及"或空数组。
2) direction 只能从以下集合中选一个最接近的：{direction_str}。无法判断就选 "PD"。
3) year 必须是4位年份字符串。
4) challenges / impressive_points / inspirations / other_points / relations / future_work 必须是字符串数组。
5) 内容用中文，title 保持原论文英文标题。

字段：
{{
  "direction": "",
  "venue": "",
  "year": "",
  "title": "",
  "paper_url": "",
  "code_open_source": "",
  "code_url": "",
  "one_sentence_summary": "",
  "one_sentence_contrib": "",
  "innovation_example": "",
  "workflow_description": "",
  "challenges": [],
  "impressive_points": [],
  "inspirations": [],
  "idea_analysis": "",
  "novelty": "",
  "hotspot": "",
  "other_points": [],
  "relations": [],
  "future_work": []
}}

PDF文件名：{pdf_name}
论文文本片段如下：
{pdf_text}
""".strip()
    content = call_chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        retries=retries,
        messages=[
            {"role": "system", "content": "你是严谨的学术信息抽取助手。仅返回合法JSON对象。"},
            {"role": "user", "content": user_prompt},
        ],
    )
    info = clean_json_block(content)
    for key in REQUIRED_KEYS:
        info.setdefault(key, [] if key in LIST_FIELDS else "未提及")
    return info


def polish_content(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    retries: int,
    extracted: dict[str, Any],
) -> dict[str, Any]:
    user_prompt = f"""
基于下方JSON信息，润色并输出同结构JSON。

要求：
1) 中文表达自然、精炼、准确，避免空话。
2) one_sentence_summary / one_sentence_contrib 必须保持“一句话”长度。
3) 数组字段必须“自适应数量”，不要凑数：
   - challenges / impressive_points / inspirations / future_work：2~4条
   - other_points / relations：1~3条
4) 每条都要是核心信息，不要重复表达同一意思。
5) 不得杜撰实验数字、数据集或链接；不确定时保留"未提及"。

输入JSON：
{json.dumps(extracted, ensure_ascii=False)}
""".strip()
    content = call_chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        retries=retries,
        messages=[
            {"role": "system", "content": "你是学术写作助手。仅返回合法JSON对象。"},
            {"role": "user", "content": user_prompt},
        ],
    )
    polished = clean_json_block(content)
    for key in REQUIRED_KEYS:
        polished.setdefault(key, extracted.get(key, "未提及"))
    return polished


def repair_fields(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    retries: int,
    data: dict[str, Any],
    fields: list[str],
) -> dict[str, Any]:
    if not fields:
        return data
    keep = {k: data.get(k) for k in REQUIRED_KEYS}
    user_prompt = f"""
你将修复JSON中指定字段，返回完整JSON对象（不要输出其他内容）。

待修复字段：{fields}
修复要求：
1) 列表字段使用高信息密度要点，不重复，不凑数。
2) 数量约束：
   - challenges / impressive_points / inspirations / future_work：2~4条
   - other_points / relations：1~3条
3) 不得编造不存在的链接或实验数据。
4) 若确实无法确认，用"未提及"或空数组。

当前JSON：
{json.dumps(keep, ensure_ascii=False)}
""".strip()
    content = call_chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        retries=retries,
        messages=[
            {"role": "system", "content": "你是学术写作修复助手，仅返回合法JSON对象。"},
            {"role": "user", "content": user_prompt},
        ],
    )
    fixed = clean_json_block(content)
    for key in REQUIRED_KEYS:
        fixed.setdefault(key, data.get(key, "未提及"))
    return fixed


def parse_choice_index(text: str, max_n: int) -> int | None:
    m = re.search(r"\b([1-9]\d*)\b", text)
    if not m:
        return None
    idx = int(m.group(1))
    if 1 <= idx <= max_n:
        return idx - 1
    return None

