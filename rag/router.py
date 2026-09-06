"""Deterministic routing between structured and semantic retrieval."""

from __future__ import annotations

import re
from typing import Literal


Route = Literal["structured", "semantic"]

_STRUCTURED_TERMS = (
    "count",
    "how many",
    "number of",
    "total",
    "year",
    "semester",
    "term",
    "academic year",
    "credit",
    "prerequisite",
    "pre-requisite",
    "required before",
    "compare",
    "comparison",
    "versus",
    " vs ",
    "difference between",
    "กี่",
    "จำนวน",
    "นับ",
    "รวม",
    "ทั้งหมด",
    "ปีที่",
    "ภาคเรียน",
    "ภาคการศึกษา",
    "เทอม",
    "หน่วยกิต",
    "วิชาบังคับก่อน",
    "บังคับก่อน",
    "ต้องเรียนก่อน",
    "เปรียบเทียบ",
    "แตกต่างกัน",
)

_SEMANTIC_TERMS = (
    "topic",
    "topics",
    "content",
    "meaning",
    "explain",
    "describe",
    "about",
    "covers",
    "similar",
    "similarity",
    "related",
    "หัวข้อ",
    "เนื้อหา",
    "ความหมาย",
    "อธิบาย",
    "เกี่ยวกับ",
    "คล้าย",
    "เหมือน",
    "สาระ",
)
_THAI_COUNT_RE = re.compile(r"กี่(?!ย)")


def route_question(question: str) -> Route:
    """Route a question using keyword rules only."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    normalized = f" {question.casefold().strip()} "
    structured = any(term in normalized for term in _STRUCTURED_TERMS if term != "กี่")
    if structured or _THAI_COUNT_RE.search(normalized):
        return "structured"
    if any(term in normalized for term in _SEMANTIC_TERMS):
        return "semantic"
    return "semantic"


__all__ = ["Route", "route_question"]
