"""Deterministic selection of curriculum retrieval capabilities."""

from __future__ import annotations

import re
from typing import Literal


Route = Literal["structured", "semantic", "hybrid"]

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
    "เรียนช่วงไหน",
    "อยู่ช่วงไหนของหลักสูตร",
    "เรียนปีไหน",
    "อยู่ปีไหน",
    "ภาคเรียน",
    "ภาคการศึกษา",
    "เทอม",
    "หน่วยกิต",
    "ช่วงเรียน",
    "กำหนดแน่นอน",
    "ยืดหยุ่น",
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
    """Select SQL, vector, or combined retrieval using keyword rules only."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    normalized = f" {question.casefold().strip()} "
    structured = any(term in normalized for term in _STRUCTURED_TERMS if term != "กี่")
    structured = structured or _THAI_COUNT_RE.search(normalized) is not None
    semantic = any(term in normalized for term in _SEMANTIC_TERMS)
    if structured and semantic:
        return "hybrid"
    if structured:
        return "structured"
    if semantic:
        return "semantic"
    return "semantic"


__all__ = ["Route", "route_question"]
