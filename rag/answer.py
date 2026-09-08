"""Grounded final-answer generation for unified curriculum QA."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any


EMPTY_ANSWER = "ไม่พบข้อมูลนี้ในเล่มหลักสูตร"
_ROUTES = {"structured", "semantic", "hybrid"}


def _structured_rows(structured_result: Any) -> list[Any]:
    if isinstance(structured_result, Mapping):
        rows = structured_result.get("rows", [])
        columns = structured_result.get("columns", []) or []
    else:
        rows = structured_result
        columns = []

    if rows is None:
        return []
    if isinstance(rows, Mapping) or isinstance(rows, (str, bytes)):
        rows = [rows]

    try:
        row_values = list(rows)
    except TypeError:
        row_values = [rows]

    normalized: list[Any] = []
    for row in row_values:
        if isinstance(row, Mapping):
            normalized.append(dict(row))
        elif columns and isinstance(row, (list, tuple)):
            normalized.append(dict(zip(columns, row)))
        else:
            normalized.append(row)
    return normalized


def _source_pages(chunk: Mapping[str, Any]) -> list[Any]:
    pages = chunk.get("source_page")
    if pages is None:
        provenance = chunk.get("provenance")
        if isinstance(provenance, Sequence) and not isinstance(provenance, (str, bytes)):
            pages = [
                reference.get("source_page")
                for reference in provenance
                if isinstance(reference, Mapping) and reference.get("source_page") is not None
            ]
    if pages is None:
        return []
    if isinstance(pages, (list, tuple, set)):
        return [page for page in pages if page is not None]
    return [pages]


def _semantic_chunks(semantic_chunks: Any) -> list[dict[str, Any]]:
    if semantic_chunks is None:
        return []
    if isinstance(semantic_chunks, Mapping):
        semantic_chunks = [semantic_chunks]
    try:
        chunks = list(semantic_chunks)
    except TypeError:
        chunks = [semantic_chunks]

    normalized: list[dict[str, Any]] = []
    for chunk in chunks:
        if isinstance(chunk, Mapping):
            normalized.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_page": _source_pages(chunk),
                    "text": chunk.get("text", ""),
                }
            )
        else:
            normalized.append({"chunk_id": None, "source_page": [], "text": str(chunk)})
    return normalized


def _evidence(
    route: str,
    structured_result: Any,
    semantic_chunks: Any,
) -> Any:
    if route == "structured":
        return _structured_rows(structured_result)
    if route == "semantic":
        return _semantic_chunks(semantic_chunks)
    return {
        "sql_rows": _structured_rows(structured_result),
        "retrieved_chunks": _semantic_chunks(semantic_chunks),
    }


def build_grounded_prompt(
    question: str,
    route: str,
    structured_result: Any = None,
    semantic_chunks: Any = None,
) -> str:
    """Build a compact prompt containing only retrieved curriculum evidence."""
    normalized_route = route.casefold() if isinstance(route, str) else route
    evidence = _evidence(normalized_route, structured_result, semantic_chunks)
    if normalized_route == "structured":
        evidence_label = "ผลลัพธ์จาก SQL (ใช้ค่าที่มีอยู่แล้วเท่านั้น)"
        rules = "ห้ามนับ รวม คำนวณ หรือสร้างค่าตัวเลขใหม่จากแถวผลลัพธ์"
    elif normalized_route == "semantic":
        evidence_label = "ชิ้นส่วนหลักฐานที่ค้นคืนได้"
        rules = "ตอบจากข้อความในชิ้นส่วนหลักฐานเท่านั้น"
    else:
        evidence_label = "ผลลัพธ์จาก SQL และชิ้นส่วนหลักฐานที่ค้นคืนได้"
        rules = "ใช้ค่าจาก SQL และข้อความในชิ้นส่วนหลักฐานเท่านั้น"

    serialized_evidence = json.dumps(
        evidence, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return "\n".join(
        (
            "ตอบคำถามต่อไปนี้เป็นภาษาไทยจากหลักฐานที่ให้เท่านั้น",
            "ห้ามแต่งเติมข้อเท็จจริงหรืออนุมานข้อมูลที่ไม่มีในหลักฐาน",
            "คงรหัสวิชา ชื่อวิชา และตัวเลขตามหลักฐานทุกประการ",
            "ใส่ source_page ในคำตอบเมื่อหลักฐานมีค่านี้",
            rules,
            f"คำถาม: {question}",
            f"เส้นทาง: {normalized_route}",
            f"{evidence_label}: {serialized_evidence}",
            "หากหลักฐานไม่ตอบคำถาม ให้ตอบ: " + EMPTY_ANSWER,
            "คำตอบ:",
        )
    )


def answer_question(
    question: str,
    route: str,
    structured_result: Any = None,
    semantic_chunks: Any = None,
    answer_model_callable: Callable[[str], str] | None = None,
) -> str:
    """Return a final answer grounded only in retrieved curriculum evidence."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(route, str) or route.casefold() not in _ROUTES:
        raise ValueError("route must be structured, semantic, or hybrid")

    normalized_route = route.casefold()
    evidence = _evidence(normalized_route, structured_result, semantic_chunks)
    if normalized_route == "hybrid":
        has_evidence = bool(evidence["sql_rows"] or evidence["retrieved_chunks"])
    else:
        has_evidence = bool(evidence)
    if not has_evidence:
        return EMPTY_ANSWER
    if not callable(answer_model_callable):
        raise ValueError("answer_model_callable is required when evidence is present")

    prompt = build_grounded_prompt(
        question,
        normalized_route,
        structured_result=structured_result,
        semantic_chunks=semantic_chunks,
    )
    answer = answer_model_callable(prompt)
    return answer if isinstance(answer, str) else str(answer)


__all__ = ["EMPTY_ANSWER", "answer_question", "build_grounded_prompt"]
