"""Grounded final-answer generation for unified curriculum QA."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any


EMPTY_ANSWER = "ไม่พบข้อมูลนี้ในเล่มหลักสูตร"
_ROUTES = {"structured", "semantic", "hybrid"}
_STRUCTURED_RESULT_METADATA = frozenset(
    {"operation", "status", "missing_plan_keys", "sql", "columns", "rows", "provenance"}
)


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


def _is_empty_answer(answer: Any) -> bool:
    return isinstance(answer, str) and answer.strip() == EMPTY_ANSWER


_FALLBACK_LIKE_PREFIXES = (
    "ไม่พบข้อมูล",
    "ไม่พบหลักฐาน",
    "ไม่พบคำตอบ",
    "ไม่พบสิ่งที่ถาม",
    "ไม่มีข้อมูล",
    "ไม่มีหลักฐาน",
    "ไม่มีคำตอบ",
    "ไม่สามารถตอบ",
    "ไม่สามารถระบุ",
    "ไม่ทราบ",
    "ไม่ปรากฏข้อมูล",
    "no data",
    "no relevant evidence",
    "not found",
    "unable to answer",
    "cannot answer",
    "no answer",
)
_COST_QUERY_TERMS = (
    "ค่าเทอม",
    "ค่าเล่าเรียน",
    "ค่าใช้จ่าย",
    "ค่าเรียน",
    "tuition",
    "fee",
    "fees",
    "cost",
    "price",
    "ราคา",
)
_COST_EVIDENCE_TERMS = _COST_QUERY_TERMS


def is_fallback_like(answer: Any) -> bool:
    """Recognize short, whole-answer no-data denials conservatively."""
    if not isinstance(answer, str):
        return False
    normalized = re.sub(r"\s+", " ", answer.casefold()).strip()
    if normalized == re.sub(r"\s+", " ", EMPTY_ANSWER.casefold()).strip():
        return True
    if len(normalized) > 240:
        return False
    normalized = re.sub(r"^[\s`*_\-:]+|[\s`*_\-:]+$", "", normalized)
    if normalized.startswith("จากหลักฐาน"):
        normalized = re.sub(r"^จากหลักฐาน(?:ที่มี)?\s*", "", normalized)
    return normalized.startswith(_FALLBACK_LIKE_PREFIXES)


def _deterministic_structured_fallback(structured_result: Any) -> str | None:
    """Expose only requested, structured rows when synthesis still denies facts."""
    rows = _structured_rows(structured_result)
    if not rows:
        return None
    serialized = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), default=str)
    return "ข้อมูลโครงสร้างหลักสูตรที่ยืนยันได้: " + serialized


def _question_has_cost_intent(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    return any(term in normalized for term in _COST_QUERY_TERMS)


def _row_has_nonempty_field(row: Mapping[str, Any], predicate: Callable[[str], bool]) -> bool:
    return any(
        predicate(str(field).casefold())
        and value is not None
        and str(value).strip() != ""
        for field, value in row.items()
    )


def _structured_rows_support_question(question: str, structured_result: Any) -> bool:
    rows = [row for row in _structured_rows(structured_result) if isinstance(row, Mapping)]
    if not rows:
        return False
    text = question.casefold()
    if _question_has_cost_intent(question):
        return any(
            _row_has_nonempty_field(
                row,
                lambda field: any(term in field for term in ("cost", "tuition", "fee", "price", "ค่า")),
            )
            for row in rows
        )

    predicates: list[Callable[[str], bool]] = []
    if "หน่วยกิต" in text or "เครดิต" in text or "credit" in text:
        predicates.append(lambda field: "credit" in field or "หน่วยกิต" in field)
    if any(term in text for term in ("ต้องผ่าน", "ก่อนลง", "เรียนก่อน", "prerequisite")):
        predicates.append(
            lambda field: field in {
                "course_code",
                "course",
                "prerequisite_code",
                "prerequisite_course_code",
                "raw_text",
            }
            or "prerequisite" in field
        )
    if any(term in text for term in ("ช่วงไหน", "ช่วงใด", "อยู่ปีไหน", "เรียนปีไหน", "เทอมไหน", "จัดวาง", "placement")):
        predicates.append(
            lambda field: field in {"year", "semester", "plan", "plan_key"}
            or "placement" in field
            or "flexible" in field
        )
    if "ชื่อ" in text or "name" in text:
        predicates.append(lambda field: field.startswith("name") or "description" in field)
    if any(term in text for term in ("อะไรบ้าง", "วิชาใด", "วิชาอะไร", "รายวิชา", "course list")):
        predicates.append(lambda field: field in {"course_code", "course", "name_th", "name_en"})
    if not predicates:
        return False
    return any(_row_has_nonempty_field(row, predicate) for row in rows for predicate in predicates)


def _evidence_supports_requested_attribute(
    question: str,
    route: str,
    structured_result: Any,
    semantic_chunks: Any,
) -> bool:
    """Fail closed for unsupported cost/tuition requests."""
    if not _question_has_cost_intent(question):
        return True
    structured_support = _structured_rows_support_question(question, structured_result)
    semantic_support = any(
        any(term in str(chunk.get("text", "")).casefold() for term in _COST_EVIDENCE_TERMS)
        for chunk in _semantic_chunks(semantic_chunks)
    )
    if route == "structured":
        return structured_support
    if route == "semantic":
        return semantic_support
    return structured_support or semantic_support


def _requested_plan_keys(question: str) -> list[str]:
    normalized = question.casefold()
    plan_keys: list[str] = []
    if "no_coop" in normalized or "ไม่สหกิจ" in normalized:
        plan_keys.append("no_coop")
    without_no_coop = normalized.replace("no_coop", "").replace("ไม่สหกิจ", "")
    if "coop" in without_no_coop or "สหกิจ" in without_no_coop:
        plan_keys.append("coop")
    return plan_keys


def _plan_prefixes(plan: str) -> tuple[str, ...]:
    return (plan, "nocoop") if plan == "no_coop" else (plan,)


def _rows_for_plan(structured_result: Any, plan: str) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    scoped_keys = tuple(
        f"{prefix}_{field}"
        for prefix in _plan_prefixes(plan)
        for field in (
            "course_code",
            "year",
            "semester",
            "flexible_year_semester_raw",
            "placement_raw",
        )
    )
    for row in _structured_rows(structured_result):
        if not isinstance(row, Mapping):
            continue
        explicit_plan = row.get("plan", row.get("plan_key"))
        if explicit_plan is not None:
            if str(explicit_plan).casefold() == plan:
                candidates.append(row)
        elif any(key in row for key in scoped_keys):
            candidates.append(row)
    return candidates


def _plan_value(row: Mapping[str, Any], plan: str, field: str) -> Any:
    explicit_plan = row.get("plan", row.get("plan_key"))
    if explicit_plan is not None and str(explicit_plan).casefold() == plan:
        return row.get(field)
    aliases = {
        "course_code": ("course_code",),
        "year": ("year",),
        "semester": ("semester",),
        "flexible_year_semester_raw": (
            "flexible_year_semester_raw",
            "placement_raw",
        ),
    }[field]
    for prefix in _plan_prefixes(plan):
        for alias in aliases:
            key = f"{prefix}_{alias}"
            if key in row:
                return row[key]
    return None


def _answer_mentions_plan(answer: Any, plan: str) -> bool:
    text = re.sub(r"\s+", " ", str(answer).casefold())
    if plan == "no_coop":
        return "no_coop" in text or "ไม่สหกิจ" in text
    without_no_coop = text.replace("no_coop", "").replace("ไม่สหกิจ", "")
    return "coop" in without_no_coop or "สหกิจ" in without_no_coop


def _answer_covers_plan_row(answer: Any, plan: str, row: Mapping[str, Any]) -> bool:
    if not _answer_mentions_plan(answer, plan):
        return False
    for field in ("year", "semester", "flexible_year_semester_raw"):
        value = _plan_value(row, plan, field)
        if value is None:
            continue
        value_text = re.sub(r"\s+", " ", str(value).casefold()).strip()
        if field == "flexible_year_semester_raw":
            pieces = [piece for piece in re.findall(r"\d+\s*/\s*\d+", value_text)]
            if value_text not in re.sub(r"\s+", " ", str(answer).casefold()) and not all(
                piece.replace(" ", "") in re.sub(r"\s+", " ", str(answer).casefold()).replace(" ", "")
                for piece in pieces
            ):
                return False
        elif value_text not in re.sub(r"\s+", " ", str(answer).casefold()):
            return False
    return True


def _missing_hybrid_plan_evidence(
    question: str,
    structured_result: Any,
    answer: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    plans = _requested_plan_keys(question)
    if len(plans) < 2:
        return [], []
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for plan in plans:
        rows = _rows_for_plan(structured_result, plan)
        if not rows:
            continue
        records.extend(
            {
                "plan": plan,
                "course_code": _plan_value(row, plan, "course_code"),
                "year": _plan_value(row, plan, "year"),
                "semester": _plan_value(row, plan, "semester"),
                "flexible_year_semester_raw": _plan_value(
                    row, plan, "flexible_year_semester_raw"
                ),
            }
            for row in rows
        )
        if not any(_answer_covers_plan_row(answer, plan, row) for row in rows):
            missing.append(plan)
    return missing, records


def _hybrid_placement_appendix(records: Sequence[Mapping[str, Any]]) -> str:
    serialized = json.dumps(list(records), ensure_ascii=False, separators=(",", ":"), default=str)
    return "ข้อมูลการจัดวางที่ยืนยันได้เพิ่มเติม: " + serialized


def _evidence(
    route: str,
    structured_result: Any,
    semantic_chunks: Any,
) -> Any:
    if route == "structured":
        rows = _structured_rows(structured_result)
        if isinstance(structured_result, Mapping):
            derived_facts = {
                key: value
                for key, value in structured_result.items()
                if key not in _STRUCTURED_RESULT_METADATA
            }
            if derived_facts:
                return {"derived_facts": derived_facts, "sql_rows": rows}
        return rows
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
            "ชื่อวิชาที่ว่างหรือเป็น 'ไม่ระบุ' เป็นเพียงข้อมูลชื่อที่ขาดหาย "
            "ไม่ใช่เหตุผลให้ปฏิเสธ หาก description มีเนื้อหาที่ตอบคำถามได้ "
            "ให้สรุปจาก description นั้นเท่านั้น",
            rules,
            f"คำถาม: {question}",
            f"เส้นทาง: {normalized_route}",
            f"{evidence_label}: {serialized_evidence}",
            "ตอบ " + EMPTY_ANSWER + " เมื่อหลักฐานไม่มีข้อมูลที่ตอบคำถามได้; "
            "ถ้าหลักฐานตอบได้เพียงบางส่วน ให้ตอบเฉพาะส่วนที่ยืนยันได้ "
            "และห้ามเดาหรือสร้างคำตอบจากหลักฐานที่เกี่ยวข้องเพียงผิวเผิน",
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
    if not _evidence_supports_requested_attribute(
        question,
        normalized_route,
        structured_result,
        semantic_chunks,
    ):
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
    answer_text = answer if isinstance(answer, str) else str(answer)
    missing_plans, placement_records = _missing_hybrid_plan_evidence(
        question,
        structured_result,
        answer_text,
    ) if normalized_route == "hybrid" else ([], [])
    if is_fallback_like(answer_text) or missing_plans:
        recovery_prompt = (
            prompt
            + "\nตรวจสอบหลักฐานอีกครั้งอย่างเคร่งครัด: "
            "ถ้าหลักฐานรองรับคำถาม ให้สรุปเฉพาะข้อเท็จจริงที่มีหลักฐานรองรับ; "
            "ถ้าหลักฐานไม่พอ ให้ตอบข้อความ fallback เดิม และห้ามเดาหรือสร้างข้อมูล"
        )
        if normalized_route == "hybrid" and missing_plans:
            recovery_prompt += (
                "\nคำถามนี้ขอข้อมูลหลายแผน ต้องรายงาน placement ของทุกแผนที่มีหลักฐาน "
                "รวมถึง flexible_year_semester_raw ห้ามตัดแผนใดออก"
            )
        recovered = answer_model_callable(recovery_prompt)
        recovered_text = recovered if isinstance(recovered, str) else str(recovered)
        if not is_fallback_like(recovered_text):
            answer_text = recovered_text
            if normalized_route == "hybrid":
                missing_plans, placement_records = _missing_hybrid_plan_evidence(
                    question,
                    structured_result,
                    answer_text,
                )
                if missing_plans and placement_records:
                    return answer_text + "\n" + _hybrid_placement_appendix(placement_records)
            else:
                return answer_text
        if normalized_route == "structured":
            structured_fallback = _deterministic_structured_fallback(structured_result)
            if structured_fallback is not None:
                return structured_fallback
        if is_fallback_like(answer_text):
            return EMPTY_ANSWER
        if normalized_route == "hybrid" and missing_plans and placement_records:
            return answer_text + "\n" + _hybrid_placement_appendix(placement_records)
    return answer_text


__all__ = ["EMPTY_ANSWER", "answer_question", "build_grounded_prompt", "is_fallback_like"]
