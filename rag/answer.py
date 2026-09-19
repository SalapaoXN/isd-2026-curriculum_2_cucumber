"""Grounded final-answer generation for unified curriculum QA."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from rag.aggregation import (
    ComparisonAggregation,
    EarliestAggregation,
    PlanComparisonAggregation,
)
from rag.grounded_answer import GroundedAnswerResult, GroundedClaim
from rag.retrieval.retrieve import SimilarityEvidence


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
_TOPIC_LIST_MARKERS = (
    "คำอธิบายรายวิชาภาษาอังกฤษ:",
    "english course description:",
    "topics:",
)
_TOPIC_LIST_SPLIT = re.compile(r",\s*|;\s*|\n+\s*[-•]\s*|•\s*|–\s*|—\s*")
_TOPIC_ITEM = re.compile(r"^[A-Z][A-Z0-9 /&()'\-]{3,}$")
_TOPIC_STOPWORDS = {"A", "AN", "AND", "FOR", "IN", "OF", "ON", "THE", "TO"}
_TOPIC_TOKEN_ALIASES = {
    "ACCESS": ("access", "เข้าถึง"),
    "CENTER": ("center", "ศูนย์ข้อมูล"),
    "CLOUD": ("cloud", "คลาวด์", "เมฆ"),
    "COMPUTING": ("computing", "ประมวลผล"),
    "DATA": ("data", "ข้อมูล"),
    "DEFINED": ("defined", "กำหนด"),
    "IDENTITY": ("identity", "ข้อมูลประจำตัว", "ตัวตน"),
    "INFRASTRUCTURE": ("infrastructure", "โครงสร้างพื้นฐาน"),
    "MANAGEMENT": ("management", "การจัดการ", "บริหาร"),
    "NETWORKING": ("networking", "เครือข่าย"),
    "PRIVATE": ("private", "ส่วนตัว"),
    "PUBLIC": ("public", "สาธารณะ"),
    "RESOURCE": ("resource", "ทรัพยากร"),
    "RESOURCES": ("resources", "ทรัพยากร"),
    "SECURITY": ("security", "ความปลอดภัย", "ความมั่นคงปลอดภัย"),
    "SERVICES": ("services", "บริการ"),
    "SERVICE": ("service", "บริการ"),
    "SOFTWARE": ("software", "ซอฟต์แวร์"),
    "STORAGE": ("storage", "จัดเก็บ", "การจัดเก็บ"),
    "SYSTEMS": ("systems", "ระบบ"),
    "SYSTEM": ("system", "ระบบ"),
    "VIRTUALIZATION": ("virtualization", "เสมือน", "เวอร์ชวล"),
}


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


def _enumerated_topic_items(semantic_chunks: Any) -> list[str]:
    items: list[str] = []
    for chunk in _semantic_chunks(semantic_chunks):
        text = str(chunk.get("text", ""))
        for marker in _TOPIC_LIST_MARKERS:
            marker_index = text.casefold().find(marker.casefold())
            if marker_index < 0:
                continue
            topic_text = text[marker_index + len(marker) :]
            for item in _TOPIC_LIST_SPLIT.split(topic_text):
                candidate = re.sub(r"\s+", " ", item).strip(" .:")
                if _TOPIC_ITEM.fullmatch(candidate) and candidate not in items:
                    items.append(candidate)
    return items


def _topic_item_is_covered(topic: str, answer: Any) -> bool:
    answer_text = str(answer).casefold()
    tokens = [
        token
        for token in re.findall(r"[A-Z]+", topic.upper())
        if token not in _TOPIC_STOPWORDS
    ]
    return bool(tokens) and all(
        any(alias.casefold() in answer_text for alias in _TOPIC_TOKEN_ALIASES.get(token, (token,)))
        for token in tokens
    )


def _semantic_concepts_are_incomplete(
    route: str,
    semantic_chunks: Any,
    answer: Any,
) -> bool:
    if route not in {"semantic", "hybrid"}:
        return False
    topics = _enumerated_topic_items(semantic_chunks)
    if len(topics) < 2:
        return False
    covered = sum(_topic_item_is_covered(topic, answer) for topic in topics)
    return covered < 2


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
    requested_plans: Sequence[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    plans = list(requested_plans) if requested_plans is not None else _requested_plan_keys(question)
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
    comparison_plans: Sequence[str] = ()
    if (
        normalized_route == "structured"
        and isinstance(structured_result, Mapping)
        and structured_result.get("operation") == "course_placement_comparison"
    ):
        upstream_plans = structured_result.get("requested_plan_keys")
        if isinstance(upstream_plans, Sequence) and not isinstance(
            upstream_plans, (str, bytes)
        ):
            comparison_plans = tuple(
                plan for plan in upstream_plans if isinstance(plan, str)
            )
    missing_plans, placement_records = _missing_hybrid_plan_evidence(
        question,
        structured_result,
        answer_text,
        comparison_plans if comparison_plans else None,
    ) if normalized_route == "hybrid" or comparison_plans else ([], [])
    incomplete_topics = _semantic_concepts_are_incomplete(
        normalized_route,
        semantic_chunks,
        answer_text,
    )
    if is_fallback_like(answer_text) or missing_plans or incomplete_topics:
        recovery_prompt = (
            prompt
            + "\nตรวจสอบหลักฐานอีกครั้งอย่างเคร่งครัด: "
            "ถ้าหลักฐานรองรับคำถาม ให้สรุปเฉพาะข้อเท็จจริงที่มีหลักฐานรองรับ; "
            "ถ้าหลักฐานไม่พอ ให้ตอบข้อความ fallback เดิม และห้ามเดาหรือสร้างข้อมูล"
        )
        if incomplete_topics:
            recovery_prompt += (
                "\nหากหลักฐานแจกแจงหัวข้อไว้ ให้สรุปหัวข้อที่รองรับคำถามอย่างครบถ้วน "
                "โดยใช้คำไทยหรือคำเทียบเท่าที่มีความหมายเดียวกันได้"
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
            if normalized_route == "hybrid" or comparison_plans:
                missing_plans, placement_records = _missing_hybrid_plan_evidence(
                    question,
                    structured_result,
                    answer_text,
                    comparison_plans if comparison_plans else None,
                )
                if missing_plans and placement_records:
                    return answer_text + "\n" + _hybrid_placement_appendix(placement_records)
            if normalized_route != "hybrid":
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


def _plain_typed_value(value: Any) -> Any:
    """Convert typed immutable evidence to deterministic renderer data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain_typed_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain_typed_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_typed_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_plain_typed_value(item) for item in value]
        return sorted(values, key=repr)
    return value


def _description_text(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    text = value.get("text", value.get("description"))
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def _description_texts(value: Any) -> tuple[str, ...]:
    """Read description text only from a grounded summary's typed evidence."""
    if isinstance(value, SimilarityEvidence):
        texts: list[str] = []
        for pair in value.pairs:
            for side in (pair.left, pair.right):
                text = _description_text(side)
                if text is not None:
                    texts.append(text)
        return tuple(texts)
    if isinstance(value, Mapping):
        direct = _description_text(value)
        if direct is not None:
            return (direct,)
        texts: list[str] = []
        for key in ("description_evidence", "options", "scored_candidates", "candidates"):
            nested = value.get(key)
            if nested is None or isinstance(nested, (str, bytes, Mapping)):
                nested = (nested,) if isinstance(nested, Mapping) else ()
            try:
                values = tuple(nested)
            except TypeError:
                values = ()
            for item in values:
                texts.extend(_description_texts(item))
        return tuple(texts)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        texts: list[str] = []
        for item in value:
            texts.extend(_description_texts(item))
        return tuple(texts)
    options = getattr(value, "options", None)
    if options is not None:
        texts: list[str] = []
        for option in options:
            texts.extend(_description_texts(option))
        return tuple(texts)
    descriptions = getattr(value, "description_evidence", None)
    if descriptions is not None:
        texts = []
        for description in descriptions:
            text = _description_text(description)
            if text is not None:
                texts.append(text)
        return tuple(texts)
    return ()


def _preference_synthesis_options(value: Any) -> tuple[Mapping[str, Any], ...]:
    options = getattr(value, "options", None)
    if options is None and isinstance(value, Mapping):
        options = value.get("options", value.get("scored_candidates", ()))
    if options is None or isinstance(options, (str, bytes, Mapping)):
        return ()
    result: list[Mapping[str, Any]] = []
    try:
        values = tuple(options)
    except TypeError:
        return ()
    for option in values:
        if not isinstance(option, Mapping):
            continue
        descriptions = tuple(
            {"text": text}
            for text in _description_texts(option)
        )
        result.append(
            {
                key: option[key]
                for key in ("program", "course_code", "course_name", "name_th", "name_en")
                if key in option
            }
            | {"descriptions": descriptions}
        )
    return tuple(result)


def _summary_synthesis_payload(claim: GroundedClaim) -> tuple[Any, ...]:
    if claim.operation == "preference":
        return _preference_synthesis_options(claim.evidence)
    return tuple({"description": text} for text in _description_texts(claim.evidence))


_SCOPE_DIMENSIONS = (("plan", "plans"), ("year", "years"), ("semester", "semesters"))


def _scope_dimension_values(scope: Any, field: str) -> tuple[Any, ...]:
    if scope is None:
        return ()
    if isinstance(scope, Mapping):
        value = scope.get(field)
        if value is None:
            value = scope.get(field[:-1])
    else:
        value = getattr(scope, field, None)
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _scope_value_text(values: tuple[Any, ...]) -> str:
    return ",".join(str(value) for value in values)


def _scope_diff_dimensions(claims: Sequence[GroundedClaim]) -> tuple[str, ...]:
    dimensions: list[str] = []
    for dimension, field in _SCOPE_DIMENSIONS:
        values = [_scope_dimension_values(claim.effective_scope, field) for claim in claims]
        if any(values) and len({repr(value) for value in values}) > 1:
            dimensions.append(dimension)
    return tuple(dimensions)


def _scope_prefix(
    claim: GroundedClaim,
    dimensions: Sequence[str],
) -> str:
    labels: list[str] = []
    for dimension, field in _SCOPE_DIMENSIONS:
        if dimension not in dimensions:
            continue
        values = _scope_dimension_values(claim.effective_scope, field)
        if values:
            labels.append(f"{dimension}={_scope_value_text(values)}")
    return f"{', '.join(labels)} | " if labels else ""


def _human_scope_prefix(
    claim: GroundedClaim,
    dimensions: Sequence[str],
) -> str:
    """Render grounded scope dimensions as Thai without raw field names."""
    scope = claim.effective_scope
    parts: list[str] = []
    program = (
        scope.get("program") if isinstance(scope, Mapping) else getattr(scope, "program", None)
    )
    if program not in (None, ""):
        parts.append(f"หลักสูตร {program}")
    for dimension, field in _SCOPE_DIMENSIONS:
        if dimension not in dimensions:
            continue
        values = tuple(
            str(value)
            for value in _scope_dimension_values(scope, field)
            if value not in (None, "")
        )
        if not values:
            continue
        if dimension == "plan":
            parts.append("แผน" + "/".join(_plan_display(value) for value in values))
        elif dimension == "year":
            parts.append("ปี " + "/".join(values))
        elif dimension == "semester":
            parts.append("ภาคเรียนที่ " + "/".join(values))
    return (" ".join(parts) + ": ") if parts else ""


def _has_scope_dimensions(scope: Any) -> bool:
    return any(
        _scope_dimension_values(scope, field)
        for _, field in _SCOPE_DIMENSIONS
    )


def _placement_choices(value: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    choices = value.get("year_semester_choices")
    if choices is not None and not isinstance(choices, (str, bytes, Mapping)):
        result: list[tuple[int, int]] = []
        try:
            for choice in choices:
                if (
                    isinstance(choice, Sequence)
                    and not isinstance(choice, (str, bytes))
                    and len(choice) == 2
                    and all(isinstance(item, int) and not isinstance(item, bool) for item in choice)
                ):
                    result.append((choice[0], choice[1]))
        except TypeError:
            return ()
        if result:
            return tuple(result)
    year = value.get("year_number", value.get("year"))
    semester = value.get("semester_number", value.get("semester"))
    if (
        isinstance(year, int)
        and not isinstance(year, bool)
        and isinstance(semester, int)
        and not isinstance(semester, bool)
    ):
        return ((year, semester),)
    return ()


def _placement_entries(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        if _placement_choices(value):
            return (value,)
        return ()
    if isinstance(value, EarliestAggregation):
        entries: list[Mapping[str, Any]] = []
        for partition in value.partitions:
            placements = partition.placements
            if placements:
                entries.extend(placements)
            else:
                partition_data = dict(_plain_typed_value(partition.partition))
                partition_data["year_semester_choices"] = (partition.value,)
                entries.append(partition_data)
        return tuple(entries)
    if isinstance(value, ComparisonAggregation):
        return _placement_entries(value.left) + _placement_entries(value.right)
    if isinstance(value, (list, tuple)):
        entries: list[Mapping[str, Any]] = []
        for item in value:
            entries.extend(_placement_entries(item))
        return tuple(entries)
    return ()


def _placement_entry_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identify one placement entry by course, plan, and timing choices."""
    return (
        entry.get("course_code"),
        entry.get("plan_key", entry.get("plan")),
        tuple(_placement_choices(entry)),
    )


def _plan_display(plan: Any) -> str:
    if plan == "coop":
        return "สหกิจ"
    if plan == "no_coop":
        return "ไม่สหกิจ"
    return str(plan) if plan not in (None, "") else "ไม่ระบุแผน"


def _placement_sentence(entry: Mapping[str, Any]) -> str | None:
    choices = _placement_choices(entry)
    if not choices:
        return None
    program = entry.get("program")
    plan = entry.get("plan_key", entry.get("plan"))
    course_code = entry.get("course_code")
    prefix = f"วิชา {course_code}" if course_code not in (None, "") else "วิชานี้"
    if program not in (None, ""):
        prefix += f"ในหลักสูตร {program}"
    if plan not in (None, ""):
        prefix += f" แผน{_plan_display(plan)}"
    if len(choices) == 1:
        timing = f"เรียนในปี {choices[0][0]} ภาคเรียนที่ {choices[0][1]}"
    else:
        options = " หรือ ".join(
            f"ปี {year} ภาคเรียนที่ {semester}" for year, semester in choices
        )
        timing = f"สามารถเรียนได้ใน{options}"
    credits = entry.get("placement_credits", entry.get("credits"))
    suffix = f" และมี {credits} หน่วยกิต" if credits not in (None, "") else ""
    return f"{prefix}{timing}{suffix}"


def _placement_text(value: Any) -> str | None:
    entries = _placement_entries(value)
    if not entries:
        return None
    if len(entries) == 2:
        left, right = entries
        left_choices = _placement_choices(left)
        right_choices = _placement_choices(right)
        left_plan = left.get("plan_key", left.get("plan"))
        right_plan = right.get("plan_key", right.get("plan"))
        if left_plan and right_plan and left_plan != right_plan and left_choices == right_choices:
            timing = " หรือ ".join(
                f"ปี {year} ภาคเรียนที่ {semester}"
                for year, semester in left_choices
            )
            timing = f"{timing}" if len(left_choices) > 1 else timing
            return (
                f"ทั้งแผน{_plan_display(left_plan)}และแผน{_plan_display(right_plan)}"
                f"เรียนใน{timing} จึงไม่ต่างกันด้านช่วงเรียน"
            )
        if left_plan == right_plan and left_choices == right_choices:
            left_code = left.get("course_code")
            right_code = right.get("course_code")
            if left_code not in (None, "") and right_code not in (None, ""):
                timing = " หรือ ".join(
                    f"ปี {year} ภาคเรียนที่ {semester}"
                    for year, semester in left_choices
                )
                program = left.get("program")
                program_text = f"ในหลักสูตร {program}" if program not in (None, "") else ""
                plan_text = f" แผน{_plan_display(left_plan)}" if left_plan else ""
                return (
                    f"วิชา {left_code} และวิชา {right_code} {program_text}{plan_text}"
                    f"เรียนใน{timing} จึงอยู่ช่วงเดียวกัน"
                )
    sentences = [_placement_sentence(entry) for entry in entries]
    rendered = [sentence for sentence in sentences if sentence]
    return "\n".join(rendered) if rendered else None


def _earliest_operand_scope_prefix(claim: GroundedClaim) -> str:
    if _has_scope_dimensions(claim.effective_scope):
        return ""
    if not isinstance(claim.value, ComparisonAggregation):
        return ""
    labels: list[str] = []
    for name, operand in (("left", claim.value.left), ("right", claim.value.right)):
        if not isinstance(operand, EarliestAggregation) or len(operand.partitions) != 1:
            return ""
        partition = operand.partitions[0].partition
        fields_text: list[str] = []
        for dimension, field in _SCOPE_DIMENSIONS:
            values = _scope_dimension_values(partition, field)
            if values:
                fields_text.append(f"{dimension}={_scope_value_text(values)}")
        if not fields_text:
            return ""
        labels.append(f"{name}[{', '.join(fields_text)}]")
    return f"{' '.join(labels)} | "


def _prerequisite_course_text(entry: Mapping[str, Any]) -> str | None:
    code = next(
        (
            entry.get(key)
            for key in ("prerequisite_code", "prerequisite_course_code", "course_code", "code")
            if entry.get(key) not in (None, "")
        ),
        None,
    )
    name = next(
        (
            entry.get(key)
            for key in (
                "prerequisite_name_en",
                "prerequisite_name_th",
                "course_name",
                "name_en",
                "name_th",
                "name",
            )
            if entry.get(key) not in (None, "")
        ),
        None,
    )
    parts = [str(value).strip() for value in (code, name) if value not in (None, "")]
    return " ".join(parts) if parts else None


def _prerequisite_entries(value: Any) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    """Collect only user-facing prerequisite entries, excluding metadata."""
    groups: list[tuple[str, tuple[Mapping[str, Any], ...]]] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            alternatives = item.get("alternative_courses")
            if isinstance(alternatives, Mapping):
                alternatives = (alternatives,)
            if isinstance(alternatives, (list, tuple)):
                alternative_entries = tuple(
                    candidate for candidate in alternatives if isinstance(candidate, Mapping)
                )
                if alternative_entries:
                    groups.append(("alternative", alternative_entries))

            if any(
                item.get(key) not in (None, "")
                for key in ("prerequisite_code", "prerequisite_course_code", "course_code", "code")
            ):
                groups.append(("required", (item,)))

            for key in ("prerequisites", "required_prerequisites", "entries", "items"):
                nested = item.get(key)
                if isinstance(nested, Mapping) or isinstance(nested, (list, tuple)):
                    visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)

    unique: list[tuple[str, tuple[Mapping[str, Any], ...]]] = []
    seen: set[tuple[Any, ...]] = set()
    for kind, entries in groups:
        readable_entries = tuple(
            entry for entry in entries if _prerequisite_course_text(entry) is not None
        )
        if not readable_entries:
            continue
        key = (
            kind,
            tuple(_prerequisite_course_text(entry) for entry in readable_entries),
        )
        if key not in seen:
            seen.add(key)
            unique.append((kind, readable_entries))
    return tuple(unique)


def _prerequisite_text(value: Any, *, valid_empty: bool = False) -> str | None:
    groups = _prerequisite_entries(value)
    if not groups:
        return "ไม่มีวิชาบังคับก่อน" if valid_empty else None
    rendered: list[str] = []
    for kind, entries in groups:
        courses = tuple(_prerequisite_course_text(entry) for entry in entries)
        courses = tuple(course for course in courses if course)
        if kind == "alternative":
            rendered.append("ต้องผ่านอย่างน้อยหนึ่งวิชาจาก: " + " หรือ ".join(courses))
        else:
            rendered.extend(f"ต้องเรียนวิชา {course} มาก่อน" for course in courses)
    return "\n".join(rendered) if rendered else ("ไม่มีวิชาบังคับก่อน" if valid_empty else None)


def _identity_entry_text(entry: Mapping[str, Any]) -> str | None:
    """Render one grounded course identity without internal fields."""
    code = next(
        (
            entry.get(key)
            for key in ("course_code", "code")
            if isinstance(entry.get(key), str) and entry.get(key).strip()
        ),
        None,
    )
    names: dict[str, str] = {}
    for label, key in (("ชื่อภาษาไทย", "name_th"), ("ชื่อภาษาอังกฤษ", "name_en")):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            names[label] = value.strip()
    if code is None and not names:
        return None
    if len(names) > 1:
        detail = "; ".join(f"{label}: {name}" for label, name in names.items())
        return f"{code} {detail}" if code is not None else detail
    if names:
        ((_, name),) = names.items()
        return f"{code} — {name}" if code is not None else name
    return str(code)


def _identity_claim_text(value: Any) -> str | None:
    """Render every distinct grounded course identity in a claim value."""
    if isinstance(value, Mapping):
        entries: tuple[Any, ...] = (value,)
    elif isinstance(value, (list, tuple)):
        entries = tuple(value)
    else:
        return None
    rendered: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        text = _identity_entry_text(entry)
        if text is None or text in seen:
            continue
        seen.add(text)
        rendered.append(text)
    return "\n".join(rendered) if rendered else None


def _program_discovery_text(value: Any) -> str | None:
    if isinstance(value, Mapping):
        entries: tuple[Any, ...] = (value,)
    elif isinstance(value, (list, tuple)):
        entries = tuple(value)
    else:
        return None

    rendered: list[str] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        program = entry.get("program")
        course_code = entry.get("course_code")
        if not isinstance(program, str) or not program.strip():
            continue
        if not isinstance(course_code, str) or not course_code.strip():
            continue
        identity = (program.strip(), course_code.strip())
        if identity in seen:
            continue
        seen.add(identity)
        name = _identity_entry_text(entry)
        if name is None:
            name = course_code.strip()
        rendered.append(f"- {program.strip()} — {name}")
    return "\n".join(rendered) if rendered else None


def _credit_claim_targets(scope: Any) -> tuple[Mapping[str, Any], ...]:
    """Return concrete course targets on a credit scope, if any."""
    targets = (
        scope.get("course_targets")
        if isinstance(scope, Mapping)
        else getattr(scope, "course_targets", ())
    )
    if not isinstance(targets, (list, tuple)):
        return ()
    return tuple(target for target in targets if isinstance(target, Mapping))


def _credit_scope_text(scope: Any, *, prefix: str) -> str:
    """Render only grounded program/plan/year/semester scope wording."""
    parts = [prefix] if prefix else []
    program = (
        scope.get("program") if isinstance(scope, Mapping) else getattr(scope, "program", None)
    )
    if program not in (None, ""):
        parts.append(f"หลักสูตร {program}" if not prefix else f"ในหลักสูตร {program}")
    plans = [str(plan) for plan in _scope_dimension_values(scope, "plans")]
    if plans:
        parts.append("แผน" + "/".join(_plan_display(plan) for plan in plans))
    years = [str(year) for year in _scope_dimension_values(scope, "years")]
    if years:
        parts.append("ปี " + "/".join(years))
    semesters = [str(semester) for semester in _scope_dimension_values(scope, "semesters")]
    if semesters:
        parts.append("ภาคเรียนที่ " + "/".join(semesters))
    return " ".join(parts)


def _sum_credits_text(claim: GroundedClaim) -> str | None:
    """Render course credits or semester totals as natural Thai."""
    value = claim.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    total: Any = int(value) if float(value).is_integer() else value
    scope = claim.effective_scope
    targets = _credit_claim_targets(scope)
    if targets:
        code = next(
            (
                target.get(key)
                for target in targets
                for key in ("course_code", "code")
                if isinstance(target.get(key), str) and target.get(key).strip()
            ),
            None,
        )
        if code is None:
            return None
        context = _credit_scope_text(scope, prefix=f"วิชา {code}")
        return f"{context}: {total} หน่วยกิต" if context else f"{total} หน่วยกิต"
    context = _credit_scope_text(scope, prefix="")
    if context:
        return f"{context} ลงทะเบียนรวม {total} หน่วยกิต"
    return f"ลงทะเบียนรวม {total} หน่วยกิต"


def _course_list_course_text(entry: Mapping[str, Any]) -> str | None:
    """Render one grounded course as a user-facing line without internals."""
    code = entry.get("course_code")
    code_text = str(code).strip() if isinstance(code, str) and code.strip() else None
    name_th = entry.get("name_th")
    name_th = name_th.strip() if isinstance(name_th, str) and name_th.strip() else None
    name_en = entry.get("name_en")
    name_en = name_en.strip() if isinstance(name_en, str) and name_en.strip() else None
    if code_text is None and name_th is None and name_en is None:
        return None
    line = code_text or ""
    if name_th is not None:
        line = f"{line} {name_th}".strip()
    if name_en is not None:
        line = f"{line} ({name_en})".strip()
    credits = next(
        (
            str(entry.get(key)).strip()
            for key in ("placement_credits", "credits", "credits_raw")
            if (
                isinstance(entry.get(key), (str, int))
                and not isinstance(entry.get(key), bool)
                and str(entry.get(key)).strip()
            )
        ),
        None,
    )
    if credits is not None:
        line = f"{line} — {credits} หน่วยกิต".strip()
    return line or None


def _course_list_alternative_text(entry: Mapping[str, Any]) -> str | None:
    """Render an alternative-course group using only grounded choice data."""
    alternatives = entry.get("alternative_courses")
    if isinstance(alternatives, Mapping):
        alternatives = (alternatives,)
    if not isinstance(alternatives, (list, tuple)):
        return None
    options: list[str] = []
    for member in alternatives:
        if not isinstance(member, Mapping):
            continue
        text = _course_list_course_text(member)
        if text is not None:
            options.append(text)
    if not options:
        return None
    minimum = entry.get("minimum_choices")
    maximum = entry.get("maximum_choices")
    if (
        isinstance(minimum, int)
        and not isinstance(minimum, bool)
        and minimum == maximum
    ):
        return f"เลือก {minimum} วิชาจาก: " + " หรือ ".join(options)
    return "วิชาทางเลือก: " + " หรือ ".join(options)


def _course_list_entry_text(entry: Any) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    alternatives = entry.get("alternative_courses")
    if isinstance(alternatives, (Mapping, list, tuple)):
        alternative_text = _course_list_alternative_text(entry)
        if alternative_text is not None:
            return alternative_text
    return _course_list_course_text(entry)


def _course_list_text(claim: GroundedClaim) -> str | None:
    """Render semester/topic course lists as readable lines without internals."""
    if claim.status == "valid_empty":
        header = _credit_scope_text(claim.effective_scope, prefix="")
        message = "ไม่พบรายวิชาตามเงื่อนไขที่ถาม"
        return f"{header}:\n{message}" if header else message
    value = claim.value
    if isinstance(value, Mapping):
        entries: tuple[Any, ...] = (value,)
    elif isinstance(value, (list, tuple)):
        entries = tuple(value)
    else:
        return None
    lines: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        text = _course_list_entry_text(entry)
        if text is None or text in seen:
            continue
        seen.add(text)
        lines.append(f"- {text}")
    if not lines:
        return None
    header = _credit_scope_text(claim.effective_scope, prefix="")
    body = "\n".join(lines)
    return f"{header}:\n{body}" if header else body


def _similarity_numeric_payload(value: SimilarityEvidence) -> Mapping[str, Any]:
    """Expose persisted similarity numbers without comparing or recalculating."""
    return {
        "status": value.status,
        "pairs": tuple(
            {
                "status": pair.status,
                "partition": pair.partition,
                "cosine_distance": pair.cosine_distance,
                "cosine_similarity": pair.cosine_similarity,
                "reason": pair.reason,
            }
            for pair in value.pairs
        ),
        "mean_distance": value.mean_distance,
        "min_distance": value.min_distance,
        "max_distance": value.max_distance,
    }


def _similarity_description_text(value: SimilarityEvidence) -> str | None:
    """Render complete pair-side descriptions without adding synthesis."""
    if value.status != "complete" or not value.pairs:
        return None
    sections: list[str] = []
    for pair in value.pairs:
        for side in (pair.left, pair.right):
            text = _description_text(side)
            if text is None:
                return None
            program = side.get("program")
            course_code = side.get("course_code")
            partition = side.get("partition")
            plan = partition.get("plan") if isinstance(partition, Mapping) else None
            label = " ".join(
                str(part)
                for part in (program, course_code)
                if isinstance(part, str) and part.strip()
            ) or "วิชา"
            if isinstance(plan, str) and plan.strip():
                label += f" ({plan})"
            sections.append(f"{label}:\n{text}")
    return "\n".join(sections)


def _plan_comparison_course_label(
    course: Mapping[str, Any] | None,
    course_key: tuple[Any, ...] | None = None,
) -> str:
    code = None
    name = None
    if isinstance(course, Mapping):
        code = course.get("course_code")
        name = next(
            (
                course.get(field)
                for field in ("name_en", "name_th", "course_name", "name")
                if course.get(field) not in (None, "")
            ),
            None,
        )
    if code in (None, "") and isinstance(course_key, tuple) and len(course_key) == 2:
        identity = course_key[1]
        if (
            course_key[0] == "course"
            and isinstance(identity, tuple)
            and len(identity) == 2
        ):
            code = identity[1]
    parts = [str(value) for value in (code, name) if value not in (None, "")]
    return " ".join(parts) if parts else "วิชาที่ไม่ระบุรหัส"


def _plan_comparison_periods_text(periods: Sequence[tuple[int, int]]) -> str:
    return ", ".join(
        f"ปี {year} ภาคเรียนที่ {semester}" for year, semester in periods
    )


def _plan_comparison_text(value: PlanComparisonAggregation) -> str | None:
    if value.status != "complete":
        return None
    lines: list[str] = []
    left_plan = _plan_display(value.left_plan)
    right_plan = _plan_display(value.right_plan)
    if value.only_left:
        lines.append(f"วิชาที่มีเฉพาะแผน{left_plan}:")
        lines.extend(
            f"- {_plan_comparison_course_label(course)}"
            for course in value.only_left
        )
    if value.only_right:
        lines.append(f"วิชาที่มีเฉพาะแผน{right_plan}:")
        lines.extend(
            f"- {_plan_comparison_course_label(course)}"
            for course in value.only_right
        )
    if value.placement_differences:
        lines.append("วิชาที่อยู่ต่างช่วงปี/ภาคเรียน:")
        for difference in value.placement_differences:
            records = difference.left_placements + difference.right_placements
            course = records[0] if records else None
            label = _plan_comparison_course_label(course, difference.course_key)
            lines.append(
                f"- {label}: แผน{left_plan}เรียนใน"
                f"{_plan_comparison_periods_text(difference.left_periods)}; "
                f"แผน{right_plan}เรียนใน"
                f"{_plan_comparison_periods_text(difference.right_periods)}"
            )
    if not lines:
        return (
            "จากข้อมูลที่มี ไม่พบความแตกต่างของรายวิชาและช่วงปี/ภาคการเรียน"
            "ระหว่างสองแผน"
        )
    return "\n".join(
        [f"เปรียบเทียบแผน{left_plan}กับแผน{right_plan}:", *lines]
    )


def _deterministic_claim_text(
    claim: GroundedClaim,
    *,
    scope_dimensions: Sequence[str] = (),
) -> str:
    def finish(text: str) -> str:
        return _scope_prefix(claim, scope_dimensions) + text

    if claim.status == "insufficient_evidence":
        return finish("หลักฐานไม่เพียงพอ")
    if claim.status not in {"complete", "valid_empty", "descriptive_only"}:
        return ""

    if claim.operation == "similarity" and isinstance(claim.value, SimilarityEvidence):
        description_text = _similarity_description_text(claim.value)
        if description_text is not None:
            return finish(description_text)
        value = _plain_typed_value(_similarity_numeric_payload(claim.value))
        return finish(
            "similarity: "
            + json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            )

    if claim.operation == "compare" and isinstance(
        claim.value, PlanComparisonAggregation
    ):
        comparison_text = _plan_comparison_text(claim.value)
        if comparison_text is not None:
            return finish(comparison_text)

    if claim.operation in {"placement", "earliest", "compare"}:
        placement_text = _placement_text(claim.value)
        if placement_text is not None:
            return _human_scope_prefix(claim, scope_dimensions) + placement_text

    if claim.operation == "prerequisite":
        prerequisite_text = _prerequisite_text(
            claim.evidence if claim.evidence is not None else claim.value,
            valid_empty=claim.status == "valid_empty",
        )
        human_prefix = _human_scope_prefix(claim, scope_dimensions)
        if prerequisite_text is not None:
            return human_prefix + prerequisite_text
        return human_prefix + "หลักฐานไม่เพียงพอ"

    if claim.operation == "identity":
        identity_text = _identity_claim_text(claim.value) or _identity_claim_text(
            claim.evidence
        )
        if identity_text is not None:
            return finish(identity_text)
        return finish("หลักฐานไม่เพียงพอ")

    if claim.operation == "program_discovery":
        discovery_text = _program_discovery_text(claim.value)
        if discovery_text is not None:
            return finish("พบวิชาในหลักสูตร:\n" + discovery_text)
        return finish("หลักฐานไม่เพียงพอ")

    if claim.operation == "sum_credits":
        credits_text = _sum_credits_text(claim)
        if credits_text is not None:
            return credits_text
        return finish("หลักฐานไม่เพียงพอ")

    if claim.operation == "list":
        list_text = _course_list_text(claim)
        if list_text is not None:
            return list_text
        return _human_scope_prefix(claim, scope_dimensions) + "หลักฐานไม่เพียงพอ"

    if claim.operation in {"describe", "topic_matches", "description_evidence"}:
        texts = _description_texts(claim.evidence)
        if texts:
            return _human_scope_prefix(claim, scope_dimensions) + "\n".join(texts)
    if claim.operation == "preference":
        options = _preference_synthesis_options(claim.evidence)
        if options:
            return finish(
                "\n".join(
                    json.dumps(_plain_typed_value(option), ensure_ascii=False, sort_keys=True)
                    for option in options
                )
            )

    value = _plain_typed_value(claim.value)
    if value is None:
        return finish(f"{claim.operation}: {claim.status}")
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return finish(
        _earliest_operand_scope_prefix(claim) + f"{claim.operation}: {serialized}"
    )


_POLISH_INSTRUCTION = """You are the final-response writer for a university curriculum QA system.

Your ONLY task is to rewrite GROUNDED_CONTENT into natural, fluent Thai.

STRICT RULES:

- Use only facts present in GROUNDED_CONTENT.
- Do not add, infer, guess, correct, or expand factual content from your own knowledge.
- Preserve course codes, program names, plan names, year/semester values, credits, prerequisite relationships, and numerical values exactly.
- Do not omit facts that directly answer USER_QUESTION.
- Combine duplicate statements naturally.
- Avoid JSON/database/internal terminology.
- Answer directly in natural Thai.
- Keep technical/course-content meaning equivalent to the supplied content.
- If GROUNDED_CONTENT says information is missing or ambiguous, preserve that limitation.
- Return only the final user-facing answer."""


def _critical_facts(text: str) -> tuple[str, ...]:
    facts: list[str] = []
    for match in re.finditer(r"(?<!\d)\d{8}(?!\d)", text):
        facts.append(f"code:{match.group(0)}")
    for match in re.finditer(r"\b(\d+)\s*(\([^\n)]*\))", text):
        facts.append(f"credit:{match.group(1)}{match.group(2)}")

    for match in re.finditer(r"ปี(?:ที่)?\s*(\d+)", text):
        facts.append(f"year:{match.group(1)}")

    semester_pattern = (
        r"(?:ภาคเรียนที่|ภาคเรียน|ภาคการศึกษาที่|ภาคการศึกษา|เทอม(?:ที่)?)\s*(\d+)"
        r"|\bsemester\s+(\d+)"
    )
    for match in re.finditer(semester_pattern, text, re.IGNORECASE):
        facts.append(f"semester:{match.group(1) or match.group(2)}")

    if re.search(r"\bno_coop\b|ไม่\s*สหกิจ", text):
        facts.append("plan:no_coop")
    elif re.search(r"\bcoop\b|(?<!ไม่)สหกิจ", text):
        facts.append("plan:coop")

    for match in re.finditer(
        r"(?<![A-Za-z0-9_])(AIT|BIT|DSBA|GENED|IT)(?![A-Za-z0-9_])", text
    ):
        facts.append(f"program:{match.group(1)}")
    return tuple(facts)


def _polish_deterministic_answer(
    question: str | None,
    deterministic_answer: str,
    answer_model_callable: Callable[[str], str] | None,
) -> str:
    if not deterministic_answer or not isinstance(question, str) or not question.strip():
        return deterministic_answer
    if not callable(answer_model_callable):
        return deterministic_answer
    prompt = "\n".join(
        (
            _POLISH_INSTRUCTION,
            f"USER_QUESTION:\n{question}",
            f"GROUNDED_CONTENT:\n{deterministic_answer}",
        )
    )
    try:
        polished = answer_model_callable(prompt)
    except Exception:
        return deterministic_answer
    if not isinstance(polished, str) or not polished.strip():
        return deterministic_answer
    polished = polished.strip()
    required_facts = set(_critical_facts(deterministic_answer))
    available_facts = set(_critical_facts(polished))
    if not required_facts.issubset(available_facts):
        return deterministic_answer
    return polished


def render_grounded_claim(
    claim: GroundedClaim,
    *,
    scope_dimensions: Sequence[str] = (),
) -> str:
    """Render one typed claim without calling a model or recomputing facts."""
    if not isinstance(claim, GroundedClaim):
        return ""
    return _deterministic_claim_text(claim, scope_dimensions=scope_dimensions)


def synthesize_grounded_claim(
    claim: GroundedClaim,
    answer_model_callable: Callable[[str], str] | None = None,
    *,
    scope_dimensions: Sequence[str] = (),
) -> str:
    """Bound synthesis for one claim, falling back to deterministic evidence text."""
    fallback = render_grounded_claim(claim, scope_dimensions=scope_dimensions)
    if not isinstance(claim, GroundedClaim) or claim.kind != "grounded_summary":
        return fallback
    if claim.status == "insufficient_evidence" or not callable(answer_model_callable):
        return fallback

    payload = _summary_synthesis_payload(claim)
    if not payload:
        return fallback
    prompt = "\n".join(
        (
            "สรุปหลักฐานที่ให้เท่านั้น โดยไม่เพิ่มข้อเท็จจริง",
            "ห้ามจัดอันดับ แนะนำสิ่งที่ดีที่สุด หรือสร้างตัวเลข/ความสัมพันธ์ใหม่",
            "ใช้ข้อมูลเฉพาะของ claim นี้และคงความหมายของข้อความเดิม",
            "หลักฐาน:",
            json.dumps(
                _plain_typed_value(payload),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "สรุป:",
        )
    )
    try:
        generated = answer_model_callable(prompt)
    except Exception:
        return fallback
    if not isinstance(generated, str) or not generated.strip() or is_fallback_like(generated):
        return fallback
    generated_text = generated.strip()
    if claim.operation == "similarity":
        return fallback + "\n" + generated_text
    return _scope_prefix(claim, scope_dimensions) + generated_text


def _scope_program_text(scope: Any) -> str:
    program = (
        scope.get("program") if isinstance(scope, Mapping) else getattr(scope, "program", None)
    )
    return str(program).strip() if isinstance(program, str) and program.strip() else ""


def _scope_category_text(scope: Any) -> str:
    category = (
        scope.get("category") if isinstance(scope, Mapping) else getattr(scope, "category", None)
    )
    return str(category).strip() if isinstance(category, str) and category.strip() else ""


def _scope_topic_text(scope: Any) -> str:
    for field in ("topic", "topics"):
        if isinstance(scope, Mapping):
            value = scope.get(field)
        else:
            value = getattr(scope, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            texts = [str(item).strip() for item in value if str(item).strip()]
            if texts:
                return "|".join(texts)
    return ""


def _scope_course_targets_text(scope: Any) -> str:
    if isinstance(scope, Mapping):
        targets = scope.get("course_targets")
    else:
        targets = getattr(scope, "course_targets", None)
    if targets is None:
        return ""
    try:
        return json.dumps(
            _plain_typed_value(targets), ensure_ascii=False, sort_keys=True, default=str
        )
    except (TypeError, ValueError):
        return repr(targets)


def _list_collapse_scope_key(claim: GroundedClaim) -> tuple[Any, ...] | None:
    """Identify one complete list scope ignoring the plan axis."""
    if claim.operation != "list" or claim.status != "complete":
        return None
    plans = tuple(
        str(plan) for plan in _scope_dimension_values(claim.effective_scope, "plans")
    )
    if sorted(plans) not in (["coop"], ["no_coop"]):
        return None
    years = tuple(
        str(value) for value in _scope_dimension_values(claim.effective_scope, "years")
    )
    semesters = tuple(
        str(value) for value in _scope_dimension_values(claim.effective_scope, "semesters")
    )
    return (
        _scope_program_text(claim.effective_scope),
        years,
        semesters,
        _scope_category_text(claim.effective_scope),
        _scope_topic_text(claim.effective_scope),
        _scope_course_targets_text(claim.effective_scope),
    )


def _course_list_body_lines(claim: GroundedClaim) -> tuple[str, ...] | None:
    """Return ordered deduped rendered entry lines without the scope header."""
    value = claim.value
    if isinstance(value, Mapping):
        entries: tuple[Any, ...] = (value,)
    elif isinstance(value, (list, tuple)):
        entries = tuple(value)
    else:
        return None
    lines: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        text = _course_list_entry_text(entry)
        if text is None or text in seen:
            continue
        seen.add(text)
        lines.append(text)
    return tuple(lines) if lines else None


_LIST_SEMANTIC_IGNORED_KEYS = frozenset(
    {
        "provenance",
        "provenance_id",
        "course_id",
        "catalog_id",
        "placement_id",
        "document_page",
        "source_page",
        "source_uri",
        "source_locator",
        "source_filename",
        "source_key",
    }
)


def _course_list_semantic_fingerprint(value: Any) -> tuple[Any, ...]:
    """Normalize grounded list semantics without using rendered text."""
    if isinstance(value, Mapping):
        items: list[tuple[str, tuple[Any, ...]]] = []
        for key in sorted(value, key=lambda item: str(item)):
            key_text = str(key)
            if key_text in _LIST_SEMANTIC_IGNORED_KEYS:
                continue
            normalized = value[key]
            if normalized is None:
                continue
            items.append((key_text, _course_list_semantic_fingerprint(normalized)))
        return ("mapping", tuple(items))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_course_list_semantic_fingerprint(item) for item in value))
    if isinstance(value, set):
        normalized = tuple(_course_list_semantic_fingerprint(item) for item in value)
        return ("set", tuple(sorted(normalized, key=repr)))
    return ("value", value)


def _course_list_body_semantic_fingerprint(claim: GroundedClaim) -> tuple[Any, ...] | None:
    value = claim.value
    if isinstance(value, Mapping):
        entries: tuple[Any, ...] = (value,)
    elif isinstance(value, (list, tuple)):
        entries = tuple(value)
    else:
        return None
    if not all(isinstance(entry, Mapping) for entry in entries):
        return None
    return tuple(_course_list_semantic_fingerprint(entry) for entry in entries)


def _collapse_identical_list_segments(
    claimed_segments: list[tuple[GroundedClaim, str]],
) -> list[tuple[GroundedClaim, str]]:
    """Merge one coop/no_coop pair sharing scope and identical list contents.

    Presentation-only: claims, evidence, and provenance are never modified.
    Only complete ``list`` claims with the same program/year/semester/
    category/topic scope and exactly identical ordered semantic entries collapse.
    Partial overlaps, differing credits/alternatives/ordering, single plans,
    and valid_empty claims keep their existing separate sections.
    """
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, (claim, _segment) in enumerate(claimed_segments):
        key = _list_collapse_scope_key(claim)
        if key is None:
            continue
        if _course_list_body_semantic_fingerprint(claim) is None:
            continue
        groups.setdefault(key, []).append(index)
    collapsed_indices: set[int] = set()
    replacements: dict[int, str] = {}
    for key, indices in groups.items():
        if len(indices) != 2:
            continue
        first_claim = claimed_segments[indices[0]][0]
        second_claim = claimed_segments[indices[1]][0]
        first_plans = tuple(
            str(plan) for plan in _scope_dimension_values(first_claim.effective_scope, "plans")
        )
        second_plans = tuple(
            str(plan) for plan in _scope_dimension_values(second_claim.effective_scope, "plans")
        )
        if {first_plans, second_plans} != {("coop",), ("no_coop",)}:
            continue
        first_fingerprint = _course_list_body_semantic_fingerprint(first_claim)
        second_fingerprint = _course_list_body_semantic_fingerprint(second_claim)
        if first_fingerprint is None or first_fingerprint != second_fingerprint:
            continue
        first_lines = _course_list_body_lines(first_claim)
        if first_lines is None:
            continue
        program, years, semesters = key[0], key[1], key[2]
        combined_scope: dict[str, Any] = {"program": program or None}
        if years:
            combined_scope["years"] = tuple(int(value) if value.isdigit() else value for value in years)
        if semesters:
            combined_scope["semesters"] = tuple(
                int(value) if value.isdigit() else value for value in semesters
            )
        combined_scope["plans"] = ("coop", "no_coop")
        header = _credit_scope_text(combined_scope, prefix="")
        body = "\n".join(f"- {line}" for line in first_lines)
        replacements[indices[0]] = f"{header}:\n{body}" if header else body
        collapsed_indices.add(indices[1])
    if not replacements:
        return claimed_segments
    return [
        (claim, replacements.get(index, segment))
        for index, (claim, segment) in enumerate(claimed_segments)
        if index not in collapsed_indices
    ]


def _suppress_covered_placement_segments(
    claimed_segments: list[tuple[GroundedClaim, str]],
) -> list[tuple[GroundedClaim, str]]:
    """Drop operand placement lines already conveyed by a comparison summary.

    A complete compare claim renders every entry it compares, either as a
    same-period summary or as per-entry sentences, so placement/earliest
    segments repeating exactly those entries add no fact.  Segments with
    entries outside every comparison summary are preserved, keeping distinct
    plan results visible.  Typed claims and provenance are never modified.
    """
    compared_keys: set[tuple[Any, ...]] = set()
    has_comparison_summary = False
    for claim, _segment in claimed_segments:
        if claim.operation != "compare" or claim.status != "complete":
            continue
        if _placement_text(claim.value) is None:
            continue
        has_comparison_summary = True
        for entry in _placement_entries(claim.value):
            if isinstance(entry, Mapping):
                compared_keys.add(_placement_entry_key(entry))
    if not has_comparison_summary:
        return claimed_segments
    kept: list[tuple[GroundedClaim, str]] = []
    for claim, segment in claimed_segments:
        if claim.operation in {"placement", "earliest"} and claim.status == "complete":
            entries = [
                entry
                for entry in _placement_entries(claim.value)
                if isinstance(entry, Mapping)
            ]
            if entries and all(
                _placement_entry_key(entry) in compared_keys for entry in entries
            ):
                continue
        kept.append((claim, segment))
    return kept


def render_grounded_answer(
    result: GroundedAnswerResult,
    answer_model_callable: Callable[[str], str] | None = None,
    *,
    question: str | None = None,
) -> GroundedAnswerResult:
    """Render deterministic evidence, then optionally polish its final text."""
    if not isinstance(result, GroundedAnswerResult):
        raise TypeError("result must be a GroundedAnswerResult")
    if not result.claims:
        return result
    scope_dimensions = _scope_diff_dimensions(result.claims)
    claimed_segments: list[tuple[GroundedClaim, str]] = []
    for claim in result.claims:
        segment = render_grounded_claim(claim, scope_dimensions=scope_dimensions)
        if segment:
            claimed_segments.append((claim, segment))
    claimed_segments = _suppress_covered_placement_segments(claimed_segments)
    claimed_segments = _collapse_identical_list_segments(claimed_segments)
    segments = [segment for _, segment in claimed_segments]
    final_answer = "\n".join(segments) if segments else result.final_answer
    final_answer = _polish_deterministic_answer(
        question, final_answer, answer_model_callable
    )
    return GroundedAnswerResult(
        status=result.status,
        answer_mode=result.answer_mode,
        final_answer=final_answer,
        claims=result.claims,
        provenance=result.provenance,
    )


__all__ = [
    "EMPTY_ANSWER",
    "answer_question",
    "build_grounded_prompt",
    "is_fallback_like",
    "render_grounded_claim",
    "synthesize_grounded_claim",
    "render_grounded_answer",
]
