"""Grounded final-answer generation for unified curriculum QA."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from rag.aggregation import ComparisonAggregation, EarliestAggregation
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


def _has_scope_dimensions(scope: Any) -> bool:
    return any(
        _scope_dimension_values(scope, field)
        for _, field in _SCOPE_DIMENSIONS
    )


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
        if pair.cosine_similarity is not None or pair.cosine_distance is not None:
            sections.append(
                "similarity: "
                + json.dumps(
                    {
                        "cosine_distance": pair.cosine_distance,
                        "cosine_similarity": pair.cosine_similarity,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            )
    return "\n".join(sections)


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

    if claim.operation in {"describe", "topic_matches", "description_evidence"}:
        texts = _description_texts(claim.evidence)
        if texts:
            return finish("\n".join(texts))
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


def render_grounded_answer(
    result: GroundedAnswerResult,
    answer_model_callable: Callable[[str], str] | None = None,
) -> GroundedAnswerResult:
    """Render a typed answer claim-by-claim without mutating its evidence."""
    if not isinstance(result, GroundedAnswerResult):
        raise TypeError("result must be a GroundedAnswerResult")
    if not result.claims:
        return result
    scope_dimensions = _scope_diff_dimensions(result.claims)
    segments: list[str] = []
    for claim in result.claims:
        if claim.kind == "grounded_summary":
            segment = synthesize_grounded_claim(
                claim,
                answer_model_callable,
                scope_dimensions=scope_dimensions,
            )
        else:
            segment = render_grounded_claim(
                claim,
                scope_dimensions=scope_dimensions,
            )
        if segment:
            segments.append(segment)
    final_answer = "\n".join(segments) if segments else result.final_answer
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
