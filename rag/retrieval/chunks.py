"""Build deterministic retrieval chunks from the structured SQLite database."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Mapping


_NO_PREREQUISITE = {
    "",
    "-",
    "n/a",
    "none",
    "no prerequisite",
    "null",
    "\u0e44\u0e21\u0e48\u0e21\u0e35",
}
_FLEXIBLE_TERM_RE = re.compile(r"^(\d+)\s*/\s*(\d+)$")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u200b", "").split()).strip()


def _is_valid(value: Any) -> bool:
    text = _clean_text(value).lower()
    return text not in _NO_PREREQUISITE


def _add_part(parts: list[str], value: Any) -> None:
    text = _clean_text(value)
    if text and _is_valid(text):
        parts.append(text)


def _flexible_term_text(value: Any) -> str:
    match = _FLEXIBLE_TERM_RE.fullmatch(_clean_text(value))
    if match is None:
        return ""
    return f"ช่วงที่สามารถลงได้: ปี {match.group(1)} ภาคเรียน {match.group(2)}"


def _provenance_map(
    connection: sqlite3.Connection, link_table: str, entity_column: str
) -> dict[int, list[dict[str, Any]]]:
    query = f"""
        SELECT
            links.{entity_column} AS entity_id,
            provenance.provenance_id,
            provenance.program,
            provenance.source_filename,
            provenance.source_page,
            provenance.document_page,
            provenance.document_category,
            provenance.source_uri,
            provenance.source_locator,
            provenance.excerpt
        FROM {link_table} AS links
        JOIN provenance
          ON provenance.provenance_id = links.provenance_id
        ORDER BY links.{entity_column}, provenance.provenance_id
    """
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute(query):
        reference = {
            key: row[key]
            for key in (
                "provenance_id",
                "program",
                "source_filename",
                "source_page",
                "document_page",
                "document_category",
                "source_uri",
                "source_locator",
                "excerpt",
            )
        }
        result[int(row["entity_id"])].append(reference)
    return result


def _merge_provenance(
    *reference_lists: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for references in reference_lists:
        for reference in references:
            provenance_id = int(reference["provenance_id"])
            if provenance_id in seen:
                continue
            seen.add(provenance_id)
            merged.append(dict(reference))
    return merged


def _course_metadata_text(
    course: Mapping[str, Any],
    placement: Mapping[str, Any] | None,
    prerequisites: Iterable[Mapping[str, Any]],
    alternative_group: Mapping[str, Any] | None = None,
    alternative_members: Iterable[Mapping[str, Any]] = (),
) -> str:
    parts: list[str] = []

    name_th = _clean_text(course.get("name_th"))
    name_en = _clean_text(course.get("name_en"))
    if name_th and name_en:
        parts.append(f"วิชา {name_th} ({name_en})")
    elif name_th:
        parts.append(f"วิชา {name_th}")
    elif name_en:
        parts.append(f"วิชา {name_en}")

    _add_part(parts, f"รหัสวิชา {course.get('course_code')}")
    _add_part(parts, f"หน่วยกิต {course.get('credits')}")

    if placement is not None:
        year = placement.get("year_number")
        semester = placement.get("semester_number")
        if year is not None and semester is not None:
            parts.append(f"เปิดสอนปีที่ {year} ภาคเรียนที่ {semester}")
        elif year is not None:
            parts.append(f"เปิดสอนปีที่ {year}")
        elif semester is not None:
            parts.append(f"เปิดสอนภาคเรียนที่ {semester}")

        _add_part(parts, placement.get("program_code"))
        _add_part(parts, placement.get("plan_code"))
        if _is_valid(placement.get("category")):
            if _is_valid(placement.get("requirement_type")):
                parts.append(
                    f"เป็น{_clean_text(placement.get('category'))} "
                    f"ประเภทวิชา{_clean_text(placement.get('requirement_type'))}"
                )
            else:
                parts.append(f"เป็น{_clean_text(placement.get('category'))}")
        flexible_term = placement.get("flexible_year_semester")
        if flexible_term is None:
            flexible_term = placement.get("notes")
        formatted_flexible_term = _flexible_term_text(flexible_term)
        if formatted_flexible_term:
            parts.append(formatted_flexible_term)
        else:
            _add_part(parts, placement.get("notes"))
        _add_part(parts, placement.get("raw_text"))
    else:
        if _is_valid(course.get("category")):
            parts.append(f"เป็น{_clean_text(course.get('category'))}")
        _add_part(parts, course.get("course_type"))

    prerequisite_rows = list(prerequisites)
    if prerequisite_rows:
        for prerequisite in prerequisite_rows:
            prerequisite_code = _clean_text(prerequisite.get("prerequisite_code"))
            group_codes = [
                _clean_text(member.get("course_code"))
                for member in prerequisite.get("alternative_members", ())
                if _clean_text(member.get("course_code"))
            ]
            if group_codes:
                parts.append(
                    "วิชาที่ต้องเรียนก่อนอย่างใดอย่างหนึ่ง "
                    + ", ".join(group_codes)
                )
            elif prerequisite_code:
                parts.append(f"วิชาที่ต้องเรียนก่อน {prerequisite_code}")
            else:
                _add_part(parts, f"วิชาที่ต้องเรียนก่อน {prerequisite.get('raw_text')}")
    elif _is_valid(course.get("prerequisite_text")):
        parts.append(
            f"วิชาที่ต้องเรียนก่อน {_clean_text(course.get('prerequisite_text'))}"
        )

    if alternative_group is not None:
        label = _clean_text(
            alternative_group.get("label") or alternative_group.get("group_key")
        )
        if label:
            parts.append(f"กลุ่มวิชาทางเลือก {label}")
        member_codes = [
            _clean_text(member.get("course_code"))
            for member in alternative_members
            if _clean_text(member.get("course_code"))
        ]
        if member_codes:
            parts.append("วิชาทางเลือก " + ", ".join(member_codes))
        minimum = alternative_group.get("minimum_choices")
        maximum = alternative_group.get("maximum_choices")
        if minimum is not None and maximum is not None:
            parts.append(f"เลือกอย่างน้อย {minimum} จาก {maximum} วิชา")

    return " ".join(parts)


def _course_description_text(course: Mapping[str, Any]) -> str:
    parts: list[str] = []
    code = _clean_text(course.get("course_code"))
    if code:
        parts.append(f"รหัสวิชา {code}")

    name_th = _clean_text(course.get("name_th"))
    name_en = _clean_text(course.get("name_en"))
    if name_th and name_en:
        parts.append(f"วิชา {name_th} ({name_en})")
    elif name_th:
        parts.append(f"วิชา {name_th}")
    elif name_en:
        parts.append(f"วิชา {name_en}")

    if _is_valid(course.get("description_th")):
        parts.append(f"คำอธิบายรายวิชา: {_clean_text(course.get('description_th'))}")
    if _is_valid(course.get("description_en")):
        parts.append(
            "คำอธิบายรายวิชาภาษาอังกฤษ: "
            f"{_clean_text(course.get('description_en'))}"
        )
    return " ".join(parts)


def _make_chunk(
    *,
    chunk_id: str,
    chunk_type: str,
    text: str,
    entity_type: str,
    entity_id: int,
    course_id: int | None,
    placement_id: int | None,
    alternative_group_id: int | None,
    provenance: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    references = [dict(reference) for reference in provenance]
    return {
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "text": text,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "course_id": course_id,
        "placement_id": placement_id,
        "alternative_group_id": alternative_group_id,
        "provenance_ids": [reference["provenance_id"] for reference in references],
        "provenance": references,
    }


def _build_chunks(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    courses = connection.execute(
        """
        SELECT
            course_id, catalog_id, course_code, name_th, name_en, credits,
            description_th, description_en, category, course_type,
            prerequisite_text, notes
        FROM courses
        ORDER BY course_id
        """
    ).fetchall()

    placements = connection.execute(
        """
        SELECT
            placements.placement_id,
            placements.plan_id,
            placements.course_id,
            placements.alternative_group_id,
            placements.year_number,
            placements.semester_number,
            placements.category,
            placements.requirement_type,
            placements.placement_order,
            placements.credits_override,
            placements.raw_text,
            placements.notes,
            placements.notes AS flexible_year_semester,
            plans.program_code,
            plans.plan_code,
            groups.group_key,
            groups.label,
            groups.minimum_choices,
            groups.maximum_choices,
            groups.notes AS group_notes
        FROM plan_placements AS placements
        LEFT JOIN curriculum_plans AS plans ON plans.plan_id = placements.plan_id
        LEFT JOIN alternative_course_groups AS groups
            ON groups.alternative_group_id = placements.alternative_group_id
        ORDER BY placements.placement_id
        """
    ).fetchall()

    alternative_members: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT
            members.alternative_group_id,
            members.alternative_group_member_id,
            members.course_id,
            members.member_order,
            courses.course_code,
            courses.name_th,
            courses.name_en
        FROM alternative_course_group_members AS members
        JOIN courses ON courses.course_id = members.course_id
        ORDER BY members.alternative_group_id, members.member_order,
                 members.alternative_group_member_id
        """
    ):
        alternative_members[int(row["alternative_group_id"])].append(dict(row))

    prerequisites: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT
            prerequisite.prerequisite_id,
            prerequisite.course_id,
            prerequisite.prerequisite_course_id,
            prerequisite.alternative_group_id,
            prerequisite.requirement_type,
            prerequisite.raw_text,
            target.course_code AS prerequisite_code
        FROM prerequisites AS prerequisite
        LEFT JOIN courses AS target
            ON target.course_id = prerequisite.prerequisite_course_id
        ORDER BY prerequisite.course_id, prerequisite.prerequisite_order,
                 prerequisite.prerequisite_id
        """
    ):
        prerequisite = dict(row)
        prerequisite["alternative_members"] = alternative_members.get(
            int(row["alternative_group_id"]), []
        ) if row["alternative_group_id"] is not None else []
        prerequisites[int(row["course_id"])].append(prerequisite)

    course_provenance = _provenance_map(
        connection, "course_provenance", "course_id"
    )
    placement_provenance = _provenance_map(
        connection, "plan_placement_provenance", "placement_id"
    )
    prerequisite_provenance = _provenance_map(
        connection, "prerequisite_provenance", "prerequisite_id"
    )
    group_provenance = _provenance_map(
        connection, "alternative_group_provenance", "alternative_group_id"
    )
    member_provenance = _provenance_map(
        connection,
        "alternative_group_member_provenance",
        "alternative_group_member_id",
    )

    placements_by_course: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    group_placements: list[Mapping[str, Any]] = []
    for row in placements:
        placement = dict(row)
        if placement["course_id"] is not None:
            placements_by_course[int(placement["course_id"])].append(placement)
        elif placement["alternative_group_id"] is not None:
            group_placements.append(placement)

    chunks: list[dict[str, Any]] = []
    for course_row in courses:
        course = dict(course_row)
        course_id = int(course["course_id"])
        course_prerequisites = prerequisites.get(course_id, [])
        prerequisite_refs = _merge_provenance(
            *(
                prerequisite_provenance.get(int(row["prerequisite_id"]), [])
                for row in course_prerequisites
            )
        )

        course_refs = course_provenance.get(course_id, [])
        course_placements = placements_by_course.get(course_id, [])
        if course_placements:
            for placement in course_placements:
                references = _merge_provenance(
                    course_refs,
                    placement_provenance.get(int(placement["placement_id"]), []),
                    prerequisite_refs,
                )
                text = _course_metadata_text(
                    course, placement, course_prerequisites
                )
                chunks.append(
                    _make_chunk(
                        chunk_id=(
                            f"course-{course_id}-placement-"
                            f"{placement['placement_id']}-metadata"
                        ),
                        chunk_type="metadata",
                        text=text,
                        entity_type="course_placement",
                        entity_id=int(placement["placement_id"]),
                        course_id=course_id,
                        placement_id=int(placement["placement_id"]),
                        alternative_group_id=None,
                        provenance=references,
                    )
                )
        else:
            chunks.append(
                _make_chunk(
                    chunk_id=f"course-{course_id}-metadata",
                    chunk_type="metadata",
                    text=_course_metadata_text(
                        course, None, course_prerequisites
                    ),
                    entity_type="course",
                    entity_id=course_id,
                    course_id=course_id,
                    placement_id=None,
                    alternative_group_id=None,
                    provenance=_merge_provenance(course_refs, prerequisite_refs),
                )
            )

        description_text = _course_description_text(course)
        if description_text:
            chunks.append(
                _make_chunk(
                    chunk_id=f"course-{course_id}-description",
                    chunk_type="description",
                    text=description_text,
                    entity_type="course",
                    entity_id=course_id,
                    course_id=course_id,
                    placement_id=None,
                    alternative_group_id=None,
                    provenance=course_refs,
                )
            )

    for placement in group_placements:
        group_id = int(placement["alternative_group_id"])
        members = alternative_members.get(group_id, [])
        member_refs = _merge_provenance(
            *(member_provenance.get(int(member["alternative_group_member_id"]), [])
              for member in members)
        )
        course_refs = _merge_provenance(
            *(course_provenance.get(int(member["course_id"]), []) for member in members)
        )
        references = _merge_provenance(
            placement_provenance.get(int(placement["placement_id"]), []),
            group_provenance.get(group_id, []),
            member_refs,
            course_refs,
        )
        group = {
            key: placement[key]
            for key in (
                "group_key",
                "label",
                "minimum_choices",
                "maximum_choices",
                "group_notes",
            )
        }
        group_text = _course_metadata_text(
            {}, placement, (), group, members
        )
        chunks.append(
            _make_chunk(
                chunk_id=f"alternative-group-{group_id}-placement-"
                f"{placement['placement_id']}-metadata",
                chunk_type="metadata",
                text=group_text,
                entity_type="alternative_placement",
                entity_id=int(placement["placement_id"]),
                course_id=None,
                placement_id=int(placement["placement_id"]),
                alternative_group_id=group_id,
                provenance=references,
            )
        )

    chunks.sort(
        key=lambda chunk: (
            chunk["course_id"] is None,
            chunk["course_id"] or 0,
            0 if chunk["chunk_type"] == "metadata" else 1,
            chunk["placement_id"] or 0,
            chunk["chunk_id"],
        )
    )
    return chunks


def build_chunks(
    database: str | Path | sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Return deterministic metadata and description chunks from a SQLite DB."""
    if isinstance(database, sqlite3.Connection):
        return _build_chunks(database)

    database_path = Path(database)
    if not database_path.exists():
        raise FileNotFoundError(database_path)
    with closing(sqlite3.connect(str(database_path))) as connection:
        connection.row_factory = sqlite3.Row
        return _build_chunks(connection)


generate_chunks = build_chunks
