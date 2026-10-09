"""Load one consolidated curriculum JSON document into the structured schema."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from contextlib import closing, nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_ALTERNATIVE_SEPARATOR = re.compile(
    r"\s*(?:\u0e2b\u0e23\u0e37\u0e2d|\bor\b|/|,)\s*", re.IGNORECASE
)
_NO_PREREQUISITE = {
    "",
    "-",
    "n/a",
    "none",
    "no prerequisite",
    "null",
    "\u0e44\u0e21\u0e48\u0e21\u0e35",
}
_DOCUMENT_CATEGORIES = {"plan", "description", "unknown"}
_SUPPLEMENTAL_DOCUMENT_CATEGORIES = {"rule", "program_requirement"}
_FLEXIBLE_YEAR_SEMESTER = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_CREDIT_UNITS = re.compile(r"^\s*(\d+)(?:\s*\([^)]*\))?\s*$")
_SOURCE_IDENTITY_FIELDS = (
    "source_document_key",
    "document_key",
    "document_id",
    "source_filename",
    "document_filename",
    "source_uri",
    "source_locator",
    "source",
)
_PRODUCTION_DOCUMENT_CATEGORIES = {"plan", "description"}


def _first_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _as_integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _flexible_year_semester_values(value: Any) -> tuple[int | None, int | None, str | None]:
    raw_value = _as_text(value)
    if raw_value is None:
        return None, None, None
    match = _FLEXIBLE_YEAR_SEMESTER.fullmatch(raw_value)
    if match is None:
        return None, None, raw_value
    return int(match.group(1)), int(match.group(2)), raw_value


def _credit_values(value: Any) -> tuple[int | None, str | None]:
    raw_value = _as_text(value)
    if raw_value is None or not raw_value.strip():
        return None, None
    match = _CREDIT_UNITS.fullmatch(raw_value)
    if match is None:
        return None, raw_value
    return int(match.group(1)), raw_value


def _source_document_key(entry: Mapping[str, Any]) -> str | None:
    for field in _SOURCE_IDENTITY_FIELDS:
        value = _as_text(entry.get(field))
        if value is not None and value.strip():
            return value.strip()
    return None


def _page_number(value: Any) -> int | None:
    if value is None:
        return None
    page = _as_integer(value)
    if page is None:
        raise ValueError(f"source page must be an integer, got {value!r}")
    return page


def _split_code_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = _first_value(value, "code", "course_code", "id")
        return _split_code_value(value)
    if isinstance(value, (list, tuple)):
        codes: list[str] = []
        for item in value:
            codes.extend(_split_code_value(item))
        return codes

    text = _as_text(value)
    if text is None:
        return []
    return [part.strip() for part in _ALTERNATIVE_SEPARATOR.split(text) if part.strip()]


def _normalized_identity(value: str) -> str:
    return value.strip().casefold()


def _normalized_course_name(value: Any, *, casefold: bool) -> Any:
    if not isinstance(value, str):
        return value
    # Compare exact text after formatting whitespace is removed; this is not
    # fuzzy matching and still distinguishes substantive title differences.
    if casefold:
        return "".join(value.split()).casefold()
    return "".join(value.split())


def _normalized_description(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


def _compatible_value(existing: Any, incoming: Any, label: str) -> Any:
    if existing is not None and incoming is not None and existing != incoming:
        raise ValueError(
            f"conflicting {label} for shared curriculum identity: "
            f"{existing!r} != {incoming!r}"
        )
    return existing if existing is not None else incoming


def _normalized_course_code(code: str) -> str:
    return _normalized_identity(code)


def _is_placeholder_course_code(code: str) -> bool:
    return "x" in code.casefold()


def _course_identity_discriminator(
    code: str, name_en: Any, name_th: Any
) -> str:
    if not _is_placeholder_course_code(code):
        return ""
    label = _as_text(name_en) or _as_text(name_th)
    if label is None or not label.strip():
        raise ValueError(
            f"placeholder course {code!r} requires a course label for identity"
        )
    normalized_label = " ".join(label.casefold().split())
    return f"placeholder:{normalized_label}"


def _course_codes(course: Mapping[str, Any]) -> list[str]:
    codes = _split_code_value(course.get("code"))
    alternatives = _first_value(
        course, "alternative_courses", "alternatives", "alternative_codes"
    )
    if alternatives is not None:
        for code in _split_code_value(alternatives):
            if code not in codes:
                codes.append(code)
    if not codes:
        raise ValueError("course record is missing a course code")
    return codes


def _member_value(value: Any, index: int, count: int, split_lines: bool = False) -> Any:
    if count == 1 or not isinstance(value, str):
        return value
    if split_lines:
        values = [line.strip() for line in value.splitlines() if line.strip()]
    else:
        values = [part.strip() for part in _ALTERNATIVE_SEPARATOR.split(value) if part.strip()]
    return values[index] if len(values) == count else value


def _provenance_entries(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, Mapping)]
    return []


def _normalized_provenance(
    entry: Mapping[str, Any],
    default_program: str | None,
    default_source_document_key: str | None,
    allowed_categories: set[str] | None = None,
) -> tuple[Any, ...]:
    category = _as_text(entry.get("document_category")) or "unknown"
    categories = _DOCUMENT_CATEGORIES if allowed_categories is None else allowed_categories
    if category not in categories:
        raise ValueError(f"unsupported document category: {category!r}")
    source_document_key = _source_document_key(entry) or default_source_document_key
    if source_document_key is None:
        if category in _PRODUCTION_DOCUMENT_CATEGORIES:
            raise ValueError(
                "production provenance record has no usable source identity "
                "(source_document_key, source_filename, source_uri, or source_locator)"
            )
        source_document_key = "unknown"
    return (
        source_document_key,
        _as_text(entry.get("program")) or default_program,
        _as_text(entry.get("source_filename")),
        _page_number(entry.get("source_page")),
        _page_number(entry.get("document_page")),
        category,
        _as_text(entry.get("source_uri")),
        _as_text(entry.get("source_locator")),
        _as_text(entry.get("excerpt")),
    )


def _provenance_ids(
    connection: sqlite3.Connection,
    value: Any,
    default_program: str | None,
    default_source_document_key: str | None,
    cache: dict[tuple[Any, ...], int],
    allowed_categories: set[str] | None = None,
) -> list[tuple[int, int]]:
    references: list[tuple[int, int]] = []
    for source_order, entry in enumerate(_provenance_entries(value)):
        normalized = _normalized_provenance(
            entry,
            default_program,
            default_source_document_key,
            allowed_categories=allowed_categories,
        )
        provenance_id = cache.get(normalized)
        if provenance_id is None:
            cursor = connection.execute(
                """
                INSERT INTO provenance (
                    source_document_key, program, source_filename, source_page, document_page,
                    document_category, source_uri, source_locator, excerpt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                normalized,
            )
            provenance_id = int(cursor.lastrowid)
            cache[normalized] = provenance_id
        references.append((provenance_id, source_order))
    return references


def _link_provenance(
    connection: sqlite3.Connection,
    table: str,
    entity_column: str,
    entity_id: int,
    references: list[tuple[int, int]],
) -> None:
    if table == "course_provenance":
        sql = (
            "INSERT OR IGNORE INTO course_provenance "
            "(course_id, provenance_id, source_order) VALUES (?, ?, ?)"
        )
        connection.executemany(
            sql, [(entity_id, provenance_id, order) for provenance_id, order in references]
        )
        return

    sql = (
        f"INSERT OR IGNORE INTO {table} "
        f"({entity_column}, provenance_id) VALUES (?, ?)"
    )
    connection.executemany(
        sql, [(entity_id, provenance_id) for provenance_id, _ in references]
    )


def _provenance_categories(
    connection: sqlite3.Connection, references: list[tuple[int, int]]
) -> set[str]:
    provenance_ids = [provenance_id for provenance_id, _ in references]
    if not provenance_ids:
        return set()
    placeholders = ", ".join("?" for _ in provenance_ids)
    return {
        str(row[0])
        for row in connection.execute(
            f"SELECT DISTINCT document_category FROM provenance "
            f"WHERE provenance_id IN ({placeholders})",
            provenance_ids,
        )
    }


def _course_provenance_categories(
    connection: sqlite3.Connection, course_id: int
) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            """SELECT DISTINCT provenance.document_category
               FROM course_provenance
               JOIN provenance USING (provenance_id)
               WHERE course_provenance.course_id = ?""",
            (course_id,),
        )
    }


def _link_catalog_and_plan(
    connection: sqlite3.Connection,
    catalog_id: int,
    plan_id: int,
    references: list[tuple[int, int]],
) -> None:
    _link_provenance(
        connection, "catalog_provenance", "catalog_id", catalog_id, references
    )
    _link_provenance(
        connection,
        "curriculum_plan_provenance",
        "plan_id",
        plan_id,
        references,
    )


def _group_values(course: Mapping[str, Any], key: str) -> Any:
    group = course.get("alternative_group")
    if isinstance(group, Mapping) and key in group:
        return group[key]
    return course.get(key)


def _create_alternative_group(
    connection: sqlite3.Connection,
    catalog_id: int,
    plan_id: int,
    course: Mapping[str, Any],
    default_key: str,
    default_label: str,
    references: list[tuple[int, int]],
    notes: Any = None,
) -> int:
    minimum = _as_integer(_group_values(course, "minimum_choices")) or 1
    maximum = _as_integer(_group_values(course, "maximum_choices")) or 1
    if minimum < 1 or maximum < minimum:
        raise ValueError("alternative group choice bounds are invalid")

    group_key = _as_text(_group_values(course, "group_key")) or default_key
    label = _as_text(_group_values(course, "label")) or default_label
    group_notes = _group_values(course, "notes")
    if group_notes is None:
        group_notes = notes

    cursor = connection.execute(
        """
        INSERT INTO alternative_course_groups (
            catalog_id, plan_id, group_key, label,
            minimum_choices, maximum_choices, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            catalog_id,
            plan_id,
            group_key,
            label,
            minimum,
            maximum,
            _as_text(group_notes),
        ),
    )
    group_id = int(cursor.lastrowid)
    _link_provenance(
        connection,
        "alternative_group_provenance",
        "alternative_group_id",
        group_id,
        references,
    )
    return group_id


def _insert_group_member(
    connection: sqlite3.Connection,
    group_id: int,
    course_id: int,
    member_order: int,
    references: list[tuple[int, int]],
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO alternative_course_group_members (
            alternative_group_id, course_id, member_order
        ) VALUES (?, ?, ?)
        """,
        (group_id, course_id, member_order),
    )
    member_id = int(cursor.lastrowid)
    _link_provenance(
        connection,
        "alternative_group_member_provenance",
        "alternative_group_member_id",
        member_id,
        references,
    )
    return member_id


def _raw_prerequisite(course: Mapping[str, Any]) -> Any:
    return _first_value(course, "prerequisite", "prerequisites", "prerequisite_text")


def _is_no_prerequisite(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)) and not value:
        return True
    text = (_as_text(value) or "").strip().lower()
    return text in _NO_PREREQUISITE


def _prerequisite_tokens(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        alternatives = _first_value(
            value, "alternatives", "alternative_courses", "course_codes", "codes"
        )
        if alternatives is not None:
            return _split_code_value(alternatives)
        value = _first_value(value, "code", "course_code", "text", "value")
    return _split_code_value(value)


def _course_id_for_code(
    code_to_ids: Mapping[str, list[int]], code: str
) -> list[int]:
    return code_to_ids.get(_normalized_course_code(code), [])


def _insert_prerequisite(
    connection: sqlite3.Connection,
    course_id: int,
    prerequisite_course_id: int | None,
    alternative_group_id: int | None,
    raw_text: str,
    requirement_type: str,
    references: list[tuple[int, int]],
) -> None:
    existing = connection.execute(
        """SELECT prerequisite_id FROM prerequisites
           WHERE course_id = ? AND prerequisite_course_id IS ?
             AND alternative_group_id IS ? AND requirement_type = ? AND raw_text = ?
           ORDER BY prerequisite_id LIMIT 1""",
        (
            course_id,
            prerequisite_course_id,
            alternative_group_id,
            requirement_type,
            raw_text,
        ),
    ).fetchone()
    if existing is None:
        cursor = connection.execute(
            """
            INSERT INTO prerequisites (
                course_id, prerequisite_course_id, alternative_group_id,
                prerequisite_order, requirement_type, raw_text
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                course_id,
                prerequisite_course_id,
                alternative_group_id,
                1,
                requirement_type,
                raw_text,
            ),
        )
        prerequisite_id = int(cursor.lastrowid)
    else:
        prerequisite_id = int(existing[0])
    _link_provenance(
        connection,
        "prerequisite_provenance",
        "prerequisite_id",
        prerequisite_id,
        references,
    )


def _load_prerequisites(
    connection: sqlite3.Connection,
    records: list[dict[str, Any]],
    code_to_ids: Mapping[str, list[int]],
    catalog_id: int,
    plan_id: int,
) -> None:
    for record in records:
        raw_value = record["prerequisite"]
        if _is_no_prerequisite(raw_value):
            continue

        raw_text = _as_text(raw_value)
        if raw_text is None:
            continue
        tokens = _prerequisite_tokens(raw_value)
        requirement_type = record["requirement_type"]
        references = record["provenance"]

        for course_id in record["course_ids"]:
            target_ids: list[int] = []
            if len(tokens) == 1:
                candidates = _course_id_for_code(code_to_ids, tokens[0])
                if len(candidates) == 1:
                    target_ids = candidates
            elif tokens:
                resolved = True
                for token in tokens:
                    candidates = _course_id_for_code(code_to_ids, token)
                    if len(candidates) != 1:
                        resolved = False
                        break
                    if candidates[0] not in target_ids:
                        target_ids.append(candidates[0])
                if not resolved:
                    target_ids = []

            if len(target_ids) == 1:
                _insert_prerequisite(
                    connection,
                    course_id,
                    target_ids[0],
                    None,
                    raw_text,
                    requirement_type,
                    references,
                )
                continue

            if len(target_ids) > 1:
                group = _create_alternative_group(
                    connection,
                    catalog_id,
                    plan_id,
                    {},
                    f"prerequisite-{record['placement_order']}-{course_id}",
                    f"Prerequisite alternatives for {record['course_code']}",
                    references,
                    raw_text,
                )
                for member_order, target_id in enumerate(target_ids, start=1):
                    _insert_group_member(
                        connection, group, target_id, member_order, references
                    )
                _insert_prerequisite(
                    connection,
                    course_id,
                    None,
                    group,
                    raw_text,
                    requirement_type,
                    references,
                )
                continue

            # Keep an unresolved prerequisite as text rather than guessing from
            # an ambiguous or absent course-code match.
            _insert_prerequisite(
                connection,
                course_id,
                None,
                None,
                raw_text,
                requirement_type,
                references,
            )


def _load_json_to_sqlite(
    input_json_path: str | Path,
    output_db_path: str | Path,
    *,
    connection: sqlite3.Connection | None = None,
    initialize_schema: bool = True,
    provenance_cache: dict[tuple[Any, ...], int] | None = None,
    ambiguous_course_fields: set[tuple[int, str]] | None = None,
    commit: bool = True,
) -> int:
    """Load one consolidated document into an initialized SQLite database."""
    input_path = Path(input_json_path)
    output_path = Path(output_db_path)
    if ambiguous_course_fields is None:
        ambiguous_course_fields = set()

    with input_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, Mapping):
        raise ValueError("consolidated JSON must contain an object at the top level")

    course_records = document.get("courses")
    if not isinstance(course_records, list):
        raise ValueError("consolidated JSON must contain a courses list")

    catalog_data = document.get("catalog")
    if not isinstance(catalog_data, Mapping):
        catalog_data = document
    plan_data = document.get("plan")
    if not isinstance(plan_data, Mapping):
        plan_data = document

    program = _as_text(_first_value(document, "program")) or "UNKNOWN"
    explicit_catalog_key = _as_text(_first_value(catalog_data, "catalog_key", "key"))
    if explicit_catalog_key is None and catalog_data is not document:
        explicit_catalog_key = _as_text(catalog_data.get("source"))
    catalog_key = explicit_catalog_key or _as_text(
        _first_value(catalog_data, "source")
    ) or program
    catalog_title = _as_text(
        _first_value(catalog_data, "title", "catalog_title", "description", "source")
    ) or program
    academic_year = _as_text(
        _first_value(catalog_data, "academic_year", "year", "version")
    )
    institution = _as_text(_first_value(catalog_data, "institution"))
    catalog_notes = _as_text(_first_value(catalog_data, "notes"))

    plan_value = document.get("plan")
    if isinstance(plan_value, Mapping):
        plan_code = _as_text(_first_value(plan_data, "plan_code", "code", "id"))
        plan_name = _as_text(_first_value(plan_data, "plan_name", "name", "title"))
    else:
        plan_code = _as_text(plan_value)
        plan_name = _as_text(_first_value(document, "plan_name")) or plan_code
    plan_version = _as_text(_first_value(plan_data, "version", "academic_year"))
    plan_notes = _as_text(_first_value(plan_data, "notes"))
    program_code_normalized = _normalized_identity(program)
    plan_key = _normalized_identity(plan_code or plan_name or "default") or "default"
    default_source_document_key = _source_document_key(document)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    connection_context = (
        closing(sqlite3.connect(str(output_path)))
        if connection is None
        else nullcontext(connection)
    )
    with connection_context as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        if initialize_schema:
            connection.executescript(schema)

        catalog_id: int | None = None
        if explicit_catalog_key is not None:
            matches = connection.execute(
                """SELECT catalog_id, title, academic_year, institution
                   FROM catalogs WHERE catalog_key = ? ORDER BY catalog_id""",
                (catalog_key,),
            ).fetchall()
            if len(matches) > 1:
                raise ValueError(
                    f"catalog_key {catalog_key!r} already identifies multiple catalogs"
                )
            if matches:
                catalog_id = int(matches[0][0])
                _, existing_title, existing_year, existing_institution = matches[0]
                merged_title = _compatible_value(
                    existing_title, catalog_title, "catalog title"
                )
                merged_year = _compatible_value(
                    existing_year, academic_year, "academic_year"
                )
                merged_institution = _compatible_value(
                    existing_institution, institution, "catalog institution"
                )
                connection.execute(
                    """UPDATE catalogs SET title = ?, academic_year = ?, institution = ?,
                       notes = COALESCE(notes, ?) WHERE catalog_id = ?""",
                    (
                        merged_title,
                        merged_year,
                        merged_institution,
                        catalog_notes,
                        catalog_id,
                    ),
                )
        if catalog_id is None:
            catalog_cursor = connection.execute(
                """
                INSERT INTO catalogs (
                    catalog_key, title, academic_year, institution, notes
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (catalog_key, catalog_title, academic_year, institution, catalog_notes),
            )
            catalog_id = int(catalog_cursor.lastrowid)

        program_row = connection.execute(
            """SELECT program_id FROM programs
               WHERE catalog_id = ? AND program_code_normalized = ?""",
            (catalog_id, program_code_normalized),
        ).fetchone()
        if program_row is None:
            program_cursor = connection.execute(
                """
                INSERT INTO programs (
                    catalog_id, program_code, program_code_normalized
                ) VALUES (?, ?, ?)
                """,
                (catalog_id, program, program_code_normalized),
            )
            program_id = int(program_cursor.lastrowid)
        else:
            program_id = int(program_row[0])

        plan_row = connection.execute(
            """SELECT plan_id, plan_code, plan_name, version FROM curriculum_plans
               WHERE catalog_id = ? AND program_id = ? AND plan_key = ?""",
            (catalog_id, program_id, plan_key),
        ).fetchone()
        if plan_row is None:
            plan_cursor = connection.execute(
                """
                INSERT INTO curriculum_plans (
                    catalog_id, program_id, program_code, plan_key,
                    plan_code, plan_name, version, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    catalog_id,
                    program_id,
                    program,
                    plan_key,
                    plan_code,
                    plan_name,
                    plan_version,
                    plan_notes,
                ),
            )
            plan_id = int(plan_cursor.lastrowid)
        else:
            plan_id = int(plan_row[0])
            merged_plan_code = _compatible_value(plan_row[1], plan_code, "plan code")
            merged_plan_name = _compatible_value(plan_row[2], plan_name, "plan name")
            merged_plan_version = _compatible_value(plan_row[3], plan_version, "plan version")
            connection.execute(
                """UPDATE curriculum_plans SET plan_code = ?, plan_name = ?, version = ?,
                   notes = COALESCE(notes, ?) WHERE plan_id = ?""",
                (merged_plan_code, merged_plan_name, merged_plan_version, plan_notes, plan_id),
            )

        if provenance_cache is None:
            provenance_cache = {}
        root_references = _provenance_ids(
            connection,
            document.get("source_provenance"),
            program,
            default_source_document_key,
            provenance_cache,
        )
        _link_catalog_and_plan(connection, catalog_id, plan_id, root_references)

        code_to_ids: defaultdict[str, list[int]] = defaultdict(list)
        identity_to_ids: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        for existing_course_id, existing_code, discriminator in connection.execute(
            """SELECT course_id, course_code_normalized, course_identity_discriminator
               FROM courses WHERE catalog_id = ?""",
            (catalog_id,),
        ):
            course_id = int(existing_course_id)
            code_to_ids[existing_code].append(course_id)
            identity_to_ids[(existing_code, discriminator)].append(course_id)
        records: list[dict[str, Any]] = []

        for placement_index, raw_course in enumerate(course_records, start=1):
            if not isinstance(raw_course, Mapping):
                raise ValueError(f"course at index {placement_index - 1} is not an object")

            codes = _course_codes(raw_course)
            raw_provenance = (
                raw_course.get("source_provenance")
                if "source_provenance" in raw_course
                else document.get("source_provenance")
            )
            references = _provenance_ids(
                connection,
                raw_provenance,
                program,
                default_source_document_key,
                provenance_cache,
            )
            source_categories = _provenance_categories(connection, references)
            current_plan_backed = "plan" in source_categories
            current_description_only = (
                bool(source_categories) and source_categories <= {"description"}
            )
            _link_catalog_and_plan(connection, catalog_id, plan_id, references)

            prerequisite = _raw_prerequisite(raw_course)
            requirement_type = (
                _as_text(
                    _first_value(
                        raw_course,
                        "prerequisite_requirement_type",
                        "requirement_type",
                    )
                )
                or "required"
            )
            course_ids: list[int] = []
            for member_index, code in enumerate(codes):
                normalized_code = _normalized_course_code(code)
                credits_value = _member_value(
                    raw_course.get("credits"), member_index, len(codes)
                )
                credit_units, credits_raw = _credit_values(credits_value)
                facts = {
                    "name_th": _member_value(raw_course.get("name_th"), member_index, len(codes), True),
                    "name_en": _member_value(raw_course.get("name_en"), member_index, len(codes), True),
                    "credits": credits_value,
                    "credit_units": credit_units,
                    "credits_raw": credits_raw,
                    "description_th": raw_course.get("desc_th", raw_course.get("description_th")),
                    "description_en": raw_course.get("desc_en", raw_course.get("description_en")),
                    "category": raw_course.get("category"),
                    "course_type": raw_course.get("type", raw_course.get("course_type")),
                    "prerequisite_text": _as_text(prerequisite),
                    "notes": raw_course.get("note", raw_course.get("notes")),
                }
                discriminator = _course_identity_discriminator(
                    code, facts["name_en"], facts["name_th"]
                )
                identity_key = (normalized_code, discriminator)
                existing_ids = identity_to_ids[identity_key]
                if existing_ids:
                    course_id = existing_ids[0]
                    existing_categories = _course_provenance_categories(
                        connection, course_id
                    )
                    existing_plan_backed = "plan" in existing_categories
                    existing_description_only = (
                        bool(existing_categories)
                        and existing_categories <= {"description"}
                    )
                    fact_columns = tuple(facts)
                    stored_facts = connection.execute(
                        f"SELECT {', '.join(fact_columns)} FROM courses WHERE course_id = ?",
                        (course_id,),
                    ).fetchone()
                    updates: dict[str, Any] = {}
                    for column, existing, incoming in zip(
                        fact_columns, stored_facts, facts.values(), strict=True
                    ):
                        if column in {
                            "name_th",
                            "name_en",
                            "credits",
                            "credit_units",
                            "credits_raw",
                        }:
                            conflict_key = (course_id, column)
                            compatible = existing == incoming
                            if column in {"name_th", "name_en"}:
                                compatible = _normalized_course_name(
                                    existing, casefold=column == "name_en"
                                ) == _normalized_course_name(
                                    incoming, casefold=column == "name_en"
                                )

                            if current_plan_backed and existing_description_only:
                                if incoming is not None:
                                    updates[column] = incoming
                                ambiguous_course_fields.discard(conflict_key)
                                continue
                            if current_description_only and existing_plan_backed:
                                continue
                            if current_description_only and existing_description_only:
                                if existing is not None and incoming is not None and not compatible:
                                    updates[column] = None
                                    ambiguous_course_fields.add(conflict_key)
                                elif (
                                    conflict_key not in ambiguous_course_fields
                                    and existing is None
                                    and incoming is not None
                                ):
                                    updates[column] = incoming
                                continue

                            if existing is not None and incoming is not None and not compatible:
                                raise ValueError(
                                    f"conflicting {column} for shared course {code!r}: "
                                    f"{existing!r} != {incoming!r}"
                                )
                            if existing is None and incoming is not None:
                                updates[column] = incoming
                            elif current_plan_backed and incoming is not None:
                                updates[column] = incoming
                            continue

                        if column in {
                            "description_th",
                            "description_en",
                            "notes",
                            "category",
                            "course_type",
                            "prerequisite_text",
                        }:
                            conflict_key = (course_id, column)
                            if conflict_key in ambiguous_course_fields:
                                continue
                            if column in {"description_th", "description_en", "notes"}:
                                existing_value = _normalized_description(existing)
                                incoming_value = _normalized_description(incoming)
                            else:
                                existing_value = existing
                                incoming_value = incoming
                            if existing_value is not None and incoming_value is not None:
                                if existing_value != incoming_value:
                                    updates[column] = None
                                    ambiguous_course_fields.add(conflict_key)
                            elif existing_value is None and incoming_value is not None:
                                updates[column] = incoming
                            continue

                        compatible = existing == incoming
                        if column in {"name_th", "name_en"}:
                            compatible = _normalized_course_name(
                                existing, casefold=column == "name_en"
                            ) == _normalized_course_name(
                                incoming, casefold=column == "name_en"
                            )
                        if existing is not None and incoming is not None and not compatible:
                            raise ValueError(
                                f"conflicting {column} for shared course {code!r}: "
                                f"{existing!r} != {incoming!r}"
                            )
                        if existing is None and incoming is not None:
                            updates[column] = incoming
                    if updates:
                        connection.execute(
                            f"UPDATE courses SET {', '.join(f'{column} = ?' for column in updates)} "
                            "WHERE course_id = ?",
                            (*updates.values(), course_id),
                        )
                else:
                    cursor = connection.execute(
                        """
                        INSERT INTO courses (
                            catalog_id, course_code, course_code_normalized,
                            course_identity_discriminator,
                            name_th, name_en, credits, credit_units, credits_raw,
                            description_th, description_en, category, course_type,
                            prerequisite_text, notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (catalog_id, code, normalized_code, discriminator, *facts.values()),
                    )
                    course_id = int(cursor.lastrowid)
                    identity_to_ids[identity_key].append(course_id)
                    code_to_ids[normalized_code].append(course_id)
                course_ids.append(course_id)
                _link_provenance(
                    connection,
                    "course_provenance",
                    "course_id",
                    course_id,
                    references,
                )

            group_id: int | None = None
            if len(course_ids) > 1:
                group_id = _create_alternative_group(
                    connection,
                    catalog_id,
                    plan_id,
                    raw_course,
                    f"placement-{placement_index}",
                    "Alternative course group",
                    references,
                    raw_course.get("note", raw_course.get("notes")),
                )
                for member_index, course_id in enumerate(course_ids, start=1):
                    _insert_group_member(
                        connection, group_id, course_id, member_index, references
                    )

            year_number = _as_integer(raw_course.get("year"))
            semester_number = _as_integer(raw_course.get("semester"))
            if year_number == 0:
                year_number = None
            if semester_number == 0:
                semester_number = None
            flexible_year_number, flexible_semester_number, flexible_year_semester_raw = (
                _flexible_year_semester_values(raw_course.get("flexible_year_semester"))
            )

            placement_cursor = connection.execute(
                """
                INSERT INTO plan_placements (
                    plan_id, course_id, alternative_group_id, year_number,
                    semester_number, flexible_year_number, flexible_semester_number,
                    flexible_year_semester_raw, category, requirement_type,
                    placement_order, credits_override, raw_text, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    course_ids[0] if group_id is None else None,
                    group_id,
                    year_number,
                    semester_number,
                    flexible_year_number,
                    flexible_semester_number,
                    flexible_year_semester_raw,
                    raw_course.get("category"),
                    raw_course.get("type", raw_course.get("course_type")),
                    placement_index,
                    raw_course.get("credits_override"),
                    raw_course.get("raw_text"),
                    raw_course.get("note", raw_course.get("notes")),
                ),
            )
            placement_id = int(placement_cursor.lastrowid)
            _link_provenance(
                connection,
                "plan_placement_provenance",
                "placement_id",
                placement_id,
                references,
            )

            records.append(
                {
                    "course_code": codes[0],
                    "course_ids": course_ids,
                    "placement_order": placement_index,
                    "prerequisite": prerequisite,
                    "requirement_type": requirement_type,
                    "provenance": references,
                }
            )

        _load_prerequisites(connection, records, code_to_ids, catalog_id, plan_id)
        if commit:
            connection.commit()
    return catalog_id


def load_json_to_sqlite(input_json_path: str | Path, output_db_path: str | Path) -> None:
    """Load one consolidated JSON file into a new structured SQLite database."""
    _load_json_to_sqlite(input_json_path, output_db_path)


def load_jsons_to_sqlite(
    input_json_paths: Iterable[str | Path],
    output_db_path: str | Path,
) -> list[int]:
    """Load consolidated JSON files into one structured SQLite database."""
    input_paths = [Path(input_path) for input_path in input_json_paths]
    if not input_paths:
        raise ValueError("at least one consolidated JSON input is required")

    output_path = Path(output_db_path)
    provenance_cache: dict[tuple[Any, ...], int] = {}
    ambiguous_course_fields: set[tuple[int, str]] = set()
    with closing(sqlite3.connect(str(output_path))) as connection:
        try:
            catalog_ids = [
                _load_json_to_sqlite(
                    input_path,
                    output_path,
                    connection=connection,
                    initialize_schema=index == 0,
                    provenance_cache=provenance_cache,
                    ambiguous_course_fields=ambiguous_course_fields,
                    commit=False,
                )
                for index, input_path in enumerate(input_paths)
            ]
            connection.commit()
            return catalog_ids
        except Exception:
            connection.rollback()
            raise
