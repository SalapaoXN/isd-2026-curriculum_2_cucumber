"""Load explicit supplemental academic-authority JSON into the RAG database."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from .loader import (
    _as_text,
    _provenance_ids,
    _source_document_key,
    _SUPPLEMENTAL_DOCUMENT_CATEGORIES,
)


_RULE_ID = re.compile(r"^rule:(?P<section>[^.]+(?:\.[^.]+)*)$")
_OPERATORS = {
    "at_least": ">=",
    "at_most": "<=",
    "below": "<",
    "above": ">",
    "equal": "=",
    "required": "required",
    "deadline": "deadline",
}


def _required_text(value: Any, field: str) -> str:
    text = _as_text(value)
    if text is None or not text.strip():
        raise ValueError(f"supplemental record is missing {field}")
    return text.strip()


def _value(value: Any, field: str) -> Any:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"supplemental record has invalid {field}")
    if isinstance(value, (str, int, float)):
        return value
    raise ValueError(f"supplemental record has invalid {field}")


def _rule_identity(rule_id: Any) -> tuple[str, str, str | None]:
    value = _required_text(rule_id, "rule_id")
    match = _RULE_ID.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid rule_id: {value!r}")
    section = match.group("section")
    parent = section.rsplit(".", 1)[0] if "." in section else None
    return value, section, f"rule:{parent}" if parent else None


def _source_references(category: Mapping[str, Any]) -> Any:
    evidence = category.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("policy category is missing evidence")
    references = evidence.get("source_provenance")
    if not isinstance(references, list) or any(
        not isinstance(reference, Mapping) for reference in references
    ):
        raise ValueError("policy category is missing source provenance")
    if not references and (category.get("values") or evidence.get("supporting_rule_text")):
        raise ValueError("policy category is missing source provenance")
    return references


def _load_policy(
    path: Path,
    connection: sqlite3.Connection,
    provenance_cache: dict[tuple[Any, ...], int],
) -> None:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, Mapping) or document.get("source") != "Academic Rules":
        raise ValueError("institution policy JSON has an invalid source")
    categories = document.get("categories")
    if not isinstance(categories, list):
        raise ValueError("institution policy JSON must contain categories")

    rules: dict[str, tuple[str, str, str]] = {}
    category_refs: dict[str, list[tuple[int, int]]] = {}
    for category in categories:
        if not isinstance(category, Mapping):
            raise ValueError("policy category must be an object")
        category_name = _required_text(category.get("category"), "category")
        references = _source_references(category)
        refs = _provenance_ids(
            connection,
            references,
            "RULE",
            _source_document_key({"source": path.name}),
            provenance_cache,
            allowed_categories=_SUPPLEMENTAL_DOCUMENT_CATEGORIES,
        )
        category_refs[category_name] = refs
        evidence = category["evidence"]
        for raw_rule in evidence.get("supporting_rule_text", []):
            if not isinstance(raw_rule, Mapping):
                raise ValueError("supporting rule must be an object")
            rule_id, section, parent = _rule_identity(raw_rule.get("rule_id"))
            rule_text = _required_text(raw_rule.get("rule_text"), "rule_text")
            snippets = raw_rule.get("snippets", [])
            if not isinstance(snippets, list):
                raise ValueError("rule snippets must be a list")
            references_json = json.dumps(
                {"source_provenance": references, "snippets": snippets},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            candidate = (section, parent or "", rule_text)
            previous = rules.get(rule_id)
            if previous is not None and previous != candidate:
                raise ValueError(f"ambiguous rule definition: {rule_id}")
            rules[rule_id] = candidate
            connection.execute(
                """
                INSERT OR IGNORE INTO regulation_rules
                    (rule_id, section_number, parent_rule_id, category,
                     rule_text, references_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (rule_id, section, parent, category_name, rule_text, references_json),
            )

    for category in categories:
        category_name = _required_text(category.get("category"), "category")
        values = category.get("values", [])
        if not isinstance(values, list):
            raise ValueError("policy values must be a list")
        for raw_fact in values:
            if not isinstance(raw_fact, Mapping):
                raise ValueError("policy fact must be an object")
            source_rule_id = _required_text(raw_fact.get("source_rule_id"), "source_rule_id")
            if source_rule_id not in rules:
                raise ValueError(f"policy fact references unknown rule: {source_rule_id}")
            condition = _required_text(
                raw_fact.get("condition") or raw_fact.get("comparator"), "condition"
            )
            operator = _OPERATORS.get(condition, condition)
            fact_value = _value(raw_fact.get("value"), "value")
            context = raw_fact.get("context") or raw_fact.get("procedure") or raw_fact.get("basis")
            refs = category_refs[category_name]
            if not refs:
                raise ValueError("policy fact has no provenance")
            cursor = connection.execute(
                """
                INSERT INTO policy_facts
                    (category, fact_key, operator, value, unit, condition,
                     context, source_rule_id, verification_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    category_name,
                    _as_text(raw_fact.get("label")),
                    operator,
                    fact_value,
                    _as_text(raw_fact.get("unit")),
                    condition,
                    _as_text(context),
                    source_rule_id,
                    _as_text(raw_fact.get("verification_status")),
                ),
            )
            fact_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO policy_fact_provenance (fact_id, provenance_id) VALUES (?, ?)",
                [(fact_id, provenance_id) for provenance_id, _ in refs],
            )


def _load_program_requirements(
    path: Path,
    connection: sqlite3.Connection,
    provenance_cache: dict[tuple[Any, ...], int],
) -> None:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, list):
        raise ValueError("program requirements JSON must be a list")
    for raw_requirement in document:
        if not isinstance(raw_requirement, Mapping):
            raise ValueError("program requirement must be an object")
        program = _required_text(raw_requirement.get("program"), "program")
        requirement_type = _required_text(
            raw_requirement.get("requirement_type"), "requirement_type"
        )
        operator = _required_text(raw_requirement.get("operator"), "operator")
        unit = _required_text(raw_requirement.get("unit"), "unit")
        value = _value(raw_requirement.get("value"), "value")
        references = raw_requirement.get("source_provenance")
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(reference, Mapping) for reference in references)
        ):
            raise ValueError("program requirement is missing source provenance")
        refs = _provenance_ids(
            connection,
            references,
            program,
            _source_document_key({"source": path.name}),
            provenance_cache,
            allowed_categories=_SUPPLEMENTAL_DOCUMENT_CATEGORIES,
        )
        if not refs:
            raise ValueError("program requirement has no provenance")
        cursor = connection.execute(
            """
            INSERT INTO program_requirements
                (program_code, requirement_type, operator, value, unit)
            VALUES (?, ?, ?, ?, ?)
            """,
            (program, requirement_type, operator, value, unit),
        )
        requirement_id = int(cursor.lastrowid)
        connection.executemany(
            """
            INSERT INTO program_requirement_provenance
                (requirement_id, provenance_id)
            VALUES (?, ?)
            """,
            [(requirement_id, provenance_id) for provenance_id, _ in refs],
        )


def load_supplemental_jsons_to_sqlite(
    institution_policy_path: str | Path,
    program_requirements_path: str | Path,
    output_db_path: str | Path,
) -> None:
    """Load the two explicit supplemental authority documents into an existing DB."""
    policy_path = Path(institution_policy_path)
    requirements_path = Path(program_requirements_path)
    output_path = Path(output_db_path)
    if not policy_path.is_file() or not requirements_path.is_file():
        raise FileNotFoundError("supplemental authority input is missing")
    with closing(sqlite3.connect(str(output_path))) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        provenance_cache: dict[tuple[Any, ...], int] = {}
        _load_policy(policy_path, connection, provenance_cache)
        _load_program_requirements(requirements_path, connection, provenance_cache)
        connection.commit()


__all__ = ["load_supplemental_jsons_to_sqlite"]
