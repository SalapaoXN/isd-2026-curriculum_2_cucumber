"""Load policy/rule artifacts into the runtime curriculum database."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .loader import _link_provenance, _provenance_ids


def _read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _collect_rules(policy: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for category in (policy or {}).get("categories", []):
        evidence = category.get("evidence") or {}
        provenance = evidence.get("source_provenance") or []
        for supporting in evidence.get("supporting_rule_text", []):
            rule_id = supporting.get("rule_id")
            rule_text = str(supporting.get("rule_text") or "").strip()
            if not rule_id or not rule_text:
                continue
            current = records.get(rule_id)
            if current and current["rule_text"] != rule_text:
                raise ValueError(f"Conflicting policy text for {rule_id}")
            if current is None:
                suffix = rule_id.split(":", 1)[-1]
                section_number = str(supporting.get("section_number") or suffix)
                parent = section_number.rsplit(".", 1)[0] if "." in section_number else None
                records[rule_id] = {
                    "rule_id": rule_id,
                    "section_number": section_number,
                    "parent_rule_id": f"rule:{parent}" if parent else None,
                    "category": category.get("category") or "Academic rules",
                    "rule_text": rule_text,
                    "references": {"snippets": supporting.get("snippets") or [],
                                   "source_provenance": []},
                }
                current = records[rule_id]
            known = {json.dumps(item, ensure_ascii=False, sort_keys=True)
                     for item in current["references"]["source_provenance"]}
            for item in provenance:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if key not in known:
                    known.add(key)
                    current["references"]["source_provenance"].append(item)
    return records


def _provenance_cache(connection: sqlite3.Connection) -> dict[tuple[Any, ...], int]:
    return {
        tuple(row[1:]): int(row[0])
        for row in connection.execute(
            "SELECT provenance_id, source_document_key, program, source_filename, source_page, "
            "document_page, document_category, source_uri, source_locator, excerpt FROM provenance"
        )
    }


def load_rule_supplements(
    database_path: str | Path,
    institution_policy_path: str | Path | None,
    program_requirements_path: str | Path | None,
) -> dict[str, int]:
    """Refresh rule/policy tables using the canonical JSON build artifacts.

    The JSON artifacts are build inputs; runtime rule search reads the resulting
    ``regulation_rules`` and ``policy_facts`` rows from SQLite.
    """
    database_path = Path(database_path)
    policy = _read_json(Path(institution_policy_path) if institution_policy_path else None)
    requirements = _read_json(Path(program_requirements_path) if program_requirements_path else None) or []
    rules = _collect_rules(policy if isinstance(policy, dict) else None)

    with closing(sqlite3.connect(str(database_path))) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"regulation_rules", "policy_facts", "policy_fact_provenance",
                    "program_requirements", "program_requirement_provenance", "provenance", "catalogs"}
        missing = sorted(required - tables)
        if missing:
            raise ValueError("runtime database is missing supplement tables: " + ", ".join(missing))

        connection.execute("DELETE FROM policy_fact_provenance")
        connection.execute("DELETE FROM program_requirement_provenance")
        connection.execute("DELETE FROM policy_facts")
        connection.execute("DELETE FROM program_requirements")
        connection.execute("DELETE FROM regulation_rules")
        provenance_cache = _provenance_cache(connection)

        for rule in rules.values():
            connection.execute(
                "INSERT INTO regulation_rules "
                "(rule_id, section_number, parent_rule_id, category, rule_text, references_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (rule["rule_id"], rule["section_number"], rule["parent_rule_id"],
                 rule["category"], rule["rule_text"],
                 json.dumps(rule["references"], ensure_ascii=False, sort_keys=True)),
            )

        for category in (policy or {}).get("categories", []) if isinstance(policy, dict) else []:
            evidence = category.get("evidence") or {}
            provenance = evidence.get("source_provenance") or []
            for fact in category.get("values", []):
                rule_id = fact.get("source_rule_id")
                value = _number(fact.get("value"))
                if rule_id not in rules or value is None:
                    continue
                condition = str(fact.get("condition") or "")
                operator = {"below": "<", "at_least": ">=", "at_most": "<=",
                            "above": ">", "equal": "="}.get(condition, condition or "=")
                cursor = connection.execute(
                    "INSERT INTO policy_facts "
                    "(category, fact_key, operator, value, unit, condition, context, source_rule_id, verification_status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (category.get("category") or "Academic rules", fact.get("label"), operator,
                     value, fact.get("unit"), condition, fact.get("source_snippet"), rule_id,
                     "verified" if any(bool(item.get("source_verified")) for item in provenance) else None),
                )
                provenance_ids = _provenance_ids(
                    connection, provenance, "RULE", None, provenance_cache,
                    allowed_categories={"rule"},
                )
                _link_provenance(connection, "policy_fact_provenance", "fact_id",
                                 int(cursor.lastrowid), provenance_ids)

        for requirement in requirements:
            if not isinstance(requirement, dict):
                continue
            catalog_key = requirement.get("catalog_key")
            catalog = connection.execute(
                "SELECT catalog_id FROM catalogs WHERE catalog_key=? ORDER BY catalog_id LIMIT 1",
                (catalog_key,),
            ).fetchone()
            value = _number(requirement.get("value"))
            if catalog is None or value is None:
                continue
            program = str(requirement.get("program") or "")
            cursor = connection.execute(
                "INSERT INTO program_requirements "
                "(catalog_id, program_code, requirement_type, operator, value, unit) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (int(catalog[0]), program, requirement.get("requirement_type") or "requirement",
                 requirement.get("operator") or "=", value, requirement.get("unit") or ""),
            )
            provenance_ids = _provenance_ids(
                connection, requirement.get("source_provenance"), program, None,
                provenance_cache, allowed_categories={"program_requirement"},
            )
            _link_provenance(connection, "program_requirement_provenance", "requirement_id",
                             int(cursor.lastrowid), provenance_ids)

        connection.commit()

    return {"regulation_rules": len(rules),
            "policy_facts": sum(len(category.get("values", []))
                                 for category in (policy or {}).get("categories", []))
            if isinstance(policy, dict) else 0,
            "program_requirements": len(requirements)}
