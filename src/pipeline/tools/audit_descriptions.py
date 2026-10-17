"""Read-only audit of canonical course descriptions against runtime SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag.structured.loader import (
    _course_codes,
    _course_identity_discriminator,
    _member_value,
    _normalized_course_code,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANONICAL_DIR = ROOT / "data" / "output" / "canonical"
DEFAULT_DATABASE = ROOT / "cucumber_outputs" / "runtime" / "curriculum.db"


def normalize_code(value: Any) -> str:
    return str(value or "").strip().casefold()


def _audit_identity_discriminator(value: Any) -> str:
    """Group fact variants by code while keeping distinct placeholder courses apart."""
    identity = str(value or "")
    if identity.startswith("placeholder:"):
        return identity.split("|variant:", 1)[0]
    return ""


def _catalog_key(document: dict[str, Any], path: Path) -> str:
    catalog = document.get("catalog")
    if isinstance(catalog, dict) and catalog.get("catalog_key"):
        return str(catalog["catalog_key"]).strip()
    if document.get("catalog_key"):
        return str(document["catalog_key"]).strip()
    program = str(document.get("program", "")).strip().lower()
    year = str(document.get("academic_year") or document.get("year") or "").strip()
    plan = str(document.get("plan", "")).strip().lower()
    if program and year:
        return "-".join(part for part in (program, year, plan) if part)
    legacy = re.match(r"([a-z]+)[_-](?=\d)|([a-z]+)(\d{4})", path.stem, re.IGNORECASE)
    if legacy:
        legacy_program = (legacy.group(1) or legacy.group(2)).lower()
        legacy_year = legacy.group(3) or re.search(r"\d{4}", path.stem).group(0)
        legacy_plan = plan or ("coop" if re.search(r"coop", path.stem, re.IGNORECASE) else "")
        return "-".join(part for part in (legacy_program, legacy_year, legacy_plan) if part)
    return path.stem


def _provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories: set[str] = set()
    files: set[str] = set()
    pages: set[str] = set()
    for record in records:
        raw = record.get("source_provenance", record.get("provenance", []))
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            continue
        if not raw and any(
            record.get(field)
            for field in ("document_category", "source_filename", "source_page")
        ):
            raw = [record]
        for item in raw:
            if not isinstance(item, dict):
                continue
            category = item.get("document_category") or item.get("category")
            filename = item.get("source_filename") or item.get("file")
            page = item.get("document_page", item.get("source_page", item.get("page")))
            if category:
                categories.add(str(category))
            if filename:
                files.add(str(filename))
            if page is not None:
                pages.add(str(page))
    return {"categories": sorted(categories), "files": sorted(files), "pages": sorted(pages)}


def _descriptions(records: list[dict[str, Any]], th: str = "desc_th", en: str = "desc_en") -> dict[str, str]:
    def value(field: str) -> str:
        vals = {str(record.get(field) or "").strip() for record in records}
        return next(iter(vals)) if len(vals) == 1 else ""

    return {"desc_th": value(th), "desc_en": value(en)}


def _member_description(
    course: dict[str, Any], code: str, field: str, index: int, count: int
) -> Any:
    descriptions = course.get("descriptions_by_code")
    if descriptions is None:
        return _member_value(course.get(field), index, count)
    if not isinstance(descriptions, dict):
        return None
    code_key = normalize_code(code)
    members = [
        value
        for key, value in descriptions.items()
        if normalize_code(key) == code_key
    ]
    if len(members) != 1 or not isinstance(members[0], dict):
        return None
    return members[0].get(field)


def _source_freshness(connection: sqlite3.Connection, canonical_files: list[Path]) -> dict[str, Any]:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "semantic_index_sources" not in tables:
        return {"available": False, "mismatches": []}
    columns = {row[1] for row in connection.execute("PRAGMA table_info(semantic_index_sources)")}
    rows = connection.execute("SELECT * FROM semantic_index_sources").fetchall()
    names = [item[0] for item in connection.execute("SELECT * FROM semantic_index_sources LIMIT 0").description]
    sources = [dict(zip(names, row)) for row in rows]
    mismatches = []
    for path in canonical_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        def is_this_source(source: dict[str, Any]) -> bool:
            if str(source.get("source_filename", "")).casefold() == path.name.casefold():
                return True
            suffixes = ("/" + path.name.casefold(), "\\" + path.name.casefold())
            return any(
                str(source.get(key, "")).casefold().endswith(suffixes)
                for key in ("source_file_identity", "source_uri", "source_locator")
            )

        matches = [source for source in sources if is_this_source(source)]
        for source in matches:
            stored = next((source[key] for key in ("source_json_fingerprint", "source_fingerprint", "fingerprint", "content_hash", "sha256") if key in columns and source.get(key)), None)
            if stored:
                normalized = str(stored).removeprefix("sha256:").lower()
                if normalized != digest:
                    mismatches.append({"file": str(path), "stored_fingerprint": str(stored), "canonical_sha256": digest})
    return {"available": True, "mismatches": mismatches, "source_rows": len(sources)}


def audit(canonical_dir: Path, database: Path) -> dict[str, Any]:
    canonical: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    files = sorted(canonical_dir.glob("*.json"))
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        key = _catalog_key(document, path)
        courses = document.get("courses", [])
        if isinstance(courses, list):
            for course in courses:
                if not isinstance(course, dict):
                    continue
                codes = _course_codes(course)
                for index, code in enumerate(codes):
                    name_th = _member_value(course.get("name_th"), index, len(codes), True)
                    name_en = _member_value(course.get("name_en"), index, len(codes), True)
                    item = {
                        **course,
                        "desc_th": _member_description(
                            course, code, "desc_th", index, len(codes)
                        ),
                        "desc_en": _member_description(
                            course, code, "desc_en", index, len(codes)
                        ),
                        "_file": path.name,
                    }
                    identity = _course_identity_discriminator(code, name_en, name_th)
                    canonical[(key, _normalized_course_code(code), identity)].append(item)

    connection = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        query = """SELECT cat.catalog_key, c.course_code, c.course_identity_discriminator,
                          c.description_th, c.description_en, c.course_id
                   FROM courses c JOIN catalogs cat ON cat.catalog_id=c.catalog_id"""
        database_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in connection.execute(query):
            identity = (
                str(row["catalog_key"] or ""),
                _normalized_course_code(row["course_code"]),
                _audit_identity_discriminator(row["course_identity_discriminator"]),
            )
            database_rows[identity].append(dict(row))
        database_provenance: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        provenance_query = """SELECT cat.catalog_key, c.course_code,
                                     c.course_identity_discriminator,
                                     p.document_category, p.source_filename,
                                     p.source_page, p.document_page
                              FROM courses c
                              JOIN catalogs cat ON cat.catalog_id=c.catalog_id
                              LEFT JOIN course_provenance cp ON cp.course_id=c.course_id
                              LEFT JOIN provenance p ON p.provenance_id=cp.provenance_id"""
        for row in connection.execute(provenance_query):
            if row["document_category"] is None and row["source_filename"] is None:
                continue
            key = (
                str(row["catalog_key"] or ""),
                _normalized_course_code(row["course_code"]),
                _audit_identity_discriminator(row["course_identity_discriminator"]),
            )
            database_provenance[key].append({
                "document_category": row["document_category"],
                "source_filename": row["source_filename"],
                "source_page": row["source_page"],
                "document_page": row["document_page"],
            })
        all_keys = sorted(set(canonical) | set(database_rows))
        comparisons = []
        counts: dict[str, int] = defaultdict(int)
        missing_with_description_source = 0
        for key in all_keys:
            left, right = canonical.get(key, []), database_rows.get(key, [])
            canonical_values = {
                tuple(_descriptions([item]).values()) for item in left
            }
            database_values = {
                tuple(_descriptions([item], "description_th", "description_en").values())
                for item in right
            }
            if len(canonical_values) > 1 or len(database_values) > 1:
                status = "ambiguous"
            elif not left:
                status = "database_only"
            elif not right:
                status = "canonical_only"
            else:
                a, b = _descriptions(left), _descriptions(right, "description_th", "description_en")
                if not any(a.values()) and not any(b.values()):
                    status = "both_missing"
                elif a != b:
                    status = "different"
                else:
                    continue
            counts[status] += 1
            a = _descriptions(left) if left else {"desc_th": "", "desc_en": ""}
            b = _descriptions(right, "description_th", "description_en") if right else {"desc_th": "", "desc_en": ""}
            canonical_provenance = _provenance(left)
            database_provenance_summary = _provenance(database_provenance[key])
            description_source_present = (
                "description" in canonical_provenance["categories"]
                or "description" in database_provenance_summary["categories"]
            )
            if status == "both_missing" and description_source_present:
                missing_with_description_source += 1
            comparisons.append({"catalog_key": key[0], "course_code": key[1], "identity_discriminator": key[2], "status": status,
                "source_status": "unknown_without_source_OCR",
                "description_source_present": description_source_present,
                "canonical": {**a, "variants": sorted({json.dumps(_descriptions([item]), ensure_ascii=False, sort_keys=True) for item in left}), "provenance": canonical_provenance, "files": sorted({str(x.get('_file')) for x in left})},
                "database": {**b, "variants": sorted({json.dumps(_descriptions([item], "description_th", "description_en"), ensure_ascii=False, sort_keys=True) for item in right}), "provenance": database_provenance_summary}})
        return {"summary": {"canonical_courses": len(canonical), "database_courses": len(database_rows), "mismatches": len(comparisons), "by_status": dict(sorted(counts.items())), "both_missing_with_description_source": missing_with_description_source},
                "source_status": "unknown_without_source_OCR", "freshness": _source_freshness(connection, files), "courses": comparisons}
    finally:
        connection.close()


def has_strict_failures(result: dict[str, Any]) -> bool:
    """Whether discrepancies or source-backed blanks should block a build."""
    if result.get("freshness", {}).get("mismatches"):
        return True
    return any(
        row.get("status") != "both_missing" or row.get("description_source_present")
        for row in result.get("courses", [])
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero for stale index sources, discrepancies, ambiguity, or blanks with description provenance",
    )
    args = parser.parse_args(argv)
    result = audit(args.canonical_dir, args.database)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(args.strict and has_strict_failures(result))


if __name__ == "__main__":
    sys.exit(main())
