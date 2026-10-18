"""Fill blank description fields from exact, unique cached-OCR extractions.

This repair path is deterministic: it never rewrites a non-empty description,
never uses fuzzy course-code matching, and retains source-page provenance.
Run without ``--apply`` to preview changes.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.structured.loader import _course_codes
from src.pipeline.datasets import DATASET_CONFIG
from src.pipeline.tools.audit_cached_descriptions import (
    CONSOLIDATED_ROOT,
    OCR_ROOT,
    ROOT,
    _consolidated_files,
    _description_courses,
)
from src.pipeline.tools.extraction.engine import merge_source_provenance


def _unique_sources(courses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for course in courses:
        code = course.get("code")
        if isinstance(code, str) and code.strip() and "x" not in code.casefold():
            grouped.setdefault(code.strip(), []).append(course)
    return {code: records[0] for code, records in grouped.items() if len(records) == 1}


def repair_document(
    document: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill only empty language fields from a unique exact-code OCR record."""
    repairs = []
    for course in document.get("courses", []):
        try:
            codes = _course_codes(course)
        except (TypeError, ValueError):
            continue
        if any("x" in code.casefold() for code in codes):
            continue

        if len(codes) > 1:
            member_sources = {code: sources.get(code.strip()) for code in codes}
            if any(source is None for source in member_sources.values()):
                continue
            descriptions_by_code = {
                code: {
                    "desc_th": source.get("desc_th"),
                    "desc_en": source.get("desc_en"),
                    "source_provenance": source.get("source_provenance") or [],
                }
                for code, source in member_sources.items()
            }
            if not any(
                value.get("desc_th") or value.get("desc_en")
                for value in descriptions_by_code.values()
            ):
                continue
            existing = course.get("descriptions_by_code")
            if existing is not None:
                # Keep a reviewed mapping untouched, even if cached OCR differs.
                continue
            course["descriptions_by_code"] = descriptions_by_code
            pages = sorted({
                entry.get("source_page")
                for source in member_sources.values()
                for entry in source.get("source_provenance", [])
                if isinstance(entry, dict)
                and entry.get("document_category") == "description"
                and entry.get("source_page") is not None
            })
            repairs.append({
                "course_code": course.get("code"),
                "fields": ["descriptions_by_code"],
                "source_pages": pages,
            })
            continue

        code = codes[0]
        source = sources.get(code.strip())
        if source is None:
            continue
        fields = [
            field
            for field in ("desc_th", "desc_en")
            if not str(course.get(field) or "").strip()
            and str(source.get(field) or "").strip()
        ]
        if not fields:
            continue
        for field in fields:
            course[field] = source[field]
        course["source_provenance"] = merge_source_provenance(course, source)
        pages = sorted({
            entry.get("source_page")
            for entry in source.get("source_provenance", [])
            if isinstance(entry, dict)
            and entry.get("document_category") == "description"
        })
        repairs.append({"course_code": code, "fields": fields, "source_pages": pages})
    return repairs


def _target_files(dataset_key: str, plan: str | None, consolidated: Path) -> list[Path]:
    stem = consolidated.name.removesuffix("_consolidated.json")
    return [
        consolidated,
        ROOT / "data" / "output" / "final" / f"{stem}_final.json",
        ROOT / "data" / "output" / "canonical" / f"{stem}_final.json",
    ]


def collect_repairs() -> tuple[dict[Path, dict[str, Any]], dict[str, Any]]:
    documents: dict[Path, dict[str, Any]] = {}
    source_cache: dict[str, dict[str, dict[str, Any]]] = {}
    report: dict[str, Any] = {"datasets": {}, "files": []}

    for key, config in DATASET_CONFIG.items():
        ocr_dir = OCR_ROOT / key
        if not ocr_dir.is_dir():
            continue
        extracted_sources = _description_courses(config)
        sources = _unique_sources(extracted_sources)
        source_cache[key] = sources
        dataset_summary = {
            "cached_description_records": len(extracted_sources),
            "unique_course_codes": len(sources),
            "patched_records": 0,
            "patched_fields": 0,
        }

        for plan, consolidated_path in _consolidated_files(config):
            for path in _target_files(key, plan, consolidated_path):
                if not path.is_file():
                    continue
                document = json.loads(path.read_text(encoding="utf-8-sig"))
                repairs = repair_document(document, sources)
                if not repairs:
                    continue
                documents[path] = document
                dataset_summary["patched_records"] += len(repairs)
                dataset_summary["patched_fields"] += sum(
                    len(repair["fields"]) for repair in repairs
                )
                report["files"].append({
                    "path": str(path.relative_to(ROOT)),
                    "plan": plan,
                    "repairs": repairs,
                })
        report["datasets"][key] = dataset_summary
    return documents, report


def _write_atomically(path: Path, document: dict[str, Any]) -> None:
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def apply_repairs(documents: dict[Path, dict[str, Any]]) -> Path:
    backup_root = (
        Path(os.getenv("TEMP", tempfile.gettempdir()))
        / "opencode"
        / f"cached-description-backup-{datetime.now():%Y%m%d-%H%M%S}"
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    for path, document in documents.items():
        backup = backup_root / path.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(path.read_bytes())
        _write_atomically(path, document)
    return backup_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write repairs to consolidated, final, and canonical JSON (default: preview)",
    )
    args = parser.parse_args(argv)
    documents, report = collect_repairs()
    report["mode"] = "apply" if args.apply else "preview"
    if args.apply and documents:
        report["backup_directory"] = str(apply_repairs(documents))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
