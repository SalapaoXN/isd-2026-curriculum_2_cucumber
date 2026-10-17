"""Find blank consolidated descriptions recoverable from cached OCR pages."""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any

from rag.structured.loader import _course_codes
from src.pipeline.datasets import DATASET_CONFIG, DatasetConfig, parse_page_spec
from src.pipeline.tools.extraction.engine import CurriculumExtractor
from src.pipeline.tools.merge.consolidator import _courses_from_persisted_description_pages


ROOT = Path(__file__).resolve().parents[3]
OCR_ROOT = ROOT / "data" / "output" / "ocr"
CONSOLIDATED_ROOT = ROOT / "data" / "output" / "consolidated"


def _consolidated_files(config: DatasetConfig) -> list[tuple[str | None, Path]]:
    result = []
    for scope in config.scopes:
        stem = config.key if scope.plan in (None, "gened") else f"{config.key}_{scope.plan}"
        path = CONSOLIDATED_ROOT / f"{stem}_consolidated.json"
        if path.is_file():
            result.append((scope.plan, path))
    return result


def _description_courses(config: DatasetConfig) -> list[dict[str, Any]]:
    pages = set()
    for scope in config.scopes:
        pages.update(parse_page_spec(scope.description_pages))
    if config.shared_description is not None:
        pages.update(parse_page_spec(config.shared_description.pages))
        pages.update(parse_page_spec(config.shared_description.description_pages))

    input_dir = OCR_ROOT / config.key
    extractor = CurriculumExtractor(
        program=config.program,
        plan=(config.shared_description or config.scopes[0]).plan,
    )
    page_records = []
    for path in sorted(input_dir.glob("*_ocr.json")):
        match = re.search(r"_page_(\d+)_ocr\.json$", path.name)
        if match is None or int(match.group(1)) not in pages:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            data = extractor.process_file(path)
        page_records.append((extractor.plan, int(match.group(1)), data))
    return _courses_from_persisted_description_pages(page_records)


def audit_cached_descriptions() -> dict[str, Any]:
    findings = []
    coverage = {}
    for config in DATASET_CONFIG.values():
        pages = set()
        for scope in config.scopes:
            pages.update(parse_page_spec(scope.description_pages))
        if config.shared_description is not None:
            pages.update(parse_page_spec(config.shared_description.pages))
            pages.update(parse_page_spec(config.shared_description.description_pages))
        ocr_files = [
            path
            for path in (OCR_ROOT / config.key).glob("*_ocr.json")
            if (match := re.search(r"_page_(\d+)_ocr\.json$", path.name))
            and int(match.group(1)) in pages
        ]
        desc_courses = _description_courses(config)
        desc_by_code: dict[str, list[dict[str, Any]]] = {}
        for course in desc_courses:
            code = course.get("code")
            if isinstance(code, str) and code.strip():
                desc_by_code.setdefault(code.strip(), []).append(course)

        coverage[config.key] = {
            "description_pages_configured": len(pages),
            "cached_ocr_pages": len(ocr_files),
            "extracted_description_records": len(desc_courses),
        }
        for plan, path in _consolidated_files(config):
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            for course in document.get("courses", []):
                try:
                    codes = _course_codes(course)
                except (TypeError, ValueError):
                    continue
                if any("x" in code.casefold() for code in codes):
                    continue

                if len(codes) > 1:
                    descriptions_by_code = course.get("descriptions_by_code")
                    for code in codes:
                        candidates = desc_by_code.get(code.strip(), [])
                        if len(candidates) != 1:
                            if candidates:
                                findings.append({
                                    "catalog_key": config.catalog_key,
                                    "plan": plan,
                                    "course_code": code,
                                    "status": "ambiguous_ocr_match",
                                    "candidate_count": len(candidates),
                                })
                            continue
                        source = candidates[0]
                        mapped = (
                            descriptions_by_code.get(code)
                            if isinstance(descriptions_by_code, dict)
                            else None
                        )
                        mapped_member = mapped if isinstance(mapped, dict) else {}
                        mismatched_fields = [
                            field
                            for field in ("desc_th", "desc_en")
                            if str(source.get(field) or "").strip()
                            and str(mapped_member.get(field) or "").strip()
                            != str(source.get(field) or "").strip()
                        ]
                        if not mismatched_fields:
                            continue
                        provenance = source.get("source_provenance") or []
                        findings.append({
                            "catalog_key": config.catalog_key,
                            "plan": plan,
                            "course_code": code,
                            "missing_fields": mismatched_fields,
                            "recoverable_fields": mismatched_fields,
                            "status": "unique_exact_code_ocr_match",
                            "source_pages": sorted({
                                item.get("source_page")
                                for item in provenance
                                if isinstance(item, dict)
                                and item.get("document_category") == "description"
                            }),
                        })
                    continue

                code = codes[0]
                missing = [
                    field
                    for field in ("desc_th", "desc_en")
                    if not str(course.get(field) or "").strip()
                ]
                if not missing:
                    continue
                candidates = desc_by_code.get(code.strip(), [])
                if len(candidates) != 1:
                    if candidates:
                        findings.append({
                            "catalog_key": config.catalog_key,
                            "plan": plan,
                            "course_code": code,
                            "missing_fields": missing,
                            "status": "ambiguous_ocr_match",
                            "candidate_count": len(candidates),
                        })
                    continue
                source = candidates[0]
                recoverable = [field for field in missing if str(source.get(field) or "").strip()]
                if not recoverable:
                    continue
                provenance = source.get("source_provenance") or []
                findings.append({
                    "catalog_key": config.catalog_key,
                    "plan": plan,
                    "course_code": code,
                    "missing_fields": missing,
                    "recoverable_fields": recoverable,
                    "status": "unique_exact_code_ocr_match",
                    "source_pages": sorted({
                        item.get("source_page")
                        for item in provenance
                        if isinstance(item, dict)
                        and item.get("document_category") == "description"
                    }),
                })
    return {
        "coverage": coverage,
        "finding_count": len(findings),
        "findings": findings,
    }


if __name__ == "__main__":
    print(json.dumps(audit_cached_descriptions(), ensure_ascii=False, indent=2))
