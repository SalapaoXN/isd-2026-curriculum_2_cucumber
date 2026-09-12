"""Build a deterministic Lab 8B curriculum conversion report from existing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "submission" / "curriculum.conversion.json"
CORRECTOR_SOURCE = ROOT / "llm_spell_corrector.py"

CANONICAL_FILES = {
    "coop": ROOT / "outputs" / "consolidated" / "it" / "coop" / "full" / "merged_it_coop_full.json",
    "no_coop": ROOT / "outputs" / "consolidated" / "it" / "no_coop" / "full" / "merged_it_no_coop_full.json",
}
CORRECTION_LOGS = {
    "coop": ROOT / "work" / "spell_correction" / "it" / "names_v2" / "merged_it_coop_full_corrections.json",
    "no_coop": ROOT / "work" / "spell_correction" / "it" / "names_v2" / "merged_it_no_coop_full_corrections.json",
}
SUBMISSION_FILE = ROOT / "submission" / "curriculum.json"
VERIFY_FILE = ROOT / "submission" / "verify.json"
EDITABLE_FIELDS = ("name_th", "name_en")
CHECK_IDS = ("CHK1", "CHK2", "CHK3", "CHK4", "CHK5", "CHK6", "CHK7")


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {relative_path(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def courses_from_canonical(document: dict[str, Any]) -> list[dict[str, Any]]:
    courses = document.get("courses")
    if not isinstance(courses, list) or not all(isinstance(item, dict) for item in courses):
        raise ValueError("Canonical artifact must contain a list of course objects")
    return courses


def courses_from_submission(document: dict[str, Any]) -> list[dict[str, Any]]:
    plans = document.get("plans")
    if not isinstance(plans, list):
        raise ValueError("Submission artifact must contain a plans list")
    return [course for plan in plans for course in plan.get("courses", [])]


def verify_corrector_contract(source: str) -> None:
    required_fragments = (
        'MODEL = "gemini-3.5-flash-lite"',
        "BATCH_SIZE = 50",
        'TEXT_FIELDS_ORDER = ("name_th", "name_en")',
        "key = (field, before)",
        'config={"temperature": 0}',
        '"unit_index", "field", "text"',
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise ValueError(f"Spell-corrector contract is not independently supported: {missing}")


def correction_log_summary(path: Path) -> dict[str, Any]:
    entries = load_json(path)
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise ValueError(f"Correction log must be a JSON array of objects: {relative_path(path)}")

    for entry in entries:
        required = {"course_code", "field", "before", "after"}
        if set(entry) != required:
            raise ValueError(f"Unexpected correction-log fields in {relative_path(path)}")
        if entry["field"] not in EDITABLE_FIELDS:
            raise ValueError(f"Immutable correction field found in {relative_path(path)}")
        if entry["before"] == entry["after"]:
            raise ValueError(f"Unchanged correction entry found in {relative_path(path)}")

    keys = {(entry["field"], entry["before"]) for entry in entries}
    return {
        "path": relative_path(path),
        "sha256": sha256(path),
        "actual_correction_count": len(entries),
        "unique_correction_units": len(keys),
        "fields": sorted({entry["field"] for entry in entries}),
        "course_codes_with_corrections": len({entry["course_code"] for entry in entries}),
        "entries": entries,
    }


def canonical_summary(path: Path) -> dict[str, Any]:
    document = load_json(path)
    courses = courses_from_canonical(document)
    if document.get("program") != "IT":
        raise ValueError(f"Unexpected canonical program in {relative_path(path)}")
    if document.get("plan") not in CANONICAL_FILES:
        raise ValueError(f"Unexpected canonical plan in {relative_path(path)}")
    if document.get("total_courses") != len(courses):
        raise ValueError(f"Canonical total_courses mismatch in {relative_path(path)}")

    target = [course for course in courses if course.get("code") == "06016414"]
    if len(target) != 1:
        raise ValueError(f"Expected one 06016414 record in {relative_path(path)}")
    duplicate_target = [course for course in courses if course.get("code") == "06016418"]
    return {
        "path": relative_path(path),
        "sha256": sha256(path),
        "source": document.get("source"),
        "description": document.get("description"),
        "program": document.get("program"),
        "plan": document.get("plan"),
        "record_count": len(courses),
        "total_courses": document.get("total_courses"),
        "source_provenance_record_count": sum(
            isinstance(course.get("source_provenance"), list)
            and len(course["source_provenance"]) > 0
            for course in courses
        ),
        "06016414_final_name_en": target[0].get("name_en"),
        "06016418_final_record_count": len(duplicate_target),
    }


def submission_summary(path: Path) -> dict[str, Any]:
    document = load_json(path)
    if document.get("program") != "IT":
        raise ValueError("Submission program must be IT")
    plans = document.get("plans")
    if not isinstance(plans, list) or [plan.get("plan") for plan in plans] != ["coop", "no_coop"]:
        raise ValueError("Submission must contain coop and no_coop plans in that order")

    plan_counts = {}
    for plan in plans:
        courses = plan.get("courses")
        if not isinstance(courses, list):
            raise ValueError("Submission plan courses must be lists")
        plan_counts[plan["plan"]] = {
            "course_count": len(courses),
            "declared_total_courses": plan.get("total_courses"),
        }
    return {
        "path": relative_path(path),
        "sha256": sha256(path),
        "program": document.get("program"),
        "plan_counts": plan_counts,
        "total_placement_records": sum(item["course_count"] for item in plan_counts.values()),
    }


def manual_normalization_summary(
    log_summaries: dict[str, dict[str, Any]],
    canonical_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    changes = []
    for plan in ("coop", "no_coop"):
        candidates = [
            entry
            for entry in log_summaries[plan]["entries"]
            if entry["course_code"] == "06016414" and entry["field"] == "name_en"
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one logged 06016414 name correction for {plan}")
        entry = candidates[0]
        final_name = canonical_summaries[plan]["06016414_final_name_en"]
        if entry["after"] == final_name:
            raise ValueError("Manual normalization delta is not present")
        if final_name != "NOSQL DATABASE SYSTEMS":
            raise ValueError(f"Unexpected final 06016414 name for {plan}: {final_name!r}")
        changes.append(
            {
                "plan": plan,
                "course_code": "06016414",
                "field": "name_en",
                "intermediate_corrected_value": entry["after"],
                "final_value": final_name,
                "evidence": {
                    "correction_log": log_summaries[plan]["path"],
                    "canonical_artifact": canonical_summaries[plan]["path"],
                },
                "historical_operator_or_timestamp": "not present in available artifacts",
            }
        )
    return changes


def deterministic_repair_summary(verify: dict[str, Any], submission: dict[str, Any]) -> dict[str, Any]:
    assertions = {
        item.get("name"): item
        for item in verify.get("regression_assertions", [])
        if isinstance(item, dict)
    }
    assertion = assertions.get("06016418_duplicate_removed_from_canonical_source")
    if assertion is None or assertion.get("status") != "PASS":
        raise ValueError("Duplicate-removal regression assertion is not supported")

    plans = {
        plan["plan"]: plan
        for plan in submission["plans"]
    }
    repairs = []
    for plan_name in ("coop", "no_coop"):
        records = plans[plan_name]["courses"]
        retained = [course for course in records if course.get("code") == "06016418"]
        if len(retained) != 1:
            raise ValueError(f"Expected one retained 06016418 record in {plan_name}")
        repairs.append(
            {
                "plan": plan_name,
                "course_code": "06016418",
                "removed_duplicate_occurrences": 1,
                "retained_occurrences_in_final_canonical": len(retained),
                "final_canonical_record_count": len(records),
                "evidence": {
                    "verification_file": relative_path(VERIFY_FILE),
                    "regression_assertion": assertion,
                },
                "pre_repair_snapshot": "not independently reconstructable from available artifacts",
            }
        )
    return {
        "event_type": "deterministic_repair",
        "status": "supported_final_state",
        "repairs": repairs,
        "provenance_preserved_in_final_canonical": True,
    }


def verification_summary(verify: dict[str, Any]) -> dict[str, Any]:
    checks = verify.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("verify.json must contain checks")
    check_results = {}
    for check_id in CHECK_IDS:
        check = checks.get(check_id)
        if not isinstance(check, dict):
            raise ValueError(f"Missing {check_id} from verify.json")
        check_results[check_id] = {
            "status": check.get("status"),
            "finding_count": len(check.get("findings", [])),
            "exception_count": len(check.get("exceptions", [])),
        }

    chk7 = checks["CHK7"]
    source_exceptions = [
        item
        for item in chk7.get("exceptions", [])
        if item.get("classification") == "source_condition_human_review"
    ]
    return {
        "event_type": "validation",
        "status": "supported",
        "source_file": relative_path(VERIFY_FILE),
        "source_sha256": sha256(VERIFY_FILE),
        "summary_status": verify.get("summary", {}).get("status"),
        "status_counts": verify.get("summary", {}).get("status_counts"),
        "checks": check_results,
        "CHK7_source_condition_human_review": {
            "count": len(source_exceptions),
            "totals": chk7.get("evidence", {}).get("semester_totals_lab_adjusted"),
            "provenance_retained": all(
                isinstance(item.get("provenance"), list) and item["provenance"]
                for item in source_exceptions
            ),
        },
        "human_review_reasons": verify.get("summary", {}).get("human_review_reasons", []),
    }


def build_report() -> dict[str, Any]:
    corrector_source = CORRECTOR_SOURCE.read_text(encoding="utf-8")
    verify_corrector_contract(corrector_source)

    canonical = {plan: canonical_summary(path) for plan, path in CANONICAL_FILES.items()}
    logs = {plan: correction_log_summary(path) for plan, path in CORRECTION_LOGS.items()}
    submission = load_json(SUBMISSION_FILE)
    submission_info = submission_summary(SUBMISSION_FILE)
    verify = load_json(VERIFY_FILE)

    all_units = {
        (entry["field"], entry["before"])
        for plan in logs.values()
        for entry in plan["entries"]
    }
    shared_units = {
        (entry["field"], entry["before"])
        for entry in logs["coop"]["entries"]
    } & {
        (entry["field"], entry["before"])
        for entry in logs["no_coop"]["entries"]
    }

    return {
        "report": "Lab 8B curriculum conversion and repair history",
        "program": "IT",
        "repair_round_count": 3,
        "determinism": {
            "serialization": "UTF-8 JSON, 4-space indentation, trailing newline, stable event/order construction",
            "timestamps": "omitted because unavailable",
            "unverifiable_history": "omitted or explicitly marked where no pre-change artifact exists",
        },
        "stages": [
            {
                "stage": "ocr_extraction_to_canonical_merged_data",
                "event_type": "extraction",
                "status": "supported_by_canonical_metadata",
                "artifacts": [canonical["coop"], canonical["no_coop"]],
                "evidence": "Each canonical artifact identifies itself as OCR-extracted curriculum data and retains source provenance.",
            },
            {
                "stage": "gemini_name_only_correction",
                "event_type": "automated_correction",
                "status": "supported_by_corrector_source_and_logs",
                "model": "gemini-3.5-flash-lite",
                "temperature": 0,
                "editable_fields": list(EDITABLE_FIELDS),
                "dedup_key": ["field_name", "exact_before_text"],
                "unique_unit_batch_size": 50,
                "multi_file": True,
                "correction_logs": {
                    "coop": {
                        key: value
                        for key, value in logs["coop"].items()
                        if key != "entries"
                    },
                    "no_coop": {
                        key: value
                        for key, value in logs["no_coop"].items()
                        if key != "entries"
                    },
                },
                "combined_logged_unique_units": len(all_units),
                "shared_logged_units_across_plans": len(shared_units),
                "evidence": {
                    "corrector_source": relative_path(CORRECTOR_SOURCE),
                    "contract_verified": True,
                    "counts_are_lengths_of_actual_correction_log_arrays": True,
                },
            },
            {
                "stage": "manual_normalization",
                "event_type": "manual_normalization",
                "status": "supported_by_correction_log_to_canonical_delta",
                "changes": manual_normalization_summary(logs, canonical),
            },
            {
                "stage": "deterministic_data_quality_repair",
                **deterministic_repair_summary(verify, submission),
            },
            verification_summary(verify),
            {
                "stage": "CHK7_source_condition_review",
                "event_type": "human_review_required",
                "status": "required",
                "rule": "9-22 credits for a fixed normal semester",
                "classification": "source_condition_human_review",
                "action_taken": "none; totals remain calculated values and no credit exception was invented",
                "evidence": "See validation stage and submission/verify.json CHK7 provenance.",
            },
        ],
        "final_artifact": submission_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report()
    serialized = json.dumps(report, ensure_ascii=False, indent=4) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(f"wrote {args.output.as_posix()}")
    print(f"coop_corrections={report['stages'][1]['correction_logs']['coop']['actual_correction_count']}")
    print(f"no_coop_corrections={report['stages'][1]['correction_logs']['no_coop']['actual_correction_count']}")
    print(f"final_placements={report['final_artifact']['total_placement_records']}")


if __name__ == "__main__":
    main()
