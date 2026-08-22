import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List
import Levenshtein


def normalize_str(text: Any) -> str:
    if text is None:
        return ""
    text_str = str(text).strip().lower()
    return re.sub(r"\s+", " ", text_str)


def calculate_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (Levenshtein.distance(s1, s2) / max_len)


def calculate_cer(gt: str, pred: str) -> float:
    if not gt:
        return 0.0
    return Levenshtein.distance(gt, pred) / len(gt)


def calculate_wer(gt: str, pred: str) -> float:
    gt_words = gt.split()
    if not gt_words:
        return 0.0
    return Levenshtein.distance(gt_words, pred.split()) / len(gt_words)


def is_plan_course(course: dict) -> bool:
    year = course.get("year")
    semester = course.get("semester")
    return year not in (None, 0, "") or semester not in (None, 0, "")


SOURCE_PROVENANCE_FIELDS = (
    "program",
    "source_filename",
    "source_page",
    "document_category",
)
VALID_DOCUMENT_CATEGORIES = {"plan", "description", "unknown"}
UNKNOWN_PREDICTION_CATEGORY = "unknown_prediction_category"
PAGE_PROVENANCE_UNAVAILABLE_REASON = (
    "Ground truth records do not contain authoritative source/page provenance."
)


def _coverage_metrics(
    gt_record_count: int,
    prediction_record_count: int,
    matched_count: int,
) -> dict:
    missing_gt_count = gt_record_count - matched_count
    extra_prediction_count = prediction_record_count - matched_count
    precision = (
        matched_count / prediction_record_count if prediction_record_count else 0.0
    )
    recall = matched_count / gt_record_count if gt_record_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "gt_record_count": gt_record_count,
        "prediction_record_count": prediction_record_count,
        "matched_count": matched_count,
        "missing_gt_count": missing_gt_count,
        "extra_prediction_count": extra_prediction_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _normalize_source_provenance(entry: Any) -> dict | None:
    if not isinstance(entry, dict):
        return None
    if any(field not in entry for field in SOURCE_PROVENANCE_FIELDS):
        return None

    program = entry.get("program")
    source_filename = entry.get("source_filename")
    source_page = entry.get("source_page")
    document_category = entry.get("document_category")
    if not isinstance(program, str) or not program.strip():
        return None
    if not isinstance(source_filename, str) or not source_filename.strip():
        return None
    if isinstance(source_page, bool):
        return None
    if isinstance(source_page, str) and source_page.strip().isdigit():
        source_page = int(source_page.strip())
    if not isinstance(source_page, int):
        return None
    if document_category not in VALID_DOCUMENT_CATEGORIES:
        return None

    return {
        "program": program,
        "source_filename": source_filename,
        "source_page": source_page,
        "document_category": document_category,
    }


def _authoritative_source_provenance(record: dict) -> List[dict]:
    entries = record.get("source_provenance")
    if not isinstance(entries, list):
        return []

    result = []
    seen = set()
    for entry in entries:
        normalized = _normalize_source_provenance(entry)
        if normalized is None:
            continue
        identity = tuple(normalized[field] for field in SOURCE_PROVENANCE_FIELDS)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(normalized)
    return result


def _source_identity(entry: dict) -> tuple:
    return tuple(entry[field] for field in SOURCE_PROVENANCE_FIELDS)


def _source_identity_key(entry: dict) -> str:
    return "|".join(str(entry[field]) for field in SOURCE_PROVENANCE_FIELDS)


def _text_stats_for_pairs(pairs: List[tuple[dict, dict]], target_fields: List[str]) -> dict:
    stats = {"cer": 0.0, "wer": 0.0, "count": 0}
    for gt_item, pred_item in pairs:
        for field in target_fields:
            if field not in gt_item:
                continue
            gt_val = normalize_str(gt_item.get(field))
            pred_val = normalize_str(pred_item.get(field))
            stats["cer"] += calculate_cer(gt_val, pred_val)
            stats["wer"] += calculate_wer(gt_val, pred_val)
            stats["count"] += 1
    return stats


def _prediction_category(record: dict) -> str:
    category = record.get("category")
    if category is None or str(category).strip() == "":
        return UNKNOWN_PREDICTION_CATEGORY
    return str(category)


def evaluate_json_structure(
    ground_truth_json: str | Path,
    prediction_json: str | Path,
    target_fields: List[str] = None,
    fuzzy_threshold: float = 0.85,
) -> dict:
    if target_fields is None:
        target_fields = ["code", "name_th", "name_en", "credits", "prerequisite"]

    gt_path = Path(ground_truth_json)
    pred_path = Path(prediction_json)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground Truth file not found: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    with gt_path.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with pred_path.open("r", encoding="utf-8") as f:
        pred_data = json.load(f)

    gt_courses: List[dict] = gt_data.get("courses", [])
    pred_courses: List[dict] = pred_data.get("courses", [])

    # ---- Course alignment by code (exact first, then fuzzy) ----- #
    def code_sim(a: dict, b: dict) -> float:
        return calculate_similarity(normalize_str(a.get("code")), normalize_str(b.get("code")))

    def exact_code(a: dict, b: dict) -> bool:
        return normalize_str(a.get("code")) == normalize_str(b.get("code")) and normalize_str(a.get("code")) != ""

    pairs: List[tuple[dict, dict]] = []
    matched_pred: List[bool] = [False] * len(pred_courses)

    for gt_item in gt_courses:
        for j, pred_item in enumerate(pred_courses):
            if matched_pred[j]:
                continue
            if exact_code(gt_item, pred_item):
                matched_pred[j] = True
                pairs.append((gt_item, pred_item))
                break

    for gt_item in gt_courses:
        if any(g is gt_item for g, _ in pairs):
            continue
        best_idx, best_sim = -1, fuzzy_threshold
        for j, pred_item in enumerate(pred_courses):
            if matched_pred[j]:
                continue
            sim = code_sim(gt_item, pred_item)
            if sim > best_sim:
                best_sim, best_idx = sim, j
        if best_idx >= 0:
            matched_pred[best_idx] = True
            pairs.append((gt_item, pred_courses[best_idx]))

    # ---- Coverage metrics from the existing one-to-one alignment ---- #
    gt_record_count = len(gt_courses)
    prediction_record_count = len(pred_courses)
    matched_count = len(pairs)
    missing_gt_count = gt_record_count - matched_count
    extra_prediction_count = prediction_record_count - matched_count
    precision = (
        matched_count / prediction_record_count if prediction_record_count else 0.0
    )
    recall = matched_count / gt_record_count if gt_record_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    # ---- CER/WER accumulators ---- #
    field_stats = {f: {"cer": 0.0, "wer": 0.0, "count": 0} for f in target_fields}
    page_stats = {"cer": 0.0, "wer": 0.0, "count": 0}
    cat_stats = {
        "plan": {"cer": 0.0, "wer": 0.0, "count": 0},
        "description": {"cer": 0.0, "wer": 0.0, "count": 0},
    }
    field_presence_stats = {
        f: {"gt": 0, "pred": 0, "matched": 0, "matched_pred_missing": 0}
        for f in target_fields
    }

    for gt_item in gt_courses:
        for field in target_fields:
            if field in gt_item:
                field_presence_stats[field]["gt"] += 1
    for pred_item in pred_courses:
        for field in target_fields:
            if field in pred_item:
                field_presence_stats[field]["pred"] += 1

    for gt_item, pred_item in pairs:
        category = "plan" if is_plan_course(gt_item) else "description"
        for field in target_fields:
            if field not in gt_item:
                continue
            if field in pred_item:
                field_presence_stats[field]["matched"] += 1
            else:
                field_presence_stats[field]["matched_pred_missing"] += 1
            gt_val = normalize_str(gt_item.get(field))
            pred_val = normalize_str(pred_item.get(field))
            cer_val = calculate_cer(gt_val, pred_val)
            wer_val = calculate_wer(gt_val, pred_val)

            field_stats[field]["cer"] += cer_val
            field_stats[field]["wer"] += wer_val
            field_stats[field]["count"] += 1

            page_stats["cer"] += cer_val
            page_stats["wer"] += wer_val
            page_stats["count"] += 1

            cat_stats[category]["cer"] += cer_val
            cat_stats[category]["wer"] += wer_val
            cat_stats[category]["count"] += 1

    def average(stat: dict) -> dict:
        n = stat["count"]
        return {
            "cer": round(stat["cer"] / n, 4) if n else 0.0,
            "wer": round(stat["wer"] / n, 4) if n else 0.0,
            "count": n,
        }

    pair_by_gt_id = {id(gt_item): pred_item for gt_item, pred_item in pairs}
    matched_pred_ids = {id(pred_item) for _, pred_item in pairs}

    rubric_field_level = {}
    for field in target_fields:
        presence = field_presence_stats[field]
        rubric_field_level[field] = {
            "text_quality": average(field_stats[field]),
            "presence_coverage": _coverage_metrics(
                presence["gt"], presence["pred"], presence["matched"]
            ),
            "matched_prediction_field_missing_count": presence[
                "matched_pred_missing"
            ],
        }

    category_groups = {}
    gt_category_missing = False
    gt_category_by_id = {}
    for gt_item in gt_courses:
        category = gt_item.get("category")
        if category is None or str(category).strip() == "":
            category = "unknown_gt_category"
            gt_category_missing = True
        else:
            category = str(category)
        gt_category_by_id[id(gt_item)] = category
        group = category_groups.setdefault(
            category, {"gt_count": 0, "matched_pairs": [], "extra_count": 0}
        )
        group["gt_count"] += 1

    for gt_item, pred_item in pairs:
        category = gt_category_by_id[id(gt_item)]
        category_groups[category]["matched_pairs"].append((gt_item, pred_item))

    for pred_item in pred_courses:
        if id(pred_item) in matched_pred_ids:
            continue
        category = _prediction_category(pred_item)
        group = category_groups.setdefault(
            category, {"gt_count": 0, "matched_pairs": [], "extra_count": 0}
        )
        group["extra_count"] += 1

    rubric_category_groups = {}
    for category, group in category_groups.items():
        matched_count_for_category = len(group["matched_pairs"])
        prediction_count_for_category = (
            matched_count_for_category + group["extra_count"]
        )
        rubric_category_groups[category] = {
            "text_quality": average(
                _text_stats_for_pairs(group["matched_pairs"], target_fields)
            ),
            "coverage": _coverage_metrics(
                group["gt_count"],
                prediction_count_for_category,
                matched_count_for_category,
            ),
        }

    rubric_category_level = {
        "status": "partial" if gt_category_missing else "available",
        "basis": "ground_truth_course_category",
        "groups": rubric_category_groups,
    }
    if gt_category_missing:
        rubric_category_level["reason"] = (
            "Some ground truth records do not contain an authoritative curriculum category."
        )

    gt_page_entries = []
    for gt_item in gt_courses:
        entries = _authoritative_source_provenance(gt_item)
        if not entries:
            gt_page_entries = None
            break
        gt_page_entries.append(entries)

    if not gt_courses or gt_page_entries is None:
        rubric_page_level = {
            "status": "unavailable",
            "reason": PAGE_PROVENANCE_UNAVAILABLE_REASON,
        }
    else:
        pred_page_entries = [
            _authoritative_source_provenance(pred_item) for pred_item in pred_courses
        ]
        page_groups = {}

        def ensure_page_group(entry: dict) -> dict:
            key = _source_identity_key(entry)
            return page_groups.setdefault(
                key,
                {
                    "source_provenance": entry,
                    "gt_indices": [],
                    "pred_indices": [],
                },
            )

        for gt_index, entries in enumerate(gt_page_entries):
            for entry in entries:
                ensure_page_group(entry)["gt_indices"].append(gt_index)
        for pred_index, entries in enumerate(pred_page_entries):
            for entry in entries:
                ensure_page_group(entry)["pred_indices"].append(pred_index)

        misplaced_match_count = 0
        for gt_item, pred_item in pairs:
            gt_ids = {
                _source_identity(entry)
                for entry in _authoritative_source_provenance(gt_item)
            }
            pred_ids = {
                _source_identity(entry)
                for entry in _authoritative_source_provenance(pred_item)
            }
            if not gt_ids.intersection(pred_ids):
                misplaced_match_count += 1

        rubric_page_groups = {}
        for key, group in page_groups.items():
            page_entry = group["source_provenance"]
            page_identity = _source_identity(page_entry)
            page_pairs = []
            page_misplaced_count = 0
            for gt_index in group["gt_indices"]:
                gt_item = gt_courses[gt_index]
                pred_item = pair_by_gt_id.get(id(gt_item))
                if pred_item is None:
                    continue
                pred_ids = {
                    _source_identity(entry)
                    for entry in _authoritative_source_provenance(pred_item)
                }
                if page_identity in pred_ids:
                    page_pairs.append((gt_item, pred_item))
                else:
                    page_misplaced_count += 1

            matched_count_for_page = len(page_pairs)
            rubric_page_groups[key] = {
                "source_provenance": page_entry,
                "text_quality": average(
                    _text_stats_for_pairs(page_pairs, target_fields)
                ),
                "coverage": _coverage_metrics(
                    len(group["gt_indices"]),
                    len(group["pred_indices"]),
                    matched_count_for_page,
                ),
                "misplaced_match_count": page_misplaced_count,
            }

        rubric_page_level = {
            "status": "available",
            "pages": rubric_page_groups,
            "misplaced_match_count": misplaced_match_count,
            "unassigned_prediction_count": sum(
                1 for entries in pred_page_entries if not entries
            ),
        }

    rubric = {
        "overall_text": average(page_stats),
        "field_level": rubric_field_level,
        "page_level": rubric_page_level,
        "category_level": rubric_category_level,
    }

    return {
        "file_name": pred_path.name,
        "total_gt_courses": len(gt_courses),
        "total_pred_courses": len(pred_courses),
        "matched_courses": len(pairs),
        "coverage": {
            "gt_record_count": gt_record_count,
            "prediction_record_count": prediction_record_count,
            "matched_count": matched_count,
            "missing_gt_count": missing_gt_count,
            "extra_prediction_count": extra_prediction_count,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "page_level": average(page_stats),
        "field_level": {f: average(field_stats[f]) for f in target_fields},
        "category_level": {
            "plan": average(cat_stats["plan"]),
            "description": average(cat_stats["description"]),
        },
        "rubric": rubric,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CLI evaluator for CER/WER between Prediction and Ground Truth JSON"
    )
    parser.add_argument(
        "prediction_json",
        type=str,
        help="Path of the Prediction JSON produced by the model/code",
    )
    parser.add_argument(
        "--gt",
        dest="ground_truth_json",
        type=str,
        required=True,
        help="Path of the Ground Truth JSON file",
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="output_json",
        type=str,
        default=None,
        help="Optional: Path to save the summary report JSON file",
    )

    args = parser.parse_args()

    try:
        result = evaluate_json_structure(
            ground_truth_json=args.ground_truth_json,
            prediction_json=args.prediction_json,
        )

        formatted_result = json.dumps(result, indent=2, ensure_ascii=False)
        print(formatted_result)

        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                f.write(formatted_result)
            print(f"\n Report saved successfully at: {out_path}")

    except Exception as e:
        print(f" Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
