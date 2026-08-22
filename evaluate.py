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

    for gt_item, pred_item in pairs:
        category = "plan" if is_plan_course(gt_item) else "description"
        for field in target_fields:
            if field not in gt_item:
                continue
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
