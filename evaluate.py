import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List
import Levenshtein


@dataclass
class FieldMetric:
    exact_matches: int = 0
    total_gt: int = 0
    total_pred: int = 0
    correct_preds: int = 0
    total_similarity: float = 0.0

    @property
    def precision(self) -> float:
        return (self.correct_preds / self.total_pred) * 100 if self.total_pred > 0 else 0.0

    @property
    def recall(self) -> float:
        return (self.correct_preds / self.total_gt) * 100 if self.total_gt > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_similarity(self) -> float:
        return (self.total_similarity / self.total_gt) * 100 if self.total_gt > 0 else 0.0


@dataclass
class JSONEvaluationResult:
    file_name: str
    page_level_exact_match: bool
    course_precision: float
    course_recall: float
    course_f1: float
    overall_field_precision: float
    overall_field_recall: float
    overall_field_f1: float
    per_field_metrics: Dict[str, Dict[str, float]]


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
    dist = Levenshtein.distance(s1, s2)
    return 1.0 - (dist / max_len)


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
        raise FileNotFoundError(f"ไม่พบไฟล์ Ground Truth: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ Prediction: {pred_path}")

    with gt_path.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with pred_path.open("r", encoding="utf-8") as f:
        pred_data = json.load(f)

    gt_courses: List[dict] = gt_data.get("courses", [])
    pred_courses: List[dict] = pred_data.get("courses", [])

    file_name = pred_path.name

    total_gt_courses = len(gt_courses)
    total_pred_courses = len(pred_courses)

    metrics: Dict[str, FieldMetric] = {f: FieldMetric() for f in target_fields}
    page_is_perfect = total_gt_courses == total_pred_courses

    max_len = max(total_gt_courses, total_pred_courses)

    for i in range(max_len):
        gt_item = gt_courses[i] if i < total_gt_courses else {}
        pred_item = pred_courses[i] if i < total_pred_courses else {}

        course_is_perfect = True

        for field in target_fields:
            gt_val = normalize_str(gt_item.get(field))
            pred_val = normalize_str(pred_item.get(field))

            m = metrics[field]

            if field in gt_item:
                m.total_gt += 1
            if field in pred_item:
                m.total_pred += 1

            sim = calculate_similarity(gt_val, pred_val)
            m.total_similarity += sim

            if gt_val == pred_val and (gt_val != ""):
                m.exact_matches += 1
                m.correct_preds += 1
            elif sim >= fuzzy_threshold:
                m.correct_preds += 1
            else:
                course_is_perfect = False

        if not course_is_perfect:
            page_is_perfect = False

    total_correct = sum(m.correct_preds for m in metrics.values())
    total_gt_fields = sum(m.total_gt for m in metrics.values())
    total_pred_fields = sum(m.total_pred for m in metrics.values())

    overall_p = (total_correct / total_pred_fields * 100) if total_pred_fields > 0 else 0.0
    overall_r = (total_correct / total_gt_fields * 100) if total_gt_fields > 0 else 0.0
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0

    matched_courses = min(total_gt_courses, total_pred_courses)
    c_p = (matched_courses / total_pred_courses * 100) if total_pred_courses > 0 else 0.0
    c_r = (matched_courses / total_gt_courses * 100) if total_gt_courses > 0 else 0.0
    c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0

    per_field_summary = {}
    for f_name, m in metrics.items():
        per_field_summary[f_name] = {
            "exact_match_count": m.exact_matches,
            "precision": round(m.precision, 2),
            "recall": round(m.recall, 2),
            "f1_score": round(m.f1_score, 2),
            "avg_similarity": round(m.avg_similarity, 2),
        }

    res = JSONEvaluationResult(
        file_name=file_name,
        page_level_exact_match=page_is_perfect,
        course_precision=round(c_p, 2),
        course_recall=round(c_r, 2),
        course_f1=round(c_f1, 2),
        overall_field_precision=round(overall_p, 2),
        overall_field_recall=round(overall_r, 2),
        overall_field_f1=round(overall_f1, 2),
        per_field_metrics=per_field_summary,
    )

    return asdict(res)


def main():
    parser = argparse.ArgumentParser(
        description="CLI Evaluator สำหรับประเมินความแม่นยำของ JSON Structured Data"
    )
    parser.add_argument(
        "prediction_json",
        type=str,
        help="Path ของไฟล์ Prediction JSON ที่รันได้จากโมเดล/โค้ด",
    )
    parser.add_argument(
        "--gt",
        dest="ground_truth_json",
        type=str,
        required=True,
        help="Path ของไฟล์ Ground Truth JSON",
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="output_json",
        type=str,
        default=None,
        help="Optional: Path สำหรับบันทึกไฟล์สรุปผลรายงาน JSON",
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
            print(f"\n บันทึกรายงานผลเรียบร้อยที่: {out_path}")

    except Exception as e:
        print(f" เกิดข้อผิดพลาด: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()