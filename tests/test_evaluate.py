import json
import tempfile
import unittest
from pathlib import Path

from evaluate import evaluate_json_structure


def course(code: str, marker: str) -> dict:
    return {
        "code": code,
        "name_th": f"TH {marker}",
        "name_en": f"EN {marker}",
        "credits": "3(3-0-6)",
        "prerequisite": "NONE",
    }


class EvaluateCoverageTests(unittest.TestCase):
    def evaluate(self, gt_courses, pred_courses):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gt_path = temp_path / "ground_truth.json"
            pred_path = temp_path / "prediction.json"
            gt_path.write_text(json.dumps({"courses": gt_courses}), encoding="utf-8")
            pred_path.write_text(
                json.dumps({"courses": pred_courses}), encoding="utf-8"
            )
            return evaluate_json_structure(gt_path, pred_path)

    def assert_coverage(self, result, **expected):
        self.assertEqual(result["coverage"], expected)

    def test_perfect_match(self):
        records = [course("A0000001", "ONE"), course("B0000002", "TWO")]

        result = self.evaluate(records, records)

        self.assert_coverage(
            result,
            gt_record_count=2,
            prediction_record_count=2,
            matched_count=2,
            missing_gt_count=0,
            extra_prediction_count=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
        )
        self.assertEqual(result["page_level"], {"cer": 0.0, "wer": 0.0, "count": 10})

    def test_one_missing_gt_record(self):
        gt_records = [course("A0000001", "ONE"), course("B0000002", "TWO")]

        result = self.evaluate(gt_records, [gt_records[0]])

        self.assert_coverage(
            result,
            gt_record_count=2,
            prediction_record_count=1,
            matched_count=1,
            missing_gt_count=1,
            extra_prediction_count=0,
            precision=1.0,
            recall=0.5,
            f1=0.6667,
        )
        self.assertEqual(result["page_level"]["count"], 5)
        self.assertEqual(result["page_level"]["cer"], 0.0)
        self.assertEqual(result["page_level"]["wer"], 0.0)

    def test_one_extra_prediction(self):
        gt_records = [course("A0000001", "ONE")]
        predictions = [gt_records[0], course("Z9999999", "EXTRA")]

        result = self.evaluate(gt_records, predictions)

        self.assert_coverage(
            result,
            gt_record_count=1,
            prediction_record_count=2,
            matched_count=1,
            missing_gt_count=0,
            extra_prediction_count=1,
            precision=0.5,
            recall=1.0,
            f1=0.6667,
        )
        self.assertEqual(result["page_level"]["count"], 5)

    def test_missing_and_extra_prediction(self):
        gt_records = [course("A0000001", "ONE"), course("B0000002", "TWO")]
        predictions = [gt_records[0], course("Z9999999", "EXTRA")]

        result = self.evaluate(gt_records, predictions)

        self.assert_coverage(
            result,
            gt_record_count=2,
            prediction_record_count=2,
            matched_count=1,
            missing_gt_count=1,
            extra_prediction_count=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
        )
        self.assertEqual(result["page_level"]["count"], 5)

    def test_repeated_codes_are_consumed_one_to_one(self):
        gt_records = [course("A0000001", "FIRST"), course("A0000001", "SECOND")]

        result = self.evaluate(gt_records, [course("A0000001", "PREDICTED")])

        self.assert_coverage(
            result,
            gt_record_count=2,
            prediction_record_count=1,
            matched_count=1,
            missing_gt_count=1,
            extra_prediction_count=0,
            precision=1.0,
            recall=0.5,
            f1=0.6667,
        )

    def test_empty_sides_are_safe(self):
        empty_result = self.evaluate([], [])
        no_gt_result = self.evaluate([], [course("A0000001", "EXTRA")])
        no_prediction_result = self.evaluate([course("A0000001", "MISSING")], [])

        self.assert_coverage(
            empty_result,
            gt_record_count=0,
            prediction_record_count=0,
            matched_count=0,
            missing_gt_count=0,
            extra_prediction_count=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
        )
        self.assert_coverage(
            no_gt_result,
            gt_record_count=0,
            prediction_record_count=1,
            matched_count=0,
            missing_gt_count=0,
            extra_prediction_count=1,
            precision=0.0,
            recall=0.0,
            f1=0.0,
        )
        self.assert_coverage(
            no_prediction_result,
            gt_record_count=1,
            prediction_record_count=0,
            matched_count=0,
            missing_gt_count=1,
            extra_prediction_count=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
        )
        self.assertEqual(empty_result["page_level"], {"cer": 0.0, "wer": 0.0, "count": 0})


if __name__ == "__main__":
    unittest.main()
