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


def sourced_course(
    code: str,
    marker: str,
    page: int,
    category: str = "CORE",
    document_category: str = "plan",
) -> dict:
    record = course(code, marker)
    record["category"] = category
    record["source_provenance"] = [
        {
            "program": "DSBA",
            "source_filename": f"dsba_page_{page:03d}.png",
            "source_page": page,
            "document_category": document_category,
        }
    ]
    return record


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

    def test_legacy_outputs_and_metrics_remain_unchanged(self):
        records = [course("A0000001", "ONE"), course("B0000002", "TWO")]

        result = self.evaluate(records, records)

        self.assertEqual(result["total_gt_courses"], 2)
        self.assertEqual(result["total_pred_courses"], 2)
        self.assertEqual(result["matched_courses"], 2)
        self.assertEqual(
            result["coverage"],
            {
                "gt_record_count": 2,
                "prediction_record_count": 2,
                "matched_count": 2,
                "missing_gt_count": 0,
                "extra_prediction_count": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
        )
        self.assertEqual(result["page_level"], {"cer": 0.0, "wer": 0.0, "count": 10})
        self.assertEqual(
            result["category_level"],
            {
                "plan": {"cer": 0.0, "wer": 0.0, "count": 0},
                "description": {"cer": 0.0, "wer": 0.0, "count": 10},
            },
        )
        self.assertEqual(result["rubric"]["overall_text"], result["page_level"])

    def test_field_presence_coverage_separates_missing_predicted_field(self):
        gt_record = course("A0000001", "ONE")
        prediction = course("A0000001", "ONE")
        del prediction["name_en"]

        result = self.evaluate([gt_record], [prediction])
        presence = result["rubric"]["field_level"]["name_en"]["presence_coverage"]

        self.assertEqual(
            presence,
            {
                "gt_record_count": 1,
                "prediction_record_count": 0,
                "matched_count": 0,
                "missing_gt_count": 1,
                "extra_prediction_count": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        )
        self.assertEqual(
            result["rubric"]["field_level"]["name_en"][
                "matched_prediction_field_missing_count"
            ],
            1,
        )
        self.assertEqual(result["field_level"]["name_en"]["count"], 1)
        self.assertEqual(result["field_level"]["name_en"]["cer"], 1.0)
        self.assertEqual(result["field_level"]["name_en"]["wer"], 1.0)

    def test_field_presence_coverage_counts_missing_gt_and_extra_prediction(self):
        gt_records = [course("A0000001", "ONE"), course("B0000002", "TWO")]
        predictions = [gt_records[0], course("Z9999999", "EXTRA")]

        result = self.evaluate(gt_records, predictions)
        presence = result["rubric"]["field_level"]["name_en"]["presence_coverage"]

        self.assertEqual(
            presence,
            {
                "gt_record_count": 2,
                "prediction_record_count": 2,
                "matched_count": 1,
                "missing_gt_count": 1,
                "extra_prediction_count": 1,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
            },
        )

    def test_authoritative_pages_have_independent_metrics(self):
        gt_records = [
            sourced_course("A0000001", "ONE", 1),
            sourced_course("B0000002", "TWO", 2),
        ]
        predictions = [
            sourced_course("A0000001", "ONE", 1),
            sourced_course("B0000002", "WRONG", 2),
        ]

        result = self.evaluate(gt_records, predictions)
        page_level = result["rubric"]["page_level"]
        pages = {
            page["source_provenance"]["source_page"]: page
            for page in page_level["pages"].values()
        }

        self.assertEqual(page_level["status"], "available")
        self.assertEqual(page_level["misplaced_match_count"], 0)
        self.assertEqual(pages[1]["coverage"]["matched_count"], 1)
        self.assertEqual(pages[2]["coverage"]["matched_count"], 1)
        self.assertEqual(pages[1]["text_quality"], {"cer": 0.0, "wer": 0.0, "count": 5})
        self.assertEqual(pages[2]["text_quality"]["count"], 5)
        self.assertGreater(pages[2]["text_quality"]["cer"], 0.0)

    def test_repeated_codes_on_different_pages_remain_one_to_one(self):
        gt_records = [
            sourced_course("A0000001", "FIRST", 1),
            sourced_course("A0000001", "SECOND", 2),
        ]
        predictions = [
            sourced_course("A0000001", "FIRST", 1),
            sourced_course("A0000001", "SECOND", 2),
        ]

        result = self.evaluate(gt_records, predictions)
        pages = {
            page["source_provenance"]["source_page"]: page
            for page in result["rubric"]["page_level"]["pages"].values()
        }

        self.assertEqual(result["coverage"]["matched_count"], 2)
        self.assertEqual(pages[1]["coverage"]["matched_count"], 1)
        self.assertEqual(pages[2]["coverage"]["matched_count"], 1)
        self.assertEqual(result["rubric"]["page_level"]["misplaced_match_count"], 0)

    def test_missing_gt_provenance_makes_rubric_pages_unavailable(self):
        record = course("A0000001", "ONE")

        result = self.evaluate([record], [record])

        self.assertEqual(
            result["rubric"]["page_level"],
            {
                "status": "unavailable",
                "reason": "Ground truth records do not contain authoritative source/page provenance.",
            },
        )

    def test_rubric_category_groups_use_gt_category_and_unknown_extra_category(self):
        gt_records = [
            sourced_course("A0000001", "ONE", 1, category="CORE"),
            sourced_course("B0000002", "TWO", 2, category="GENERAL"),
        ]
        predictions = [
            sourced_course("A0000001", "ONE", 1, category="CORE"),
            sourced_course("B0000002", "TWO", 2, category="GENERAL"),
            course("Z9999999", "EXTRA"),
        ]

        result = self.evaluate(gt_records, predictions)
        category_level = result["rubric"]["category_level"]

        self.assertEqual(category_level["status"], "available")
        self.assertEqual(category_level["basis"], "ground_truth_course_category")
        self.assertEqual(
            category_level["groups"]["CORE"]["coverage"]["matched_count"], 1
        )
        self.assertEqual(
            category_level["groups"]["GENERAL"]["coverage"]["matched_count"], 1
        )
        self.assertEqual(
            category_level["groups"]["unknown_prediction_category"]["coverage"],
            {
                "gt_record_count": 0,
                "prediction_record_count": 1,
                "matched_count": 0,
                "missing_gt_count": 0,
                "extra_prediction_count": 1,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        )

    def test_misplaced_global_match_is_reported_without_changing_global_metrics(self):
        gt_record = sourced_course("A0000001", "ONE", 1)
        prediction = sourced_course("A0000001", "ONE", 2)

        result = self.evaluate([gt_record], [prediction])
        page_level = result["rubric"]["page_level"]
        pages = {
            page["source_provenance"]["source_page"]: page
            for page in page_level["pages"].values()
        }

        self.assertEqual(result["coverage"]["matched_count"], 1)
        self.assertEqual(result["page_level"], {"cer": 0.0, "wer": 0.0, "count": 5})
        self.assertEqual(page_level["misplaced_match_count"], 1)
        self.assertEqual(pages[1]["coverage"]["matched_count"], 0)
        self.assertEqual(pages[1]["coverage"]["missing_gt_count"], 1)
        self.assertEqual(pages[2]["coverage"]["matched_count"], 0)
        self.assertEqual(pages[2]["coverage"]["extra_prediction_count"], 1)


    def test_prerequisite_absence_representations_are_semantically_equal(self):
        gt_record = course("A0000001", "ONE")
        prediction = course("A0000001", "ONE")

        gt_record["prerequisite"] = "ไม่มี"
        prediction["prerequisite"] = "NONE"

        result = self.evaluate(
            [gt_record],
            [prediction],
        )

        prerequisite = result["field_level"]["prerequisite"]

        self.assertEqual(prerequisite["cer"], 0.0)
        self.assertEqual(prerequisite["wer"], 0.0)
        self.assertEqual(prerequisite["count"], 1)

    def test_prerequisite_alternative_separators_are_semantically_equal(self):
        gt_record = course("A0000001", "ONE")
        prediction = course("A0000001", "ONE")

        gt_record["prerequisite"] = "06036119 หรือ 06036122"
        prediction["prerequisite"] = "06036119, 06036122"

        result = self.evaluate([gt_record], [prediction])

        prerequisite = result["field_level"]["prerequisite"]
        self.assertEqual(prerequisite["cer"], 0.0)
        self.assertEqual(prerequisite["wer"], 0.0)
        self.assertEqual(prerequisite["count"], 1)


    def test_real_prerequisite_mismatch_is_still_penalized(self):
        gt_record = course("A0000001", "ONE")
        prediction = course("A0000001", "ONE")

        gt_record["prerequisite"] = "ไม่มี"
        prediction["prerequisite"] = "06016413"

        result = self.evaluate(
            [gt_record],
            [prediction],
        )

        prerequisite = result["field_level"]["prerequisite"]

        self.assertGreater(prerequisite["cer"], 0.0)
        self.assertGreater(prerequisite["wer"], 0.0)

    def test_equivalent_repeated_prediction_is_collapsed_for_canonical_gt(self):
        gt_record = course("06016418", "WEB")

        first = course("06016418", "WEB")
        first["prerequisite"] = "06016408"
        first["year"] = 3
        first["semester"] = 1

        second = course("06016418", "WEB")
        second["prerequisite"] = "06016408"
        second["year"] = 3
        second["semester"] = 1

        gt_record["prerequisite"] = "06016408"

        result = self.evaluate(
            [gt_record],
            [first, second],
        )

        self.assertEqual(
            result["coverage"]["prediction_record_count"],
            1,
        )
        self.assertEqual(
            result["coverage"]["matched_count"],
            1,
        )
        self.assertEqual(
            result["coverage"]["extra_prediction_count"],
            0,
        )

        self.assertEqual(
            result["prediction_view"],
            {
                "raw_record_count": 2,
                "evaluated_record_count": 1,
                "collapsed_equivalent_repeated_records": 1,
            },
        )


    def test_conflicting_repeated_prediction_is_not_collapsed(self):
        gt_record = course("06016418", "WEB")
        gt_record["prerequisite"] = "06016408"

        first = course("06016418", "WEB")
        first["prerequisite"] = "06016408"

        second = course("06016418", "WEB")
        second["prerequisite"] = "06016413"

        result = self.evaluate(
            [gt_record],
            [first, second],
        )

        self.assertEqual(
            result["coverage"]["prediction_record_count"],
            2,
        )
        self.assertEqual(
            result["coverage"]["matched_count"],
            1,
        )
        self.assertEqual(
            result["coverage"]["extra_prediction_count"],
            1,
        )
        self.assertEqual(
            result["prediction_view"][
                "collapsed_equivalent_repeated_records"
            ],
            0,
        )

if __name__ == "__main__":
    unittest.main()
