import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import evaluate as evaluator
from evaluate import (
    calculate_cer,
    calculate_field_wer,
    calculate_wer,
    evaluate_json_structure,
    evaluate_pair,
    write_evaluation_reports,
)


SUMMARY_COLUMNS = [
    "program",
    "plan",
    "gt_total",
    "pred_total",
    "tp",
    "fn",
    "fp",
    "precision",
    "recall",
    "f1",
    "precision_percent",
    "recall_percent",
    "f1_percent",
]
FIELD_COLUMNS = [
    "program",
    "plan",
    "field",
    "sample_count",
    "cer",
    "character_accuracy_percent",
    "wer",
    "word_accuracy_percent",
]
ERROR_COLUMNS = [
    "program",
    "plan",
    "alignment_status",
    "code",
    "field",
    "gt_value",
    "pred_value",
    "cer",
    "wer",
]


def course(code, name_th="ภาษาไทย", name_en="THAI ENGLISH"):
    return {
        "code": code,
        "name_th": name_th,
        "name_en": name_en,
        "credits": "3(3-0-6)",
        "prerequisite": "ไม่มี",
    }


def write_pair(directory, name, gt_courses, pred_courses, program, plan):
    gt_path = directory / f"{name}_gt.json"
    pred_path = directory / f"{name}_pred.json"
    gt_path.write_text(
        json.dumps(
            {"program": program, "plan": plan, "courses": gt_courses},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pred_path.write_text(
        json.dumps(
            {"program": program, "plan": plan, "courses": pred_courses},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return gt_path, pred_path


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class EvaluationReportTests(unittest.TestCase):
    def test_thai_newmm_wer_and_english_whitespace_wer(self):
        thai_wer = calculate_field_wer("name_th", "ฉันรักแมว", "ฉันรักหมา")
        english_wer = calculate_field_wer(
            "name_en", "HELLO WORLD", "HELLO THERE"
        )

        self.assertAlmostEqual(thai_wer, 1 / 3)
        self.assertEqual(calculate_wer("ฉันรักแมว", "ฉันรักหมา"), 1.0)
        self.assertEqual(english_wer, 0.5)

    def test_cer_is_unchanged(self):
        self.assertAlmostEqual(calculate_cer("ABC", "AB"), 1 / 3)

    def test_report_metrics_use_percentages_and_leave_non_text_wer_blank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_path, pred_path = write_pair(
                root,
                "thai",
                [course("A0000001", "ฉันรักแมว")],
                [course("A0000001", "ฉันรักหมา")],
                "DSBA",
                "coop",
            )
            case = evaluate_pair(gt_path, pred_path)
            write_evaluation_reports([case], root / "reports" / "evaluation")

            rows = read_csv(root / "reports/evaluation/field_metrics.csv")
            by_field = {row["field"]: row for row in rows}

        self.assertAlmostEqual(float(by_field["name_th"]["wer"]), 0.3333, places=4)
        self.assertAlmostEqual(
            float(by_field["name_th"]["word_accuracy_percent"]), 66.67, places=2
        )
        self.assertEqual(by_field["name_th"]["character_accuracy_percent"], "77.78")
        self.assertEqual(by_field["code"]["wer"], "")
        self.assertEqual(by_field["code"]["word_accuracy_percent"], "")
        self.assertEqual(by_field["credits"]["wer"], "")

    def test_year_and_semester_are_wer_na_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_path, pred_path = write_pair(
                root,
                "placement",
                [{**course("A0000001"), "year": 1, "semester": 2}],
                [{**course("A0000001"), "year": 1, "semester": 2}],
                "DSBA",
                "coop",
            )
            result, details = evaluate_json_structure(
                gt_path,
                pred_path,
                target_fields=["code", "name_th", "credits", "year", "semester"],
                return_details=True,
            )
            case = {
                "program": "DSBA",
                "plan": "coop",
                "result": result,
                "details": details,
            }
            write_evaluation_reports([case], root / "reports" / "evaluation")
            rows = read_csv(root / "reports/evaluation/field_metrics.csv")

        by_field = {row["field"]: row for row in rows}
        self.assertEqual(by_field["year"]["wer"], "")
        self.assertEqual(by_field["semester"]["word_accuracy_percent"], "")

    def test_summary_has_tp_fn_fp_and_no_tn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_path, pred_path = write_pair(
                root,
                "coverage",
                [course("A0000001"), course("B0000002")],
                [course("A0000001"), course("C0000003")],
                "IT",
                "no_coop",
            )
            case = evaluate_pair(gt_path, pred_path)
            payload = write_evaluation_reports([case], root / "reports" / "evaluation")
            rows = read_csv(root / "reports/evaluation/evaluation_summary.csv")

        self.assertEqual(
            {key: rows[0][key] for key in ("tp", "fn", "fp")},
            {"tp": "1", "fn": "1", "fp": "1"},
        )
        self.assertEqual(rows[0]["precision_percent"], "50.0")
        self.assertEqual(rows[0]["recall_percent"], "50.0")
        self.assertEqual(rows[0]["f1_percent"], "50.0")
        self.assertNotIn("tn", rows[0])
        self.assertNotIn("tn", payload["results"][0])

    def test_error_csv_has_exact_schema_and_omits_matching_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_path, pred_path = write_pair(
                root,
                "errors",
                [course("A0000001", "ฉันรักแมว"), course("B0000002")],
                [course("A0000001", "ฉันรักหมา"), course("C0000003")],
                "AIT",
                None,
            )
            case = evaluate_pair(gt_path, pred_path)
            write_evaluation_reports([case], root / "reports" / "evaluation")
            path = root / "reports/evaluation/evaluation_errors.csv"
            rows = read_csv(path)

        self.assertEqual(list(rows[0].keys()), ERROR_COLUMNS)
        statuses = {row["alignment_status"] for row in rows}
        self.assertEqual(statuses, {"matched", "missing", "extra"})
        self.assertIn(
            ("matched", "name_th"),
            {(row["alignment_status"], row["field"]) for row in rows},
        )
        self.assertNotIn(
            ("matched", "name_en"),
            {(row["alignment_status"], row["field"]) for row in rows},
        )

    def test_csv_files_are_utf8_sig_and_have_exact_schemas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_path, pred_path = write_pair(
                root,
                "schema",
                [course("A0000001", "แมว")],
                [course("A0000001", "หมา")],
                "GENED",
                "gened",
            )
            case = evaluate_pair(gt_path, pred_path)
            report_dir = root / "reports/evaluation"
            write_evaluation_reports([case], report_dir)

            summary_path = report_dir / "evaluation_summary.csv"
            field_path = report_dir / "field_metrics.csv"
            error_path = report_dir / "evaluation_errors.csv"
            self.assertEqual(summary_path.read_bytes()[:3], b"\xef\xbb\xbf")
            self.assertEqual(field_path.read_bytes()[:3], b"\xef\xbb\xbf")
            self.assertEqual(error_path.read_bytes()[:3], b"\xef\xbb\xbf")
            self.assertEqual(list(read_csv(summary_path)[0].keys()), SUMMARY_COLUMNS)
            self.assertEqual(list(read_csv(field_path)[0].keys()), FIELD_COLUMNS)
            self.assertEqual(list(read_csv(error_path)[0].keys()), ERROR_COLUMNS)
            self.assertIn("แมว", error_path.read_text(encoding="utf-8-sig"))

    def test_description_fields_are_reported_only_when_gt_has_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt = {**course("A0000001"), "desc_th": "คำอธิบาย"}
            pred = {**gt, "desc_th": "คำอธิบายผิด"}
            gt_path, pred_path = write_pair(
                root, "description", [gt], [pred], "GENED", "gened"
            )
            case = evaluate_pair(gt_path, pred_path)
            write_evaluation_reports([case], root / "reports" / "evaluation")
            fields = {
                row["field"]
                for row in read_csv(root / "reports/evaluation/field_metrics.csv")
            }

        self.assertIn("desc_th", fields)
        self.assertNotIn("desc_en", fields)

    def test_batch_pairs_coexist_and_second_invocation_overwrites_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_gt, first_pred = write_pair(
                root, "first", [course("A0000001")], [course("A0000001")], "DSBA", "coop"
            )
            second_gt, second_pred = write_pair(
                root, "second", [course("B0000002")], [course("B0000002")], "IT", "coop"
            )
            report_dir = root / "reports/evaluation"

            with patch.object(evaluator, "REPORT_DIR", report_dir), patch.object(
                sys,
                "argv",
                [
                    "evaluate.py",
                    "--pair",
                    str(first_pred),
                    str(first_gt),
                    "--pair",
                    str(second_pred),
                    str(second_gt),
                ],
            ), redirect_stdout(io.StringIO()):
                evaluator.main()

            self.assertEqual(len(read_csv(report_dir / "evaluation_summary.csv")), 2)
            self.assertEqual(
                [item["program"] for item in json.loads(
                    (report_dir / "evaluation.json").read_text(encoding="utf-8")
                )["results"]],
                ["DSBA", "IT"],
            )

            with patch.object(evaluator, "REPORT_DIR", report_dir), patch.object(
                sys,
                "argv",
                ["evaluate.py", str(first_pred), "--gt", str(first_gt)],
            ), redirect_stdout(io.StringIO()):
                evaluator.main()

            summary_rows = read_csv(report_dir / "evaluation_summary.csv")
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["program"], "DSBA")
            self.assertEqual(
                len(json.loads(
                    (report_dir / "evaluation.json").read_text(encoding="utf-8")
                )["results"]),
                1,
            )

    def test_single_pair_cli_and_legacy_out_remain_usable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_path, pred_path = write_pair(
                root, "single", [course("A0000001")], [course("A0000001")], "DSBA", "coop"
            )
            report_dir = root / "reports/evaluation"
            legacy_out = root / "legacy.json"

            with patch.object(evaluator, "REPORT_DIR", report_dir), patch.object(
                sys,
                "argv",
                [
                    "evaluate.py",
                    str(pred_path),
                    "--gt",
                    str(gt_path),
                    "--out",
                    str(legacy_out),
                ],
            ), redirect_stdout(io.StringIO()):
                evaluator.main()

            legacy = json.loads(legacy_out.read_text(encoding="utf-8"))
            combined = json.loads(
                (report_dir / "evaluation.json").read_text(encoding="utf-8")
            )

        self.assertIn("coverage", legacy)
        self.assertNotIn("results", legacy)
        self.assertEqual(len(combined["results"]), 1)


if __name__ == "__main__":
    unittest.main()
