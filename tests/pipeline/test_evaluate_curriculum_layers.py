import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.tools.evaluation.layers import (
    Scope,
    align_records,
    compare_layers,
    evaluate_records,
    field_matches,
    write_layer_report,
)


def course(code, **values):
    result = {
        "code": code,
        "name_th": values.pop("name_th", "ชื่อ"),
        "name_en": values.pop("name_en", "Name"),
        "credits": values.pop("credits", "3(3-0-6)"),
        "year": values.pop("year", 1),
        "semester": values.pop("semester", 1),
        "prerequisite": values.pop("prerequisite", "ไม่มี"),
    }
    result.update(values)
    return result


class DualEvaluatorTests(unittest.TestCase):
    def test_exact_one_to_one_match(self):
        result = evaluate_records([course("00000001")], [course("00000001")], "IT", "coop", "consolidated")
        self.assertEqual(result["matched_gt_rows"], 1)
        self.assertEqual(result["missing_output_rows"], 0)
        self.assertEqual(result["extra_output_rows"], 0)
        self.assertEqual(result["field_metrics"]["credits"]["normalized_correct"], 1)

    def test_missing_output_row(self):
        result = evaluate_records([course("00000001")], [], "IT", "coop", "consolidated")
        self.assertEqual(result["missing_output_rows"], 1)
        self.assertEqual(result["field_metrics"]["code"]["normalized_incorrect"], 1)

    def test_extra_output_row(self):
        result = evaluate_records([course("00000001")], [course("00000001"), course("00000002")], "IT", "coop", "consolidated")
        self.assertEqual(result["extra_output_rows"], 1)

    def test_duplicate_occurrence_ordinal(self):
        gt = [course("xxxxxxxx", year=1), course("xxxxxxxx", year=2)]
        pred = [course("xxxxxxxx", year=1), course("xxxxxxxx", year=2)]
        alignment = align_records(gt, pred, "AIT", None)
        self.assertEqual(len(alignment["pairs"]), 2)
        self.assertEqual([pair[0].occurrence for pair in alignment["pairs"]], [1, 2])

    def test_it_style_duplicate_shape(self):
        gt = [course("06016418")]
        pred = [course("06016418"), course("06016418")]
        result = evaluate_records(gt, pred, "IT", "coop", "consolidated")
        self.assertEqual(result["matched_gt_rows"], 1)
        self.assertEqual(result["extra_output_rows"], 1)

    def test_gened_missing_placement_excluded(self):
        item = course("90641003")
        item.pop("year")
        item.pop("semester")
        result = evaluate_records([item], [item.copy()], "GENED", "gened", "consolidated")
        self.assertEqual(result["field_metrics"]["year"]["denominator"], 0)
        self.assertEqual(result["field_metrics"]["semester"]["denominator"], 0)
        self.assertEqual(result["field_metrics"]["code"]["denominator"], 1)

    def test_strict_and_normalized_name_comparison(self):
        strict, normalized = field_matches("name_en", "A  Name", "a name")
        self.assertFalse(strict)
        self.assertTrue(normalized)

    def test_cer_wer_are_reported(self):
        result = evaluate_records([course("00000001", name_en="A Name")], [course("00000001", name_en="A Nme")], "IT", "coop", "consolidated")
        self.assertGreater(result["field_metrics"]["name_en"]["cer"], 0)
        self.assertGreater(result["field_metrics"]["name_en"]["wer"], 0)

    def test_structured_credit_equality(self):
        strict, normalized = field_matches("credits", "3(3-0-6)", "3 (3-0-6)")
        self.assertFalse(strict)
        self.assertTrue(normalized)

    def test_malformed_credit_is_not_normalized_into_correctness(self):
        _, normalized = field_matches("credits", "3(3-0-6)", "3(3-0)")
        self.assertFalse(normalized)

    def test_wildcard_credit_remains_distinct(self):
        _, normalized = field_matches("credits", "3(x-x-x)", "3(3-0-6)")
        self.assertFalse(normalized)

    def test_alternative_credit_order_is_equivalent(self):
        _, normalized = field_matches("credits", "3(3-0-6) หรือ 3(2-2-5)", "3(2-2-5) หรือ 3(3-0-6)")
        self.assertTrue(normalized)

    def test_prerequisite_and_or_distinction(self):
        _, normalized = field_matches("prerequisite", "00000001 หรือ 00000002", "00000001 และ 00000002")
        self.assertFalse(normalized)

    def test_prerequisite_thai_or_matches_code_comma_alternatives(self):
        _, normalized = field_matches("prerequisite", "06036119 หรือ 06036122", "06036119, 06036122")
        self.assertTrue(normalized)

    def test_prerequisite_code_alternatives_are_order_insensitive(self):
        _, normalized = field_matches("prerequisite", "06036119 หรือ 06036122", "06036122, 06036119")
        self.assertTrue(normalized)

    def test_prerequisite_and_expression_remains_distinct(self):
        _, normalized = field_matches("prerequisite", "06036119 และ 06036122", "06036119, 06036122")
        self.assertFalse(normalized)

    def test_prerequisite_arbitrary_comma_text_is_not_or(self):
        _, normalized = field_matches("prerequisite", "COURSE A, COURSE B", "COURSE A หรือ COURSE B")
        self.assertFalse(normalized)

    def test_prerequisite_empty_and_single_values_unchanged(self):
        _, empty_normalized = field_matches("prerequisite", "ไม่มี", "")
        _, single_normalized = field_matches("prerequisite", "06036119", "06036119")
        self.assertTrue(empty_normalized)
        self.assertTrue(single_normalized)

    def test_wrong_to_correct_transition(self):
        gt = {Scope("IT", "coop"): [course("00000001", name_en="Correct")]}
        con = {Scope("IT", "coop"): [course("00000001", name_en="Wrong")]}
        llm = {Scope("IT", "coop"): [course("00000001", name_en="Correct")]}
        comparison = compare_layers(gt, con, llm)
        row = next(row for row in comparison["field_deltas"] if row["field"] == "name_en")
        self.assertEqual(row["improvements"], 1)

    def test_correct_to_wrong_transition(self):
        gt = {Scope("IT", "coop"): [course("00000001", name_en="Correct")]}
        con = {Scope("IT", "coop"): [course("00000001", name_en="Correct")]}
        llm = {Scope("IT", "coop"): [course("00000001", name_en="Wrong")]}
        comparison = compare_layers(gt, con, llm)
        row = next(row for row in comparison["field_deltas"] if row["field"] == "name_en")
        self.assertEqual(row["regressions"], 1)

    def test_non_name_layer_difference_is_flagged(self):
        gt = {Scope("IT", "coop"): [course("00000001", credits="3(3-0-6)")]}
        con = {Scope("IT", "coop"): [course("00000001", credits="3(3-0-6)")]}
        llm = {Scope("IT", "coop"): [course("00000001", credits="4(3-0-6)")]}
        comparison = compare_layers(gt, con, llm)
        self.assertEqual(len(comparison["non_name_layer_differences"]), 1)

    def test_deterministic_report_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_root = root / "ground_truth"
            con_root = root / "outputs" / "consolidated"
            llm_root = root / "outputs" / "llm"
            scopes = {
                ("AIT", None): ("ait_no_plan", "AIT/AIT_academic_plan.json"),
                ("BIT", "coop"): ("bit_coop", "BIT/BIT_academic_plan_coop.json"),
                ("BIT", "no_coop"): ("bit_no_coop", "BIT/BIT_academic_plan_no_coop.json"),
                ("DSBA", "coop"): ("dsba_coop", "DSBA/DSBA_academic_plan_coop.json"),
                ("DSBA", "no_coop"): ("dsba_no_coop", "DSBA/DSBA_academic_plan_no_coop.json"),
                ("GENED", "gened"): ("gened_gened", "general_education_ground_truth.json"),
                ("IT", "coop"): ("it_coop", "IT/IT_academic_plan_coop.json"),
                ("IT", "no_coop"): ("it_no_coop", "IT/IT_academic_plan_no_coop.json"),
            }
            for (program, plan), (stem, gt_rel) in scopes.items():
                gt_path = gt_root / gt_rel
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                gt_path.write_text(json.dumps({"courses": [course(program + "0000001")]}, ensure_ascii=False), encoding="utf-8")
                con_path = con_root / "full" / f"merged_{stem}_full.json"
                con_path.parent.mkdir(parents=True, exist_ok=True)
                con_path.write_text(json.dumps({"courses": [course(program + "0000001")]}, ensure_ascii=False), encoding="utf-8")
                llm_path = llm_root / f"merged_{stem}_full_corrected.json"
                llm_path.parent.mkdir(parents=True, exist_ok=True)
                llm_path.write_text(json.dumps({"courses": [course(program + "0000001")]}, ensure_ascii=False), encoding="utf-8")
            result = evaluate_records([course("00000001")], [course("00000001")], "IT", "coop", "consolidated")
            report_dir = root / "reports"
            write_layer_report(result, report_dir)
            self.assertTrue((report_dir / "summary.json").is_file())
            with (report_dir / "field_metrics.csv").open(encoding="utf-8") as handle:
                self.assertEqual(next(csv.DictReader(handle))["field"], "code")
            self.assertEqual(result["matched_gt_rows"], 1)


if __name__ == "__main__":
    unittest.main()
