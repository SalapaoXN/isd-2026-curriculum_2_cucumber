"""Regression checks for side-local grounding of semantic comparisons."""

import unittest

from rag.semantic.prompts import build_semantic_interpreter_prompt
from rag.semantic.schema import ComparisonSpec, SemanticIntent
from rag.semantic.validation import validate_semantic_intent


class ComparisonOperandLocalityTests(unittest.TestCase):
    def intent(self, left, right):
        return SemanticIntent(
            task="compare",
            subject="course",
            comparison=ComparisonSpec(
                left=tuple(left.items()),
                right=tuple(right.items()),
                measure="course_count",
                operation="overlap",
            ),
            requested_fields=("code", "name"),
        )

    def validate(self, question, left, right):
        return validate_semantic_intent(self.intent(left, right), question)

    def test_exact_cross_catalog_year_operands_are_valid(self):
        result = self.validate(
            "DSBA 2565 ปี 1 กับ AIT 2566 ปี 2 มีวิชาไหนเรียนร่วมกันบ้าง",
            {"program": "DSBA", "catalog": "2565", "year": 1},
            {"program": "AIT", "catalog": "2566", "year": 2},
        )
        self.assertTrue(result.valid, result.reason)

    def test_swapped_years_are_rejected_even_when_both_numbers_exist(self):
        result = self.validate(
            "DSBA 2565 ปี 1 กับ AIT 2566 ปี 2 มีวิชาไหนเรียนร่วมกันบ้าง",
            {"program": "DSBA", "catalog": "2565", "year": 2},
            {"program": "AIT", "catalog": "2566", "year": 1},
        )
        self.assertFalse(result.valid)
        self.assertIn("opposite operand", result.reason or "")

    def test_swapped_catalogs_are_rejected_even_when_both_editions_exist(self):
        result = self.validate(
            "DSBA 2565 ปี 1 กับ AIT 2566 ปี 2 มีวิชาไหนเรียนร่วมกันบ้าง",
            {"program": "DSBA", "catalog": "2566", "year": 1},
            {"program": "AIT", "catalog": "2565", "year": 2},
        )
        self.assertFalse(result.valid)
        self.assertIn("opposite operand", result.reason or "")

    def test_whole_operand_swap_is_rejected_by_mention_order(self):
        result = self.validate(
            "DSBA 2565 ปี 1 กับ AIT 2566 ปี 2 มีวิชาไหนเรียนร่วมกันบ้าง",
            {"program": "AIT", "catalog": "2566", "year": 2},
            {"program": "DSBA", "catalog": "2565", "year": 1},
        )
        self.assertFalse(result.valid)
        self.assertIn("side order", result.reason or "")

    def test_shared_year_on_both_sides_remains_valid(self):
        result = self.validate(
            "DSBA 2565 ปี 1 กับ AIT 2566 ปี 1 มีวิชาไหนเรียนร่วมกันบ้าง",
            {"program": "DSBA", "catalog": "2565", "year": 1},
            {"program": "AIT", "catalog": "2566", "year": 1},
        )
        self.assertTrue(result.valid, result.reason)

    def test_shared_common_year_with_distinct_semesters_remains_valid(self):
        result = self.validate(
            "ปี 3 เทอม 1 กับ เทอม 2 อันไหนมีวิชาเรียนร่วมกัน",
            {"year": 3, "semester": 1},
            {"year": 3, "semester": 2},
        )
        self.assertTrue(result.valid, result.reason)

    def test_year_cue_does_not_match_catalog_digits(self):
        result = self.validate(
            "DSBA 2565 กับ AIT 2566 ปี 2 มีวิชาไหนเรียนร่วมกันบ้าง",
            {"program": "DSBA", "catalog": "2565", "year": 5},
            {"program": "AIT", "catalog": "2566", "year": 2},
        )
        self.assertFalse(result.valid)
        self.assertIn("year is not grounded", result.reason or "")

    def test_prompt_requires_side_local_operand_binding(self):
        prompt = build_semantic_interpreter_prompt("compare two scopes")
        self.assertIn("Preserve operand mention order", prompt)
        self.assertIn("same local operand phrase", prompt)


if __name__ == "__main__":
    unittest.main()
