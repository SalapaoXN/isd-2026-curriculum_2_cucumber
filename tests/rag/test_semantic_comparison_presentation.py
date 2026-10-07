"""Typed numeric-comparison presentation and answer-guard controls."""

import unittest

from rag.semantic.answerer import render_semantic_answer, render_verified_comparison
from rag.semantic.schema import (
    VerifiedNumericComparison,
    VerifiedNumericComparisonSide,
    VerifiedResult,
)


def _comparison(
    *,
    operation: str,
    relation: str,
    left: VerifiedNumericComparisonSide,
    right: VerifiedNumericComparisonSide,
    difference: int | float,
) -> VerifiedNumericComparison:
    return VerifiedNumericComparison(
        measure="credits",
        requested_operation=operation,
        actual_relation=relation,
        left=left,
        right=right,
        absolute_difference=difference,
    )


def _verified(comparison: VerifiedNumericComparison) -> VerifiedResult:
    return VerifiedResult(
        status="answer",
        summary_facts=(
            f"{comparison.left.label}: {comparison.left.value} หน่วยกิต",
            f"{comparison.right.label}: {comparison.right.value} หน่วยกิต",
        ),
        provenance=({"source_filename": "verified-control"},),
        numeric_comparison=comparison,
    )


class SemanticComparisonPresentationTests(unittest.TestCase):
    def test_equal_course_sides_keep_course_specific_wording(self):
        comparison = _comparison(
            operation="equal",
            relation="equal",
            left=VerifiedNumericComparisonSide(
                label="CALCULUS 1 (06046400)",
                course_code="06046400",
                course_name="CALCULUS 1",
                value=3,
            ),
            right=VerifiedNumericComparisonSide(
                label="CALCULUS 2 (06046401)",
                course_code="06046401",
                course_name="CALCULUS 2",
                value=3,
            ),
            difference=0,
        )
        verified = _verified(comparison)
        answer = render_verified_comparison(verified)
        self.assertIn("ทั้งสองวิชามีจำนวนหน่วยกิตเท่ากัน", answer)
        guarded_answer, mode = render_semantic_answer(
            "compare course credits", verified, lambda _prompt: answer
        )
        self.assertEqual(mode, "grounded_synthesis")
        self.assertEqual(guarded_answer, answer)

    def test_equal_non_course_sides_use_generic_wording(self):
        comparison = _comparison(
            operation="equal",
            relation="equal",
            left=VerifiedNumericComparisonSide(
                label="IT แผน coop ปี 3 เทอม 1",
                course_code=None,
                course_name=None,
                value=33,
            ),
            right=VerifiedNumericComparisonSide(
                label="IT แผน no_coop ปี 3 เทอม 1",
                course_code=None,
                course_name=None,
                value=33,
            ),
            difference=0,
        )
        verified = _verified(comparison)
        answer = render_verified_comparison(verified)
        self.assertIn("ทั้งสองฝั่งมีจำนวนหน่วยกิตเท่ากัน", answer)
        self.assertNotIn("ทั้งสองวิชา", answer)
        guarded_answer, mode = render_semantic_answer(
            "compare plan credits", verified, lambda _prompt: answer
        )
        self.assertEqual(mode, "grounded_synthesis")
        self.assertEqual(guarded_answer, answer)

    def test_unequal_course_winner_remains_tied_to_verified_relation(self):
        comparison = _comparison(
            operation="greater",
            relation="left_greater",
            left=VerifiedNumericComparisonSide(
                label="Course A",
                course_code="10000001",
                course_name="Course A",
                value=4,
            ),
            right=VerifiedNumericComparisonSide(
                label="Course B",
                course_code="10000002",
                course_name="Course B",
                value=3,
            ),
            difference=1,
        )
        answer = render_verified_comparison(_verified(comparison))
        self.assertIn("Course A มีหน่วยกิตมากกว่า Course B", answer)
        self.assertNotIn("Course B มีหน่วยกิตมากกว่า Course A", answer)

    def test_difference_uses_verified_absolute_difference(self):
        comparison = _comparison(
            operation="difference",
            relation="left_greater",
            left=VerifiedNumericComparisonSide(
                label="Course A",
                course_code="10000001",
                course_name="Course A",
                value=5,
            ),
            right=VerifiedNumericComparisonSide(
                label="Course B",
                course_code="10000002",
                course_name="Course B",
                value=3,
            ),
            difference=2,
        )
        verified = _verified(comparison)
        answer = render_verified_comparison(verified)
        self.assertIn("มีผลต่าง 2 หน่วยกิต", answer)
        guarded_answer, mode = render_semantic_answer(
            "compare credit difference", verified, lambda _prompt: answer
        )
        self.assertEqual(mode, "grounded_synthesis")
        self.assertEqual(guarded_answer, answer)


if __name__ == "__main__":
    unittest.main()
