"""P0.1 adversarial answer-fact-integrity RED tests (held-out correctness audit).

These tests prove the confirmed gaps BEFORE any production repair:

- P0-A: ``validate_answer_text`` preserves identifiers but does not bind
  every factual value (credits/placement/prerequisite/negation/extra claim)
  to its canonical fact.
- P0-B: ``_comparison_answer_valid`` requires both side values to appear
  but does not prove each value stays attached to its correct side
  (swapped sides), and ``difference`` verdicts do not check winner wording.

Red cases assert the SAFE behavior (reject forged answer -> deterministic
fallback). They MUST fail under current code for the expected reason
(validator accepts the forged text). Green cases document behavior that
must keep working after the repair.
"""

import unittest
from types import SimpleNamespace

from rag.semantic.answerer import (
    render_semantic_answer,
    render_verified_comparison,
    render_verified_fallback,
    validate_answer_text,
)
from rag.semantic.answerer import _comparison_answer_valid
from rag.semantic.schema import (
    VerifiedNumericComparison,
    VerifiedNumericComparisonSide,
    VerifiedResult,
)


def _single_course_verified(
    code="06016405",
    title="CALCULUS 1",
    credits=3,
    extra_fact=None,
):
    fact = f"{code} {title}: {credits} \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
    if extra_fact:
        fact = f"{fact} | {extra_fact}"
    claim = SimpleNamespace(
        operation="identity",
        status="complete",
        value={"course_code": code, "name_en": title, "credits": credits},
    )
    return VerifiedResult(
        status="answer",
        summary_facts=(fact,),
        claims=(claim,),
        provenance=({"source_filename": "p0-red-control"},),
    )


def _list_verified(rows):
    from rag.semantic.executor import _claim_line

    claim = SimpleNamespace(operation="list", status="complete", value=tuple(rows))
    return VerifiedResult(
        status="answer",
        summary_facts=(_claim_line("list", rows),),
        claims=(claim,),
        provenance=({"source_page": 1},),
    )


def _program_comparison(left_value=129, right_value=132, operation="difference"):
    left = VerifiedNumericComparisonSide(
        label="IT", course_code=None, course_name=None, value=left_value
    )
    right = VerifiedNumericComparisonSide(
        label="DSBA", course_code=None, course_name=None, value=right_value
    )
    if left_value == right_value:
        relation = "equal"
    elif left_value > right_value:
        relation = "left_greater"
    else:
        relation = "right_greater"
    comparison = VerifiedNumericComparison(
        measure="credits",
        requested_operation=operation,
        actual_relation=relation,
        left=left,
        right=right,
        absolute_difference=abs(left_value - right_value),
    )
    verified = VerifiedResult(
        status="answer",
        summary_facts=(
            f"IT: {left_value} \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15",
            f"DSBA: {right_value} \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15",
        ),
        provenance=({"source_filename": "verified-control"},),
        numeric_comparison=comparison,
    )
    return comparison, verified


def _course_comparison():
    left = VerifiedNumericComparisonSide(
        label="CALCULUS 1 (06046400)",
        course_code="06046400",
        course_name="CALCULUS 1",
        value=3,
    )
    right = VerifiedNumericComparisonSide(
        label="CALCULUS 2 (06046401)",
        course_code="06046401",
        course_name="CALCULUS 2",
        value=4,
    )
    comparison = VerifiedNumericComparison(
        measure="credits",
        requested_operation="difference",
        actual_relation="right_greater",
        left=left,
        right=right,
        absolute_difference=1,
    )
    verified = VerifiedResult(
        status="answer",
        summary_facts=(
            "CALCULUS 1 (06046400): 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15",
            "CALCULUS 2 (06046401): 4 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15",
        ),
        provenance=({"source_filename": "verified-control"},),
        numeric_comparison=comparison,
    )
    return comparison, verified


class GenericFactBindingTests(unittest.TestCase):
    def test_correct_code_and_credits_accepted(self):
        verified = _single_course_verified()
        answer = "06016405 CALCULUS 1 \u0e21\u0e35 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        self.assertTrue(validate_answer_text(answer, verified))

    def test_changed_credit_value_rejected(self):
        verified = _single_course_verified()
        forged = "06016405 CALCULUS 1 \u0e21\u0e35 99 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        self.assertFalse(validate_answer_text(forged, verified))

    def test_invented_requirement_rejected(self):
        verified = _single_course_verified()
        forged = (
            "06016405 CALCULUS 1 \u0e21\u0e35 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15 "
            "\u0e41\u0e25\u0e30\u0e15\u0e49\u0e2d\u0e07\u0e2a\u0e2d\u0e1a\u0e20\u0e32\u0e29\u0e32\u0e2d\u0e31\u0e07\u0e01\u0e24\u0e29\u0e40\u0e1e\u0e34\u0e48\u0e21"
        )
        self.assertFalse(validate_answer_text(forged, verified))

    def test_wrong_placement_rejected(self):
        verified = _single_course_verified(
            extra_fact="\u0e0a\u0e31\u0e49\u0e19\u0e1b\u0e35\u0e17\u0e35\u0e48 1 \u0e20\u0e32\u0e04\u0e01\u0e32\u0e23\u0e28\u0e36\u0e01\u0e29\u0e32\u0e17\u0e35\u0e48 1"
        )
        forged = (
            "06016405 CALCULUS 1 \u0e2d\u0e22\u0e39\u0e48\u0e0a\u0e31\u0e49\u0e19\u0e1b\u0e35\u0e17\u0e35\u0e48 3 "
            "\u0e20\u0e32\u0e04\u0e01\u0e32\u0e23\u0e28\u0e36\u0e01\u0e29\u0e32\u0e17\u0e35\u0e48 2"
        )
        self.assertFalse(validate_answer_text(forged, verified))

    def test_wrong_prerequisite_rejected(self):
        verified = _single_course_verified(
            extra_fact="06016405 \u0e21\u0e35\u0e27\u0e34\u0e0a\u0e32\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a\u0e01\u0e48\u0e2d\u0e19: 06016400"
        )
        forged = (
            "06016405 CALCULUS 1 \u0e21\u0e35\u0e27\u0e34\u0e0a\u0e32\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a\u0e01\u0e48\u0e2d\u0e19: 06016999"
        )
        self.assertFalse(validate_answer_text(forged, verified))

    def test_negated_verified_fact_rejected(self):
        verified = _single_course_verified()
        forged = "06016405 CALCULUS 1 \u0e44\u0e21\u0e48\u0e43\u0e0a\u0e48 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        self.assertFalse(validate_answer_text(forged, verified))

    def test_harmless_thai_paraphrase_accepted(self):
        verified = _single_course_verified()
        answer = "\u0e27\u0e34\u0e0a\u0e32 06016405 CALCULUS 1 \u0e21\u0e35\u0e08\u0e33\u0e19\u0e27\u0e19 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\u0e04\u0e23\u0e31\u0e1a"
        self.assertTrue(validate_answer_text(answer, verified))

    def test_formatting_changes_accepted(self):
        verified = _single_course_verified()
        answer = "06016405\nCALCULUS 1\n3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        self.assertTrue(validate_answer_text(answer, verified))

    def test_forged_credit_falls_back_to_deterministic(self):
        verified = _single_course_verified()
        forged = "06016405 CALCULUS 1 \u0e21\u0e35 99 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        expected_fallback = render_verified_fallback("q", verified)
        answer, mode = render_semantic_answer("q", verified, lambda _p: forged)
        self.assertEqual(mode, "deterministic")
        self.assertEqual(answer, expected_fallback)
        self.assertNotIn("99", answer)


class ListValueBindingTests(unittest.TestCase):
    def test_all_pairs_correct_accepted(self):
        rows = [
            {"course_code": "10000001", "name_th": "\u0e0a\u0e37\u0e48\u0e2d\u0e2b\u0e19\u0e36\u0e48\u0e07"},
            {"course_code": "10000002", "name_th": "\u0e0a\u0e37\u0e48\u0e2d\u0e2a\u0e2d\u0e07"},
        ]
        verified = _list_verified(rows)
        answer = "- 10000001 \u2014 \u0e0a\u0e37\u0e48\u0e2d\u0e2b\u0e19\u0e36\u0e48\u0e07\n- 10000002 \u2014 \u0e0a\u0e37\u0e48\u0e2d\u0e2a\u0e2d\u0e07"
        self.assertTrue(validate_answer_text(answer, verified))

    def test_title_paired_with_other_code_rejected(self):
        rows = [
            {"course_code": "10000001", "name_th": "\u0e0a\u0e37\u0e48\u0e2d\u0e2b\u0e19\u0e36\u0e48\u0e07"},
            {"course_code": "10000002", "name_th": "\u0e0a\u0e37\u0e48\u0e2d\u0e2a\u0e2d\u0e07"},
        ]
        verified = _list_verified(rows)
        forged = "- 10000001 \u2014 \u0e0a\u0e37\u0e48\u0e2d\u0e2a\u0e2d\u0e07\n- 10000002 \u2014 \u0e0a\u0e37\u0e48\u0e2d\u0e2b\u0e19\u0e36\u0e48\u0e07"
        self.assertFalse(validate_answer_text(forged, verified))

    def test_credit_moved_to_other_course_rejected(self):
        rows = [
            {"course_code": "10000001", "name_th": "\u0e0a\u0e37\u0e48\u0e2d\u0e2b\u0e19\u0e36\u0e48\u0e07", "credits": 3},
            {"course_code": "10000002", "name_th": "\u0e0a\u0e37\u0e48\u0e2d\u0e2a\u0e2d\u0e07", "credits": 1},
        ]
        verified = _list_verified(rows)
        forged = (
            "- 10000001 \u2014 \u0e0a\u0e37\u0e48\u0e2d\u0e2b\u0e19\u0e36\u0e48\u0e07 1 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- 10000002 \u2014 \u0e0a\u0e37\u0e48\u0e2d\u0e2a\u0e2d\u0e07 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        )
        self.assertFalse(validate_answer_text(forged, verified))

    def test_late_item_swap_rejected(self):
        rows = [
            {"course_code": f"{10000000 + i:08d}", "name_th": f"\u0e0a\u0e37\u0e48\u0e2d {i}"}
            for i in range(10)
        ]
        verified = _list_verified(rows)
        # Late-position title cross-pairing: codes stay in order, but the
        # 9th/10th titles are swapped. Per-line code-title binding must
        # still reject this.
        lines = []
        for i in range(10):
            title_index = 9 - (i - 8) if i >= 8 else i
            lines.append(f"- {10000000 + i:08d} \u2014 \u0e0a\u0e37\u0e48\u0e2d {title_index}")
        forged = "\n".join(lines)
        self.assertFalse(validate_answer_text(forged, verified))

    def test_missing_title_fallback_accepted(self):
        rows = [{"course_code": "10000003"}]
        verified = _list_verified(rows)
        answer, mode = render_semantic_answer("list", verified, None)
        self.assertTrue(validate_answer_text(answer, verified))

    def test_subset_footer_preserved(self):
        rows = [
            {"course_code": f"{10000000 + i:08d}", "name_th": f"\u0e0a\u0e37\u0e48\u0e2d {i}"}
            for i in range(21)
        ]
        verified = _list_verified(rows)
        answer, _ = render_semantic_answer("list", verified, None)
        self.assertIn("\u0e41\u0e2a\u0e14\u0e07 20 \u0e08\u0e32\u0e01 21 \u0e23\u0e32\u0e22\u0e27\u0e34\u0e0a\u0e32", answer)


class NumericComparisonBindingTests(unittest.TestCase):
    def test_correct_program_values_accepted(self):
        comparison, verified = _program_comparison()
        answer = render_verified_comparison(verified)
        self.assertIn("129", answer)
        self.assertIn("132", answer)
        self.assertTrue(_comparison_answer_valid(answer, comparison))
        self.assertTrue(validate_answer_text(answer, verified))

    def test_swapped_side_values_rejected(self):
        comparison, verified = _program_comparison()
        forged = (
            "\u0e40\u0e1b\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e17\u0e35\u0e22\u0e1a\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- IT: 132 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- DSBA: 129 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "\n\u0e2a\u0e23\u0e38\u0e1b: \u0e21\u0e35\u0e1c\u0e25\u0e15\u0e48\u0e32\u0e07 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        )
        self.assertFalse(_comparison_answer_valid(forged, comparison))

    def test_correct_values_wrong_winner_rejected(self):
        comparison, verified = _program_comparison()
        _ = verified
        forged = (
            "\u0e40\u0e1b\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e17\u0e35\u0e22\u0e1a\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- IT: 129 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- DSBA: 132 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "\n\u0e2a\u0e23\u0e38\u0e1b: IT \u0e21\u0e35\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\u0e21\u0e32\u0e01\u0e01\u0e27\u0e48\u0e32 DSBA "
            "\u0e21\u0e35\u0e1c\u0e25\u0e15\u0e48\u0e32\u0e07 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        )
        self.assertFalse(_comparison_answer_valid(forged, comparison))

    def test_wrong_difference_rejected(self):
        comparison, _ = _program_comparison()
        forged = (
            "\u0e40\u0e1b\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e17\u0e35\u0e22\u0e1a\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- IT: 129 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- DSBA: 132 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "\n\u0e2a\u0e23\u0e38\u0e1b: \u0e21\u0e35\u0e1c\u0e25\u0e15\u0e48\u0e32\u0e07 5 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        )
        self.assertFalse(_comparison_answer_valid(forged, comparison))

    def test_correct_difference_one_side_wrong_rejected(self):
        comparison, _ = _program_comparison()
        forged = (
            "\u0e40\u0e1b\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e17\u0e35\u0e22\u0e1a\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- IT: 130 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- DSBA: 132 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "\n\u0e2a\u0e23\u0e38\u0e1b: \u0e21\u0e35\u0e1c\u0e25\u0e15\u0e48\u0e32\u0e07 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        )
        self.assertFalse(_comparison_answer_valid(forged, comparison))

    def test_equal_claiming_greater_rejected(self):
        comparison, _ = _program_comparison(
            left_value=130, right_value=130, operation="equal"
        )
        forged = (
            "\u0e40\u0e1b\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e17\u0e35\u0e22\u0e1a\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- IT: 130 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- DSBA: 130 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "\n\u0e2a\u0e23\u0e38\u0e1b: IT \u0e21\u0e35\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\u0e21\u0e32\u0e01\u0e01\u0e27\u0e48\u0e32 DSBA"
        )
        self.assertFalse(_comparison_answer_valid(forged, comparison))

    def test_course_identity_bound_to_value(self):
        comparison, _ = _course_comparison()
        forged = (
            "\u0e40\u0e1b\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e17\u0e35\u0e22\u0e1a\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- CALCULUS 1 (06046400): 4 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- CALCULUS 2 (06046401): 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "\n\u0e2a\u0e23\u0e38\u0e1b: \u0e21\u0e35\u0e1c\u0e25\u0e15\u0e48\u0e32\u0e07 1 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        )
        self.assertFalse(_comparison_answer_valid(forged, comparison))

    def test_program_scope_bound_to_value(self):
        comparison, _ = _program_comparison()
        forged = (
            "\u0e40\u0e1b\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e17\u0e35\u0e22\u0e1a\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- DSBA: 129 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- IT: 132 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "\n\u0e2a\u0e23\u0e38\u0e1b: \u0e21\u0e35\u0e1c\u0e25\u0e15\u0e48\u0e32\u0e07 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        )
        self.assertFalse(_comparison_answer_valid(forged, comparison))

    def test_swapped_comparison_falls_back_to_deterministic(self):
        comparison, verified = _program_comparison()
        forged = (
            "\u0e40\u0e1b\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e17\u0e35\u0e22\u0e1a\u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- IT: 132 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "- DSBA: 129 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15\n"
            "\n\u0e2a\u0e23\u0e38\u0e1b: \u0e21\u0e35\u0e1c\u0e25\u0e15\u0e48\u0e32\u0e07 3 \u0e2b\u0e19\u0e48\u0e27\u0e22\u0e01\u0e34\u0e15"
        )
        expected = render_verified_comparison(verified)
        answer, mode = render_semantic_answer(
            "compare", verified, lambda _p: forged,
            numeric_comparison=comparison,
        )
        self.assertEqual(mode, "deterministic")
        self.assertEqual(answer, expected)


if __name__ == "__main__":
    unittest.main()
