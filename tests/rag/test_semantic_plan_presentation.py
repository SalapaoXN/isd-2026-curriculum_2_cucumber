"""Display plan keys in Thai without changing canonical state or numbers."""

import unittest

from backend.schemas import AskRequest
from rag.semantic.answerer import (
    _comparison_operand_label, render_semantic_answer, validate_answer_text,
)
from rag.semantic.executor import _claim_line, _side_label
from rag.semantic.schema import (
    ResolvedOperand, ResolvedScope, VerifiedNumericComparison,
    VerifiedNumericComparisonSide, VerifiedResult,
)


def comparison():
    left = ResolvedOperand(scope=ResolvedScope(program="IT", catalog_key="it-2565",
                                              plan="coop", years=(1,), semesters=(1,)))
    right = ResolvedOperand(scope=ResolvedScope(program="DSBA", catalog_key="dsba-2565",
                                               plan="no_coop", years=(1,), semesters=(1,)))
    numeric = VerifiedNumericComparison(measure="credits", requested_operation="difference",
        actual_relation="equal", left=VerifiedNumericComparisonSide(label=_side_label(left), course_code=None, course_name=None, value=18),
        right=VerifiedNumericComparisonSide(label=_side_label(right), course_code=None, course_name=None, value=18), absolute_difference=0)
    result = VerifiedResult(status="answer", numeric_comparison=numeric,
        summary_facts=(f"{numeric.left.label}: 18 หน่วยกิต", f"{numeric.right.label}: 18 หน่วยกิต"))
    return left, right, result


class PlanPresentationTests(unittest.TestCase):
    def test_comparison_fallback_maps_both_plans_and_preserves_numbers(self):
        left, right, result = comparison()
        answer, mode = render_semantic_answer("compare", result, None)
        self.assertEqual(mode, "deterministic")
        self.assertIn("IT แผนสหกิจ", answer)
        self.assertIn("DSBA แผนไม่สหกิจ", answer)
        self.assertNotIn("coop", answer)
        self.assertEqual(answer.count(": 18 หน่วยกิต"), 2)
        self.assertIn("มีผลต่าง 0 หน่วยกิต", answer)
        self.assertEqual((left.scope.plan, right.scope.plan), ("coop", "no_coop"))
        self.assertEqual((result.numeric_comparison.left.value, result.numeric_comparison.right.value,
                          result.numeric_comparison.absolute_difference), (18, 18, 0))

    def test_synthesis_inputs_and_accepted_answer_use_thai_plan_labels(self):
        _, _, result = comparison()
        prompts = []
        def provider(prompt):
            prompts.append(prompt)
            return (f"- {result.numeric_comparison.left.label}: 18 หน่วยกิต\n"
                    f"- {result.numeric_comparison.right.label}: 18 หน่วยกิต\n"
                    "สรุป: ทั้งสองฝั่งมีจำนวนหน่วยกิตเท่ากัน มีผลต่าง 0 หน่วยกิต")
        answer, mode = render_semantic_answer("เปรียบเทียบหน่วยกิต", result, provider)
        self.assertEqual(mode, "grounded_synthesis")
        self.assertIn("แผนสหกิจ", prompts[0])
        self.assertIn("แผนไม่สหกิจ", prompts[0])
        self.assertNotIn("no_coop", prompts[0])
        self.assertNotIn("แผน coop", prompts[0])
        self.assertTrue(validate_answer_text(answer, result))

    def test_raw_or_dropped_plan_labels_reject_synthesis_without_rewriting(self):
        _, _, result = comparison()
        raw = "- IT แผน coop: 18 หน่วยกิต\n- DSBA แผน no_coop: 18 หน่วยกิต\nสรุป: ทั้งสองฝั่งมีจำนวนหน่วยกิตเท่ากัน"
        self.assertFalse(validate_answer_text(raw, result))
        self.assertFalse(validate_answer_text("ทั้งสองฝั่งมีจำนวนหน่วยกิตเท่ากัน 18 หน่วยกิต", result))
        answer, mode = render_semantic_answer("compare", result, lambda prompt: raw)
        self.assertEqual(mode, "deterministic")
        self.assertIn("แผนสหกิจ", answer)
        self.assertIn("แผนไม่สหกิจ", answer)
        self.assertNotIn("coop", answer)

    def test_placement_and_scope_formatters_share_the_display_policy(self):
        left, right, _ = comparison()
        self.assertIn("แผนสหกิจ", _comparison_operand_label(left))
        self.assertIn("แผนไม่สหกิจ", _comparison_operand_label(right))
        for plan, label in (("coop", "แผนสหกิจ"), ("no_coop", "แผนไม่สหกิจ")):
            value = {"course_code": "06016454", "year_number": 1,
                     "semester_number": 1, "plan_key": plan}
            line = _claim_line("placement", value)
            self.assertIn(label, line)
            self.assertNotIn("coop", line)
            self.assertEqual(value["plan_key"], plan)

    def test_clarification_resolution_transport_stays_canonical(self):
        for plan in ("coop", "no_coop"):
            request = AskRequest(question="comparison retry", home_program="IT",
                conversation_context={"program": "IT", "plan": "coop"},
                clarification_resolution={"dimension": "plan", "program": "DSBA",
                                          "operand": "right", "value": plan})
            self.assertEqual(request.clarification_resolution.value, plan)
            self.assertEqual(request.conversation_context["plan"], "coop")

    def test_nonplan_and_exact_course_presentation_is_unchanged(self):
        self.assertEqual(_claim_line("identity", {"course_code": "06016454", "name_en": "UX TOOLS"}),
                         "06016454 UX TOOLS")
        result = VerifiedResult(status="answer", summary_facts=("จำนวน: 10",))
        self.assertEqual(render_semantic_answer("count", result, None), ("- จำนวน: 10", "deterministic"))
        # The presentation guard must not globally ban genuine course-title text.
        self.assertTrue(validate_answer_text("COOP", VerifiedResult(status="answer", summary_facts=("COOP",))))
