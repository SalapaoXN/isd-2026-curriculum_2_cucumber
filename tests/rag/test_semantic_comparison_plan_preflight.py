"""Six controls for the existing numeric term-comparison plan requirement."""

import unittest

from rag.semantic.planner import missing_comparison_plan
from rag.semantic.schema import (
    AggregationSpec, ComparisonSpec, ResolvedIntent, ResolvedOperand,
    ResolvedScope, SemanticIntent,
)
from tests.rag.test_semantic_scope_clarification import run_intent


class ComparisonPlanPreflightTests(unittest.TestCase):
    def intent(self, left_plan=None, right_plan=None, term=True, courses=False):
        left = [("program", "IT"), ("catalog", "it-2565")]
        right = [("program", "DSBA"), ("catalog", "dsba-2565")]
        for side, plan in ((left, left_plan), (right, right_plan)):
            if term:
                side.append(("year", 1))
            if plan is not None:
                side.append(("plan", plan))
        if courses:
            left.append(("course", "06016454"))
            right.append(("course", "06026201"))
        return SemanticIntent(task="compare", subject="program", comparison=ComparisonSpec(
            left=tuple(left), right=tuple(right), measure="credits", operation="difference"))

    def run_comparison(self, **kwargs):
        return run_intent(
            self.intent(**kwargs),
            "เปรียบเทียบ IT it-2565 coop กับ DSBA dsba-2565 no_coop ปี 1"
            + (" 06016454 06026201" if kwargs.get("courses") else ""))

    def test_right_missing_plan(self):
        response, planner, normal, comparison = self.run_comparison(left_plan="coop")
        self.assertEqual((response["status"], response["action"]),
                         ("clarification_required", "plan_required"))
        self.assertIn("DSBA", response["answer"])
        self.assertIn("ฝั่งขวา", response["answer"])
        planner.assert_not_called()
        normal.assert_not_called()
        comparison.assert_not_called()

    def test_left_missing_plan(self):
        response, planner, normal, comparison = self.run_comparison(right_plan="no_coop")
        self.assertEqual(response["action"], "plan_required")
        self.assertIn("IT", response["answer"])
        self.assertIn("ฝั่งซ้าย", response["answer"])
        planner.assert_not_called()
        normal.assert_not_called()
        comparison.assert_not_called()

    def test_both_plans_proceed(self):
        response, planner, _, comparison = self.run_comparison(left_plan="coop", right_plan="no_coop")
        self.assertEqual(response["status"], "answer")
        planner.assert_called_once()
        comparison.assert_called_once()

    def test_exact_courses_need_no_plan(self):
        response, _, _, comparison = self.run_comparison(courses=True)
        self.assertEqual(response["status"], "answer")
        comparison.assert_called_once()

    def test_bare_program_totals_need_no_plan(self):
        response, _, _, comparison = self.run_comparison(term=False)
        self.assertEqual(response["status"], "answer")
        comparison.assert_called_once()

    def test_planless_supported_families(self):
        scope = ResolvedScope(program="IT", catalog_key="it-2565", years=(1,))
        intents = (
            SemanticIntent(task="list"),
            SemanticIntent(task="aggregate", aggregation=AggregationSpec(function="sum", measure="credits")),
            SemanticIntent(task="compare", comparison=ComparisonSpec(
                measure="course_count", operation="overlap")),
        )
        for intent in intents:
            with self.subTest(task=intent.task):
                resolved = ResolvedIntent(intent=intent, scope=scope,
                    comparison_sides=(ResolvedOperand(scope=scope), ResolvedOperand(scope=scope)))
                self.assertIsNone(missing_comparison_plan(resolved))
