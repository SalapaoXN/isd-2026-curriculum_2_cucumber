"""Six bounded comparison-operation contract and fallback controls."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.semantic.executor import execute_comparison
from rag.semantic.modes import semantic_ask_response
from rag.semantic.prompts import build_semantic_interpreter_prompt
from rag.semantic.schema import ComparisonSpec, ScopeMention, SemanticIntent


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"
QUESTION = "เปรียบเทียบหน่วยกิต IT หลักสูตร it-2565 แผน coop ปี 1 เทอม 1 กับ DSBA หลักสูตร dsba-2565 แผน coop ปี 1 เทอม 1"


class ComparisonOperationTests(unittest.TestCase):
    def setUp(self):
        self.context = {"program": "IT", "catalog_key": "it-2565", "plan": "coop",
                        "result_courses": [{"course_code": "06016454", "program": "IT", "catalog_key": "it-2565"}],
                        "result_scope_program": "IT"}

    def intent(self, operation):
        return SemanticIntent(task="compare", subject="semester", comparison=ComparisonSpec(
            left=(("program", "IT"), ("catalog", "it-2565"), ("plan", "coop"), ("year", 1), ("semester", 1)),
            right=(("program", "DSBA"), ("catalog", "dsba-2565"), ("plan", "coop"), ("year", 1), ("semester", 1)),
            measure="credits", operation=operation))

    def run_intent(self, intent, question=QUESTION):
        context = deepcopy(self.context)
        with patch("rag.semantic.pipeline.interpret_semantic_intent", return_value=(intent, "test")), \
             patch("rag.semantic.pipeline.resolve_semantic_intent", wraps=__import__(
                 "rag.semantic.resolver", fromlist=["resolve_semantic_intent"]).resolve_semantic_intent) as resolver, \
             patch("rag.semantic.pipeline.plan_semantic_query", wraps=__import__(
                 "rag.semantic.planner", fromlist=["plan_semantic_query"]).plan_semantic_query) as planner, \
             patch("rag.semantic.pipeline.execute_comparison", wraps=execute_comparison) as comparison, \
             patch("rag.semantic.pipeline.execute_deterministic") as normal:
            response = semantic_ask_response(DB, question, context, home_program="IT",
                                             interpret_callable=lambda prompt: "unused")
        self.assertEqual(context, self.context)
        return response, resolver, planner, comparison, normal

    def test_prompt_constrains_explicit_quantitative_not_arbitrary_comparison(self):
        prompt = build_semantic_interpreter_prompt(QUESTION)
        self.assertIn("explicitly asks to compare credit amounts", prompt)
        self.assertIn("Do not omit operation", prompt)
        self.assertIn("do not justify difference: leave operation null", prompt)

    def test_valid_difference_preserves_real_comparison(self):
        response, resolver, planner, comparison, normal = self.run_intent(self.intent("difference"))
        self.assertEqual(response["status"], "answer")
        self.assertIn("18", response["answer"])
        resolver.assert_called_once()
        planner.assert_called_once()
        comparison.assert_called_once()
        normal.assert_not_called()

    def test_null_operation_clarifies_before_factual_stages(self):
        intent = self.intent(None)
        response, resolver, planner, comparison, normal = self.run_intent(intent)
        self.assertEqual((response["status"], response["action"]),
                         ("clarification_required", "comparison_operation_required"))
        self.assertIn("ส่วนต่าง", response["answer"])
        self.assertIsNone(intent.comparison.operation)
        for spy in (resolver, planner, comparison, normal): spy.assert_not_called()

    def test_truly_underspecified_comparison_does_not_get_default(self):
        intent = SemanticIntent(task="compare", subject="program", comparison=ComparisonSpec(
            left=(("program", "IT"),), right=(("program", "DSBA"),),
            measure="credits", operation=None))
        response, resolver, planner, comparison, normal = self.run_intent(intent, "เปรียบเทียบ IT กับ DSBA")
        self.assertEqual(response["action"], "comparison_operation_required")
        self.assertIsNone(intent.comparison.operation)
        for spy in (resolver, planner, comparison, normal): spy.assert_not_called()

    def test_null_operation_preserves_home_normal_results(self):
        response, _, _, _, _ = self.run_intent(self.intent(None))
        self.assertEqual(response["next_context"], self.context)
        self.assertEqual(response["next_context"]["program"], "IT")
        self.assertNotIn("comparison", response["next_context"])

    def test_d1_unsupported_unchanged(self):
        response, resolver, planner, comparison, normal = self.run_intent(
            SemanticIntent(task="unknown", scope=ScopeMention(program="IT")), "IT 06016454 เรียนยากไหม")
        self.assertEqual((response["status"], response["action"]), ("unsupported", "unsupported"))
        for spy in (resolver, planner, comparison, normal): spy.assert_not_called()
