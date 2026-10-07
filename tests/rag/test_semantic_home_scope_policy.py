"""Seven deterministic pipeline policy controls; no provider/evidence calls."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.semantic.pipeline import semantic_answer
from rag.semantic.schema import ComparisonSpec, ScopeMention, SemanticIntent, VerifiedResult


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


class HomeScopePolicyTests(unittest.TestCase):
    def setUp(self):
        self.prior = {
            "program": "IT", "catalog_key": "it-2565", "plan": "coop",
            "years": [1], "semesters": [1],
            "result_courses": [{"course_code": "06016454", "program": "IT",
                                "catalog_key": "it-2565"}],
            "result_scope_program": "IT",
        }

    def run_intent(self, intent, question, home="IT", prior=None, comparison_status="answer"):
        context = deepcopy(self.prior if prior is None else prior)
        before = deepcopy(context)
        with patch("rag.semantic.pipeline.interpret_semantic_intent", return_value=(intent, "test")), \
             patch("rag.semantic.pipeline.plan_semantic_query", wraps=__import__(
                 "rag.semantic.planner", fromlist=["plan_semantic_query"]).plan_semantic_query) as planner, \
             patch("rag.semantic.pipeline.execute_deterministic", return_value=VerifiedResult(status="answer")) as normal, \
             patch("rag.semantic.pipeline.execute_comparison", return_value=VerifiedResult(status=comparison_status)) as comparison, \
             patch("rag.semantic.pipeline.render_semantic_answer", return_value=("verified", "deterministic")):
            result = semantic_answer(DB, question, context, home_program=home,
                                     interpret_callable=lambda prompt: "unused")
        self.assertEqual(context, before)
        return result, planner, normal, comparison

    def test_implicit_home(self):
        result, planner, normal, _ = self.run_intent(
            SemanticIntent(task="list", scope=ScopeMention(year=2, semester=1)),
            "ปี 2 เทอม 1 มีอะไรบ้าง")
        self.assertEqual(result.result.status, "answer")
        self.assertEqual(result.trace.resolved_intent["program"], "IT")
        planner.assert_called_once()
        normal.assert_called_once()

    def test_explicit_same_program(self):
        result, _, normal, _ = self.run_intent(
            SemanticIntent(task="list", scope=ScopeMention(program="IT")), "IT มีอะไรบ้าง")
        self.assertEqual(result.result.status, "answer")
        normal.assert_called_once()

    def test_foreign_normal_stops_before_execution(self):
        result, planner, normal, comparison = self.run_intent(
            SemanticIntent(task="list", scope=ScopeMention(program="DSBA")), "DSBA มีอะไรบ้าง")
        self.assertEqual(result.result.status, "context_conflict")
        self.assertIn("IT", result.result.final_answer)
        self.assertEqual(result.next_context, self.prior)
        planner.assert_not_called()
        normal.assert_not_called()
        comparison.assert_not_called()

    def comparison_intent(self):
        return SemanticIntent(task="compare", subject="program", comparison=ComparisonSpec(
            left=(("program", "IT"), ("catalog", "it-2565"), ("plan", "coop"),
                  ("year", 1), ("semester", 1)),
            right=(("program", "DSBA"), ("catalog", "dsba-2565"), ("plan", "coop"),
                   ("year", 1), ("semester", 1)),
            measure="credits", operation="difference"))

    def test_cross_program_comparison_allowed(self):
        result, planner, normal, comparison = self.run_intent(
            self.comparison_intent(), "เปรียบเทียบ IT it-2565 coop ปี 1 เทอม 1 กับ DSBA dsba-2565 coop ปี 1 เทอม 1")
        self.assertEqual(result.result.status, "answer")
        planner.assert_called_once()
        normal.assert_not_called()
        comparison.assert_called_once()
        sides = comparison.call_args.args[1].comparison_sides
        self.assertEqual([side.scope.program for side in sides], ["IT", "DSBA"])
        self.assertTrue(all(not side.unresolved for side in sides))
        self.assertEqual(result.next_context, self.prior)

    def test_failed_comparison_preserves_normal_state(self):
        result, _, _, comparison = self.run_intent(
            self.comparison_intent(), "เปรียบเทียบ IT it-2565 coop ปี 1 เทอม 1 กับ DSBA dsba-2565 coop ปี 1 เทอม 1",
            comparison_status="missing_data")
        comparison.assert_called_once()
        self.assertNotEqual(result.result.status, "answer")
        self.assertEqual(result.next_context, self.prior)

    def test_unscoped_switch_allowed(self):
        result, _, normal, _ = self.run_intent(
            SemanticIntent(task="list", scope=ScopeMention(program="DSBA", catalog="dsba-2565")),
            "DSBA dsba-2565 มีอะไรบ้าง", home=None)
        self.assertEqual(result.result.status, "answer")
        normal.assert_called_once()
        context = normal.call_args.args[2]
        self.assertEqual(context.program, "DSBA")
        self.assertIsNone(context.plan)
        self.assertEqual(context.semesters, ())

    def test_inconsistent_scoped_context_rejected(self):
        for updates in ({"program": "DSBA"}, {"catalog_key": "dsba-2565"}, {"plan": "invalid-plan"}):
            with self.subTest(updates=updates):
                prior = {**self.prior, **updates}
                result, planner, normal, comparison = self.run_intent(
                    SemanticIntent(task="list"), "มีอะไรบ้าง", prior=prior)
                self.assertEqual(result.result.status, "context_conflict")
                planner.assert_not_called()
                normal.assert_not_called()
                comparison.assert_not_called()
