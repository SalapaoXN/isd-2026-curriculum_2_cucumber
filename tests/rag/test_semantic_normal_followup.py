"""Six bounded normal-list continuation controls."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.semantic.pipeline import semantic_answer
from rag.semantic.schema import ComparisonSpec, ScopeMention, SemanticIntent, VerifiedResult


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


class NormalFollowupTests(unittest.TestCase):
    def setUp(self):
        self.prior = {"program": "IT", "catalog_key": "it-2565", "plan": "coop",
                      "years": [2], "semesters": [1]}
        self.prior["last_normal_operation"] = {"kind": "list_courses", **self.prior}

    def run_turn(self, question, intent, prior=None, home="IT"):
        context = deepcopy(self.prior if prior is None else prior)
        with patch("rag.semantic.pipeline.interpret_semantic_intent", return_value=(intent, "test")) as interpreter, \
             patch("rag.semantic.pipeline.execute_deterministic", return_value=VerifiedResult(status="answer")) as execute, \
             patch("rag.semantic.pipeline.execute_comparison", return_value=VerifiedResult(status="answer")) as compare, \
             patch("rag.semantic.pipeline.render_semantic_answer", return_value=("verified", "deterministic")):
            result = semantic_answer(DB, question, context, home_program=home,
                                     interpret_callable=lambda prompt: "unused")
        return result, execute, compare, interpreter

    def test_basic_temporal_followup(self):
        first, _, _, _ = self.run_turn("ปี 2 เทอม 1 มีอะไรบ้าง",
            SemanticIntent(task="list", scope=ScopeMention(year=2, semester=1)),
            prior={"program": "IT", "catalog_key": "it-2565", "plan": "coop"})
        self.assertEqual(first.result.status, "answer")
        self.assertEqual(first.next_context["last_normal_operation"]["kind"], "list_courses")
        result, execute, _, interpreter = self.run_turn("แล้วเทอม 2 ล่ะ",
            SemanticIntent(task="list", scope=ScopeMention(semester=2)), prior=first.next_context)
        self.assertEqual(result.result.status, "answer")
        scope = execute.call_args.args[2]
        self.assertEqual((scope.program, scope.years, scope.semesters), ("IT", (2,), (2,)))
        self.assertEqual(interpreter.call_args.kwargs["last_normal_operation"],
                         first.next_context["last_normal_operation"])

    def test_explicit_supersession(self):
        result, execute, _, _ = self.run_turn("IT ปี 3 เทอม 1 มีอะไรบ้าง",
            SemanticIntent(task="list", scope=ScopeMention(program="IT", year=3, semester=1)))
        self.assertEqual(result.result.status, "answer")
        self.assertEqual(execute.call_args.args[2].years, (3,))
        self.assertEqual(result.next_context["last_normal_operation"]["years"], [3])

    def test_program_switch_cannot_reuse_operation(self):
        result, execute, _, _ = self.run_turn("DSBA แล้วปี 2 ล่ะ",
            SemanticIntent(task="list", scope=ScopeMention(program="DSBA", year=2)), home=None)
        self.assertNotEqual(result.result.status, "answer")
        execute.assert_not_called()

    def test_scoped_foreign_stays_blocked(self):
        result, execute, compare, _ = self.run_turn("DSBA ปี 2 มีอะไรบ้าง",
            SemanticIntent(task="list", scope=ScopeMention(program="DSBA", year=2)))
        self.assertEqual(result.result.status, "context_conflict")
        execute.assert_not_called()
        compare.assert_not_called()

    def test_comparison_disables_normal_continuation(self):
        intent = SemanticIntent(task="compare", subject="program", comparison=ComparisonSpec(
            left=(("program", "IT"), ("catalog", "it-2565"), ("plan", "coop")),
            right=(("program", "DSBA"), ("catalog", "dsba-2565"), ("plan", "coop")),
            measure="credits", operation="difference"))
        result, _, compare, _ = self.run_turn(
            "เปรียบเทียบ IT it-2565 coop กับ DSBA dsba-2565 coop", intent)
        self.assertEqual(result.result.status, "answer")
        compare.assert_called_once()
        self.assertNotIn("last_normal_operation", result.next_context)
        self.assertEqual(result.next_context["program"], "IT")
        followup, execute, _, _ = self.run_turn("แล้วปี 2 ล่ะ",
            SemanticIntent(task="list", scope=ScopeMention(year=2)), prior=result.next_context)
        self.assertNotEqual(followup.result.status, "answer")
        execute.assert_not_called()

    def test_absent_or_malformed_operation_fails_closed(self):
        for operation in (None, {"kind": "list_courses", "text": "arbitrary history"},
                          {**self.prior["last_normal_operation"], "program": "DSBA"}):
            with self.subTest(operation=operation):
                prior = {**self.prior, "last_normal_operation": operation}
                result, execute, _, interpreter = self.run_turn("แล้วเทอม 2 ล่ะ",
                    SemanticIntent(task="list", scope=ScopeMention(semester=2)), prior=prior)
                self.assertNotEqual(result.result.status, "answer")
                execute.assert_not_called()
                self.assertIsNone(interpreter.call_args.kwargs["last_normal_operation"])
