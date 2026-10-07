"""Six existing-scope failure adaptation controls; no live providers."""

from pathlib import Path
import unittest
from unittest.mock import patch

from rag.semantic.modes import semantic_ask_response
from rag.semantic.planner import plan_semantic_query
from rag.semantic.schema import ComparisonSpec, ScopeMention, SemanticIntent, VerifiedResult


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


def run_intent(intent, question, context=None, home=None, verified=None):
    with patch("rag.semantic.pipeline.interpret_semantic_intent", return_value=(intent, "test")), \
         patch("rag.semantic.pipeline.plan_semantic_query", wraps=plan_semantic_query) as planner, \
         patch("rag.semantic.pipeline.execute_deterministic", return_value=verified or VerifiedResult(status="answer")) as normal, \
         patch("rag.semantic.pipeline.execute_comparison", return_value=verified or VerifiedResult(status="answer")) as comparison, \
         patch("rag.semantic.pipeline.render_semantic_answer", return_value=("verified", "deterministic")):
        response = semantic_ask_response(
            DB, question, context, home_program=home, interpret_callable=lambda prompt: "unused")
    return response, planner, normal, comparison


class ScopeClarificationTests(unittest.TestCase):
    def test_missing_it_catalog(self):
        response, planner, normal, comparison = run_intent(
            SemanticIntent(task="list", scope=ScopeMention(program="IT")), "IT มีอะไรบ้าง")
        self.assertEqual((response["status"], response["action"]),
                         ("clarification_required", "catalog_required"))
        self.assertIn("IT", response["answer"])
        self.assertIn("ปีหลักสูตร", response["answer"])
        planner.assert_not_called()
        normal.assert_not_called()
        comparison.assert_not_called()

    def test_right_dsba_catalog(self):
        intent = SemanticIntent(task="compare", subject="program", comparison=ComparisonSpec(
            left=(("program", "IT"), ("catalog", "it-2565")),
            right=(("program", "DSBA"),), measure="credits", operation="difference"))
        prior = {"program": "IT", "catalog_key": "it-2565", "plan": "coop"}
        response, planner, normal, comparison = run_intent(
            intent, "เปรียบเทียบ IT it-2565 กับ DSBA", prior, "IT")
        self.assertEqual(response["action"], "catalog_required")
        self.assertIn("DSBA", response["answer"])
        self.assertIn("ฝั่งขวา", response["answer"])
        self.assertEqual(response["next_context"], prior)
        planner.assert_not_called()
        normal.assert_not_called()
        comparison.assert_not_called()

    def test_standalone_gened_catalog(self):
        response, planner, normal, comparison = run_intent(
            SemanticIntent(task="list", scope=ScopeMention(program="GENED")), "GENED มีอะไรบ้าง")
        self.assertEqual(response["action"], "catalog_required")
        self.assertIn("GENED", response["answer"])
        planner.assert_not_called()
        normal.assert_not_called()
        comparison.assert_not_called()

    def test_unsupported_not_catalog_clarification(self):
        response, planner, normal, comparison = run_intent(
            SemanticIntent(task="unknown", scope=ScopeMention(program="IT")), "IT เรียนยากไหม")
        self.assertEqual((response["status"], response["action"]), ("unsupported", "unsupported"))
        planner.assert_not_called()
        normal.assert_not_called()
        comparison.assert_not_called()

    def test_genuine_missing_data_unchanged(self):
        response, _, normal, _ = run_intent(
            SemanticIntent(task="list", scope=ScopeMention(program="IT")), "IT มีอะไรบ้าง",
            {"program": "IT", "catalog_key": "it-2565"},
            verified=VerifiedResult(status="missing_data", missing_information=("evidence unavailable",)))
        self.assertEqual((response["status"], response["action"]),
                         ("insufficient_evidence", "insufficient_evidence"))
        normal.assert_called_once()

    def test_home_conflict_unchanged(self):
        prior = {"program": "IT", "catalog_key": "it-2565"}
        response, planner, normal, comparison = run_intent(
            SemanticIntent(task="list", scope=ScopeMention(program="DSBA")),
            "DSBA มีอะไรบ้าง", prior, "IT")
        self.assertEqual((response["status"], response["action"]), ("context_conflict", "context_conflict"))
        self.assertEqual(response["next_context"], prior)
        planner.assert_not_called()
        normal.assert_not_called()
        comparison.assert_not_called()
