"""Bounded auxiliary clarification transport without relaxed intent parsing."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.semantic.interpreter import parse_semantic_intent_payload, SemanticSchemaError
from rag.semantic.modes import semantic_ask_response
from rag.semantic.schema import MAX_TEXT_LEN


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"
QUESTION = "เปรียบเทียบ IT หลักสูตร it-2565 กับ DSBA หลักสูตร dsba-2565"
LONG_TEXT = "No quantitative measure or supported relation specified for comparing these two curricula."
CONTEXT = {
    "program": "IT", "catalog_key": "it-2565", "plan": "coop",
    "result_courses": [{"course_code": "06016454", "program": "IT", "catalog_key": "it-2565"}],
    "result_scope_program": "IT",
}


def payload(clarification=LONG_TEXT, operation=None):
    return {
        "task": "compare", "subject": "program", "relation": None,
        "target": {"kind": "none", "raw_text": None, "normalized_hint": None, "ordinal": None},
        "scope": {"program": None, "catalog": None, "plan": None, "plan_hint": None, "year": None, "semester": None},
        "filters": [], "aggregation": None, "ranking": None,
        "comparison": {
            "left": {"program": "IT", "catalog": "it-2565", "plan": None, "plan_hint": None, "year": None, "semester": None, "course": None},
            "right": {"program": "DSBA", "catalog": "dsba-2565", "plan": None, "plan_hint": None, "year": None, "semester": None, "course": None},
            "measure": "credits", "operation": operation,
        },
        "requested_fields": [], "clarification": clarification,
        "policy_topic": None, "observed_value": None,
    }


def parse(data):
    return parse_semantic_intent_payload(json.dumps(data, ensure_ascii=False))


def exact_c3_control():
    """One real interpreter -> pipeline -> API adaptation control, no provider."""
    context = deepcopy(CONTEXT)
    raw = json.dumps(payload(), ensure_ascii=False)
    with patch("rag.semantic.pipeline.resolve_semantic_intent") as resolver, \
         patch("rag.semantic.pipeline.plan_semantic_query") as planner, \
         patch("rag.semantic.pipeline.execute_comparison") as comparison, \
         patch("rag.semantic.pipeline.execute_deterministic") as normal:
        result = semantic_ask_response(DB, QUESTION, context, home_program="IT",
                                       interpret_callable=lambda prompt: raw)
    assert (result["status"], result["action"]) == ("clarification_required", "comparison_operation_required")
    assert result["answer"] and result["provenance"] == []
    assert result["next_context"] == CONTEXT and context == CONTEXT
    assert result.get("comparison") is None
    for spy in (resolver, planner, comparison, normal):
        spy.assert_not_called()
    return result


class ClarificationTransportTests(unittest.TestCase):
    def test_overlength_clarification_preserves_null_operation(self):
        self.assertGreater(len(LONG_TEXT), MAX_TEXT_LEN)
        intent = parse(payload())
        self.assertIsNone(intent.comparison.operation)
        self.assertEqual(intent.clarification, LONG_TEXT[:MAX_TEXT_LEN].strip())

    def test_short_and_optional_clarification_preserved(self):
        self.assertEqual(parse(payload("Choose a comparison operation")).clarification,
                         "Choose a comparison operation")
        self.assertIsNone(parse(payload(None)).clarification)
        self.assertIsNone(parse(payload("  ")).clarification)

    def test_malformed_clarification_types_rejected(self):
        for value in ({"text": "choose"}, ["choose"], 42, True):
            with self.subTest(value=value), self.assertRaisesRegex(
                SemanticSchemaError, "clarification must be a string or null"
            ):
                parse(payload(value))

    def test_structural_and_semantic_text_protections_remain_hard(self):
        data = payload()
        data["unexpected"] = True
        with self.assertRaisesRegex(SemanticSchemaError, "invalid schema"):
            parse(data)
        data = payload()
        data["target"]["raw_text"] = "x" * (MAX_TEXT_LEN + 1)
        with self.assertRaisesRegex(SemanticSchemaError, "exceeds"):
            parse(data)
        data = payload()
        data["comparison"]["left"] = []
        with self.assertRaises(SemanticSchemaError):
            parse(data)

    def test_missing_operation_preserves_context_without_factual_claims(self):
        from rag.semantic.pipeline import semantic_answer
        context = deepcopy(CONTEXT)
        with patch("rag.semantic.pipeline.resolve_semantic_intent") as resolver, \
             patch("rag.semantic.pipeline.plan_semantic_query") as planner, \
             patch("rag.semantic.pipeline.execute_comparison") as executor:
            outcome = semantic_answer(DB, QUESTION, context, home_program="IT",
                                      interpret_callable=lambda prompt: json.dumps(payload()))
        self.assertEqual(outcome.next_context, CONTEXT)
        self.assertEqual(context, CONTEXT)
        self.assertEqual(outcome.result.claims, ())
        self.assertEqual(outcome.result.provenance, ())
        self.assertIsNone(parse(payload()).comparison.operation)
        for spy in (resolver, planner, executor):
            spy.assert_not_called()

    def test_valid_difference_still_executes_canonical_comparison(self):
        data = payload("Compare credits", "difference")
        data["subject"] = "semester"
        for side in ("left", "right"):
            data["comparison"][side].update(plan="coop", plan_hint="coop", year=1, semester=1)
        result = semantic_ask_response(DB,
            "เปรียบเทียบหน่วยกิต IT หลักสูตร it-2565 แผน coop ปี 1 เทอม 1 กับ DSBA หลักสูตร dsba-2565 แผน coop ปี 1 เทอม 1",
            deepcopy(CONTEXT), home_program="IT",
            interpret_callable=lambda prompt: json.dumps(data))
        self.assertEqual(result["status"], "answer")
        self.assertIn("18", result["answer"])
        self.assertTrue(result["provenance"])
        self.assertEqual(result["next_context"], CONTEXT)
