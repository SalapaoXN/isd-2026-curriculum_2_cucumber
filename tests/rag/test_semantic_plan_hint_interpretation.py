"""Producer and consumer controls for grounded raw plan + canonical hint."""

import json
from pathlib import Path
import unittest

from rag.semantic.context import merge_semantic_context
from rag.semantic.interpreter import interpret_semantic_intent, parse_semantic_intent_payload
from rag.semantic.prompts import (
    SEMANTIC_INTERPRETER_PROMPT_VERSION,
    build_semantic_interpreter_prompt,
)
from rag.semantic.resolver import resolve_comparison_operand, resolve_semantic_intent
from rag.semantic.validation import validate_semantic_intent


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


def _payload(*, question, plan=None, plan_hint=None, program="IT", catalog=None,
             year=2, semester=2, comparison=None):
    task = "compare" if comparison is not None else "aggregate"
    return json.dumps(
        {
            "task": task,
            "subject": "semester" if comparison is None else "program",
            "relation": None,
            "target": {"kind": "none", "raw_text": None,
                       "normalized_hint": None, "ordinal": None},
            "scope": {"program": program, "catalog": catalog,
                      "plan": plan, "plan_hint": plan_hint,
                      "year": year, "semester": semester},
            "filters": [],
            "aggregation": (None if comparison is not None else
                            {"function": "sum", "measure": "credits", "group_by": []}),
            "ranking": None,
            "comparison": comparison,
            "requested_fields": [],
            "clarification": None,
            "policy_topic": None,
            "observed_value": None,
        },
        ensure_ascii=False,
    )


class SemanticPlanHintInterpretationTests(unittest.TestCase):
    def _interpret(self, question, raw_plan, hint):
        payload = _payload(question=question, plan=raw_plan, plan_hint=hint,
                           program="IT", catalog=None, year=2, semester=2)
        intent, _ = interpret_semantic_intent(question, lambda _prompt: payload)
        return intent, validate_semantic_intent(intent, question)

    def _resolve(self, intent):
        merged = merge_semantic_context(
            intent, {"program": "IT", "catalog_key": "it-2565"}
        )
        return resolve_semantic_intent(DB, intent, merged)

    def test_thai_coop_raw_phrase_plus_hint_resolves(self):
        for raw in ("แบบสหกิจ", "แผนสหกิจ", "สหกิจ"):
            with self.subTest(raw=raw):
                question = f"IT {raw} ปี 2 เทอม 2 รวมหน่วยกิตเท่าไร"
                intent, validation = self._interpret(question, raw, "coop")
                self.assertTrue(validation.valid, validation.reason)
                self.assertIn(intent.scope.plan, question)
                self.assertEqual(self._resolve(intent).scope.plan, "coop")

    def test_thai_no_coop_raw_phrase_plus_hint_resolves(self):
        for raw in ("แบบไม่สหกิจ", "แผนไม่สหกิจ", "ไม่สหกิจ"):
            with self.subTest(raw=raw):
                question = f"IT {raw} ปี 2 เทอม 2 รวมหน่วยกิตเท่าไร"
                intent, validation = self._interpret(question, raw, "no_coop")
                self.assertTrue(validation.valid, validation.reason)
                self.assertIn(intent.scope.plan, question)
                self.assertEqual(self._resolve(intent).scope.plan, "no_coop")

    def test_contradictory_top_level_hints_fail_validation_and_resolver(self):
        cases = (
            ("IT แบบสหกิจ ปี 2 เทอม 2", "แบบสหกิจ", "no_coop"),
            ("IT แบบไม่สหกิจ ปี 2 เทอม 2", "แบบไม่สหกิจ", "coop"),
        )
        for question, raw, hint in cases:
            with self.subTest(raw=raw, hint=hint):
                intent, validation = self._interpret(question, raw, hint)
                self.assertFalse(validation.valid)
                self.assertIn("plan_hint", validation.reason)
                resolved = self._resolve(intent)
                self.assertTrue(resolved.needs_clarification)

    def test_hint_only_ungrounded_and_unsupported_phrase_fail_closed(self):
        question, raw, hint = "IT ปี 2 เทอม 2", None, "coop"
        intent, validation = self._interpret(question, raw, hint)
        self.assertFalse(validation.valid)
        self.assertTrue(self._resolve(intent).needs_clarification)

        question, raw, hint = "IT แบบสหกิจ ปี 2 เทอม 2", "แบบสหกิจ", "coop"
        intent, validation = self._interpret(question, raw, hint)
        self.assertTrue(validation.valid, validation.reason)
        self.assertEqual(self._resolve(intent).scope.plan, "coop")

        question, raw, hint = (
            "IT cooperative learning route ปี 2 เทอม 2",
            "cooperative learning route",
            "coop",
        )
        intent, validation = self._interpret(question, raw, hint)
        self.assertFalse(validation.valid)
        self.assertTrue(self._resolve(intent).needs_clarification)

    def test_ungrounded_raw_plan_fails_current_question_validation(self):
        question, raw, hint = "IT ปี 2 เทอม 2", "สหกิจ", "coop"
        intent, validation = self._interpret(question, raw, hint)
        self.assertFalse(validation.valid)


    def test_canonical_raw_keys_resolve_without_hint(self):
        for raw in ("coop", "no_coop"):
            with self.subTest(raw=raw):
                question = f"IT {raw} ปี 2 เทอม 2 รวมหน่วยกิตเท่าไร"
                intent, validation = self._interpret(question, raw, None)
                self.assertTrue(validation.valid, validation.reason)
                self.assertEqual(self._resolve(intent).scope.plan, raw)

    def test_wrong_hint_cannot_override_canonical_raw_key(self):
        question = "IT coop ปี 2 เทอม 2"
        intent, validation = self._interpret(question, "coop", "no_coop")
        self.assertFalse(validation.valid)
        self.assertTrue(self._resolve(intent).needs_clarification)

    def test_supported_phrase_unavailable_in_selected_scope_fails_closed(self):
        question = "IT default ปี 2 เทอม 2"
        intent, validation = self._interpret(question, "default", "default")
        self.assertTrue(validation.valid, validation.reason)
        self.assertTrue(self._resolve(intent).needs_clarification)

    def test_comparison_sides_use_independent_raw_plan_hints(self):
        question = "IT แผนสหกิจกับแบบไม่สหกิจ ปี 2 เทอม 2 อันไหนหน่วยกิตมากกว่า"
        comparison = {
            "left": {"program": "IT", "plan": "แผนสหกิจ", "plan_hint": "coop",
                     "year": 2, "semester": 2},
            "right": {"program": "IT", "plan": "แบบไม่สหกิจ", "plan_hint": "no_coop",
                      "year": 2, "semester": 2},
            "measure": "credits", "operation": "greater",
        }
        intent = parse_semantic_intent_payload(_payload(
            question=question, plan=None, plan_hint=None, program="IT",
            year=2, semester=2, comparison=comparison,
        ))
        validation = validate_semantic_intent(intent, question)
        self.assertTrue(validation.valid, validation.reason)
        left = resolve_comparison_operand(
            DB, intent.comparison.left, "IT", "it-2565"
        )
        right = resolve_comparison_operand(
            DB, intent.comparison.right, "IT", "it-2565"
        )
        self.assertEqual((left.unresolved, left.scope.plan), (False, "coop"))
        self.assertEqual((right.unresolved, right.scope.plan), (False, "no_coop"))

    def test_wrong_comparison_side_hint_fails_without_cross_side_leakage(self):
        question = "IT แผนสหกิจกับแบบไม่สหกิจ ปี 2 เทอม 2 อันไหนหน่วยกิตมากกว่า"
        comparison = {
            "left": {"program": "IT", "plan": "แผนสหกิจ", "plan_hint": "no_coop"},
            "right": {"program": "IT", "plan": "แบบไม่สหกิจ", "plan_hint": "no_coop"},
            "measure": "credits", "operation": "greater",
        }
        intent = parse_semantic_intent_payload(_payload(
            question=question, plan=None, plan_hint=None, program="IT",
            year=2, semester=2, comparison=comparison,
        ))
        validation = validate_semantic_intent(intent, question)
        self.assertFalse(validation.valid)
        left = resolve_comparison_operand(DB, intent.comparison.left, "IT", "it-2565")
        right = resolve_comparison_operand(DB, intent.comparison.right, "IT", "it-2565")
        self.assertTrue(left.unresolved)
        self.assertEqual(right.scope.plan, "no_coop")

    def test_comparison_side_without_plan_does_not_inherit_other_side_plan(self):
        left = resolve_comparison_operand(
            DB,
            (("program", "IT"), ("plan", "แผนสหกิจ"), ("plan_hint", "coop")),
            "IT",
            "it-2565",
        )
        right = resolve_comparison_operand(
            DB,
            (("program", "IT"),),
            "IT",
            "it-2565",
        )
        self.assertEqual(left.scope.plan, "coop")
        self.assertIsNone(right.scope.plan)

    def test_prompt_v11_quotes_raw_phrase_and_teaches_both_plan_directions(self):
        prompt = build_semantic_interpreter_prompt(
            "synthetic plan prompt check", canonical_plan_keys=("coop", "no_coop")
        )
        self.assertEqual(SEMANTIC_INTERPRETER_PROMPT_VERSION, "semantic-interpreter/v11")
        self.assertIn("RAW PLAN IS A QUOTE-LIKE GROUNDING FIELD", prompt)
        self.assertIn('"plan":"แผนสหกิจ"', prompt)
        self.assertIn('"plan_hint":"coop"', prompt)
        self.assertIn('"plan":"แบบไม่สหกิจ"', prompt)
        self.assertIn('"plan_hint":"no_coop"', prompt)
        self.assertIn("unless the student literally wrote that key", prompt)


if __name__ == "__main__":
    unittest.main()
