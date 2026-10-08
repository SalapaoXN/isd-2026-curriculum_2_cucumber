"""Focused top-level and comparison plan-hint authority contract tests."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.semantic.answerer import render_verified_comparison
from rag.semantic.context import merge_semantic_context
from rag.semantic.executor import execute_comparison
from rag.semantic.interpreter import parse_semantic_intent_payload
from rag.semantic.pipeline import semantic_answer
from rag.semantic.pipeline import _canonical_plan_key_candidates
from rag.semantic.prompts import build_semantic_interpreter_prompt
from rag.semantic.resolver import resolve_comparison_operand, resolve_semantic_intent
from rag.semantic.validation import validate_semantic_intent


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


def _payload(*, scope, comparison=None):
    return json.dumps(
        {
            "task": "compare" if comparison is not None else "unknown",
            "subject": "program",
            "relation": None,
            "target": {
                "kind": "none",
                "raw_text": None,
                "normalized_hint": None,
                "ordinal": None,
            },
            "scope": {
                "program": None,
                "catalog": None,
                "plan": None,
                "plan_hint": None,
                "year": None,
                "semester": None,
                **scope,
            },
            "filters": [],
            "aggregation": None,
            "ranking": None,
            "comparison": comparison,
            "requested_fields": [],
            "clarification": None,
            "policy_topic": None,
            "observed_value": None,
        }
    )


def _scope(program="DSBA", catalog="dsba-2565", plan=None, plan_hint=None):
    return {
        "program": program,
        "catalog": catalog,
        "plan": plan,
        "plan_hint": plan_hint,
    }


class PlanHintAuthorityTests(unittest.TestCase):
    def _parse_validate(self, question, scope, comparison=None):
        intent = parse_semantic_intent_payload(
            _payload(scope=scope, comparison=comparison)
        )
        return intent, validate_semantic_intent(intent, question)

    def test_grounded_top_level_plan_hint_resolves_against_catalog(self):
        question = "DSBA dsba-2565 แผนสหกิจ"
        intent, validation = self._parse_validate(
            question,
            _scope(plan="แผนสหกิจ", plan_hint="coop"),
        )
        self.assertTrue(validation.valid, validation.reason)
        merged = merge_semantic_context(intent, None)
        resolved = resolve_semantic_intent(DB_PATH, intent, merged)
        self.assertFalse(resolved.needs_clarification)
        self.assertEqual(resolved.scope.plan, "coop")

    def test_top_level_hint_only_and_ungrounded_raw_plan_fail_closed(self):
        hint_only, hint_validation = self._parse_validate(
            "DSBA dsba-2565 compare plans", _scope(plan_hint="coop")
        )
        self.assertFalse(hint_validation.valid)
        hint_resolved = resolve_semantic_intent(
            DB_PATH, hint_only, merge_semantic_context(hint_only, None)
        )
        self.assertTrue(hint_resolved.needs_clarification)

        ungrounded, ungrounded_validation = self._parse_validate(
            "DSBA dsba-2565 compare plans",
            _scope(plan="cooperative learning route", plan_hint="coop"),
        )
        self.assertFalse(ungrounded_validation.valid)

    def test_invalid_and_wrong_scope_hints_do_not_establish_scope(self):
        question = "DSBA dsba-2565 แผนสหกิจ"
        invalid, validation = self._parse_validate(
            question,
            _scope(plan="แผนสหกิจ", plan_hint="not-a-plan"),
        )
        self.assertFalse(validation.valid)
        invalid_resolved = resolve_semantic_intent(
            DB_PATH, invalid, merge_semantic_context(invalid, None)
        )
        self.assertTrue(invalid_resolved.needs_clarification)

        wrong_scope, validation = self._parse_validate(
            "AIT ait-2566 แผนสหกิจ",
            _scope(
                program="AIT",
                catalog="ait-2566",
                plan="แผนสหกิจ",
                plan_hint="coop",
            ),
        )
        self.assertTrue(validation.valid, validation.reason)
        wrong_scope_resolved = resolve_semantic_intent(
            DB_PATH, wrong_scope, merge_semantic_context(wrong_scope, None)
        )
        self.assertTrue(wrong_scope_resolved.needs_clarification)

    def test_h04_common_term_scope_reaches_both_comparison_sides_end_to_end(self):
        question = (
            "IT แผนสหกิจกับแผนปกติ ปี 3 เทอม 1 อันไหนหน่วยกิตเยอะกว่า"
        )
        comparison = {
            "left": {"plan": "แผนสหกิจ", "plan_hint": "coop"},
            "right": {"plan": "แผนปกติ", "plan_hint": "no_coop"},
            "measure": "credits",
            "operation": "greater",
        }
        common_scope = {
            "program": "IT",
            "catalog": None,
            "plan": None,
            "plan_hint": None,
            "year": 3,
            "semester": 1,
        }
        payload = _payload(scope=common_scope, comparison=comparison)
        response = None
        captured = []
        execute = execute_comparison

        def capture_execution(db_path, resolved):
            verified = execute(db_path, resolved)
            captured.append((resolved, verified))
            return verified

        with patch(
            "rag.semantic.pipeline.execute_comparison",
            side_effect=capture_execution,
        ):
            response = semantic_answer(
                DB_PATH,
                question,
                conversation_context={
                    "program": "IT",
                    "catalog_key": "it-2565",
                    # A common/context plan must not override either side.
                    "plan": "coop",
                },
                interpret_callable=lambda _prompt: payload,
            )

        self.assertTrue(response.trace.validation["valid"])
        self.assertEqual(response.result.status, "answer")
        self.assertEqual(len(captured), 1)
        resolved, verified = captured[0]
        left, right = resolved.comparison_sides
        self.assertEqual(resolved.scope.years, (3,))
        self.assertEqual(resolved.scope.semesters, (1,))
        self.assertEqual(
            (left.scope.program, left.scope.catalog_key, left.scope.plan,
             left.scope.years, left.scope.semesters),
            ("IT", "it-2565", "coop", (3,), (1,)),
        )
        self.assertEqual(
            (right.scope.program, right.scope.catalog_key, right.scope.plan,
             right.scope.years, right.scope.semesters),
            ("IT", "it-2565", "no_coop", (3,), (1,)),
        )
        self.assertEqual(verified.status, "answer")
        self.assertIsNotNone(verified.numeric_comparison)
        self.assertTrue(verified.provenance)
        self.assertEqual(
            response.result.final_answer,
            render_verified_comparison(verified),
        )
        self.assertIn(
            "ทั้งสองฝั่งมีจำนวนหน่วยกิตเท่ากัน",
            response.result.final_answer,
        )
        self.assertNotIn("ทั้งสองวิชา", response.result.final_answer)

    def test_plan_key_candidates_are_read_from_the_current_program_edition(self):
        question = "IT แผนสหกิจกับแผนปกติ ปี 3 เทอม 1 อันไหนหน่วยกิตเยอะกว่า"
        candidates = _canonical_plan_key_candidates(
            DB_PATH,
            question,
            {"program": "IT", "catalog_key": "it-2565"},
            ("IT",),
        )
        self.assertEqual(set(candidates), {"coop", "no_coop"})
        prompt = build_semantic_interpreter_prompt(
            question, canonical_plan_keys=candidates
        )
        self.assertIn("CANONICAL PLAN-KEY CANDIDATES", prompt)
        self.assertIn("coop, no_coop", prompt)
        self.assertIn("plan_hint", prompt)

    def test_comparison_side_scope_precedence_and_identity_isolation(self):
        left = resolve_comparison_operand(
            DB_PATH,
            (
                ("plan", "แผนสหกิจ"),
                ("plan_hint", "coop"),
                ("year", 2),
                ("semester", 2),
            ),
            "IT",
            "it-2565",
            default_years=(3,),
            default_semesters=(1,),
        )
        right = resolve_comparison_operand(
            DB_PATH,
            (("plan", "แผนปกติ"), ("plan_hint", "no_coop")),
            "IT",
            "it-2565",
            default_years=(3,),
            default_semesters=(1,),
        )
        self.assertFalse(left.unresolved, left.reason)
        self.assertFalse(right.unresolved, right.reason)
        self.assertEqual((left.scope.years, left.scope.semesters), ((2,), (2,)))
        self.assertEqual((right.scope.years, right.scope.semesters), ((3,), (1,)))
        self.assertEqual((left.scope.plan, right.scope.plan), ("coop", "no_coop"))

        course_left = resolve_comparison_operand(
            DB_PATH,
            (
                ("plan", "แผนสหกิจ"),
                ("plan_hint", "coop"),
                ("course", "06026200"),
            ),
            "DSBA",
            "dsba-2565",
        )
        course_right = resolve_comparison_operand(
            DB_PATH,
            (("plan", "แผนปกติ"), ("plan_hint", "no_coop")),
            "DSBA",
            "dsba-2565",
        )
        self.assertFalse(course_left.unresolved, course_left.reason)
        self.assertFalse(course_right.unresolved, course_right.reason)
        self.assertEqual(course_left.target.course_code, "06026200")
        self.assertIsNone(course_right.target.course_code)
        self.assertEqual(course_right.target.kind, "none")

    def test_comparison_hint_only_and_ungrounded_sides_fail_validation(self):
        comparison = {
            "left": {"program": "DSBA", "plan_hint": "coop"},
            "right": {"program": "DSBA", "plan": "no_coop"},
            "measure": "credits",
            "operation": "equal",
        }
        question = "DSBA compare plans"
        hint_only, validation = self._parse_validate(
            question,
            _scope(program="DSBA"),
            comparison,
        )
        self.assertFalse(validation.valid)
        unresolved = resolve_comparison_operand(
            DB_PATH, hint_only.comparison.left, "DSBA", "dsba-2565"
        )
        self.assertTrue(unresolved.unresolved)

        ungrounded = {
            "left": {
                "program": "DSBA",
                "plan": "แผนสหกิจ",
                "plan_hint": "coop",
            },
            "right": {"program": "DSBA", "plan": "no_coop"},
            "measure": "credits",
            "operation": "equal",
        }
        intent, validation = self._parse_validate(
            question, _scope(program="DSBA"), ungrounded
        )
        self.assertFalse(validation.valid)
        unresolved = resolve_comparison_operand(
            DB_PATH, intent.comparison.left, "DSBA", "dsba-2565"
        )
        # The resolver has no question argument; current-turn grounding is
        # enforced by validation, while the supported raw phrase determines
        # its meaning when the resolver is called directly.
        self.assertFalse(unresolved.unresolved)

    def test_comparison_hint_not_valid_in_canonical_scope_fails_closed(self):
        unresolved = resolve_comparison_operand(
            DB_PATH,
            (("plan", "แผนสหกิจ"), ("plan_hint", "not-a-plan")),
            "IT",
            "it-2565",
        )
        self.assertTrue(unresolved.unresolved)
        self.assertEqual(
            unresolved.reason,
            "plan hint contradicts deterministic raw plan meaning",
        )

    def test_already_canonical_raw_plan_remains_supported(self):
        question = "DSBA dsba-2565 coop"
        intent, validation = self._parse_validate(
            question, _scope(plan="coop")
        )
        self.assertTrue(validation.valid, validation.reason)
        resolved = resolve_semantic_intent(
            DB_PATH, intent, merge_semantic_context(intent, None)
        )
        self.assertFalse(resolved.needs_clarification)
        self.assertEqual(resolved.scope.plan, "coop")


if __name__ == "__main__":
    unittest.main()
