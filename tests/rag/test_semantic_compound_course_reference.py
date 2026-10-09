"""Regression contracts for exact course title plus embedded code references."""

import json
from pathlib import Path
import unittest

from rag.hybrid_demo import DEFAULT_CURRICULUM_DB_PATH
from rag.semantic.context import MergedContext
from rag.semantic.pipeline import semantic_answer
from rag.semantic.resolver import resolve_comparison_operand, resolve_semantic_intent
from rag.semantic.schema import SemanticIntent, SemanticTarget, ScopeMention


DB_PATH = Path(DEFAULT_CURRICULUM_DB_PATH)


def _intent_payload(*, relation, target, requested_field, scope=None):
    return json.dumps(
        {
            "task": "lookup",
            "subject": "course",
            "relation": relation,
            "target": {
                "kind": "literal",
                "raw_text": target,
                "normalized_hint": None,
                "ordinal": None,
            },
            "scope": scope or {
                "program": None,
                "catalog": None,
                "plan": None,
                "year": None,
                "semester": None,
            },
            "filters": [],
            "aggregation": None,
            "ranking": None,
            "comparison": None,
            "requested_fields": [requested_field],
            "clarification": None,
            "policy_topic": None,
            "observed_value": None,
        },
        ensure_ascii=False,
    )


class SemanticCompoundCourseReferenceTests(unittest.TestCase):
    def resolve_literal(self, reference, *, program="IT", catalog="it-2565"):
        intent = SemanticIntent(
            task="lookup",
            subject="course",
            relation="placement",
            target=SemanticTarget(kind="literal", raw_text=reference),
            scope=ScopeMention(program=program, catalog=catalog, plan="no_coop"),
        )
        merged = MergedContext(
            program=program, catalog_key=catalog, plan="no_coop",
        )
        return resolve_semantic_intent(DB_PATH, intent, merged)

    def assert_resolves(self, reference, course_code):
        resolved = self.resolve_literal(reference)
        self.assertFalse(resolved.needs_clarification)
        self.assertEqual(resolved.target.course_code, course_code)
        self.assertEqual(resolved.target.program, "IT")
        self.assertEqual(resolved.target.catalog_key, "it-2565")
        return resolved

    def test_project_title_and_code_resolve_canonical_identity(self):
        self.assert_resolves("PROJECT 1 (06016406)", "06016406")
        self.assert_resolves("  project   1 ( 06016406 )  ", "06016406")

    def test_server_side_title_and_code_resolve_under_no_coop_scope(self):
        resolved = self.assert_resolves(
            "SERVER SIDE WEB DEVELOPMENT (06016418)", "06016418",
        )
        self.assertEqual(resolved.scope.plan, "no_coop")

    def test_nosql_title_and_code_resolve_canonical_identity(self):
        self.assert_resolves("NOSQL DATABASE SYSTEMS (06016414)", "06016414")

    def test_conflicting_title_with_valid_code_fails_closed(self):
        resolved = self.resolve_literal("NOSQL DATABASE SYSTEMS (06016406)")

        self.assertTrue(resolved.needs_clarification)
        self.assertIsNone(resolved.target.course_code)

    def test_embedded_code_cannot_escape_program_or_catalog_scope(self):
        wrong_program = self.resolve_literal(
            "PROJECT 1 (06016406)", program="DSBA", catalog="dsba-2565",
        )
        wrong_catalog = self.resolve_literal(
            "PROJECT 1 (06016406)", catalog="it-2560",
        )

        for resolved in (wrong_program, wrong_catalog):
            with self.subTest(reason=resolved.clarification_reason):
                self.assertTrue(resolved.needs_clarification)
                self.assertIsNone(resolved.target.course_code)

    def test_malformed_or_multiple_embedded_codes_fail_closed(self):
        for reference in (
            "PROJECT 1 (0601640)",
            "PROJECT 1 (106016406)",
            "PROJECT 1 (06016406) and 06016418",
        ):
            with self.subTest(reference=reference):
                resolved = self.resolve_literal(reference)
                self.assertTrue(resolved.needs_clarification)
                self.assertIsNone(resolved.target.course_code)

    def test_pure_code_and_exact_title_resolution_remain_supported(self):
        self.assert_resolves("06016406", "06016406")
        self.assert_resolves("PROJECT 1", "06016406")

    def test_partial_title_discovery_remains_supported(self):
        resolved = self.resolve_literal("SERVER SIDE WEB")

        self.assertFalse(resolved.needs_clarification)
        self.assertEqual(resolved.target.course_code, "06016418")

    def test_comparison_operand_uses_same_scoped_compound_reference_policy(self):
        operand = resolve_comparison_operand(
            DB_PATH,
            (
                ("course", "SERVER SIDE WEB DEVELOPMENT (06016418)"),
                ("program", "IT"),
                ("catalog", "it-2565"),
                ("plan", "no_coop"),
            ),
            default_program=None,
            default_catalog_key=None,
        )

        self.assertFalse(operand.unresolved)
        self.assertEqual(operand.scope.program, "IT")
        self.assertEqual(operand.scope.catalog_key, "it-2565")
        self.assertEqual(operand.scope.plan, "no_coop")
        self.assertEqual(operand.target.course_code, "06016418")

    def test_project_compound_reference_reaches_verified_description(self):
        question = "PROJECT 1 (06016406) ให้นักศึกษาทำงานและศึกษาประเด็นลักษณะใดบ้าง?"
        payload = _intent_payload(
            relation="description",
            target="PROJECT 1 (06016406)",
            requested_field="description",
        )
        outcome = semantic_answer(
            DB_PATH,
            question,
            {"program": "IT", "catalog_key": "it-2565", "plan": "no_coop"},
            home_program="IT",
            interpret_callable=lambda _prompt: payload,
        )

        self.assertEqual(outcome.result.status, "answer")
        descriptions = [
            claim for claim in outcome.result.claims
            if claim.operation == "describe"
        ]
        self.assertTrue(descriptions)
        description_evidence = " ".join(
            str(evidence)
            for claim in descriptions
            for evidence in claim.evidence
        )
        self.assertIn("06016406", description_evidence)
        self.assertIn("STUDY OR RESEARCH OF INTERESTING CURRENT TOPICS", description_evidence)
        self.assertIn("06016406", outcome.result.final_answer)
        self.assertIn("PROJECT 1", outcome.result.final_answer)
        self.assertIn("STUDY OR RESEARCH OF INTERESTING CURRENT TOPICS", outcome.result.final_answer)

    def test_server_side_compound_reference_reaches_no_coop_placement(self):
        question = (
            "SERVER SIDE WEB DEVELOPMENT (06016418) ใน IT "
            "แผนไม่สหกิจอยู่ปีไหน เทอมไหน?"
        )
        payload = _intent_payload(
            relation="placement",
            target="SERVER SIDE WEB DEVELOPMENT (06016418)",
            requested_field="placement",
            scope={
                "program": "IT",
                "catalog": None,
                "plan": "แผนไม่สหกิจ",
                "plan_hint": "no_coop",
                "year": None,
                "semester": None,
            },
        )
        outcome = semantic_answer(
            DB_PATH,
            question,
            {"program": "IT", "catalog_key": "it-2565", "plan": "no_coop"},
            home_program="IT",
            interpret_callable=lambda _prompt: payload,
        )

        self.assertEqual(outcome.result.status, "answer")
        placements = [
            claim for claim in outcome.result.claims
            if claim.operation == "placement"
        ]
        self.assertTrue(placements)
        evidence = " ".join(
            str(row) for claim in placements for row in claim.evidence
        )
        self.assertIn("06016418", evidence)
        self.assertIn("'plan_key': 'no_coop'", evidence)
        self.assertIn("'year_number': 3", evidence)
        self.assertIn("'semester_number': 1", evidence)
        self.assertIn("06016418", outcome.result.final_answer)
        self.assertIn("3", outcome.result.final_answer)
        self.assertIn("1", outcome.result.final_answer)

    def test_nosql_credit_control_remains_three_credits(self):
        question = (
            "NOSQL DATABASE SYSTEMS (06016414) "
            "มีหน่วยกิตเท่าไรในหลักสูตร IT?"
        )
        payload = _intent_payload(
            relation="credits",
            target="NOSQL DATABASE SYSTEMS (06016414)",
            requested_field="credits",
            scope={
                "program": "IT",
                "catalog": None,
                "plan": None,
                "year": None,
                "semester": None,
            },
        )
        outcome = semantic_answer(
            DB_PATH,
            question,
            {"program": "IT", "catalog_key": "it-2565"},
            home_program="IT",
            interpret_callable=lambda _prompt: payload,
        )

        self.assertEqual(outcome.result.status, "answer")
        credit_claims = [
            claim for claim in outcome.result.claims
            if claim.operation == "sum_credits"
        ]
        self.assertTrue(credit_claims)
        self.assertTrue(any(
            "3" in str(claim.value) for claim in credit_claims
        ))
        self.assertIn("3", outcome.result.final_answer)


if __name__ == "__main__":
    unittest.main()
