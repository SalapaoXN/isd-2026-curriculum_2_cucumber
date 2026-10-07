"""Fail-closed direction guard for unsupported reverse prerequisite queries."""

import unittest
from pathlib import Path
from unittest.mock import patch

from rag.qa import ask
from rag.query_spec import parse_query_spec
from rag.resolution import QueryContext


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)
IT_CONTEXT = QueryContext(program="IT", catalog_key="it-2565")


class ReversePrerequisiteDirectionGuardTests(unittest.TestCase):
    def test_upstream_thai_course_prerequisite_query_remains_supported(self):
        question = "06016454 ต้องผ่านอะไรมาก่อน"
        spec = parse_query_spec(question, has_validated_context_scope=True)

        self.assertNotEqual(spec.judgement, "unsupported")
        self.assertEqual(spec.course_codes, ("06016454",))
        self.assertEqual(spec.operations, ("prerequisite",))
        result = ask(DB_PATH, question, context=IT_CONTEXT)["result"]
        self.assertEqual(result.status, "answer")
        prerequisite_claims = [
            claim for claim in result.claims if claim.operation == "prerequisite"
        ]
        self.assertTrue(prerequisite_claims)
        for claim in prerequisite_claims:
            self.assertEqual(claim.status, "valid_empty")
            self.assertTrue(claim.provenance)
            self.assertTrue(claim.evidence)
            for row in claim.evidence:
                self.assertEqual(row["prerequisite_state"], "explicit_none")
                self.assertEqual(row["prerequisite_text"], "ไม่มี")

    def test_explicit_course_code_upstream_lookup_remains_supported(self):
        question = "IT 06016454 มี prerequisite ไหม"
        spec = parse_query_spec(question, has_validated_context_scope=True)

        self.assertNotEqual(spec.judgement, "unsupported")
        self.assertEqual(spec.course_codes, ("06016454",))
        result = ask(DB_PATH, question, context=IT_CONTEXT)["result"]
        self.assertEqual(result.status, "answer")
        self.assertIn("prerequisite", [claim.operation for claim in result.claims])

    def test_reverse_thai_required_before_form_fails_before_prerequisite_lookup(self):
        self._assert_reverse_fails_closed(
            "มีวิชาอะไรที่ต้องผ่าน 06016454 ก่อน"
        )

    def test_reverse_prerequisite_loanword_form_fails_closed(self):
        self._assert_reverse_fails_closed(
            "มีวิชาไหนใช้ 06016454 เป็น prerequisite"
        )

    def test_generic_course_name_reverse_form_fails_closed(self):
        self._assert_reverse_fails_closed(
            "มีวิชาไหนต้องผ่าน Calculus 1 ก่อน"
        )

    def _assert_reverse_fails_closed(self, question):
        spec = parse_query_spec(question, has_validated_context_scope=True)
        self.assertEqual(spec.judgement, "unsupported")

        with patch("rag.evidence_executor.prerequisite_state") as state_lookup:
            result = ask(DB_PATH, question, context=IT_CONTEXT)["result"]

        status = result.get("status") if isinstance(result, dict) else result.status
        claims = result.get("claims", ()) if isinstance(result, dict) else result.claims
        provenance = (
            result.get("provenance", ())
            if isinstance(result, dict)
            else result.provenance
        )
        self.assertNotEqual(status, "answer")
        self.assertFalse(claims)
        self.assertFalse(provenance)
        state_lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
