from __future__ import annotations

import unittest

from rag.answer import render_grounded_answer
from rag.evidence_planner import StructuralScope
from rag.grounded_answer import GroundedAnswerResult, GroundedClaim


class H14CountSynthesisTests(unittest.TestCase):
    def _result(self, *, value=16, scope=None):
        scope = scope or StructuralScope(
            program="IT", plans=("coop",), years=(2,), semesters=(1, 2)
        )
        claim = GroundedClaim(
            claim_id="count",
            operation="count",
            effective_scope=scope,
            value=value,
            provenance=({"source_page": 12},),
        )
        return GroundedAnswerResult(
            status="answer", answer_mode="deterministic", claims=(claim,),
            provenance=claim.provenance,
        )

    def test_valid_count_rewrite_is_accepted_and_claim_is_unchanged(self):
        result = self._result()
        calls = []
        rendered = render_grounded_answer(
            result,
            lambda prompt: calls.append(prompt) or "หลักสูตร IT แผน coop ปี 2 มี 16 วิชา",
            question="IT แผนสหกิจ ปี 2 มีกี่วิชา",
            synthesize_answer=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("16 วิชา", rendered.final_answer)
        self.assertEqual(rendered.status, result.status)
        self.assertEqual(rendered.claims, result.claims)
        self.assertEqual(rendered.provenance, result.provenance)
        self.assertIn('"count":16', calls[0])
        self.assertNotIn("source_page", calls[0])

    def test_opt_in_false_keeps_exact_deterministic_count_without_call(self):
        result = self._result()
        deterministic = render_grounded_answer(result).final_answer
        calls = []
        rendered = render_grounded_answer(
            result,
            lambda prompt: calls.append(prompt) or "rewritten",
            question="IT แผนสหกิจ ปี 2 มีกี่วิชา",
            synthesize_answer=False,
        )
        self.assertEqual(calls, [])
        self.assertEqual(rendered.final_answer, deterministic)

    def test_provider_absent_keeps_deterministic_count(self):
        result = self._result()
        self.assertEqual(
            render_grounded_answer(result, synthesize_answer=True).final_answer,
            render_grounded_answer(result).final_answer,
        )

    def test_count_change_is_rejected(self):
        result = self._result()
        deterministic = render_grounded_answer(result).final_answer
        rendered = render_grounded_answer(
            result,
            lambda _prompt: "หลักสูตร IT แผน coop ปี 2 มี 17 วิชา",
            question="IT แผนสหกิจ ปี 2 มีกี่วิชา",
            synthesize_answer=True,
        )
        self.assertEqual(rendered.final_answer, deterministic)

    def test_conflicting_quantity_and_scope_are_rejected(self):
        result = self._result()
        deterministic = render_grounded_answer(result).final_answer
        for answer in (
            "หลักสูตร IT แผน coop ปี 2 มี 16 วิชา รวม 3 หน่วยกิต",
            "หลักสูตร IT แผน coop ปี 3 มี 16 วิชา",
        ):
            rendered = render_grounded_answer(
                result, lambda _prompt, answer=answer: answer,
                question="IT แผนสหกิจ ปี 2 มีกี่วิชา",
                synthesize_answer=True,
            )
            self.assertEqual(rendered.final_answer, deterministic)

    def test_unbounded_count_never_calls_model(self):
        result = self._result(scope=StructuralScope(program="IT"))
        calls = []
        rendered = render_grounded_answer(
            result,
            lambda prompt: calls.append(prompt) or "IT มี 16 วิชา",
            question="IT มีกี่วิชา",
            synthesize_answer=True,
        )
        self.assertEqual(calls, [])
        self.assertEqual(rendered.final_answer, render_grounded_answer(result).final_answer)

    def test_provider_failure_falls_back_without_retry(self):
        result = self._result()
        calls = []

        def fail(_prompt):
            calls.append(True)
            raise RuntimeError("provider unavailable")

        rendered = render_grounded_answer(
            result, fail, question="IT แผนสหกิจ ปี 2 มีกี่วิชา", synthesize_answer=True
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(rendered.final_answer, render_grounded_answer(result).final_answer)


if __name__ == "__main__":
    unittest.main()
