from __future__ import annotations

import unittest

from rag.aggregation import ComponentAggregation
from rag.answer import render_grounded_answer
from rag.evidence_planner import StructuralScope
from rag.grounded_answer import GroundedAnswerResult, GroundedClaim


class H13BAnswerSynthesisTests(unittest.TestCase):
    def _claim(self, operation: str, value, evidence=None):
        return GroundedClaim(
            claim_id="atomic",
            operation=operation,
            effective_scope=StructuralScope(
                program="IT",
                plans=("coop",),
                course_targets=({"course_code": "06016420", "program": "IT"},),
            ),
            value=value,
            evidence=evidence,
            provenance=({"source_page": 10},),
        )

    def _result(self, claim):
        return GroundedAnswerResult(
            status="answer",
            answer_mode="deterministic",
            claims=(claim,),
            provenance=claim.provenance,
        )

    def test_credit_synthesis_accepts_only_grounded_credit_shape(self):
        evidence = ComponentAggregation(
            operation="sum_credits",
            status="complete",
            value=3,
            components=({"credits_raw": "3(2-2-5)"},),
        )
        calls = []
        result = render_grounded_answer(
            self._result(self._claim("sum_credits", 3, evidence)),
            lambda prompt: calls.append(prompt) or "วิชา 06016420 ในหลักสูตร IT แผน coop มี 3(2-2-5) หน่วยกิต",
            question="วิชา 06016420 มีกี่หน่วยกิต",
            synthesize_answer=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("06016420", result.final_answer)
        self.assertIn("3(2-2-5)", result.final_answer)
        self.assertEqual(result.claims[0].value, 3)
        self.assertEqual(result.provenance, self._result(self._claim("sum_credits", 3, evidence)).provenance)

    def test_credit_guard_rejects_changed_value_and_falls_back(self):
        evidence = ComponentAggregation(
            operation="sum_credits", status="complete", value=3,
            components=({"credits_raw": "3(2-2-5)"},),
        )
        deterministic = render_grounded_answer(
            self._result(self._claim("sum_credits", 3, evidence)),
            synthesize_answer=False,
        ).final_answer
        result = render_grounded_answer(
            self._result(self._claim("sum_credits", 3, evidence)),
            lambda _prompt: "วิชา 06016420 มี 4(4-0-8) หน่วยกิต",
            question="วิชา 06016420 มีกี่หน่วยกิต",
            synthesize_answer=True,
        )
        self.assertEqual(result.final_answer, deterministic)

    def test_existence_synthesis_preserves_positive_polarity_and_target(self):
        calls = []
        result = render_grounded_answer(
            self._result(self._claim("existence", True)),
            lambda prompt: calls.append(prompt) or "มีวิชา 06016420 ในแผน IT coop",
            question="IT มีวิชา 06016420 ไหม",
            synthesize_answer=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("มีวิชา 06016420", result.final_answer)

    def test_existence_guard_rejects_target_mutation(self):
        calls = []
        result = render_grounded_answer(
            self._result(self._claim("existence", True)),
            lambda prompt: calls.append(prompt) or "มีวิชา 06016421",
            question="IT มีวิชา 06016420 ไหม",
            synthesize_answer=True,
        )
        self.assertEqual(len(calls), 1)
        deterministic = render_grounded_answer(
            self._result(self._claim("existence", True)), synthesize_answer=False
        ).final_answer
        self.assertEqual(result.final_answer, deterministic)
        self.assertNotIn("06016421", result.final_answer)

    def test_existence_synthesis_preserves_negative_polarity(self):
        result = render_grounded_answer(
            self._result(self._claim("existence", False)),
            lambda _prompt: "ไม่พบวิชา 06016420 ในหลักสูตร IT แผน coop",
            question="IT แผนสหกิจ มีวิชา 06016420 ไหม",
            synthesize_answer=True,
        )
        self.assertIn("ไม่พบวิชา 06016420", result.final_answer)

    def test_non_atomic_claim_does_not_synthesize(self):
        calls = []
        claim = self._claim("list", [{"course_code": "06016420"}])
        result = render_grounded_answer(
            self._result(claim),
            lambda prompt: calls.append(prompt) or "rewritten",
            question="มีวิชาอะไรบ้าง",
            synthesize_answer=True,
        )
        self.assertEqual(calls, [])
        self.assertNotEqual(result.final_answer, "rewritten")

    def test_prompt_contains_only_sanitized_atomic_facts(self):
        evidence = ComponentAggregation(
            operation="sum_credits", status="complete", value=3,
            components=({
                "credits_raw": "3(2-2-5)",
                "course_id": 999,
                "provenance": ({"source_page": 77},),
            },),
        )
        prompts = []
        render_grounded_answer(
            self._result(self._claim("sum_credits", 3, evidence)),
            lambda prompt: prompts.append(prompt) or "วิชา 06016420 ในหลักสูตร IT แผน coop มี 3(2-2-5) หน่วยกิต",
            question="วิชา 06016420 มีกี่หน่วยกิต",
            synthesize_answer=True,
        )
        self.assertEqual(len(prompts), 1)
        prompt = prompts[0]
        self.assertIn("06016420", prompt)
        self.assertIn("3(2-2-5)", prompt)
        self.assertNotIn("999", prompt)
        self.assertNotIn("source_page", prompt)
        self.assertNotIn("provenance", prompt)


if __name__ == "__main__":
    unittest.main()
