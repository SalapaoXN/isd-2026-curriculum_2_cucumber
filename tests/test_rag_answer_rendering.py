import unittest
from unittest.mock import patch

from rag.grounded_answer import GroundedAnswerResult, GroundedClaim
from rag.judgement import JudgementEvidence
from rag.retrieval.retrieve import SimilarityEvidence, SimilarityPair
from rag.answer import (
    render_grounded_answer,
    render_grounded_claim,
    synthesize_grounded_claim,
)


class GroundedAnswerRenderingTests(unittest.TestCase):
    def _description_claim(self, text="คำอธิบายที่ยืนยันแล้ว"):
        evidence = {
            "course_code": "06016414",
            "text": text,
            "unrelated": "ห้ามส่งข้อมูลนี้ให้โมเดล",
            "provenance": [{"source_page": 12}],
        }
        return GroundedClaim(
            "claim_001",
            "describe",
            kind="grounded_summary",
            value=evidence,
            evidence=evidence,
            provenance=tuple(evidence["provenance"]),
        )

    def test_deterministic_claim_uses_python_only(self):
        calls = []
        claim = GroundedClaim("claim_001", "count", value={"count": 2})

        with patch("rag.answer.answer_question", side_effect=AssertionError):
            rendered = render_grounded_claim(claim)

        self.assertIn('"count":2', rendered)
        self.assertEqual(calls, [])

    def test_summary_synthesis_receives_only_that_claim_description(self):
        prompts = []
        claim = self._description_claim()

        rendered = synthesize_grounded_claim(
            claim,
            lambda prompt: prompts.append(prompt) or "สรุปจากหลักฐาน",
        )

        self.assertEqual(rendered, "สรุปจากหลักฐาน")
        self.assertEqual(len(prompts), 1)
        self.assertIn("คำอธิบายที่ยืนยันแล้ว", prompts[0])
        self.assertNotIn("ห้ามส่งข้อมูลนี้ให้โมเดล", prompts[0])
        self.assertNotIn("06016414", prompts[0])

    def test_empty_non_string_or_failed_synthesis_uses_grounded_fallback(self):
        claim = self._description_claim("fallback text")
        self.assertEqual(
            synthesize_grounded_claim(claim, lambda prompt: ""),
            "fallback text",
        )
        self.assertEqual(
            synthesize_grounded_claim(claim, lambda prompt: 123),
            "fallback text",
        )
        self.assertEqual(
            synthesize_grounded_claim(
                claim,
                lambda prompt: (_ for _ in ()).throw(RuntimeError("model failed")),
            ),
            "fallback text",
        )

    def test_preference_preserves_option_order_and_never_ranks_by_distance(self):
        evidence = JudgementEvidence(
            "preference",
            "supported",
            options=(
                {
                    "program": "IT",
                    "course_code": "06016414",
                    "course_name": "First",
                    "distance": 0.8,
                    "description_evidence": (
                        {"text": "first description"},
                    ),
                    "partition": {"plan": "coop"},
                    "provenance": [{"source_page": 1}],
                },
                {
                    "program": "IT",
                    "course_code": "06016419",
                    "course_name": "Second",
                    "distance": 0.1,
                    "description_evidence": (
                        {"text": "second description"},
                    ),
                    "partition": {"plan": "coop"},
                    "provenance": [{"source_page": 2}],
                },
            ),
        )
        claim = GroundedClaim(
            "claim_001",
            "preference",
            kind="grounded_summary",
            value=evidence,
            evidence=evidence,
        )

        rendered = render_grounded_claim(claim)
        self.assertLess(rendered.index("06016414"), rendered.index("06016419"))
        self.assertNotIn("best", rendered.casefold())
        self.assertNotIn("0.1", rendered)

    def test_similarity_renders_numeric_facts_and_synthesis_gets_descriptions_only(self):
        left = {
            "program": "IT",
            "course_code": "06016414",
            "chunk_id": "left",
            "chunk_type": "description",
            "text": "left description",
            "provenance": [{"source_page": 3}],
        }
        right = {
            "program": "IT",
            "course_code": "06016419",
            "chunk_id": "right",
            "chunk_type": "description",
            "text": "right description",
            "provenance": [{"source_page": 4}],
        }
        pair = SimilarityPair(
            "complete",
            {"plan": "coop"},
            left,
            right,
            cosine_distance=0.2,
            cosine_similarity=0.8,
        )
        evidence = SimilarityEvidence(
            "complete",
            pairs=(pair,),
            mean_distance=0.2,
            min_distance=0.2,
            max_distance=0.2,
        )
        claim = GroundedClaim(
            "claim_001",
            "similarity",
            kind="grounded_summary",
            value=evidence,
            evidence=evidence,
        )

        numeric = render_grounded_claim(claim)
        prompts = []
        synthesized = synthesize_grounded_claim(
            claim,
            lambda prompt: prompts.append(prompt) or "description comparison",
        )

        self.assertIn("0.2", numeric)
        self.assertIn("0.8", numeric)
        self.assertIn("description comparison", synthesized)
        self.assertIn("0.2", synthesized)
        self.assertIn("left description", prompts[0])
        self.assertIn("right description", prompts[0])
        self.assertNotIn("0.2", prompts[0])
        self.assertNotIn("0.8", prompts[0])
        rendered_answer = render_grounded_answer(
            GroundedAnswerResult(
                "answer",
                "grounded_synthesis",
                claims=(claim,),
            ),
            lambda prompt: "description comparison",
        )
        self.assertIn("0.2", rendered_answer.final_answer)
        self.assertIn("description comparison", rendered_answer.final_answer)

    def test_mixed_answer_renders_claims_in_order_without_mutation(self):
        deterministic = GroundedClaim(
            "claim_001",
            "count",
            value={"count": 2},
            provenance=({"source_page": 1},),
        )
        summary = self._description_claim("second segment")
        source = GroundedAnswerResult(
            "answer",
            "mixed",
            claims=(deterministic, summary),
            provenance=({"source_page": 1}, {"source_page": 12}),
        )

        rendered = render_grounded_answer(source, lambda prompt: "synthesized segment")

        self.assertTrue(rendered.final_answer.startswith('count: {"count":2}'))
        self.assertTrue(rendered.final_answer.endswith("synthesized segment"))
        self.assertEqual(rendered.claims, source.claims)
        self.assertEqual(rendered.status, source.status)
        self.assertEqual(rendered.provenance, source.provenance)
        self.assertEqual(source.final_answer, "")

    def test_insufficient_claim_does_not_expose_value_or_call_model(self):
        calls = []
        claim = GroundedClaim(
            "claim_001",
            "count",
            status="insufficient_evidence",
            value={"count": 99},
        )

        rendered = synthesize_grounded_claim(
            claim,
            lambda prompt: calls.append(prompt) or "must not run",
        )

        self.assertEqual(rendered, "หลักฐานไม่เพียงพอ")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
