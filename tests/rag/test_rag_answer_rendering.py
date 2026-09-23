import unittest
from unittest.mock import patch

from rag.aggregation import ComparisonAggregation, EarliestAggregation, EarliestPartition
from rag.evidence_planner import StructuralScope
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

    def test_single_scoped_claim_keeps_existing_rendering(self):
        # Frozen user-facing rendering uses Thai human scope prefixes
        # (living pins: test_rag_answer.py scoped-sum assertions).
        claim = GroundedClaim(
            "claim_001",
            "sum_credits",
            effective_scope=StructuralScope(
                program="IT", plans=("coop",), years=(2,)
            ),
            value=18,
        )

        rendered = render_grounded_answer(
            GroundedAnswerResult("answer", "deterministic", claims=(claim,))
        )

        self.assertEqual(
            rendered.final_answer,
            "หลักสูตร IT แผนสหกิจ ปี 2 ลงทะเบียนรวม 18 หน่วยกิต",
        )

    def test_sibling_claims_prefix_only_differing_plan(self):
        claims = (
            GroundedClaim(
                "claim_001",
                "sum_credits",
                effective_scope=StructuralScope(
                    program="IT", plans=("coop",), years=(2,)
                ),
                value=48,
            ),
            GroundedClaim(
                "claim_002",
                "sum_credits",
                effective_scope=StructuralScope(
                    program="IT", plans=("no_coop",), years=(2,)
                ),
                value=45,
            ),
        )

        rendered = render_grounded_answer(
            GroundedAnswerResult("answer", "deterministic", claims=claims)
        )

        self.assertEqual(
            rendered.final_answer,
            "หลักสูตร IT แผนสหกิจ ปี 2 ลงทะเบียนรวม 48 หน่วยกิต\n"
            "หลักสูตร IT แผนไม่สหกิจ ปี 2 ลงทะเบียนรวม 45 หน่วยกิต",
        )
        self.assertNotIn("year=", rendered.final_answer)
        self.assertNotIn("program=", rendered.final_answer)

    def test_sibling_claims_prefix_year_and_semester_in_fixed_order(self):
        claims = (
            GroundedClaim(
                "claim_001",
                "sum_credits",
                effective_scope=StructuralScope(
                    program="IT", plans=("coop",), years=(2,), semesters=(1,)
                ),
                value=21,
            ),
            GroundedClaim(
                "claim_002",
                "sum_credits",
                effective_scope=StructuralScope(
                    program="IT", plans=("coop",), years=(3,), semesters=(2,)
                ),
                value=21,
            ),
        )

        rendered = render_grounded_answer(
            GroundedAnswerResult("answer", "deterministic", claims=claims)
        )

        self.assertEqual(
            rendered.final_answer,
            "หลักสูตร IT แผนสหกิจ ปี 2 ภาคเรียนที่ 1 ลงทะเบียนรวม 21 หน่วยกิต\n"
            "หลักสูตร IT แผนสหกิจ ปี 3 ภาคเรียนที่ 2 ลงทะเบียนรวม 21 หน่วยกิต",
        )

    def test_sibling_claims_prefix_all_differing_dimensions_and_preserve_order(self):
        first = GroundedClaim(
            "claim_001",
            "sum_credits",
            effective_scope=StructuralScope(
                program="IT", plans=("coop",), years=(2,), semesters=(1,)
            ),
            value=48,
            provenance=({"source_page": 1},),
        )
        second = GroundedClaim(
            "claim_002",
            "sum_credits",
            effective_scope=StructuralScope(
                program="IT", plans=("no_coop",), years=(3,), semesters=(2,)
            ),
            value=45,
            provenance=({"source_page": 2},),
        )
        source = GroundedAnswerResult(
            "answer",
            "deterministic",
            claims=(first, second),
            provenance=first.provenance + second.provenance,
        )

        rendered = render_grounded_answer(source)

        self.assertEqual(
            rendered.final_answer,
            "หลักสูตร IT แผนสหกิจ ปี 2 ภาคเรียนที่ 1 ลงทะเบียนรวม 48 หน่วยกิต\n"
            "หลักสูตร IT แผนไม่สหกิจ ปี 3 ภาคเรียนที่ 2 ลงทะเบียนรวม 45 หน่วยกิต",
        )
        self.assertEqual(rendered.claims, source.claims)
        self.assertEqual(rendered.provenance, source.provenance)

    def test_earliest_comparison_labels_existing_operand_scopes(self):
        left = EarliestAggregation(
            "complete",
            (
                EarliestPartition(
                    {
                        "program": "IT",
                        "plans": ("coop",),
                        "years": (2,),
                        "semesters": (2,),
                    },
                    (2, 2),
                    ({"provenance": ({"source_page": 11},)},),
                    ({"source_page": 11},),
                ),
            ),
        )
        right = EarliestAggregation(
            "complete",
            (
                EarliestPartition(
                    {
                        "program": "IT",
                        "plans": ("no_coop",),
                        "years": (4,),
                        "semesters": (1,),
                    },
                    (4, 1),
                    ({"provenance": ({"source_page": 22},)},),
                    ({"source_page": 22},),
                ),
            ),
        )
        claim = GroundedClaim(
            "claim_001",
            "compare",
            value=ComparisonAggregation("complete", "less", left, right),
            provenance=({"source_page": 11}, {"source_page": 22}),
        )

        rendered = render_grounded_claim(claim)

        self.assertTrue(
            rendered.startswith(
                "left[plan=coop, year=2, semester=2] "
                "right[plan=no_coop, year=4, semester=1] | compare: "
            )
        )
        self.assertIn('"relation":"less"', rendered)
        self.assertEqual(claim.provenance, ({"source_page": 11}, {"source_page": 22}))

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

        # Frozen similarity rendering is descriptions-only (living pin:
        # test_rag_answer complete_similarity_renders_grounded_course_*
        # asserts NotIn cosine_*); numerics stay in evidence for gating.
        self.assertIn("06016414", numeric)
        self.assertIn("06016419", numeric)
        self.assertIn("left description", numeric)
        self.assertIn("right description", numeric)
        self.assertNotIn("cosine_similarity", numeric)
        self.assertNotIn("cosine_distance", numeric)
        self.assertIn("description comparison", synthesized)
        self.assertIn("left description", prompts[0])
        self.assertIn("right description", prompts[0])
        self.assertNotIn("0.2", prompts[0])
        self.assertNotIn("0.8", prompts[0])
        answer_calls: list = []
        rendered_answer = render_grounded_answer(
            GroundedAnswerResult(
                "answer",
                "grounded_synthesis",
                claims=(claim,),
            ),
            answer_calls.append,
        )
        # Same H26 gating as above: no question, no polish call — the
        # deterministic descriptions-only text stands.
        self.assertEqual(answer_calls, [])
        self.assertIn("left description", rendered_answer.final_answer)
        self.assertIn("right description", rendered_answer.final_answer)
        self.assertNotIn("description comparison", rendered_answer.final_answer)

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

        model_calls: list = []

        rendered = render_grounded_answer(source, model_calls.append)

        self.assertTrue(rendered.final_answer.startswith('count: {"count":2}'))
        # Frozen H26 gating: without a question the polish path is a
        # no-op, so the describe claim keeps its deterministic text and
        # the model is never called.
        self.assertTrue(rendered.final_answer.endswith("second segment"))
        self.assertEqual(model_calls, [])
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
