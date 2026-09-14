import dataclasses
import unittest

from rag.aggregation import ComponentAggregation, CourseSetAggregation
from rag.evidence_executor import EvidenceBundle, EvidenceExecutionResult
from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.grounded_answer import (
    GroundedAnswerResult,
    GroundedClaim,
    compose_grounded_answer,
    compose_grounded_claims,
)
from rag.judgement import JudgementEvidence
from rag.retrieval.retrieve import SimilarityEvidence


class GroundedAnswerCompositionTests(unittest.TestCase):
    def _bundle(self, *results):
        scope = StructuralScope(program="IT", plans=("coop",))
        requests = tuple(result.planned_request for result in results)
        plan = EvidencePlan(scope=scope, requests=requests)
        return EvidenceBundle(plan=plan, results=results)

    def _result(self, request_id, kind, payload, *, status="complete", scope=None):
        scope = scope or StructuralScope(program="IT", plans=("coop",))
        request = EvidenceRequest(request_id=request_id, kind=kind, scope=scope)
        return EvidenceExecutionResult(
            request_id=request_id,
            kind=kind,
            planned_request=request,
            effective_scope=scope,
            status=status,
            payload=payload,
        )

    def _course(self, code, page):
        return {
            "program": "IT",
            "course_code": code,
            "provenance": [{"source_page": page}],
        }

    def test_public_contracts_are_immutable_and_have_exact_fields(self):
        self.assertEqual(
            {field.name for field in dataclasses.fields(GroundedClaim)},
            {
                "claim_id",
                "operation",
                "effective_scope",
                "status",
                "kind",
                "value",
                "evidence",
                "provenance",
            },
        )
        self.assertEqual(
            {field.name for field in dataclasses.fields(GroundedAnswerResult)},
            {"status", "answer_mode", "final_answer", "claims", "provenance"},
        )
        claim = GroundedClaim("claim_001", "list", value=("x",))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            claim.operation = "count"

    def test_stable_claim_ids_order_and_first_seen_provenance(self):
        first = self._result(
            "r1",
            "placement_facts",
            {"placement": "first", "provenance": [{"source_page": 10}]},
        )
        second = self._result(
            "r2",
            "credit_facts",
            {"credits": 3, "provenance": [{"source_page": 10}, {"source_page": 11}]},
        )
        result = compose_grounded_answer(
            self._bundle(first, second),
            operation_by_request={"r1": "placement", "r2": "sum_credits"},
        )
        self.assertEqual([claim.claim_id for claim in result.claims], ["claim_001", "claim_002"])
        self.assertEqual([claim.operation for claim in result.claims], ["placement", "sum_credits"])
        self.assertEqual(
            [reference["source_page"] for reference in result.provenance],
            [10, 11],
        )

    def test_list_count_and_existence_preserve_one_typed_course_set(self):
        courses = (self._course("0101", 20), self._course("0102", 21))
        payload = CourseSetAggregation("complete", courses, count=2, exists=True)
        result = self._result("r1", "course_set", payload)
        composed = compose_grounded_answer(
            self._bundle(result),
            operation_by_request={"r1": "count"},
        )
        self.assertEqual(composed.status, "answer")
        self.assertEqual(composed.claims[0].value.count, 2)
        self.assertTrue(composed.claims[0].value.exists)
        self.assertEqual(composed.claims[0].provenance[0]["source_page"], 20)

    def test_valid_zero_and_false_are_not_rewritten_as_no_data(self):
        payload = CourseSetAggregation("valid_empty", count=0, exists=False)
        result = self._result("r1", "course_set", payload, status="valid_empty")
        composed = compose_grounded_answer(
            self._bundle(result),
            operation_by_request={"r1": "existence"},
        )
        self.assertEqual(composed.status, "valid_empty")
        self.assertFalse(composed.claims[0].value.exists)
        self.assertEqual(composed.claims[0].value.count, 0)
        self.assertNotEqual(composed.status, "no_data")

    def test_scalar_credit_placement_prerequisite_earliest_and_compare_remain_typed(self):
        payloads = (
            ("credits", "credit_facts", {"counted_credit_units": 3, "provenance": [{"id": "c"}]}),
            ("placement", "placement_facts", {"year": 2, "semester": 1, "provenance": [{"id": "p"}]}),
            ("prereq", "prerequisite_facts", {"required": (), "provenance": [{"id": "r"}]}),
            ("earliest", "placement_facts", {"value": (2, 1), "provenance": [{"id": "e"}]}),
            ("compare", "placement_facts", {"relation": "less", "provenance": [{"id": "x"}]}),
        )
        results = tuple(
            self._result(request_id, kind, payload)
            for request_id, kind, payload in payloads
        )
        composed = compose_grounded_answer(
            self._bundle(*results),
            operation_by_request={
                "credits": "sum_credits",
                "placement": "placement",
                "prereq": "prerequisite",
                "earliest": "earliest",
                "compare": "compare",
            },
        )
        self.assertEqual(composed.status, "answer")
        self.assertEqual(composed.claims[0].value["counted_credit_units"], 3)
        self.assertEqual(composed.claims[-1].value["relation"], "less")

    def test_quantity_and_workload_judgement_evidence_are_typed(self):
        course_set = CourseSetAggregation("complete", (self._course("0101", 30),), 1, True)
        quantity = JudgementEvidence(
            "quantity",
            "descriptive_only",
            facts={"count": course_set},
            provenance=({"source_page": 30},),
        )
        workload = JudgementEvidence(
            "workload",
            "supported",
            facts={"course_count": course_set},
            provenance=({"source_page": 30},),
        )
        result = compose_grounded_answer(
            judgement_evidence=(quantity, workload),
        )
        self.assertEqual(result.status, "answer")
        self.assertEqual(result.answer_mode, "deterministic")
        self.assertEqual(result.claims[0].status, "descriptive_only")
        self.assertEqual(result.claims[0].value.status, "descriptive_only")
        self.assertEqual([claim.operation for claim in result.claims], ["quantity", "workload"])

    def test_identity_preserves_conflicting_name_variants(self):
        identity_result = {
            "status": "answer",
            "identities": [
                {
                    "program": "IT",
                    "course_code": "0101",
                    "name_th_variants": ("ชื่อหนึ่ง", "ชื่อสอง"),
                    "provenance": [{"source_page": 40}],
                }
            ],
        }
        result = compose_grounded_answer(identity_result=identity_result)
        self.assertEqual(result.status, "answer")
        self.assertEqual(
            result.claims[0].value[0]["name_th_variants"],
            ("ชื่อหนึ่ง", "ชื่อสอง"),
        )

    def test_partitioned_results_remain_separate(self):
        coop = StructuralScope(program="IT", plans=("coop",))
        no_coop = StructuralScope(program="IT", plans=("no_coop",))
        first = self._result(
            "coop",
            "course_set",
            CourseSetAggregation("complete", (self._course("0101", 50),), 1, True),
            scope=coop,
        )
        second = self._result(
            "no_coop",
            "course_set",
            CourseSetAggregation("complete", (self._course("0101", 50),), 1, True),
            scope=no_coop,
        )
        result = compose_grounded_answer(self._bundle(first, second))
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(
            [claim.effective_scope.plans for claim in result.claims],
            [("coop",), ("no_coop",)],
        )

    def test_description_creates_grounded_summary_shell_without_rendering(self):
        result = self._result(
            "description",
            "description_evidence",
            {"course_code": "0101", "text": "grounded", "provenance": [{"id": "d"}]},
        )
        composed = compose_grounded_answer(self._bundle(result))
        self.assertEqual(composed.answer_mode, "grounded_synthesis")
        self.assertEqual(composed.claims[0].kind, "grounded_summary")
        self.assertEqual(composed.final_answer, "")

    def test_similarity_is_typed_numeric_evidence_without_binary_judgement(self):
        similarity = object.__new__(SimilarityEvidence)
        object.__setattr__(similarity, "status", "valid_empty")
        object.__setattr__(similarity, "pairs", ())
        object.__setattr__(similarity, "unmatched_partitions", ())
        object.__setattr__(similarity, "mean_distance", None)
        object.__setattr__(similarity, "min_distance", None)
        object.__setattr__(similarity, "max_distance", None)
        result = compose_grounded_answer(similarity_evidence=(similarity,))
        self.assertEqual(result.status, "valid_empty")
        self.assertEqual(result.claims[0].operation, "similarity")
        self.assertIs(result.claims[0].value, similarity)
        self.assertFalse(hasattr(result.claims[0].value, "similar"))

    def test_complete_similarity_preserves_numeric_facts_partition_and_both_provenances(self):
        left = {
            "program": "IT",
            "course_code": "0101",
            "chunk_id": "left_chunk",
            "chunk_type": "description",
            "text": "left description",
            "provenance": [{"source_page": 70}],
        }
        right = {
            "program": "IT",
            "course_code": "0102",
            "chunk_id": "right_chunk",
            "chunk_type": "description",
            "text": "right description",
            "provenance": [{"source_page": 71}],
        }
        pair = SimilarityEvidence.__annotations__
        self.assertIn("pairs", pair)
        from rag.retrieval.retrieve import SimilarityPair

        similarity_pair = SimilarityPair(
            status="complete",
            partition={"plan": "coop"},
            left=left,
            right=right,
            cosine_distance=0.2,
            cosine_similarity=0.8,
        )
        similarity = SimilarityEvidence(
            status="complete",
            pairs=(similarity_pair,),
            mean_distance=0.2,
            min_distance=0.2,
            max_distance=0.2,
        )
        result = compose_grounded_answer(similarity_evidence=similarity)
        self.assertEqual(result.status, "answer")
        self.assertEqual(result.claims[0].value.pairs[0].partition["plan"], "coop")
        self.assertEqual(result.claims[0].value.pairs[0].cosine_distance, 0.2)
        self.assertEqual(result.claims[0].value.mean_distance, 0.2)
        self.assertEqual(
            [reference["source_page"] for reference in result.claims[0].provenance],
            [70, 71],
        )
        self.assertFalse(hasattr(result.claims[0].value, "similar"))

    def test_complete_plus_insufficient_preserves_verified_claim(self):
        complete = self._result(
            "good",
            "credit_facts",
            {"counted_credit_units": 3, "provenance": [{"id": "good"}]},
        )
        failed = self._result(
            "bad",
            "placement_facts",
            None,
            status="insufficient_evidence",
        )
        result = compose_grounded_answer(
            self._bundle(complete, failed),
            operation_by_request={"good": "sum_credits", "bad": "placement"},
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.claims[0].value["counted_credit_units"], 3)
        self.assertIsNone(result.claims[1].value)

    def test_all_valid_empty_is_distinct(self):
        empty = self._result(
            "empty",
            "course_set",
            None,
            status="valid_empty",
        )
        result = compose_grounded_answer(self._bundle(empty))
        self.assertEqual(result.status, "valid_empty")

    def test_malformed_composition_fails_closed_without_exposing_input(self):
        result = compose_grounded_answer(
            identity_result={
                "status": "answer",
                "identities": [{"program": "IT", "course_code": "0101", "provenance": [123]}],
            }
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.claims, ())
        self.assertEqual(result.provenance, ())

    def test_resolution_blocked_status_bypasses_claim_composition(self):
        result = compose_grounded_answer(resolution_status="context_conflict")
        self.assertEqual(result.status, "context_conflict")
        self.assertEqual(result.claims, ())
        self.assertEqual(result.final_answer, "")

    def test_input_evidence_is_not_mutated(self):
        payload = {"value": 3, "provenance": [{"source_page": 60}]}
        before = {"value": 3, "provenance": [{"source_page": 60}]}
        result = self._result("r1", "credit_facts", payload)
        compose_grounded_claims(self._bundle(result), operation_by_request={"r1": "sum_credits"})
        self.assertEqual(payload, before)
        with self.assertRaises(TypeError):
            result.payload["value"] = 4


if __name__ == "__main__":
    unittest.main()
