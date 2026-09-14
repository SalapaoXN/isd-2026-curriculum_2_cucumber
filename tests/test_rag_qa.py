import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag.aggregation import ComparisonAggregation, ComponentAggregation
from rag.evidence_executor import EvidenceBundle, EvidenceExecutionResult
from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.grounded_answer import GroundedAnswerResult, GroundedClaim
from rag.qa import _compose_evidence_claims, _course_set_aggregate, ask
from rag.query_spec import parse_query_spec
from rag.retrieval.retrieve import (
    ConstrainedTopicRetrievalResult,
    SimilarityEvidence,
    SimilarityPair,
)
from rag.resolution import (
    CourseReferenceResolution,
    QueryContext,
    ResolutionOutcome,
)


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


class RagQaTest(unittest.TestCase):
    def test_answerable_path_uses_typed_bundle_and_ignores_legacy_arguments(self):
        course = {
            "program": "IT",
            "course_code": "06016414",
            "name_en": "Course",
            "name_th": "วิชา",
            "provenance": ({"source_page": 1},),
        }
        scope = StructuralScope(program="IT", plans=("coop",), years=(1,), semesters=(1,))
        request = EvidenceRequest("course_set", "course_set", scope)
        plan = EvidencePlan(scope, (request,))
        bundle = EvidenceBundle(
            plan,
            (EvidenceExecutionResult(
                "course_set", "course_set", request, scope, "complete", {"courses": [course]}
            ),),
        )
        forbidden_model = lambda _prompt: self.fail("legacy model must not be called")

        with patch("rag.qa.plan_evidence", return_value=plan) as planner, patch(
            "rag.qa.execute_evidence_plan", return_value=bundle
        ) as executor:
            result = ask(
                DB_PATH,
                "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
                forbidden_model,
                top_k=99,
            )

        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].claims[0].operation, "list")
        planner.assert_called_once()
        executor.assert_called_once_with(DB_PATH, plan)

    def test_blocked_resolution_returns_without_route_or_evidence_work(self):
        blocked_questions = {
            "วิชาไหนยากที่สุด": "unsupported",
            "06019999 เรียนอะไร": "no_data",
            "วิชา NOSQL เรียนเรื่องอะไรบ้าง": "clarify_program",
            "มีวิชาเกี่ยวกับ database อะไรบ้าง": "clarify_program",
        }

        def forbidden(_prompt):
            self.fail("blocked requests must not call a model")

        for question, action in blocked_questions.items():
            with self.subTest(question=question):
                with patch(
                    "rag.qa.plan_evidence",
                    side_effect=AssertionError("planner must not be called"),
                ) as planner, patch(
                    "rag.qa.execute_evidence_plan",
                    side_effect=AssertionError("executor must not be called"),
                ) as executor:
                    result = ask(DB_PATH, question, forbidden)

                self.assertIsNone(result["route"])
                self.assertEqual(result["result"]["status"], action)
                self.assertEqual(result["result"]["action"], action)
                planner.assert_not_called()
                executor.assert_not_called()

    def test_context_conflict_stops_before_qa_work(self):
        with patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("planner must not be called"),
        ) as planner, patch(
            "rag.qa.execute_evidence_plan",
            side_effect=AssertionError("executor must not be called"),
        ) as executor:
            result = ask(
                DB_PATH,
                "AIT ปี 2 เรียนอะไรบ้าง",
                context=QueryContext(program="IT"),
            )

        self.assertIsNone(result["route"])
        self.assertEqual(result["result"]["status"], "context_conflict")
        self.assertEqual(result["result"]["action"], "context_conflict")
        self.assertEqual(result["result"]["context_conflicts"], ("program",))
        self.assertEqual(result["result"]["blocking_ambiguity"], ())
        planner.assert_not_called()
        executor.assert_not_called()

    def test_identity_returns_typed_exact_evidence_without_qa_paths(self):
        with patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("identity must not plan"),
        ) as planner, patch(
            "rag.qa.execute_evidence_plan",
            side_effect=AssertionError("identity must not execute"),
        ) as executor:
            result = ask(
                DB_PATH,
                "วิชา Calculus 1 รหัสวิชาอะไร",
                context=QueryContext(program="AIT"),
            )

        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].claims[0].operation, "identity")
        self.assertEqual(result["result"].status, "answer")
        self.assertEqual(
            [
                (item["program"], item["course_code"])
                for item in result["result"].claims[0].value
            ],
            [("AIT", "06046400")],
        )
        self.assertTrue(result["result"].provenance)
        planner.assert_not_called()
        executor.assert_not_called()

    def test_identity_unknown_code_keeps_no_data_guard(self):
        result = ask(DB_PATH, "06019999 ชื่ออะไร")

        self.assertIsNone(result["route"])
        self.assertEqual(result["result"]["status"], "no_data")
        self.assertEqual(result["result"]["action"], "no_data")

    def test_identity_code_returns_canonical_name_without_model(self):
        with patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("identity must not plan"),
        ) as planner, patch(
            "rag.qa.execute_evidence_plan",
            side_effect=AssertionError("identity must not execute"),
        ) as executor:
            result = ask(DB_PATH, "06046400 ชื่ออะไร")

        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        identity = result["result"].claims[0].value
        self.assertEqual(len(identity), 1)
        self.assertEqual(identity[0]["program"], "AIT")
        self.assertEqual(identity[0]["course_code"], "06046400")
        self.assertEqual(identity[0]["name_en"], "CALCULUS 1")
        planner.assert_not_called()
        executor.assert_not_called()

    def test_answerable_path_does_not_require_legacy_structured_callable(self):
        result = ask(DB_PATH, "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง")
        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)

    def test_similarity_path_calls_bundle_bridge_once_and_preserves_typed_claim(self):
        left_target = {
            "program": "AIT",
            "course_code": "06046400",
            "course_id": 1,
        }
        right_target = {
            "program": "AIT",
            "course_code": "06046401",
            "course_id": 2,
        }
        scope = StructuralScope(
            program="AIT",
            plans=("coop",),
            years=(1,),
            semesters=(1,),
            course_targets=(left_target, right_target),
        )
        left_request = EvidenceRequest(
            "similarity_description_1",
            "description_evidence",
            scope,
            course_targets=(left_target,),
        )
        right_request = EvidenceRequest(
            "similarity_description_2",
            "description_evidence",
            scope,
            course_targets=(right_target,),
        )
        plan = EvidencePlan(scope, (left_request, right_request))
        left_description = {
            **left_target,
            "chunk_id": "left-description",
            "chunk_type": "description",
            "text": "Left description",
            "provenance": ({"source_page": 1},),
        }
        right_description = {
            **right_target,
            "chunk_id": "right-description",
            "chunk_type": "description",
            "text": "Right description",
            "provenance": ({"source_page": 2},),
        }
        bundle = EvidenceBundle(
            plan,
            (
                EvidenceExecutionResult(
                    left_request.request_id,
                    left_request.kind,
                    left_request,
                    scope,
                    "complete",
                    (left_description,),
                ),
                EvidenceExecutionResult(
                    right_request.request_id,
                    right_request.kind,
                    right_request,
                    scope,
                    "complete",
                    (right_description,),
                ),
            ),
        )
        partition = {
            "program": "AIT",
            "plan": "coop",
            "plans": ("coop",),
            "years": (1,),
            "semesters": (1,),
            "category": None,
            "group_by": (),
        }
        pair = SimilarityPair(
            status="complete",
            partition=partition,
            left=left_description,
            right=right_description,
            cosine_distance=0.2,
            cosine_similarity=0.8,
        )
        similarity = SimilarityEvidence(
            status="complete",
            pairs=(pair,),
            mean_distance=0.2,
            min_distance=0.2,
            max_distance=0.2,
        )
        spec = self._spec(("similarity",))
        outcome = ResolutionOutcome(
            action="answer",
            blocking_ambiguity=(),
            resolved_program="AIT",
            course_references=(
                CourseReferenceResolution("course_code", "06046400", (left_target,)),
                CourseReferenceResolution("course_code", "06046401", (right_target,)),
            ),
            resolved_plans=("coop",),
        )
        forbidden_model = lambda _prompt: self.fail("similarity must not synthesize")

        with patch("rag.qa.parse_query_spec", return_value=spec), patch(
            "rag.qa.resolve_query_spec", return_value=outcome
        ), patch("rag.qa.plan_evidence", return_value=plan), patch(
            "rag.qa.execute_evidence_plan", return_value=bundle
        ), patch(
            "rag.qa.execute_exact_similarity_from_bundle", return_value=similarity
        ) as bridge:
            result = ask(DB_PATH, "ignored", answer_model_callable=forbidden_model)

        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claim = result["result"].claims[0]
        self.assertEqual(claim.operation, "similarity")
        self.assertIs(claim.value, similarity)
        self.assertEqual(claim.effective_scope, scope)
        self.assertEqual(
            [item["source_page"] for item in claim.provenance], [1, 2]
        )
        bridge.assert_called_once_with(
            DB_PATH,
            bundle,
            "similarity_description_1",
            "similarity_description_2",
            selected_plan="coop",
        )

    def test_similarity_claim_keeps_multiple_partitions_without_scope_merge(self):
        planned_scope = StructuralScope(
            program="AIT",
            plans=("coop", "no_coop"),
            years=(1,),
            semesters=(1,),
        )
        coop_scope = replace(planned_scope, plans=("coop",))
        no_coop_scope = replace(planned_scope, plans=("no_coop",))
        left_target = {"program": "AIT", "course_code": "06046400", "course_id": 1}
        right_target = {"program": "AIT", "course_code": "06046401", "course_id": 2}
        left_request = EvidenceRequest(
            "similarity_description_1",
            "description_evidence",
            planned_scope,
            course_targets=(left_target,),
        )
        right_request = EvidenceRequest(
            "similarity_description_2",
            "description_evidence",
            planned_scope,
            course_targets=(right_target,),
        )
        def description(target, chunk_id, page, scope):
            return {
                **target,
                "chunk_id": chunk_id,
                "chunk_type": "description",
                "text": chunk_id,
                "partition": {"program": "AIT", "plan": scope.plans[0]},
                "provenance": ({"source_page": page},),
            }

        bundle = EvidenceBundle(
            EvidencePlan(planned_scope, (left_request, right_request)),
            (
                EvidenceExecutionResult(
                    left_request.request_id,
                    left_request.kind,
                    left_request,
                    coop_scope,
                    "complete",
                    (description(left_target, "left-coop", 1, coop_scope),),
                ),
                EvidenceExecutionResult(
                    right_request.request_id,
                    right_request.kind,
                    right_request,
                    coop_scope,
                    "complete",
                    (description(right_target, "right-coop", 2, coop_scope),),
                ),
                EvidenceExecutionResult(
                    left_request.request_id,
                    left_request.kind,
                    left_request,
                    no_coop_scope,
                    "complete",
                    (description(left_target, "left-no-coop", 3, no_coop_scope),),
                ),
                EvidenceExecutionResult(
                    right_request.request_id,
                    right_request.kind,
                    right_request,
                    no_coop_scope,
                    "complete",
                    (description(right_target, "right-no-coop", 4, no_coop_scope),),
                ),
            ),
        )
        def pair(plan, left_page, right_page):
            partition = {
                "program": "AIT",
                "plan": plan,
                "plans": (plan,),
                "years": (1,),
                "semesters": (1,),
                "category": None,
                "group_by": (),
            }
            return SimilarityPair(
                status="complete",
                partition=partition,
                left={"provenance": ({"source_page": left_page},), "text": "left"},
                right={"provenance": ({"source_page": right_page},), "text": "right"},
                cosine_distance=float(left_page) / 10,
                cosine_similarity=1 - float(left_page) / 10,
            )

        evidence = SimilarityEvidence(
            status="complete",
            pairs=(pair("coop", 1, 2), pair("no_coop", 3, 4)),
            mean_distance=0.2,
            min_distance=0.1,
            max_distance=0.3,
        )
        claims = _compose_evidence_claims(
            self._spec(("similarity",)),
            bundle,
            similarity_evidence=evidence,
            similarity_request_ids=(
                "similarity_description_1",
                "similarity_description_2",
            ),
        )

        self.assertEqual(len(claims), 1)
        self.assertIsNone(claims[0].effective_scope)
        self.assertIs(claims[0].value, evidence)
        self.assertEqual([pair.partition["plan"] for pair in evidence.pairs], ["coop", "no_coop"])
        self.assertEqual(
            [item["source_page"] for item in claims[0].provenance], [1, 2, 3, 4]
        )

    def _bundle(self, requests_and_payloads):
        scope = StructuralScope(program="IT", plans=("coop",), years=(1,), semesters=(1,))
        requests = tuple(
            EvidenceRequest(request_id, kind, scope)
            for request_id, kind, _payload, _status in requests_and_payloads
        )
        results = tuple(
            EvidenceExecutionResult(
                request_id=request_id,
                kind=kind,
                planned_request=request,
                effective_scope=scope,
                status=status,
                payload=payload,
            )
            for request, (request_id, kind, payload, status) in zip(
                requests, requests_and_payloads
            )
        )
        return EvidenceBundle(EvidencePlan(scope, requests), results)

    def _spec(self, operations, *, topic=None, judgement="none"):
        spec = parse_query_spec("IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง")
        return replace(spec, operations=tuple(operations), topic=topic, judgement=judgement)

    def _course(self, code="06016414", page=1):
        return {
            "program": "IT",
            "course_code": code,
            "name_en": "Course",
            "name_th": "วิชา",
            "provenance": ({"source_page": page},),
        }

    def test_private_adapters_reuse_one_relation_for_list_count_existence(self):
        bundle = self._bundle(
            (("course_set", "course_set", {"courses": [self._course()]}, "complete"),)
        )
        with patch(
            "rag.qa._course_set_aggregate", wraps=_course_set_aggregate
        ) as aggregate:
            claims = _compose_evidence_claims(
                self._spec(("list", "count", "existence")), bundle
            )

        self.assertEqual(aggregate.call_count, 1)

        self.assertEqual(
            [claim.operation for claim in claims], ["list", "count", "existence"]
        )
        self.assertEqual(claims[0].value[0]["course_code"], "06016414")
        self.assertEqual(claims[1].value, 1)
        self.assertTrue(claims[2].value)
        self.assertTrue(all(isinstance(claim, GroundedClaim) for claim in claims))
        self.assertEqual([claim.claim_id for claim in claims], ["claim_001", "claim_002", "claim_003"])

    def test_topic_operations_use_topic_matches_candidates_not_course_set(self):
        topic_course = {
            **self._course("06016414", 7),
            "description_evidence": (
                {
                    "chunk_id": "06016414-description",
                    "chunk_type": "description",
                    "text": "Database systems",
                    "program": "IT",
                    "course_code": "06016414",
                    "partition": {
                        "program": "IT",
                        "plans": ("coop",),
                        "years": (1,),
                        "semesters": (1,),
                        "category": None,
                        "group_by": (),
                    },
                    "provenance": ({"source_page": 7},),
                },
            ),
        }
        topic = ConstrainedTopicRetrievalResult(
            "scored",
            candidates=(topic_course,),
            scored_candidates=(topic_course,),
        )
        unrelated = self._course("06019999", 8)
        bundle = self._bundle(
            (
                ("course_set", "course_set", {"courses": [unrelated]}, "complete"),
                ("topic_matches", "topic_matches", topic, "complete"),
            )
        )
        claims = _compose_evidence_claims(
            self._spec(("list", "count", "preference"), topic="database", judgement="preference"),
            bundle,
        )

        self.assertEqual([claim.operation for claim in claims], ["list", "count", "preference"])
        self.assertEqual(claims[0].value[0]["course_code"], "06016414")
        self.assertEqual(claims[1].value, 1)
        self.assertEqual(claims[2].status, "complete")
        self.assertEqual(claims[2].value.options[0]["course_code"], "06016414")

    def test_direct_adapters_preserve_partition_and_typed_provenance(self):
        scope = StructuralScope(program="IT", plans=("coop",), years=(2,), semesters=(1,))
        request_data = (
            ("credit_facts", "credit_facts", {"components": [{
                "program": "IT",
                "course_code": "06016414",
                "counted_credit_units": 3,
                "provenance": ({"source_page": 12},),
            }]}, "complete"),
            ("placement_facts", "placement_facts", {"courses": [{
                "program": "IT",
                "course_code": "06016414",
                "year": 2,
                "semester": 1,
                "provenance": ({"source_page": 13},),
            }]}, "complete"),
            ("description_evidence", "description_evidence", ({
                "chunk_id": "06016414-description",
                "text": "Database systems",
                "provenance": ({"source_page": 14},),
            },), "complete"),
        )
        requests = tuple(EvidenceRequest(item[0], item[1], scope) for item in request_data)
        results = tuple(
            EvidenceExecutionResult(item[0], item[1], request, scope, item[3], item[2])
            for request, item in zip(requests, request_data)
        )
        bundle = EvidenceBundle(EvidencePlan(scope, requests), results)
        claims = _compose_evidence_claims(
            self._spec(("sum_credits", "placement", "earliest", "describe")), bundle
        )

        self.assertEqual(
            [claim.operation for claim in claims],
            ["sum_credits", "placement", "earliest", "describe"],
        )
        self.assertEqual(claims[0].value, 3)
        self.assertEqual(claims[0].effective_scope, scope)
        self.assertEqual(claims[0].provenance[0]["source_page"], 12)
        self.assertEqual(claims[2].value.value, (2, 1))
        self.assertEqual(claims[3].kind, "grounded_summary")

    def test_judgement_adapters_preserve_valid_zero_and_workload_proxies(self):
        empty_scope = StructuralScope(program="IT", plans=("coop",), years=(1,), semesters=(1,))
        request = EvidenceRequest("course_set", "course_set", empty_scope)
        result = EvidenceExecutionResult(
            "course_set", "course_set", request, empty_scope, "valid_empty", {"courses": []}
        )
        bundle = EvidenceBundle(EvidencePlan(empty_scope, (request,)), (result,))
        claims = _compose_evidence_claims(
            self._spec(("count", "existence", "quantity"), judgement="quantity"), bundle
        )

        self.assertEqual([claims[0].value, claims[1].value], [0, False])
        self.assertEqual(claims[2].value.status, "descriptive_only")
        self.assertEqual(claims[2].status, "descriptive_only")

    def test_compare_adaptation_uses_existing_comparison_without_recomputing(self):
        scope = StructuralScope(program="IT", plans=("coop",), years=(1,), semesters=(1,))
        left = ComponentAggregation("sum_credits", "complete", 3, ({"provenance": ({"source_page": 1},)},))
        right = ComponentAggregation("sum_credits", "complete", 6, ({"provenance": ({"source_page": 2},)},))
        comparison = ComparisonAggregation("complete", "less", left, right)
        request = EvidenceRequest("comparison", "credit_facts", scope)
        result = EvidenceExecutionResult(
            "comparison", "credit_facts", request, scope, "complete", comparison
        )
        bundle = EvidenceBundle(EvidencePlan(scope, (request,)), (result,))
        claims = _compose_evidence_claims(self._spec(("compare",)), bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].value.relation, "less")
        self.assertEqual(
            [reference["source_page"] for reference in claims[0].provenance], [1, 2]
        )


if __name__ == "__main__":
    unittest.main()
