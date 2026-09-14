import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag.aggregation import ComparisonAggregation, ComponentAggregation
from rag.evidence_executor import EvidenceBundle, EvidenceExecutionResult
from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.grounded_answer import GroundedClaim
from rag.qa import _compose_evidence_claims, _course_set_aggregate, ask
from rag.query_spec import parse_query_spec
from rag.retrieval.retrieve import ConstrainedTopicRetrievalResult
from rag.resolution import QueryContext


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


class RagQaTest(unittest.TestCase):
    def test_structured_route_requires_and_uses_injected_callable(self):
        model_callable = lambda _prompt: "SELECT 1"
        structured_result = {"sql": "SELECT 1", "columns": ["x"], "rows": [(1,)]}

        with patch("rag.qa.ask_structured", return_value=structured_result) as structured:
            result = ask("curriculum.db", "How many credits?", model_callable)

        self.assertEqual(result, {"route": "structured", "result": structured_result})
        structured.assert_called_once()
        self.assertEqual(structured.call_args.args[0:2], ("curriculum.db", "How many credits?"))
        self.assertEqual(structured.call_args.args[3], model_callable)

    def test_semantic_route_uses_retrieval_and_top_k(self):
        semantic_result = [{"chunk_id": "course-1-description", "distance": 0.1}]

        with patch("rag.qa.retrieve", return_value=semantic_result) as retrieve:
            result = ask("curriculum.db", "What topics does this course cover?", top_k=3)

        self.assertEqual(result, {"route": "semantic", "result": semantic_result})
        retrieve.assert_called_once_with(
            "curriculum.db", "What topics does this course cover?", k=3
        )

    def test_combined_question_uses_sql_and_semantic_evidence(self):
        model_callable = lambda _prompt: "SELECT 1"
        structured_result = {"sql": "SELECT 1", "columns": ["x"], "rows": [(1,)]}
        semantic_result = [{"chunk_id": "chunk-1", "distance": 0.1}]

        with patch(
            "rag.qa.ask_structured", return_value=structured_result
        ) as structured, patch("rag.qa.retrieve", return_value=semantic_result) as retrieve:
            result = ask(
                "curriculum.db",
                "What database topics are offered in IT year 1?",
                model_callable,
                top_k=3,
            )

        self.assertEqual(
            result,
            {
                "route": "hybrid",
                "result": {
                    "structured": structured_result,
                    "semantic": semantic_result,
                },
            },
        )
        structured.assert_called_once()
        retrieve.assert_called_once_with(
            "curriculum.db",
            "What database topics are offered in IT year 1?",
            k=3,
        )

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
                    "rag.qa.route_question",
                    side_effect=AssertionError("route must not be called"),
                ) as route, patch(
                    "rag.qa.ask_structured",
                    side_effect=AssertionError("structured QA must not be called"),
                ) as structured, patch(
                    "rag.qa.retrieve",
                    side_effect=AssertionError("retrieval must not be called"),
                ) as retrieve:
                    result = ask(DB_PATH, question, forbidden)

                self.assertIsNone(result["route"])
                self.assertEqual(result["result"]["status"], action)
                self.assertEqual(result["result"]["action"], action)
                route.assert_not_called()
                structured.assert_not_called()
                retrieve.assert_not_called()

    def test_context_conflict_stops_before_qa_work(self):
        with patch(
            "rag.qa.route_question",
            side_effect=AssertionError("route must not be called"),
        ) as route, patch(
            "rag.qa.ask_structured",
            side_effect=AssertionError("structured QA must not be called"),
        ) as structured, patch(
            "rag.qa.retrieve",
            side_effect=AssertionError("retrieval must not be called"),
        ) as retrieve:
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
        route.assert_not_called()
        structured.assert_not_called()
        retrieve.assert_not_called()

    def test_identity_returns_typed_exact_evidence_without_qa_paths(self):
        with patch(
            "rag.qa.route_question",
            side_effect=AssertionError("identity must not route"),
        ) as route, patch(
            "rag.qa.ask_structured",
            side_effect=AssertionError("identity must not use structured QA"),
        ) as structured, patch(
            "rag.qa.retrieve",
            side_effect=AssertionError("identity must not retrieve"),
        ) as retrieve:
            result = ask(
                DB_PATH,
                "วิชา Calculus 1 รหัสวิชาอะไร",
                context=QueryContext(program="AIT"),
            )

        self.assertIsNone(result["route"])
        self.assertEqual(result["result"]["operation"], "identity")
        self.assertEqual(result["result"]["status"], "answer")
        self.assertEqual(
            [
                (item["program"], item["course_code"])
                for item in result["result"]["identities"]
            ],
            [("AIT", "06046400")],
        )
        self.assertTrue(result["result"]["identities"][0]["provenance"])
        route.assert_not_called()
        structured.assert_not_called()
        retrieve.assert_not_called()

    def test_identity_unknown_code_keeps_no_data_guard(self):
        result = ask(DB_PATH, "06019999 ชื่ออะไร")

        self.assertIsNone(result["route"])
        self.assertEqual(result["result"]["status"], "no_data")
        self.assertEqual(result["result"]["action"], "no_data")

    def test_identity_code_returns_canonical_name_without_model(self):
        with patch(
            "rag.qa.route_question",
            side_effect=AssertionError("identity must not route"),
        ) as route, patch(
            "rag.qa.retrieve",
            side_effect=AssertionError("identity must not retrieve"),
        ) as retrieve:
            result = ask(DB_PATH, "06046400 ชื่ออะไร")

        self.assertIsNone(result["route"])
        identity = result["result"]["identities"]
        self.assertEqual(len(identity), 1)
        self.assertEqual(identity[0]["program"], "AIT")
        self.assertEqual(identity[0]["course_code"], "06046400")
        self.assertEqual(identity[0]["name_en"], "CALCULUS 1")
        route.assert_not_called()
        retrieve.assert_not_called()

    def test_structured_route_without_callable_fails(self):
        with self.assertRaises(ValueError):
            ask("curriculum.db", "What are the prerequisites?")

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
