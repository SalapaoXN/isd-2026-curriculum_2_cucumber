import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag.aggregation import (
    ComparisonAggregation,
    ComponentAggregation,
    EarliestAggregation,
    aggregate_sum_credits,
    compare_aggregates,
)
from rag.evidence_executor import EvidenceBundle, EvidenceExecutionResult
from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.grounded_answer import GroundedAnswerResult, GroundedClaim
from rag.qa import _compose_evidence_claims, _course_set_aggregate, ask
from rag.query_spec import parse_query_spec
from rag.retrieval.retrieve import (
    ConstrainedTopicRetrievalResult,
    SimilarityEvidence,
    SimilarityPair,
    aggregate_exact_course_similarity,
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
            "มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง": "clarify_program",
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

    def test_context_program_executes_year_semester_list_without_question_program(self):
        result = ask(
            DB_PATH,
            "ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
            context=QueryContext(program="IT"),
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].status, "answer")
        self.assertEqual(
            [claim.operation for claim in result["result"].claims],
            ["list", "list"],
        )
        self.assertEqual(
            [claim.effective_scope.plans for claim in result["result"].claims],
            [("coop",), ("no_coop",)],
        )

    def test_context_program_topic_query_uses_existing_topic_matches_path(self):
        explicit = ask(DB_PATH, "IT มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง")
        contextual = ask(
            DB_PATH,
            "มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง",
            context=QueryContext(program="IT"),
        )

        for result in (explicit, contextual):
            self.assertIsInstance(result["result"], GroundedAnswerResult)
            self.assertEqual(result["result"].status, "answer")
            self.assertTrue(result["result"].claims)
            self.assertTrue(
                all(claim.operation == "list" for claim in result["result"].claims)
            )
            self.assertTrue(
                all(claim.effective_scope.program == "IT" for claim in result["result"].claims)
            )

        self.assertEqual(
            [claim.value for claim in explicit["result"].claims],
            [claim.value for claim in contextual["result"].claims],
        )

    def test_contextless_year_semester_list_still_clarifies_program(self):
        result = ask(DB_PATH, "ปี 1 เทอม 1 มีวิชาอะไรบ้าง")

        self.assertEqual(result["result"]["status"], "clarify_program")
        self.assertEqual(result["result"]["action"], "clarify_program")

    def test_context_plan_restricts_each_runtime_claim_to_selected_plan(self):
        for plan in ("coop", "no_coop"):
            with self.subTest(plan=plan):
                result = ask(
                    DB_PATH,
                    "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
                    context=QueryContext(program="IT", plan=plan),
                )

                self.assertIsInstance(result["result"], GroundedAnswerResult)
                self.assertEqual(result["result"].status, "answer")
                self.assertTrue(result["result"].claims)
                self.assertTrue(
                    all(
                        claim.effective_scope.plans == (plan,)
                        for claim in result["result"].claims
                    )
                )

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

    def test_program_discovery_bypasses_planner_and_preserves_all_matches(self):
        with patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("program discovery must not plan"),
        ) as planner, patch(
            "rag.qa.execute_evidence_plan",
            side_effect=AssertionError("program discovery must not execute"),
        ) as executor:
            result = ask(DB_PATH, "CHARM SCHOOL มีอยู่ในหลักสูตรอะไรบ้าง?")

        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].claims[0].operation, "program_discovery")
        self.assertEqual(result["result"].status, "answer")
        self.assertEqual(
            [
                (item["program"], item["course_code"])
                for item in result["result"].claims[0].value
            ],
            [
                ("BIT", "96641001"),
                ("DSBA", "90641001"),
                ("GENED", "90641001"),
                ("IT", "90641001"),
            ],
        )
        self.assertIn("BIT", result["result"].final_answer)
        self.assertIn("IT", result["result"].final_answer)
        self.assertTrue(result["result"].provenance)
        planner.assert_not_called()
        executor.assert_not_called()

    def test_identity_code_without_program_requests_clarification(self):
        with patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("clarify must not plan"),
        ) as planner, patch(
            "rag.qa.execute_evidence_plan",
            side_effect=AssertionError("clarify must not execute"),
        ) as executor:
            result = ask(DB_PATH, "06046400 ชื่ออะไร")

        self.assertIsNone(result["route"])
        self.assertEqual(result["result"]["status"], "clarify_program")
        self.assertEqual(result["result"]["action"], "clarify_program")
        self.assertEqual(result["result"]["blocking_ambiguity"], ("program",))
        self.assertIsNone(result["result"]["resolved_program"])
        (reference,) = result["result"]["course_references"]
        self.assertEqual(reference["reference"], "06046400")
        self.assertEqual(
            [
                (candidate["program"], candidate["course_code"])
                for candidate in reference["candidates"]
            ],
            [("AIT", "06046400")],
        )
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

    def test_real_similarity_query_without_program_requests_clarification(self):
        with patch(
            "rag.evidence_executor.aggregate_exact_course_similarity",
            side_effect=AssertionError("clarify must not aggregate"),
        ) as aggregate:
            result = ask(
                DB_PATH,
                "06016414 กับ 06016419 เนื้อหาคล้ายกันไหม",
            )

        self.assertIsNone(result["route"])
        self.assertEqual(result["result"]["status"], "clarify_program")
        self.assertEqual(result["result"]["action"], "clarify_program")
        self.assertEqual(result["result"]["blocking_ambiguity"], ("program",))
        self.assertIsNone(result["result"]["resolved_program"])
        self.assertEqual(
            {ref["reference"] for ref in result["result"]["course_references"]},
            {"06016414", "06016419"},
        )
        aggregate.assert_not_called()

    def test_real_similarity_query_without_context_keeps_references_and_clarifies(self):
        with patch(
            "rag.evidence_executor.aggregate_exact_course_similarity",
            side_effect=AssertionError("clarify must not aggregate"),
        ) as aggregate:
            result = ask(
                DB_PATH,
                "06016414 กับ 06016419 เนื้อหาคล้ายกันไหม",
            )

        self.assertIsNone(result["route"])
        self.assertEqual(result["result"]["status"], "clarify_program")
        self.assertEqual(result["result"]["action"], "clarify_program")
        self.assertEqual(result["result"]["blocking_ambiguity"], ("program",))
        self.assertIsNone(result["result"]["resolved_program"])
        self.assertEqual(
            {ref["reference"] for ref in result["result"]["course_references"]},
            {"06016414", "06016419"},
        )
        aggregate.assert_not_called()

    def test_plan_specific_single_course_describe_preserves_no_coop_evidence(self):
        result = ask(
            DB_PATH,
            "06016414 เรียนเกี่ยวกับอะไร",
            context=QueryContext(program="IT", plan="no_coop"),
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claims = [
            claim
            for claim in result["result"].claims
            if claim.effective_scope.plans == ("no_coop",)
        ]
        self.assertEqual(len(claims), 1)
        claim = claims[0]
        self.assertEqual(claim.operation, "describe")
        self.assertEqual(len(claim.evidence), 1)
        evidence = claim.evidence[0]
        self.assertEqual(evidence["course_id"], 736)
        self.assertEqual(evidence["plan"], "no_coop")
        self.assertEqual(
            evidence["source_filename"],
            "merged_it_no_coop_full_corrected.json",
        )
        self.assertEqual(
            {reference["provenance_id"] for reference in evidence["provenance"]},
            {194, 236},
        )

    def test_course_targeted_semantic_detail_reaches_typed_description_evidence(self):
        result = ask(
            DB_PATH,
            "วิชา 06016406 ของ IT แบบไม่สหกิจเป็นโครงงานลักษณะไหน และผู้เรียนต้องทำหรือแสดงผลลัพธ์อะไรบ้าง?",
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].status, "answer")
        claims = [claim for claim in result["result"].claims if claim.operation == "describe"]
        self.assertEqual(len(claims), 1)
        self.assertTrue(claims[0].evidence)
        self.assertTrue(claims[0].provenance)

    def test_multi_operation_placement_wording_produces_supported_claims(self):
        result = ask(
            DB_PATH,
            "วิชา 06016419 ของ IT แบบไม่สหกิจอยู่ปีไหน เทอมไหน กี่หน่วยกิต และเนื้อหาเกี่ยวกับเครือข่ายกับความปลอดภัยอย่างไร?",
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claims = result["result"].claims
        self.assertEqual(
            {claim.operation for claim in claims},
            {"placement", "sum_credits", "describe"},
        )
        self.assertTrue(all(claim.status == "complete" for claim in claims))
        self.assertTrue(all(claim.provenance for claim in claims))

    def test_final_polish_receives_original_question_without_mutating_claims(self):
        question = (
            "วิชา 06016420 ของ IT แบบไม่สหกิจอยู่ช่วงไหนของหลักสูตร "
            "และเรียนเกี่ยวกับอะไรบ้าง?"
        )
        deterministic = ask(DB_PATH, question)["result"]
        prompts = []

        def polish(prompt):
            prompts.append(prompt)
            return (
                "วิชา 06016420 ในหลักสูตร IT แผน no_coop เรียนในปี 2 "
                "ภาคเรียนที่ 2 มี 3(2-2-5) หน่วยกิต และมีเนื้อหาตามหลักฐาน"
            )

        polished = ask(DB_PATH, question, answer_model_callable=polish)["result"]

        self.assertEqual(len(prompts), 1)
        self.assertIn(question, prompts[0])
        self.assertIn("GROUNDED_CONTENT", prompts[0])
        self.assertEqual(polished.claims, deterministic.claims)
        self.assertEqual(polished.provenance, deterministic.provenance)
        self.assertIn("06016420", polished.final_answer)

    def test_residual_course_targeted_wording_produces_complete_claims(self):
        cases = (
            "สำหรับ IT แบบสหกิจ วิชา 06016418 เรียนช่วงไหนของหลักสูตร และเนื้อหาครอบคลุมเรื่องใดเกี่ยวกับฐานข้อมูลบ้าง?",
            "วิชา 06036115 ใน BIT แบบสหกิจเรียนปีไหน เทอมไหน และเนื้อหาช่วยจัดการความปลอดภัยของระบบสารสนเทศเรื่องใดบ้าง?",
        )
        for question in cases:
            with self.subTest(question=question):
                result = ask(DB_PATH, question)

                self.assertIsInstance(result["result"], GroundedAnswerResult)
                self.assertEqual(result["result"].status, "answer")
                claims = result["result"].claims
                self.assertEqual(
                    {claim.operation for claim in claims},
                    {"placement", "describe"},
                )
                self.assertTrue(all(claim.status == "complete" for claim in claims))
                self.assertTrue(all(claim.evidence for claim in claims))
                self.assertTrue(all(claim.provenance for claim in claims))

    def test_existing_dsba_multi_operation_remains_complete(self):
        result = ask(
            DB_PATH,
            "วิชา 06026259 ของ DSBA แบบสหกิจเรียนปีไหน เทอมไหน กี่หน่วยกิต และมีเนื้อหาที่ทำงานร่วมกับสถานประกอบการอย่างไร?",
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claims = result["result"].claims
        self.assertEqual(
            {claim.operation for claim in claims},
            {"placement", "sum_credits", "describe"},
        )
        for claim in claims:
            if claim.operation in {"placement", "describe"}:
                with self.subTest(operation=claim.operation):
                    self.assertEqual(claim.status, "complete")
                    self.assertTrue(claim.provenance)
        # Exact-course credit scope: only the 4/2 term carries the course,
        # so the other evaluated terms are successfully-empty relations.
        credit_by_term = {
            (
                tuple(claim.effective_scope.years),
                tuple(claim.effective_scope.semesters),
            ): claim
            for claim in claims
            if claim.operation == "sum_credits"
        }
        self.assertEqual(
            sorted(credit_by_term),
            [((3,), (1,)), ((3,), (2,)), ((4,), (1,)), ((4,), (2,))],
        )
        for term in (((3,), (1,)), ((3,), (2,)), ((4,), (1,))):
            with self.subTest(term=term):
                self.assertEqual(credit_by_term[term].status, "valid_empty")
                self.assertEqual(credit_by_term[term].value, 0)
        with self.subTest(term=((4,), (2,))):
            self.assertEqual(credit_by_term[((4,), (2,))].status, "complete")
            self.assertEqual(credit_by_term[((4,), (2,))].value, 6)
            self.assertTrue(credit_by_term[((4,), (2,))].provenance)

    def test_flexible_only_exact_course_credit_uses_direct_course_fact(self):
        result = ask(DB_PATH, "IT วิชา 06016465 มีกี่หน่วยกิต?")

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claims = result["result"].claims
        self.assertEqual(len(claims), 1)
        claim = claims[0]
        self.assertEqual(claim.operation, "sum_credits")
        self.assertEqual(claim.status, "complete")
        self.assertEqual(claim.value, 3)
        self.assertEqual(claim.effective_scope.years, ())
        self.assertEqual(claim.effective_scope.semesters, ())
        self.assertEqual(
            tuple(target["course_code"] for target in claim.effective_scope.course_targets),
            ("06016465",),
        )
        self.assertEqual(claim.evidence.components[0]["course_code"], "06016465")
        self.assertTrue(claim.provenance)

    def test_exact_course_name_and_credit_without_program_requests_clarification(self):
        result = ask(DB_PATH, "วิชา 06016420 ชื่ออะไรและมีกี่หน่วยกิต?")

        self.assertIsNone(result["route"])
        self.assertEqual(result["result"]["status"], "clarify_program")
        self.assertEqual(result["result"]["action"], "clarify_program")
        self.assertEqual(result["result"]["blocking_ambiguity"], ("program",))
        self.assertIsNone(result["result"]["resolved_program"])
        (reference,) = result["result"]["course_references"]
        self.assertEqual(reference["reference"], "06016420")
        self.assertEqual(
            [
                (candidate["program"], candidate["course_code"])
                for candidate in reference["candidates"]
            ],
            [("IT", "06016420")],
        )

    def test_explicit_plan_comparison_uses_complete_placement_operands(self):
        result = ask(
            DB_PATH,
            "วิชา 06036103 ใน BIT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหน เทอมไหน และรายละเอียดการจัดวางต่างกันอย่างไร?",
        )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        claims = result["result"].claims
        placement = [claim for claim in claims if claim.operation == "placement"]
        comparison = [claim for claim in claims if claim.operation == "compare"]
        self.assertTrue(placement)
        self.assertTrue(all(claim.status == "complete" for claim in placement))
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison[0].status, "complete")
        self.assertEqual(comparison[0].value.relation, "equal")

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

    def _earliest_spec(self, course_code="06016465"):
        return replace(
            self._spec(("placement", "earliest", "compare")),
            course_codes=(course_code,),
            group_by=("plan",),
        )

    def _earliest_bundle(
        self,
        placements_by_plan,
        *,
        course_code="06016465",
        statuses=None,
        target_codes=None,
    ):
        target = {"program": "IT", "course_code": course_code, "course_id": 65}
        plan_keys = tuple(placements_by_plan)
        bundle_scope = StructuralScope(
            program="IT",
            plans=plan_keys,
            course_targets=(target,),
            group_by=("plan",),
        )
        requests = []
        results = []
        statuses = statuses or {}
        target_codes = target_codes or {}
        for plan in plan_keys:
            effective_target = {
                **target,
                "course_code": target_codes.get(plan, course_code),
            }
            scope = StructuralScope(
                program="IT",
                plans=(plan,),
                course_targets=(effective_target,),
                group_by=("plan",),
            )
            request = EvidenceRequest(
                f"placement_{plan}",
                "placement_facts",
                scope,
                course_targets=(effective_target,),
            )
            requests.append(request)
            results.append(
                EvidenceExecutionResult(
                    request_id=request.request_id,
                    kind=request.kind,
                    planned_request=request,
                    effective_scope=scope,
                    status=statuses.get(plan, "complete"),
                    payload={"courses": placements_by_plan[plan]},
                )
            )
        return EvidenceBundle(
            EvidencePlan(bundle_scope, tuple(requests), group_by=("plan",)),
            tuple(results),
        )

    def _placement_comparison_bundle(
        self,
        records_by_plan,
        target_codes,
        *,
        group_by,
        program="IT",
    ):
        targets = tuple(
            {"program": program, "course_code": code, "course_id": index}
            for index, code in enumerate(target_codes, start=1)
        )
        plans = tuple(records_by_plan)
        bundle_scope = StructuralScope(
            program=program,
            plans=plans,
            course_targets=targets,
            group_by=tuple(group_by),
        )
        requests = []
        results = []
        for plan in plans:
            scope = StructuralScope(
                program=program,
                plans=(plan,),
                course_targets=targets,
                group_by=tuple(group_by),
            )
            for index, record in enumerate(records_by_plan[plan], start=1):
                request = EvidenceRequest(
                    f"placement_{plan}_{index}",
                    "placement_facts",
                    scope,
                    course_targets=targets,
                )
                requests.append(request)
                results.append(
                    EvidenceExecutionResult(
                        request_id=request.request_id,
                        kind=request.kind,
                        planned_request=request,
                        effective_scope=scope,
                        status="complete",
                        payload={"courses": [record]},
                    )
                )
        return EvidenceBundle(
            EvidencePlan(bundle_scope, tuple(requests), group_by=tuple(group_by)),
            tuple(results),
        )

    def _placement(self, plan, year, semester, page, *, course_code="06016465"):
        return {
            "program": "IT",
            "course_code": course_code,
            "plan_key": plan,
            "year": year,
            "semester": semester,
            "provenance": ({"source_page": page},),
        }

    def _year_credit_spec(self, *, semesters=()):
        return replace(
            self._spec(("sum_credits",)),
            years=(2,),
            semesters=tuple(semesters),
            group_by=(),
        )

    def _credit_component(
        self,
        plan,
        semester,
        page,
        *,
        course_code="06016420",
        counted_credit_units=3,
    ):
        return {
            "program": "IT",
            "course_code": course_code,
            "counted_credit_units": counted_credit_units,
            "partition": {
                "program": "IT",
                "plans": (plan,),
                "years": (2,),
                "semesters": (semester,),
                "category": None,
                "group_by": (),
            },
            "provenance": ({"source_page": page},),
        }

    def _credit_group(self, plan, semester, page, member_page):
        return {
            "program": "IT",
            "alternative_group_id": 7,
            "counted_credit_units": 6,
            "alternative_courses": (
                {
                    "program": "IT",
                    "course_code": "06016421",
                    "provenance": ({"source_page": member_page},),
                },
            ),
            "partition": {
                "program": "IT",
                "plans": (plan,),
                "years": (2,),
                "semesters": (semester,),
                "category": None,
                "group_by": (),
            },
            "provenance": ({"source_page": page},),
        }

    def _credit_bundle(self, entries):
        plan_keys = tuple(dict.fromkeys(plan for plan, _semester, _status, _components in entries))
        bundle_scope = StructuralScope(
            program="IT",
            plans=plan_keys,
            years=(2,),
            group_by=(),
        )
        requests = []
        results = []
        for index, (plan, semester, status, components) in enumerate(entries, start=1):
            scope = StructuralScope(
                program="IT",
                plans=(plan,),
                years=(2,),
                semesters=(semester,),
                group_by=(),
            )
            request = EvidenceRequest(
                f"credit_{index}",
                "credit_facts",
                scope,
            )
            requests.append(request)
            results.append(
                EvidenceExecutionResult(
                    request_id=request.request_id,
                    kind=request.kind,
                    planned_request=request,
                    effective_scope=scope,
                    status=status,
                    payload={"components": components},
                )
            )
        return EvidenceBundle(
            EvidencePlan(bundle_scope, tuple(requests)),
            tuple(results),
        )

    def _greatest_credit_spec(self):
        return replace(
            self._spec(("sum_credits", "compare")),
            years=(),
            semesters=(),
            course_codes=(),
            group_by=("semester",),
        )

    def _greatest_credit_component(
        self,
        plan,
        year,
        semester,
        page,
        counted_credit_units=3,
        *,
        alternative_group_id=None,
        alternative_courses=(),
    ):
        component = {
            "program": "IT",
            "course_code": f"00000{page:03d}",
            "counted_credit_units": counted_credit_units,
            "partition": {
                "program": "IT",
                "plans": (plan,),
                "years": (year,),
                "semesters": (semester,),
                "category": None,
                "group_by": ("semester",),
            },
            "provenance": ({"source_page": page},),
        }
        if alternative_group_id is not None:
            component["alternative_group_id"] = alternative_group_id
            component["alternative_courses"] = alternative_courses
        return component

    def _greatest_credit_bundle(self, entries, *, years, semesters):
        plan_keys = tuple(dict.fromkeys(plan for plan, _year, _semester, _status, _components in entries))
        bundle_scope = StructuralScope(
            program="IT",
            plans=plan_keys,
            years=tuple(years),
            semesters=tuple(semesters),
            group_by=("semester",),
        )
        requests = []
        results = []
        for index, (plan, year, semester, status, components) in enumerate(entries, start=1):
            scope = StructuralScope(
                program="IT",
                plans=(plan,),
                years=(year,),
                semesters=(semester,),
                group_by=("semester",),
            )
            request = EvidenceRequest(f"greatest_{index}", "credit_facts", scope)
            requests.append(request)
            results.append(
                EvidenceExecutionResult(
                    request_id=request.request_id,
                    kind=request.kind,
                    planned_request=request,
                    effective_scope=scope,
                    status=status,
                    payload={"components": components},
                )
            )
        return EvidenceBundle(
            EvidencePlan(bundle_scope, tuple(requests)),
            tuple(results),
        )

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

    def test_typed_course_set_claims_preserve_alternative_parent(self):
        alternative_parent = {
            "program": "IT",
            "course_code": None,
            "alternative_group_id": 8,
            "minimum_choices": 1,
            "maximum_choices": 1,
            "counted_credit_units": 6,
            "alternative_courses": [
                {"program": "IT", "course_code": "06016481"},
                {"program": "IT", "course_code": "06016482"},
            ],
            "partition": {"plan": "coop", "year": 3, "semester": 2},
            "provenance": ({"group": 8}, {"source_page": 20}),
        }
        bundle = self._bundle(
            (
                (
                    "course_set",
                    "course_set",
                    {"courses": [alternative_parent]},
                    "complete",
                ),
            )
        )

        claims = _compose_evidence_claims(
            self._spec(("list", "count", "existence")), bundle
        )

        self.assertEqual([claim.status for claim in claims], ["complete"] * 3)
        item = claims[0].value[0]
        self.assertIsNone(item["course_code"])
        self.assertEqual(item["alternative_group_id"], 8)
        self.assertEqual(
            {member["course_code"] for member in item["alternative_courses"]},
            {"06016481", "06016482"},
        )
        self.assertEqual(claims[1].value, 1)
        self.assertTrue(claims[2].value)
        self.assertIn({"group": 8}, claims[0].provenance)
        self.assertIn({"source_page": 20}, claims[0].provenance)

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

    def test_coarse_year_credit_rollup_groups_each_plan_once(self):
        bundle = self._credit_bundle(
            (
                (
                    "coop",
                    1,
                    "complete",
                    [self._credit_component("coop", 1, 101)],
                ),
                (
                    "coop",
                    2,
                    "complete",
                    [
                        self._credit_component("coop", 2, 102),
                        self._credit_group("coop", 2, 103, 104),
                    ],
                ),
                (
                    "no_coop",
                    1,
                    "complete",
                    [self._credit_component("no_coop", 1, 201, counted_credit_units=5)],
                ),
                (
                    "no_coop",
                    2,
                    "complete",
                    [self._credit_component("no_coop", 2, 202, counted_credit_units=6)],
                ),
            )
        )

        with patch("rag.qa.aggregate_sum_credits", wraps=aggregate_sum_credits) as aggregate:
            claims = _compose_evidence_claims(self._year_credit_spec(), bundle)

        self.assertEqual(len(claims), 2)
        self.assertEqual(
            [(claim.effective_scope.plans, claim.value) for claim in claims],
            [(('coop',), 12), (('no_coop',), 11)],
        )
        self.assertEqual(aggregate.call_count, 2)
        self.assertTrue(all(claim.effective_scope.semesters == () for claim in claims))
        coop_components = claims[0].evidence.components
        grouped = next(
            component
            for component in coop_components
            if component.get("alternative_group_id") == 7
        )
        self.assertEqual(grouped["alternative_courses"][0]["provenance"], ({"source_page": 104},))
        self.assertEqual(
            [reference["source_page"] for reference in claims[0].provenance],
            [101, 103, 102],
        )
        self.assertEqual(
            {component["partition"]["semesters"] for component in coop_components},
            {(1,), (2,)},
        )

    def test_coarse_year_credit_rollup_complete_plus_valid_empty_is_complete(self):
        bundle = self._credit_bundle(
            (
                ("coop", 1, "complete", [self._credit_component("coop", 1, 101)]),
                ("coop", 2, "valid_empty", []),
            )
        )

        claims = _compose_evidence_claims(self._year_credit_spec(), bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].status, "complete")
        self.assertEqual(claims[0].value, 3)
        self.assertEqual(claims[0].effective_scope.semesters, ())

    def test_coarse_year_credit_rollup_all_valid_empty_keeps_zero_semantics(self):
        bundle = self._credit_bundle(
            (
                ("coop", 1, "valid_empty", []),
                ("coop", 2, "valid_empty", []),
            )
        )

        claims = _compose_evidence_claims(self._year_credit_spec(), bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].status, "valid_empty")
        self.assertEqual(claims[0].value, 0)
        self.assertEqual(claims[0].evidence.status, "valid_empty")

    def test_coarse_year_credit_rollup_insufficient_semester_fails_closed(self):
        bundle = self._credit_bundle(
            (
                ("coop", 1, "complete", [self._credit_component("coop", 1, 101)]),
                ("coop", 2, "insufficient_evidence", []),
            )
        )

        claims = _compose_evidence_claims(self._year_credit_spec(), bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].status, "insufficient_evidence")
        self.assertIsNone(claims[0].value)
        self.assertIsNone(claims[0].evidence)
        self.assertEqual(claims[0].provenance, ())

    def test_explicit_semester_credit_query_keeps_semester_level_behavior(self):
        bundle = self._credit_bundle(
            (("coop", 1, "complete", [self._credit_component("coop", 1, 101)]),)
        )

        claims = _compose_evidence_claims(
            self._year_credit_spec(semesters=(1,)), bundle
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].value, 3)
        self.assertEqual(claims[0].effective_scope.semesters, (1,))

    def test_greatest_credit_selects_highest_concrete_term(self):
        entries = tuple(
            (
                "coop",
                year,
                semester,
                "complete",
                [self._greatest_credit_component("coop", year, semester, page, credits)],
            )
            for year, semester, page, credits in (
                (1, 1, 301, 5),
                (1, 2, 302, 9),
                (2, 1, 303, 7),
                (2, 2, 304, 4),
            )
        )
        bundle = self._greatest_credit_bundle(entries, years=(1, 2), semesters=(1, 2))

        claims = _compose_evidence_claims(self._greatest_credit_spec(), bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].operation, "sum_credits")
        self.assertEqual(claims[0].value, 9)
        self.assertEqual(claims[0].effective_scope.plans, ("coop",))
        self.assertEqual(claims[0].effective_scope.years, (1,))
        self.assertEqual(claims[0].effective_scope.semesters, (2,))

    def test_greatest_credit_reduces_plans_independently(self):
        entries = []
        for plan, values in (("coop", (5, 9)), ("no_coop", (8, 6))):
            for year, credits in enumerate(values, start=1):
                entries.append(
                    (
                        plan,
                        year,
                        1,
                        "complete",
                        [self._greatest_credit_component(plan, year, 1, 310 + year, credits)],
                    )
                )
        bundle = self._greatest_credit_bundle(entries, years=(1, 2), semesters=(1,))

        claims = _compose_evidence_claims(self._greatest_credit_spec(), bundle)

        self.assertEqual(
            [(claim.effective_scope.plans, claim.value) for claim in claims],
            [(('coop',), 9), (('no_coop',), 8)],
        )

    def test_greatest_credit_delegates_candidate_arithmetic(self):
        entries = tuple(
            (
                "coop",
                year,
                1,
                "complete",
                [self._greatest_credit_component("coop", year, 1, 320 + year, credits)],
            )
            for year, credits in ((1, 5), (2, 9), (3, 7))
        )
        bundle = self._greatest_credit_bundle(entries, years=(1, 2, 3), semesters=(1,))

        with patch("rag.qa.aggregate_sum_credits", wraps=aggregate_sum_credits) as aggregate:
            claims = _compose_evidence_claims(self._greatest_credit_spec(), bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].value, 9)
        self.assertEqual(aggregate.call_count, 3)

    def test_greatest_credit_valid_empty_is_a_zero_candidate(self):
        bundle = self._greatest_credit_bundle(
            (
                ("coop", 1, 1, "valid_empty", []),
                ("coop", 2, 1, "valid_empty", []),
            ),
            years=(1, 2),
            semesters=(1,),
        )

        claims = _compose_evidence_claims(self._greatest_credit_spec(), bundle)

        self.assertEqual(len(claims), 2)
        self.assertEqual([claim.status for claim in claims], ["valid_empty", "valid_empty"])
        self.assertEqual([claim.value for claim in claims], [0, 0])
        self.assertEqual(
            [claim.effective_scope.years for claim in claims],
            [(1,), (2,)],
        )
        self.assertTrue(all(claim.evidence.status == "valid_empty" for claim in claims))

    def test_greatest_credit_insufficient_candidate_fails_plan_closed(self):
        bundle = self._greatest_credit_bundle(
            (
                ("coop", 1, 1, "complete", [self._greatest_credit_component("coop", 1, 1, 330, 9)]),
                ("coop", 2, 1, "insufficient_evidence", []),
            ),
            years=(1, 2),
            semesters=(1,),
        )

        claims = _compose_evidence_claims(self._greatest_credit_spec(), bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].status, "insufficient_evidence")
        self.assertIsNone(claims[0].value)
        self.assertIsNone(claims[0].evidence)

    def test_greatest_credit_missing_candidate_is_not_zero(self):
        bundle = self._greatest_credit_bundle(
            (
                ("coop", 1, 1, "complete", [self._greatest_credit_component("coop", 1, 1, 340, 9)]),
                ("coop", 1, 2, "complete", [self._greatest_credit_component("coop", 1, 2, 341, 5)]),
                ("coop", 2, 1, "complete", [self._greatest_credit_component("coop", 2, 1, 342, 7)]),
            ),
            years=(1, 2),
            semesters=(1, 2),
        )

        claims = _compose_evidence_claims(self._greatest_credit_spec(), bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].status, "insufficient_evidence")
        self.assertIsNone(claims[0].value)

    def test_greatest_credit_returns_all_ties_in_term_order_with_provenance(self):
        alt = (
            {
                "program": "IT",
                "course_code": "00000001",
                "provenance": ({"source_page": 350},),
            },
        )
        entries = (
            (
                "coop",
                3,
                1,
                "complete",
                [
                    self._greatest_credit_component(
                        "coop", 3, 1, 351, 21,
                        alternative_group_id=7,
                        alternative_courses=alt,
                    )
                ],
            ),
            ("coop", 1, 1, "complete", [self._greatest_credit_component("coop", 1, 1, 352, 4)]),
            ("coop", 1, 2, "complete", [self._greatest_credit_component("coop", 1, 2, 353, 21)]),
            ("coop", 3, 2, "complete", [self._greatest_credit_component("coop", 3, 2, 354, 6)]),
        )
        bundle = self._greatest_credit_bundle(entries, years=(1, 3), semesters=(1, 2))

        claims = _compose_evidence_claims(self._greatest_credit_spec(), bundle)

        self.assertEqual(
            [(claim.effective_scope.years, claim.effective_scope.semesters, claim.value) for claim in claims],
            [((1,), (2,), 21), ((3,), (1,), 21)],
        )
        self.assertEqual(claims[0].evidence.components[0]["course_code"], "00000353")
        self.assertEqual(claims[0].provenance, ({"source_page": 353},))
        self.assertEqual(claims[1].evidence.components[0]["alternative_group_id"], 7)
        self.assertEqual(claims[1].evidence.components[0]["alternative_courses"], alt)
        self.assertEqual(
            [reference["source_page"] for reference in claims[1].provenance],
            [351],
        )

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

    def test_compare_without_comparison_payload_fails_closed(self):
        bundle = self._bundle(
            (("course_set", "course_set", {"courses": []}, "valid_empty"),)
        )

        claims = _compose_evidence_claims(self._spec(("compare",)), bundle)

        self.assertEqual(len(claims), 1)
        claim = claims[0]
        self.assertEqual(claim.operation, "compare")
        self.assertEqual(claim.status, "insufficient_evidence")
        self.assertIsNone(claim.value)
        self.assertIsNone(claim.evidence)
        self.assertEqual(claim.provenance, ())

    def test_normal_two_course_placement_comparison_is_partition_local(self):
        spec = parse_query_spec("06016414 กับ 06016419 วิชาไหนเรียนก่อน?")
        bundle = self._placement_comparison_bundle(
            {
                "coop": [
                    self._placement("coop", 2, 2, 11, course_code="06016414"),
                    self._placement("coop", 2, 2, 12, course_code="06016419"),
                ],
                "no_coop": [
                    self._placement("no_coop", 2, 2, 21, course_code="06016414"),
                    self._placement("no_coop", 2, 2, 22, course_code="06016419"),
                ],
            },
            ("06016414", "06016419"),
            group_by=("course",),
        )

        claims = _compose_evidence_claims(spec, bundle)
        comparisons = [claim for claim in claims if claim.operation == "compare"]
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(claim.status == "complete" for claim in comparisons))
        self.assertTrue(all(claim.value.relation == "equal" for claim in comparisons))
        self.assertEqual(
            {
                partition.partition["plans"]
                for claim in comparisons
                for aggregate in (claim.value.left, claim.value.right)
                for partition in aggregate.partitions
            },
            {("coop",), ("no_coop",)},
        )
        self.assertEqual(
            len({reference["source_page"] for claim in comparisons for reference in claim.provenance}),
            4,
        )

    def test_normal_two_course_placement_comparison_orders_operands(self):
        spec = parse_query_spec("06016414 กับ 06016419 วิชาไหนเรียนก่อน?")
        earlier = self._placement_comparison_bundle(
            {
                "coop": [
                    self._placement("coop", 1, 1, 11, course_code="06016414"),
                    self._placement("coop", 2, 1, 12, course_code="06016419"),
                ]
            },
            ("06016414", "06016419"),
            group_by=("course",),
        )
        later = self._placement_comparison_bundle(
            {
                "coop": [
                    self._placement("coop", 2, 1, 11, course_code="06016414"),
                    self._placement("coop", 1, 1, 12, course_code="06016419"),
                ]
            },
            ("06016414", "06016419"),
            group_by=("course",),
        )

        self.assertEqual(
            _compose_evidence_claims(spec, earlier)[-1].value.relation,
            "less",
        )
        self.assertEqual(
            _compose_evidence_claims(spec, later)[-1].value.relation,
            "greater",
        )

    def test_normal_same_course_plan_comparison_preserves_plan_operands(self):
        spec = parse_query_spec(
            "06016401 ในแผนสหกิจกับไม่สหกิจ เรียนช่วงเดียวกันไหม?"
        )
        bundle = self._placement_comparison_bundle(
            {
                "coop": [self._placement("coop", 1, 1, 11, course_code="06016401")],
                "no_coop": [
                    self._placement("no_coop", 1, 1, 22, course_code="06016401")
                ],
            },
            ("06016401",),
            group_by=("plan",),
        )

        claims = _compose_evidence_claims(spec, bundle)
        comparison = next(claim for claim in claims if claim.operation == "compare")
        self.assertEqual(comparison.status, "complete")
        self.assertEqual(comparison.value.relation, "equal")
        self.assertEqual(
            [
                operand.partitions[0].partition["plans"]
                for operand in (comparison.value.left, comparison.value.right)
            ],
            [("coop",), ("no_coop",)],
        )

    def test_normal_same_course_plan_comparison_reports_unequal_periods(self):
        spec = parse_query_spec(
            "06016401 ในแผนสหกิจกับไม่สหกิจ เรียนช่วงเดียวกันไหม?"
        )
        bundle = self._placement_comparison_bundle(
            {
                "coop": [self._placement("coop", 1, 1, 11, course_code="06016401")],
                "no_coop": [
                    self._placement("no_coop", 2, 1, 22, course_code="06016401")
                ],
            },
            ("06016401",),
            group_by=("plan",),
        )

        comparison = next(
            claim
            for claim in _compose_evidence_claims(spec, bundle)
            if claim.operation == "compare"
        )
        self.assertEqual(comparison.status, "complete")
        self.assertEqual(comparison.value.relation, "less")

    def test_normal_placement_comparison_missing_or_incompatible_operand_fails_closed(self):
        spec = parse_query_spec("06016414 กับ 06016419 วิชาไหนเรียนก่อน?")
        missing = self._placement_comparison_bundle(
            {
                "coop": [self._placement("coop", 2, 2, 11, course_code="06016414")]
            },
            ("06016414", "06016419"),
            group_by=("course",),
        )
        incompatible_record = self._placement(
            "coop", 2, 2, 11, course_code="06016414"
        )
        incompatible_record["program"] = "DSBA"
        incompatible = self._placement_comparison_bundle(
            {"coop": [incompatible_record, self._placement("coop", 2, 2, 12, course_code="06016419")]},
            ("06016414", "06016419"),
            group_by=("course",),
        )

        for bundle in (missing, incompatible):
            with self.subTest(bundle=bundle):
                comparison = next(
                    claim
                    for claim in _compose_evidence_claims(spec, bundle)
                    if claim.operation == "compare"
                )
                self.assertEqual(comparison.status, "insufficient_evidence")
                self.assertIsNone(comparison.value)

    def test_earliest_plan_comparison_preserves_isolated_operands_and_provenance(self):
        bundle = self._earliest_bundle(
            {
                "coop": [self._placement("coop", 2, 2, 11)],
                "no_coop": [self._placement("no_coop", 4, 1, 22)],
            }
        )

        with patch("rag.qa.compare_aggregates", wraps=compare_aggregates) as compare:
            claims = _compose_evidence_claims(self._earliest_spec(), bundle)

        comparison = next(claim for claim in claims if claim.operation == "compare")
        self.assertEqual(comparison.status, "complete")
        self.assertIsInstance(comparison.value, ComparisonAggregation)
        self.assertEqual(comparison.value.relation, "less")
        self.assertTrue(compare.called)
        left = comparison.value.left
        right = comparison.value.right
        self.assertIsInstance(left, EarliestAggregation)
        self.assertIsInstance(right, EarliestAggregation)
        self.assertEqual(left.value, (2, 2))
        self.assertEqual(right.value, (4, 1))
        self.assertEqual(left.partitions[0].partition["plans"], ("coop",))
        self.assertEqual(right.partitions[0].partition["plans"], ("no_coop",))
        self.assertEqual(
            [reference["source_page"] for reference in comparison.provenance],
            [11, 22],
        )

    def test_earliest_plan_comparison_equal_values_produce_equal(self):
        bundle = self._earliest_bundle(
            {
                "coop": [self._placement("coop", 2, 2, 11)],
                "no_coop": [self._placement("no_coop", 2, 2, 22)],
            }
        )

        claims = _compose_evidence_claims(self._earliest_spec(), bundle)

        comparison = next(claim for claim in claims if claim.operation == "compare")
        self.assertEqual(comparison.status, "complete")
        self.assertEqual(comparison.value.relation, "equal")

    def test_earliest_plan_comparison_missing_operand_fails_closed(self):
        bundle = self._earliest_bundle(
            {
                "coop": [self._placement("coop", 2, 2, 11)],
                "no_coop": [],
            },
            statuses={"no_coop": "insufficient_evidence"},
        )

        claims = _compose_evidence_claims(self._earliest_spec(), bundle)

        comparison = next(claim for claim in claims if claim.operation == "compare")
        self.assertEqual(comparison.status, "insufficient_evidence")
        self.assertIsNone(comparison.value)
        self.assertIsNone(comparison.evidence)
        self.assertEqual(comparison.provenance, ())

    def test_earliest_plan_comparison_identity_mismatch_fails_closed(self):
        bundle = self._earliest_bundle(
            {
                "coop": [self._placement("coop", 2, 2, 11)],
                "no_coop": [
                    self._placement("no_coop", 4, 1, 22, course_code="06016414")
                ],
            },
            target_codes={"no_coop": "06016414"},
        )

        claims = _compose_evidence_claims(self._earliest_spec(), bundle)

        comparison = next(claim for claim in claims if claim.operation == "compare")
        self.assertEqual(comparison.status, "insufficient_evidence")
        self.assertIsNone(comparison.value)

    def test_unsupported_comparison_shape_remains_fail_closed(self):
        bundle = self._bundle(
            (("credit_facts", "credit_facts", {"components": []}, "valid_empty"),)
        )

        claims = _compose_evidence_claims(
            self._spec(("sum_credits", "compare")), bundle
        )

        comparison = next(claim for claim in claims if claim.operation == "compare")
        self.assertEqual(comparison.status, "insufficient_evidence")
        self.assertIsNone(comparison.value)
        self.assertIsNone(comparison.evidence)
        self.assertEqual(comparison.provenance, ())


if __name__ == "__main__":
    unittest.main()
