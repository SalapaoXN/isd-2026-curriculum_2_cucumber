import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag.aggregation import (
    ComparisonAggregation,
    ComponentAggregation,
    EarliestAggregation,
    PlanComparisonAggregation,
    aggregate_sum_credits,
    compare_aggregates,
)
from rag.evidence_executor import EvidenceBundle, EvidenceExecutionResult
from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.grounded_answer import GroundedAnswerResult, GroundedClaim
from rag.structured.fallback import (
    GroundedCourseCreditResult,
    GroundedCourseListResult,
    GroundedPlacementResult,
    StructuredFallbackResult,
)
from rag.qa import (
    _classify_structured_parse_completeness,
    _compose_evidence_claims,
    _course_set_aggregate,
    _fallback_scope,
    _is_placement_fallback_candidate,
    _should_use_intent_interpreter,
    ask,
)
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
    resolve_query_spec,
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

    def test_english_y2_list_keeps_year_scope_in_grounded_claims(self):
        result = ask(DB_PATH, "DSBA Y2 มีวิชาอะไรบ้าง")

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertTrue(result["result"].claims)
        self.assertTrue(
            all(claim.effective_scope.years == (2,) for claim in result["result"].claims)
        )

    def test_invalid_english_year_fails_closed_without_scope_widening(self):
        with patch("rag.qa.run_structured_fallback") as fallback:
            result = ask(
                DB_PATH,
                "DSBA Y5 มีวิชาอะไรบ้าง",
                structured_model_callable=lambda prompt: self.fail(
                    "invalid year must not enter fallback"
                ),
            )

        self.assertEqual(result["result"]["status"], "unsupported")
        fallback.assert_not_called()

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

    def test_workload_credit_wording_without_program_still_clarifies(self):
        with patch("rag.qa.run_structured_fallback") as fallback:
            result = ask(
                DB_PATH,
                "วิชา 06026212 หนักกี่หน่วย",
                structured_model_callable=lambda prompt: self.fail(
                    "program-free workload must not use credit fallback"
                ),
            )

        self.assertEqual(result["result"]["status"], "clarify_program")
        fallback.assert_not_called()

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

    def test_identity_code_without_program_answers_directly(self):
        with patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("clarify must not plan"),
        ) as planner, patch(
            "rag.qa.execute_evidence_plan",
            side_effect=AssertionError("clarify must not execute"),
        ) as executor:
            result = ask(DB_PATH, "06046400 ชื่ออะไร")

        self.assertIsNone(result["route"])
        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].status, "answer")
        self.assertEqual(result["result"].claims[0].operation, "identity")
        self.assertEqual(
            [
                (candidate["program"], candidate["course_code"])
                for candidate in result["result"].claims[0].value
            ],
            [("AIT", "06046400")],
        )
        planner.assert_not_called()
        executor.assert_not_called()

    def test_pure_identity_queries_bypass_planner_executor_and_model(self):
        questions = (
            "Calculus 1 มีรหัสวิชาอะไร",
            "06016420 ชื่อวิชาอะไร",
            "DSBA วิชา Calculus 1 มีรหัสวิชาอะไร",
        )
        with patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("identity must not plan"),
        ) as planner, patch(
            "rag.qa.execute_evidence_plan",
            side_effect=AssertionError("identity must not execute"),
        ) as executor:
            for question in questions:
                with self.subTest(question=question):
                    result = ask(DB_PATH, question)
                    self.assertIsInstance(result["result"], GroundedAnswerResult)
                    self.assertEqual(result["result"].status, "answer")
                    self.assertTrue(result["result"].final_answer)
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

    def test_program_free_calculus_prerequisite_answers_by_consensus(self):
        result = ask(DB_PATH, "Calculus 2 มีวิชาบังคับก่อนคืออะไร")

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].status, "answer")
        self.assertIn("CALCULUS 1", result["result"].final_answer)
        self.assertTrue(result["result"].provenance)
        self.assertEqual(
            [claim.operation for claim in result["result"].claims],
            ["prerequisite"],
        )

    def test_structured_parse_completeness_preserves_deterministic_fast_paths(self):
        for question in (
            "DSBA ปี 2 มีวิชาศึกษาทั่วไปอะไรบ้าง",
            "IT ปี 2 เทอม 2 ลงทะเบียนรวมกี่หน่วยกิต?",
            "วิชา IT 06016420 มีหน่วยกิตเท่าไร?",
        ):
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH)
                result = _classify_structured_parse_completeness(spec, resolution)
                self.assertEqual(result.classification, "complete")

    def test_structured_parse_completeness_detects_bounded_partial_filters(self):
        cases = (
            (
                "DSBA ปี 2 มีวิชา Gen Ed อะไรบ้าง",
                ("category",),
            ),
            (
                "DSBA ปี 2 มีวิชาบังคับ 3 หน่วยกิตอะไรบ้าง",
                ("requirement_type", "credit_units"),
            ),
        )
        for question, missing_filters in cases:
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH)
                result = _classify_structured_parse_completeness(spec, resolution)
                self.assertEqual(result.classification, "partial")
                self.assertEqual(result.missing_filters, missing_filters)
                self.assertEqual(result.program, "DSBA")

    def test_structured_residue_is_detected_before_ineligible_operation_checks(self):
        cases = (
            (
                "IT แผนสหกิจกับไม่สหกิจ วิชาบังคับต่างกันยังไง",
                ("requirement_type",),
            ),
            (
                "IT ปี 2 แผนสหกิจกับไม่สหกิจ 3 หน่วยกิตต่างกันยังไง",
                ("credit_units",),
            ),
            (
                "IT วิชาบังคับมีวิชาอะไรเกี่ยวกับ database บ้าง",
                ("requirement_type",),
            ),
        )
        for question, missing_filters in cases:
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH)
                result = _classify_structured_parse_completeness(spec, resolution)
                self.assertEqual(result.classification, "partial")
                self.assertEqual(result.missing_filters, missing_filters)

                with patch(
                    "rag.qa.plan_evidence",
                    side_effect=AssertionError(
                        "unresolved structured residue must not reach planner"
                    ),
                ) as planner:
                    answer = ask(DB_PATH, question)

                self.assertEqual(answer["result"].status, "insufficient_evidence")
                planner.assert_not_called()

    def test_structured_residue_free_ineligible_queries_keep_existing_classification(self):
        cases = (
            ("IT แผนสหกิจกับไม่สหกิจต่างกันยังไง", ("compare",)),
            ("IT มีวิชาอะไรเกี่ยวกับ database บ้าง", ("list",)),
            (
                "IT วิชา 06016402 กับ 06026207 เนื้อหาต่างกันอย่างไร",
                ("similarity", "describe"),
            ),
        )
        for question, operations in cases:
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH)
                result = _classify_structured_parse_completeness(spec, resolution)
                self.assertEqual(tuple(spec.operations), operations)
                self.assertEqual(result.classification, "not_eligible")
                self.assertEqual(result.missing_filters, ())

    def test_parsed_categories_are_not_treated_as_unresolved_residue(self):
        for question in (
            "DSBA ปี 2 มีวิชาศึกษาทั่วไปอะไรบ้าง",
            "DSBA ปี 2 มีวิชาเลือกอะไรบ้าง",
        ):
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH)
                result = _classify_structured_parse_completeness(spec, resolution)
                self.assertIsNotNone(spec.category)
                self.assertNotIn("category", result.missing_filters)

    def test_structured_parse_completeness_uses_ui_program_for_partial_query(self):
        spec = parse_query_spec("ปี 2 มีวิชา Gen Ed อะไรบ้าง")
        context = QueryContext(program="DSBA")
        resolution = resolve_query_spec(spec, DB_PATH, context=context)

        result = _classify_structured_parse_completeness(spec, resolution, context)

        self.assertEqual(result.classification, "partial")
        self.assertEqual(result.program, "DSBA")

    def test_structured_parse_completeness_rejects_semantic_unsupported_and_blocked(self):
        cases = (
            ("IT วิชา 06016404 เรียนเกี่ยวกับอะไรบ้าง?", None),
            ("IT วิชา 06016420 ยากไหม?", None),
            ("วิชา 06016404 เรียนเกี่ยวกับอะไรบ้าง?", None),
        )
        for question, context in cases:
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH, context=context)
                result = _classify_structured_parse_completeness(
                    spec, resolution, context
                )
                self.assertEqual(result.classification, "not_eligible")

    def test_structured_parse_completeness_marks_unrecognized_structured_shape(self):
        original = parse_query_spec("DSBA ปี 2 มีวิชาอะไรบ้าง")
        spec = replace(original, operations=())
        resolution = resolve_query_spec(spec, DB_PATH)

        result = _classify_structured_parse_completeness(spec, resolution)

        self.assertEqual(result.classification, "unrecognized_structured")
        self.assertEqual(result.program, "DSBA")

    def test_intent_interpreter_accepts_only_bounded_operation_free_long_tail(self):
        cases = (
            ("IT 06016404 เน้น data", True),
            ("IT 06016404 เรียนเกี่ยวกับอะไรบ้าง", False),
            ("IT ปี 2 เทอม 1 มีวิชาอะไรบ้าง", False),
            ("IT 06016404 ยากไหม", False),
            ("06016404 เน้น data", False),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH)
                completeness = _classify_structured_parse_completeness(
                    spec,
                    resolution,
                )
                self.assertEqual(
                    _should_use_intent_interpreter(
                        spec,
                        completeness,
                        resolution,
                    ),
                    expected,
                )

    def test_intent_interpreter_compiles_and_reaches_planner_once(self):
        model_calls = []
        interpreted_payload = (
            '{"intent":"course_description",'
            '"proposed_program":null,"proposed_plans":[],'
            '"proposed_years":[],"proposed_semesters":[],'
            '"course_codes":[],"topic":null,'
            '"requested_facts":["course_description"],'
            '"judgement_dimension":null,"unresolved":[]}'
        )
        plan = object()

        def intent_model(prompt):
            model_calls.append(prompt)
            return interpreted_payload

        with patch("rag.qa.plan_evidence", return_value=plan) as planner, patch(
            "rag.qa.execute_evidence_plan", return_value=object()
        ), patch(
            "rag.qa._compose_evidence_claims", return_value=()
        ), patch(
            "rag.qa.render_grounded_answer", return_value={"status": "answer"}
        ):
            result = ask(
                DB_PATH,
                "IT 06016404 เน้น data",
                intent_model_callable=intent_model,
            )

        self.assertEqual(result["result"]["status"], "answer")
        self.assertEqual(len(model_calls), 1)
        planner.assert_called_once()
        compiled_spec = planner.call_args.args[0]
        self.assertEqual(compiled_spec.program, "IT")
        self.assertEqual(compiled_spec.course_codes, ("06016404",))
        self.assertEqual(compiled_spec.operations, ("describe",))

    def test_intent_interpreter_failure_is_fail_closed_without_planner(self):
        with patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("failed intent must not plan"),
        ) as planner:
            result = ask(
                DB_PATH,
                "IT 06016404 เน้น data",
                intent_model_callable=lambda _prompt: "not json",
            )

        self.assertEqual(result["result"].status, "insufficient_evidence")
        planner.assert_not_called()

    def test_operation_bearing_deterministic_query_never_invokes_intent_model(self):
        plan = EvidencePlan(
            StructuralScope(program="IT", years=(2,), semesters=(1,)),
            (),
        )
        with patch(
            "rag.qa.plan_evidence",
            return_value=plan,
        ), patch(
            "rag.qa.execute_evidence_plan",
            return_value=EvidenceBundle(plan, ()),
        ), patch(
            "rag.qa._compose_evidence_claims",
            return_value=(),
        ), patch(
            "rag.qa.render_grounded_answer",
            return_value={"status": "answer"},
        ):
            result = ask(
                DB_PATH,
                "IT ปี 2 เทอม 1 มีวิชาอะไรบ้าง",
                intent_model_callable=lambda _prompt: self.fail(
                    "complete deterministic queries must not interpret"
                ),
            )

        self.assertEqual(result["result"]["status"], "answer")

    @staticmethod
    def fallback_course_record():
        return {
            "program": "DSBA",
            "plan_key": "no_coop",
            "year_number": 2,
            "semester_number": 1,
            "course_id": 101,
            "course_code": "90642067",
            "name_th": "ซอฟต์บอลและเบสบอล",
            "name_en": "SOFTBALL AND BASEBALL",
            "credits": "3(3-0-6)",
            "placement_credits": "3(3-0-6)",
            "category": "หมวดวิชาศึกษาทั่วไป",
            "requirement_type": "required",
            "is_alternative": False,
            "alternative_group_id": None,
            "placement_id": 201,
            "provenance": ({"provenance_id": 301, "source_page": 2},),
        }

    @staticmethod
    def fallback_placement_record():
        return {
            "program": "DSBA",
            "plan_key": "no_coop",
            "year_number": 2,
            "semester_number": 1,
            "course_id": 212,
            "course_code": "06026212",
            "name_th": "COURSE TH",
            "name_en": "CANONICAL PLACEMENT NAME",
            "credits": "3(3-0-6)",
            "placement_id": 401,
            "provenance": ({"provenance_id": 501, "source_page": 7},),
            "is_alternative": False,
            "alternative_group_id": None,
        }

    def _patch_list_fallback(self, *, status="complete"):
        sql_result = StructuredFallbackResult(
            status="success",
            sql="SELECT course_id FROM courses LIMIT 100",
            columns=("course_id",),
            rows=((101,),),
        )
        grounded_result = GroundedCourseListResult(
            status=status,
            records=(self.fallback_course_record(),) if status == "complete" else (),
        )
        return sql_result, grounded_result

    def test_partial_category_list_uses_one_fallback_and_canonical_claim(self):
        _, grounded_result = self._patch_list_fallback()
        structured_calls = []
        answer_calls = []

        def structured_model(prompt):
            structured_calls.append(prompt)
            return "SELECT course_id FROM courses"

        def forbidden_answer_model(prompt):
            answer_calls.append(prompt)
            self.fail("SQL fallback answers must not invoke answer polishing")

        with patch(
            "rag.qa.ground_course_list", return_value=grounded_result
        ) as ground, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("partial list must bypass planner"),
        ):
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชา Gen Ed อะไรบ้าง",
                structured_model_callable=structured_model,
                answer_model_callable=forbidden_answer_model,
                intent_model_callable=lambda _prompt: self.fail(
                    "approved SQL fallback must run before intent interpretation"
                ),
            )

        self.assertEqual(len(structured_calls), 1)
        ground.assert_called_once()
        self.assertEqual(result["result"].status, "answer")
        self.assertEqual(result["result"].claims[0].operation, "list")
        self.assertIn("90642067", result["result"].final_answer)
        self.assertNotIn("SELECT course_id", result["result"].final_answer)
        self.assertEqual(answer_calls, [])
        scope = ground.call_args.args[2]
        self.assertEqual(scope.program, "DSBA")
        self.assertEqual(scope.years, (2,))

    def test_partial_requirement_and_credit_filter_uses_fallback_once(self):
        _, grounded_result = self._patch_list_fallback()
        model_calls = []

        def structured_model(prompt):
            model_calls.append(prompt)
            return "SELECT course_id FROM courses"

        with patch(
            "rag.qa.ground_course_list", return_value=grounded_result
        ) as ground, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("partial list must bypass planner"),
        ):
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชาบังคับ 3 หน่วยกิตอะไรบ้าง",
                structured_model_callable=structured_model,
            )

        self.assertEqual(len(model_calls), 1)
        ground.assert_called_once()
        self.assertEqual(result["result"].status, "answer")

    def test_partial_sum_credits_uses_selected_targets_with_existing_credit_executor(self):
        model_calls = []
        selected_targets = (
            {
                "program": "IT",
                "course_id": 101,
                "course_code": "06016414",
                "catalog_id": 4,
            },
        )
        grounded_result = GroundedCourseListResult(
            status="complete",
            selected_targets=selected_targets,
            records=(self.fallback_course_record(),),
        )

        def execute_filtered_credit(_db_path, plan):
            request = plan.requests[0]
            self.assertEqual(request.kind, "credit_facts")
            self.assertEqual(request.course_targets, selected_targets)
            self.assertEqual(request.scope.course_targets, selected_targets)
            effective_scope = replace(
                request.scope,
                plans=("no_coop",),
                expand_applicable=(),
                unconstrained=tuple(
                    axis for axis in request.scope.unconstrained if axis != "plan"
                ),
            )
            component = {
                "program": "IT",
                "plan_key": "no_coop",
                "year": 2,
                "semester": 1,
                "course_id": 101,
                "course_code": "06016414",
                "counted_credit_units": 3,
                "provenance": ({"source_page": 11},),
            }
            return EvidenceBundle(
                plan,
                (
                    EvidenceExecutionResult(
                        "credit_facts",
                        "credit_facts",
                        request,
                        effective_scope,
                        "complete",
                        {
                            "status": "ok",
                            "components": (component,),
                            "provenance": component["provenance"],
                        },
                    ),
                ),
            )

        def run_selector(_db_path, _question, _scope, model):
            model("selector prompt")
            return StructuredFallbackResult(
                status="success",
                columns=("course_id",),
                rows=((101,),),
            )

        with patch(
            "rag.qa.run_structured_fallback", side_effect=run_selector
        ) as fallback, patch(
            "rag.qa.ground_course_list", return_value=grounded_result
        ), patch(
            "rag.qa.execute_evidence_plan", side_effect=execute_filtered_credit
        ) as executor, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("filtered credit must not use broad planner"),
        ):
            result = ask(
                DB_PATH,
                "IT ปี 2 เทอม 1 มีวิชาบังคับรวมกี่หน่วยกิต",
                structured_model_callable=lambda prompt: model_calls.append(prompt)
                or "SELECT course_id FROM courses",
                answer_model_callable=lambda prompt: self.fail(
                    "filtered credit fallback must not polish"
                ),
            )

        fallback.assert_called_once()
        self.assertEqual(len(model_calls), 1)
        executor.assert_called_once()
        claim = result["result"].claims[0]
        self.assertEqual(claim.operation, "sum_credits")
        self.assertEqual(claim.status, "complete")
        self.assertEqual(claim.value, 3)

    def test_partial_sum_credits_valid_empty_does_not_execute_unfiltered_credit(self):
        with patch(
            "rag.qa.run_structured_fallback",
            return_value=StructuredFallbackResult(
                status="success",
                columns=("course_id",),
                rows=(),
            ),
        ) as fallback, patch(
            "rag.qa.ground_course_list",
            return_value=GroundedCourseListResult(status="valid_empty"),
        ), patch(
            "rag.qa.execute_evidence_plan",
            side_effect=AssertionError("empty selector must not execute credit facts"),
        ) as executor:
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชาบังคับรวมกี่หน่วยกิต",
                structured_model_callable=lambda prompt: "SELECT course_id FROM courses",
            )

        fallback.assert_called_once()
        executor.assert_not_called()
        claim = result["result"].claims[0]
        self.assertEqual(claim.operation, "sum_credits")
        self.assertEqual(claim.status, "valid_empty")
        self.assertEqual(claim.value, 0)

    def test_partial_sum_credits_grounding_failure_does_not_plan_broad_scope(self):
        with patch(
            "rag.qa.run_structured_fallback",
            return_value=StructuredFallbackResult(status="error"),
        ) as fallback, patch(
            "rag.qa.ground_course_list",
            return_value=GroundedCourseListResult(status="insufficient_evidence"),
        ), patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("failed filtered credit must not plan broadly"),
        ) as planner:
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชาบังคับรวมกี่หน่วยกิต",
                structured_model_callable=lambda prompt: "unused",
            )

        fallback.assert_called_once()
        planner.assert_not_called()
        self.assertEqual(result["result"].status, "insufficient_evidence")
        self.assertEqual(result["result"].claims[0].operation, "sum_credits")

    def test_partial_count_uses_course_list_fallback_and_count_aggregate(self):
        grounded_result = GroundedCourseListResult(
            status="complete",
            records=(self.fallback_course_record(),),
        )
        model_calls = []

        with patch(
            "rag.qa.ground_course_list", return_value=grounded_result
        ) as ground, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("partial count must bypass planner"),
        ):
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชาบังคับกี่วิชา",
                structured_model_callable=lambda prompt: model_calls.append(prompt)
                or "SELECT course_id FROM courses",
                answer_model_callable=lambda prompt: self.fail(
                    "count fallback must not polish"
                ),
            )

        self.assertEqual(len(model_calls), 1)
        ground.assert_called_once()
        claim = result["result"].claims[0]
        self.assertEqual(claim.operation, "count")
        self.assertEqual(claim.status, "complete")
        self.assertEqual(claim.value, 1)

    def test_partial_existence_uses_course_list_fallback_and_exists_aggregate(self):
        grounded_result = GroundedCourseListResult(
            status="complete",
            records=(self.fallback_course_record(),),
        )
        model_calls = []

        with patch(
            "rag.qa.ground_course_list", return_value=grounded_result
        ) as ground, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("partial existence must bypass planner"),
        ):
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชาบังคับไหม",
                structured_model_callable=lambda prompt: model_calls.append(prompt)
                or "SELECT course_id FROM courses",
                answer_model_callable=lambda prompt: self.fail(
                    "existence fallback must not polish"
                ),
            )

        self.assertEqual(len(model_calls), 1)
        ground.assert_called_once()
        claim = result["result"].claims[0]
        self.assertEqual(claim.operation, "existence")
        self.assertEqual(claim.status, "complete")
        self.assertTrue(claim.value)

    def test_partial_count_and_existence_preserve_valid_empty_values(self):
        for question, operation, expected in (
            ("DSBA ปี 2 มีวิชาบังคับกี่วิชา", "count", 0),
            ("DSBA ปี 2 มีวิชาบังคับไหม", "existence", False),
        ):
            with self.subTest(question=question):
                with patch(
                    "rag.qa.run_structured_fallback",
                    return_value=StructuredFallbackResult(
                        status="success",
                        sql="SELECT course_id FROM courses LIMIT 100",
                        columns=("course_id",),
                        rows=(),
                    ),
                ) as fallback, patch(
                    "rag.qa.ground_course_list",
                    return_value=GroundedCourseListResult(status="valid_empty"),
                ) as ground, patch(
                    "rag.qa.plan_evidence",
                    side_effect=AssertionError("valid empty must bypass planner"),
                ):
                    result = ask(
                        DB_PATH,
                        question,
                        structured_model_callable=lambda prompt: "unused",
                    )

                claim = result["result"].claims[0]
                self.assertEqual(claim.operation, operation)
                self.assertEqual(claim.status, "valid_empty")
                self.assertEqual(claim.value, expected)
                fallback.assert_called_once()
                ground.assert_called_once()

    def test_incomplete_structured_queries_without_model_fail_closed_before_planner(self):
        questions = (
            "DSBA ปี 2 มีวิชา Gen Ed อะไรบ้าง",
            "DSBA ปี 2 มีวิชาบังคับกี่วิชา",
            "DSBA ปี 2 มีวิชาบังคับไหม",
        )
        for question in questions:
            with self.subTest(question=question):
                with patch(
                    "rag.qa.plan_evidence",
                    side_effect=AssertionError(
                        "incomplete structured query must not reach planner"
                    ),
                ) as planner:
                    result = ask(DB_PATH, question)

                self.assertEqual(result["result"].status, "insufficient_evidence")
                planner.assert_not_called()

    def test_unrecognized_structured_queries_without_model_fail_closed_before_planner(self):
        for question in (
            "DSBA 06026212 สามารถลงได้ช่วงไหนบ้าง",
            "DSBA 06026212 กี่หน่วย",
        ):
            with self.subTest(question=question):
                with patch(
                    "rag.qa.plan_evidence",
                    side_effect=AssertionError(
                        "unrecognized structured query must not reach planner"
                    ),
                ) as planner:
                    result = ask(DB_PATH, question)

                self.assertEqual(result["result"].status, "insufficient_evidence")
                planner.assert_not_called()

    def test_incomplete_fallback_scope_failure_fails_closed_before_planner(self):
        with patch(
            "rag.qa._fallback_scope", return_value=None
        ), patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("fallback preparation failure must not plan"),
        ) as planner:
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชาบังคับกี่วิชา",
                structured_model_callable=lambda prompt: "unused",
            )

        self.assertEqual(result["result"].status, "insufficient_evidence")
        planner.assert_not_called()

    def test_partial_count_grounding_failure_fails_closed(self):
        with patch(
            "rag.qa.run_structured_fallback",
            return_value=StructuredFallbackResult(
                status="error",
                error_category="relation_guard",
                error="disallowed relation",
            ),
        ) as fallback, patch(
            "rag.qa.ground_course_list",
            return_value=GroundedCourseListResult(status="insufficient_evidence"),
        ) as ground, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("failed count must not plan broadly"),
        ):
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชาบังคับกี่วิชา",
                structured_model_callable=lambda prompt: "unused",
            )

        self.assertEqual(result["result"].status, "insufficient_evidence")
        self.assertEqual(result["result"].claims[0].operation, "count")
        self.assertNotIn("SELECT", result["result"].final_answer)
        fallback.assert_called_once()
        ground.assert_called_once()

    def test_complete_count_and_existence_stay_deterministic(self):
        for question in (
            "IT ปี 3 ต้องเรียนกี่วิชา",
            "IT ปี 2 เทอม 1 มีวิชา 06016414 ไหม",
        ):
            with self.subTest(question=question):
                with patch("rag.qa.run_structured_fallback") as fallback:
                    result = ask(
                        DB_PATH,
                        question,
                        structured_model_callable=lambda prompt: self.fail(
                            "complete structured query must not fallback"
                        ),
                    )

                self.assertIsInstance(result["result"], GroundedAnswerResult)
                fallback.assert_not_called()

    def test_colloquial_single_course_credit_uses_one_fallback(self):
        model_calls = []
        grounded_result = GroundedCourseCreditResult(
            status="complete",
            records=(
                {
                    "program": "DSBA",
                    "course_id": 205,
                    "course_code": "06026212",
                    "credit_units": 3,
                    "credits": "3(3-0-6)",
                    "provenance": ({"source_page": 37},),
                },
            ),
            credit_units=3,
            credits="3(3-0-6)",
            provenance=({"source_page": 37},),
        )

        def structured_model(prompt):
            model_calls.append(prompt)
            return "SELECT DISTINCT course_id AS course_id FROM courses"

        with patch(
            "rag.qa.ground_course_credit", return_value=grounded_result
        ) as ground, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("credit fallback must bypass planner"),
        ):
            result = ask(
                DB_PATH,
                "DSBA 06026212 กี่หน่วย",
                structured_model_callable=structured_model,
            )

        self.assertEqual(len(model_calls), 1)
        ground.assert_called_once()
        self.assertEqual(result["result"].status, "answer")
        self.assertEqual(result["result"].claims[0].operation, "sum_credits")
        self.assertEqual(result["result"].claims[0].value, 3)
        self.assertIn("3", result["result"].final_answer)

    def test_complete_single_course_credit_stays_on_deterministic_path(self):
        model = lambda prompt: self.fail("complete credit must not call SQL model")
        with patch("rag.qa.run_structured_fallback") as fallback:
            result = ask(
                DB_PATH,
                "DSBA วิชา 06026212 มีกี่หน่วยกิต",
                structured_model_callable=model,
            )

        self.assertEqual(result["result"].status, "answer")
        self.assertTrue(
            any(
                claim.operation == "sum_credits" and claim.value == 3
                for claim in result["result"].claims
            )
        )
        fallback.assert_not_called()

    def test_course_credit_fallback_uses_canonical_credit_and_provenance(self):
        model_calls = []

        def structured_model(prompt):
            model_calls.append(prompt)
            return (
                "SELECT DISTINCT p.course_id AS course_id "
                "FROM v_plan_courses AS p "
                "WHERE p.program = 'DSBA' AND p.course_code = '06026212'"
            )

        result = ask(
            DB_PATH,
            "DSBA 06026212 กี่หน่วย",
            structured_model_callable=structured_model,
        )["result"]

        self.assertEqual(len(model_calls), 1)
        self.assertEqual(result.status, "answer")
        claim = result.claims[0]
        self.assertEqual(claim.operation, "sum_credits")
        self.assertEqual(claim.status, "complete")
        self.assertEqual(claim.value, 3)
        self.assertTrue(claim.provenance)
        self.assertNotIn("SELECT", result.final_answer)

    def test_course_credit_fallback_uses_exact_code_scope_and_no_polish(self):
        fallback_result = StructuredFallbackResult(
            status="success",
            sql="SELECT DISTINCT course_id AS course_id FROM courses",
            columns=("course_id",),
            rows=((205,),),
        )
        grounded_result = GroundedCourseCreditResult(
            status="complete",
            records=(
                {
                    "program": "DSBA",
                    "course_id": 205,
                    "course_code": "06026212",
                    "credit_units": 3,
                    "credits": "3(3-0-6)",
                    "provenance": ({"source_page": 37},),
                },
            ),
            credit_units=3,
            credits="3(3-0-6)",
            provenance=({"source_page": 37},),
        )

        def forbidden_answer_model(prompt):
            self.fail("course-credit fallback must not invoke answer polishing")

        with patch(
            "rag.qa.run_structured_fallback", return_value=fallback_result
        ) as fallback, patch(
            "rag.qa.ground_course_credit", return_value=grounded_result
        ) as ground, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("credit fallback must bypass planner"),
        ):
            result = ask(
                DB_PATH,
                "DSBA 06026212 กี่หน่วย",
                structured_model_callable=lambda prompt: "unused",
                answer_model_callable=forbidden_answer_model,
            )

        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.kwargs["selector_mode"], "course_credit")
        scope = fallback.call_args.args[2]
        self.assertEqual(scope.program, "DSBA")
        self.assertEqual(scope.plans, ())
        self.assertEqual(scope.course_ids, ())
        self.assertEqual(scope.course_codes, ("06026212",))
        ground.assert_called_once()
        self.assertEqual(result["result"].claims[0].value, 3)

    def test_semester_total_does_not_use_course_credit_fallback(self):
        with patch("rag.qa.run_structured_fallback") as fallback:
            result = ask(
                DB_PATH,
                "DSBA ปี 2 เทอม 2 ลงทะเบียนรวมกี่หน่วยกิต",
                structured_model_callable=lambda prompt: self.fail(
                    "semester totals must not use course-credit fallback"
                ),
            )

        self.assertEqual(result["result"].status, "answer")
        fallback.assert_not_called()

    def test_course_credit_fallback_failure_fails_closed(self):
        failed = GroundedCourseCreditResult(status="insufficient_evidence")
        with patch(
            "rag.qa.run_structured_fallback",
            return_value=StructuredFallbackResult(
                status="error",
                error_category="relation_guard",
                error="disallowed relation",
            ),
        ) as fallback, patch(
            "rag.qa.ground_course_credit", return_value=failed
        ) as ground, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("credit fallback failure must not plan"),
        ):
            result = ask(
                DB_PATH,
                "DSBA 06026212 กี่หน่วย",
                structured_model_callable=lambda prompt: "unused",
            )

        self.assertEqual(result["result"].status, "insufficient_evidence")
        self.assertEqual(result["result"].claims[0].status, "insufficient_evidence")
        fallback.assert_called_once()
        ground.assert_called_once()

    def test_complete_list_stays_on_deterministic_path_without_model_call(self):
        model = lambda prompt: self.fail("complete query must not call SQL model")
        with patch("rag.qa.run_structured_fallback") as fallback:
            result = ask(
                DB_PATH,
                "DSBA ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
                structured_model_callable=model,
            )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        fallback.assert_not_called()

    def test_complete_placement_stays_on_deterministic_path_without_model_call(self):
        model = lambda prompt: self.fail("complete placement must not call SQL model")
        with patch("rag.qa.run_structured_fallback") as fallback:
            result = ask(
                DB_PATH,
                "DSBA วิชา 06026212 เรียนปีไหน เทอมไหน",
                structured_model_callable=model,
            )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertTrue(
            any(claim.operation == "placement" for claim in result["result"].claims)
        )
        fallback.assert_not_called()

    def test_unrecognized_placement_uses_one_fallback_and_canonical_claim(self):
        structured_calls = []
        answer_calls = []
        grounded_result = GroundedPlacementResult(
            status="complete",
            records=(self.fallback_placement_record(),),
        )

        def structured_model(prompt):
            structured_calls.append(prompt)
            return "SELECT 401 AS placement_id"

        def forbidden_answer_model(prompt):
            answer_calls.append(prompt)
            self.fail("SQL placement fallback must not invoke answer polishing")

        with patch(
            "rag.qa.ground_placement", return_value=grounded_result
        ) as ground, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("placement fallback must bypass planner"),
        ):
            result = ask(
                DB_PATH,
                "DSBA 06026212 สามารถลงได้ช่วงไหนบ้าง",
                structured_model_callable=structured_model,
                answer_model_callable=forbidden_answer_model,
            )

        self.assertEqual(len(structured_calls), 1)
        ground.assert_called_once()
        self.assertEqual(result["result"].status, "answer")
        self.assertEqual(result["result"].claims[0].operation, "placement")
        self.assertEqual(result["result"].claims[0].status, "complete")
        self.assertEqual(
            result["result"].claims[0].evidence[0]["year_number"],
            2,
        )
        self.assertIn("06026212", result["result"].final_answer)
        self.assertEqual(answer_calls, [])
        self.assertEqual(ground.call_args.args[2].program, "DSBA")
        self.assertEqual(ground.call_args.args[2].course_codes, ("06026212",))

    def test_colloquial_long_rien_placement_uses_one_fallback(self):
        grounded_result = GroundedPlacementResult(
            status="complete",
            records=(self.fallback_placement_record(),),
        )
        for question in (
            "DSBA 06026212 ลงเรียนช่วงไหนบ้าง",
            "DSBA 06026212 ลงเรียนช่วงไหน",
        ):
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH)
                completeness = _classify_structured_parse_completeness(
                    spec, resolution
                )
                self.assertEqual(spec.operations, ())
                self.assertEqual(
                    completeness.classification, "unrecognized_structured"
                )
                self.assertTrue(
                    _is_placement_fallback_candidate(spec, completeness)
                )
                structured_calls = []

                def structured_model(prompt):
                    structured_calls.append(prompt)
                    return "SELECT 401 AS placement_id"

                def forbidden_answer_model(prompt):
                    self.fail(
                        "SQL placement fallback must not invoke answer polishing"
                    )

                with patch(
                    "rag.qa.ground_placement", return_value=grounded_result
                ) as ground, patch(
                    "rag.qa.plan_evidence",
                    side_effect=AssertionError(
                        "placement fallback must bypass planner"
                    ),
                ):
                    result = ask(
                        DB_PATH,
                        question,
                        structured_model_callable=structured_model,
                        answer_model_callable=forbidden_answer_model,
                    )

                self.assertEqual(len(structured_calls), 1)
                ground.assert_called_once()
                self.assertEqual(result["result"].status, "answer")
                self.assertEqual(
                    result["result"].claims[0].operation, "placement"
                )
                self.assertEqual(result["result"].claims[0].status, "complete")
                self.assertEqual(ground.call_args.args[2].program, "DSBA")
                self.assertEqual(
                    ground.call_args.args[2].course_codes, ("06026212",)
                )

    def test_long_rien_ton_nai_stays_on_deterministic_placement(self):
        model = lambda prompt: self.fail(
            "complete placement must not call SQL model"
        )
        with patch("rag.qa.run_structured_fallback") as fallback:
            result = ask(
                DB_PATH,
                "DSBA 06026212 ลงเรียนตอนไหน",
                structured_model_callable=model,
            )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].status, "answer")
        self.assertTrue(
            all(
                claim.operation == "placement"
                for claim in result["result"].claims
            )
        )
        self.assertTrue(
            all(
                claim.status == "complete"
                for claim in result["result"].claims
            )
        )
        fallback.assert_not_called()

    def test_non_placement_wording_does_not_take_placement_seam(self):
        for question in (
            "DSBA 06026212 กี่หน่วย",
            "DSBA ปี 2 มีวิชาบังคับกี่วิชา",
        ):
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH)
                completeness = _classify_structured_parse_completeness(
                    spec, resolution
                )
                self.assertFalse(
                    _is_placement_fallback_candidate(spec, completeness)
                )

    def test_program_free_long_rien_placement_never_reaches_fallback(self):
        cases = (
            ("06026212 เรียนปีไหน", "clarify_program"),
            ("06026212 ลงเรียนช่วงไหนบ้าง", "insufficient_evidence"),
        )
        for question, expected_status in cases:
            with self.subTest(question=question):
                with patch("rag.qa.run_structured_fallback") as fallback:
                    result = ask(
                        DB_PATH,
                        question,
                        structured_model_callable=lambda prompt: self.fail(
                            "program-free placement must not use SQL fallback"
                        ),
                    )

                fallback.assert_not_called()
                outcome = result["result"]
                status = (
                    outcome["status"]
                    if isinstance(outcome, dict)
                    else outcome.status
                )
                self.assertEqual(status, expected_status)

    def test_unconstrained_placement_scope_preserves_code_across_catalogs(self):
        question = "DSBA 06026212 สามารถลงได้ช่วงไหนบ้าง"
        spec = parse_query_spec(question)
        resolution = resolve_query_spec(spec, DB_PATH)
        completeness = _classify_structured_parse_completeness(
            spec, resolution
        )
        scope = _fallback_scope(
            completeness,
            resolution,
            placement_code_identity=True,
        )

        self.assertIsNotNone(scope)
        self.assertEqual(scope.program, "DSBA")
        self.assertEqual(scope.plans, ())
        self.assertEqual(scope.course_ids, ())
        self.assertEqual(scope.course_codes, ("06026212",))

        model_calls = []

        def structured_model(prompt):
            model_calls.append(prompt)
            return (
                "SELECT DISTINCT p.placement_id AS placement_id "
                "FROM v_plan_courses AS p "
                "WHERE p.course_code = '06026212'"
            )

        result = ask(
            DB_PATH,
            question,
            structured_model_callable=structured_model,
        )["result"]
        claim = result.claims[0]

        self.assertEqual(len(model_calls), 1)
        self.assertEqual(claim.status, "complete")
        self.assertEqual(
            [(record["placement_id"], record["plan_key"]) for record in claim.evidence],
            [(211, "coop"), (300, "no_coop")],
        )

    def test_explicit_placement_plan_keeps_only_that_plan(self):
        cases = (
            ("DSBA แบบสหกิจ 06026212 สามารถลงได้ช่วงไหนบ้าง", 211, "coop"),
            ("DSBA แบบไม่สหกิจ 06026212 สามารถลงได้ช่วงไหนบ้าง", 300, "no_coop"),
        )
        for question, placement_id, plan_key in cases:
            with self.subTest(question=question):
                result = ask(
                    DB_PATH,
                    question,
                    structured_model_callable=lambda prompt, placement_id=placement_id: (
                        f"SELECT {placement_id} AS placement_id"
                    ),
                )["result"]
                claim = result.claims[0]

                self.assertEqual(claim.status, "complete")
                self.assertEqual(
                    [(record["placement_id"], record["plan_key"]) for record in claim.evidence],
                    [(placement_id, plan_key)],
                )

    def test_placement_fallback_rejects_selector_from_another_program(self):
        result = ask(
            DB_PATH,
            "DSBA 06026212 สามารถลงได้ช่วงไหนบ้าง",
            structured_model_callable=lambda prompt: (
                "SELECT DISTINCT p.placement_id AS placement_id "
                "FROM v_plan_courses AS p WHERE p.program = 'IT'"
            ),
        )["result"]

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.claims[0].status, "insufficient_evidence")

    def test_placement_fallback_failure_returns_insufficient_evidence(self):
        fallback_result = StructuredFallbackResult(
            status="error",
            error_category="relation_guard",
            error="disallowed relation",
        )
        failed = GroundedPlacementResult(status="insufficient_evidence")
        with patch(
            "rag.qa.run_structured_fallback", return_value=fallback_result
        ) as fallback, patch(
            "rag.qa.ground_placement", return_value=failed
        ), patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("placement fallback failure must not plan"),
        ):
            result = ask(
                DB_PATH,
                "DSBA 06026212 สามารถลงได้ช่วงไหนบ้าง",
                structured_model_callable=lambda prompt: "unused",
            )

        self.assertEqual(result["result"].status, "insufficient_evidence")
        self.assertEqual(result["result"].claims[0].status, "insufficient_evidence")
        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.kwargs["selector_mode"], "placement")

    def test_placement_fallback_valid_empty_uses_existing_empty_status(self):
        fallback_result = StructuredFallbackResult(
            status="success",
            sql="SELECT placement_id FROM v_plan_courses",
            columns=("placement_id",),
            rows=(),
        )
        empty = GroundedPlacementResult(status="valid_empty")
        with patch(
            "rag.qa.run_structured_fallback", return_value=fallback_result
        ), patch(
            "rag.qa.ground_placement", return_value=empty
        ):
            result = ask(
                DB_PATH,
                "DSBA 06026212 สามารถลงได้ช่วงไหนบ้าง",
                structured_model_callable=lambda prompt: "unused",
            )

        self.assertEqual(result["result"].status, "valid_empty")
        self.assertEqual(result["result"].claims[0].status, "valid_empty")

    def test_semantic_and_unsupported_queries_do_not_use_sql_fallback(self):
        model = lambda prompt: self.fail("non-list query must not call SQL model")
        for question in (
            "DSBA มีวิชาอะไรเกี่ยวกับ database บ้าง",
            "DSBA ปี 2 เรียนยากไหม",
        ):
            with self.subTest(question=question):
                with patch("rag.qa.run_structured_fallback") as fallback:
                    ask(
                        DB_PATH,
                        question,
                        structured_model_callable=model,
                    )
                fallback.assert_not_called()

    def test_topic_with_unresolved_filter_fails_closed_before_sql_or_planner(self):
        question = "IT วิชาบังคับมีวิชาอะไรเกี่ยวกับ database บ้าง"
        spec = parse_query_spec(question)
        resolution = resolve_query_spec(spec, DB_PATH)
        completeness = _classify_structured_parse_completeness(spec, resolution)
        self.assertEqual(completeness.classification, "partial")
        self.assertEqual(completeness.missing_filters, ("requirement_type",))

        with patch("rag.qa.run_structured_fallback") as fallback, patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError(
                "topic-bearing partial query must not reach planner"
            ),
        ) as planner:
            result = ask(
                DB_PATH,
                question,
                structured_model_callable=lambda prompt: "unused",
            )

        self.assertEqual(result["result"].status, "insufficient_evidence")
        fallback.assert_not_called()
        planner.assert_not_called()

    def test_fallback_grounding_failure_returns_insufficient_evidence(self):
        sql_result, _ = self._patch_list_fallback()
        failed = GroundedCourseListResult(status="insufficient_evidence")
        with patch(
            "rag.qa.run_structured_fallback", return_value=sql_result
        ), patch(
            "rag.qa.ground_course_list", return_value=failed
        ), patch(
            "rag.qa.plan_evidence",
            side_effect=AssertionError("fallback failure must not plan broadly"),
        ):
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชา Gen Ed อะไรบ้าง",
                structured_model_callable=lambda prompt: "SELECT course_id FROM courses",
            )

        self.assertEqual(result["result"].status, "insufficient_evidence")
        self.assertEqual(result["result"].claims[0].status, "insufficient_evidence")

    def test_fallback_valid_empty_uses_existing_empty_status(self):
        sql_result, _ = self._patch_list_fallback()
        empty = GroundedCourseListResult(status="valid_empty")
        with patch(
            "rag.qa.run_structured_fallback", return_value=sql_result
        ), patch(
            "rag.qa.ground_course_list", return_value=empty
        ):
            result = ask(
                DB_PATH,
                "DSBA ปี 2 มีวิชา Gen Ed อะไรบ้าง",
                structured_model_callable=lambda prompt: "SELECT course_id FROM courses",
            )

        self.assertEqual(result["result"].status, "valid_empty")
        self.assertEqual(result["result"].claims[0].status, "valid_empty")

    def test_program_free_prerequisite_with_divergent_candidates_still_clarifies(self):
        for question in (
            "Advanced Database Systems มีวิชาบังคับก่อนคืออะไร",
            "Database System Maintenance and Administration มีวิชาบังคับก่อนคืออะไร",
        ):
            with self.subTest(question=question):
                result = ask(DB_PATH, question)
                self.assertEqual(result["result"]["status"], "clarify_program")
                self.assertEqual(result["result"]["action"], "clarify_program")
                self.assertEqual(result["result"]["blocking_ambiguity"], ("program",))

    def test_program_free_prerequisite_with_unknown_candidate_still_clarifies(self):
        result = ask(DB_PATH, "Design Thinking มีวิชาบังคับก่อนคืออะไร")

        self.assertEqual(result["result"]["status"], "clarify_program")
        self.assertEqual(result["result"]["action"], "clarify_program")

    def test_explicit_unknown_prerequisite_is_insufficient_evidence(self):
        result = ask(DB_PATH, "GENED 90641001 มีวิชาบังคับก่อนคืออะไร")

        self.assertEqual(result["result"].status, "insufficient_evidence")

    def test_program_free_unanimous_explicit_none_prerequisite_is_valid_empty(self):
        result = ask(DB_PATH, "Discrete Mathematics มีวิชาบังคับก่อนคืออะไร")
        typed = result["result"]

        self.assertEqual(typed.status, "valid_empty")
        self.assertIn("ไม่มีวิชาบังคับก่อน", typed.final_answer)
        self.assertTrue(typed.provenance)

    def test_explicit_program_prerequisite_does_not_use_consensus_exception(self):
        result = ask(DB_PATH, "AIT 06046401 มีวิชาบังคับก่อนคืออะไร")

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["result"].status, "answer")
        self.assertIn("CALCULUS 1", result["result"].final_answer)

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

    def test_whole_plan_comparison_uses_typed_aggregate_and_preserves_both_plans(self):
        model = lambda _prompt: self.fail("whole-plan comparison must not use SQL fallback")
        with patch("rag.qa.run_structured_fallback") as fallback:
            result = ask(
                DB_PATH,
                "IT สหกิจกับไม่สหกิจต่างกันยังไง",
                structured_model_callable=model,
            )["result"]

        self.assertEqual(result.status, "answer")
        self.assertEqual(len(result.claims), 1)
        claim = result.claims[0]
        self.assertEqual(claim.operation, "compare")
        self.assertIsInstance(claim.value, PlanComparisonAggregation)
        self.assertEqual(claim.status, "complete")
        self.assertTrue(claim.provenance)
        self.assertIn("สหกิจ", result.final_answer)
        self.assertIn("ไม่สหกิจ", result.final_answer)
        fallback.assert_not_called()

    def test_whole_plan_comparison_incomplete_evidence_fails_closed(self):
        spec = replace(
            self._spec(("compare",)),
            plans=("coop", "no_coop"),
            group_by=("plan",),
            course_codes=(),
            course_name=None,
        )
        bundle = self._whole_plan_bundle(
            {
                "coop": (("00000001", 1, 1, 11),),
                "no_coop": (("00000001", 1, 1, 22),),
            },
            statuses={"no_coop": "insufficient_evidence"},
        )

        claims = _compose_evidence_claims(spec, bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].operation, "compare")
        self.assertEqual(claims[0].status, "insufficient_evidence")
        self.assertIsNone(claims[0].value)
        self.assertEqual(claims[0].provenance, ())

    def _whole_plan_bundle(self, placements_by_plan, *, statuses=None):
        statuses = statuses or {}
        plans = tuple(placements_by_plan)
        bundle_scope = StructuralScope(
            program="IT",
            plans=plans,
            group_by=("plan",),
        )
        requests = []
        results = []
        for plan in plans:
            course_records = []
            placement_records = []
            for index, (code, year, semester, page) in enumerate(
                placements_by_plan[plan], start=1
            ):
                course_records.append(
                    {
                        "program": "IT",
                        "course_code": code,
                        "name_en": f"COURSE {code}",
                        "course_id": index,
                        "provenance": ({"source_page": page},),
                    }
                )
                placement_records.append(
                    {
                        "program": "IT",
                        "course_code": code,
                        "name_en": f"COURSE {code}",
                        "plan_key": plan,
                        "year": year,
                        "semester": semester,
                        "provenance": ({"source_page": page + 100},),
                    }
                )
            scope = StructuralScope(
                program="IT",
                plans=(plan,),
                group_by=("plan",),
            )
            for kind, payload in (
                ("course_set", {"courses": course_records}),
                ("placement_facts", {"courses": placement_records}),
            ):
                request = EvidenceRequest(f"{kind}_{plan}", kind, scope)
                requests.append(request)
                results.append(
                    EvidenceExecutionResult(
                        request_id=request.request_id,
                        kind=kind,
                        planned_request=request,
                        effective_scope=scope,
                        status=statuses.get(plan, "complete"),
                        payload=payload,
                    )
                )
        return EvidenceBundle(
            EvidencePlan(bundle_scope, tuple(requests), group_by=("plan",)),
            tuple(results),
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
