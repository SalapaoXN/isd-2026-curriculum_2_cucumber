"""Focused tests for NLFLEX's bounded query-structure proposal seam."""

import json
import unittest
from pathlib import Path

from rag.intent_compiler import IntentCompilerError, compile_intent_to_query_spec
from rag.intent_interpreter import (
    IntentValidationError,
    interpret_question_intent,
)
from rag.query_spec import parse_query_spec
from rag.qa import ask
from rag.resolution import QueryContext


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


def _proposal(**changes):
    payload = {
        "operations": ["list"],
        "predicate": "has_prerequisite",
        "course_name_span": None,
    }
    payload.update(changes)
    return json.dumps(payload, ensure_ascii=False)


class NLFlexProposalContractTests(unittest.TestCase):
    def test_query_structure_mode_accepts_only_the_closed_proposal_schema(self):
        calls = []
        interpretation = interpret_question_intent(
            "วิชาที่มีวิชาบังคับก่อนมีอะไรบ้าง",
            lambda prompt: calls.append(prompt) or _proposal(),
            proposal_kind="query_structure",
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(interpretation.operations, ("list",))
        self.assertEqual(interpretation.predicate, "has_prerequisite")
        self.assertIsNone(interpretation.course_name_span)

    def test_query_structure_mode_rejects_unknown_or_malformed_fields(self):
        for changes in (
            {"program": "IT"},
            {"predicate": "has_high_quality"},
            {"operations": "list"},
            {"judgement_dimension": "difficulty"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(IntentValidationError):
                    interpret_question_intent(
                        "IT มีวิชาอะไรบ้างที่มีวิชาบังคับก่อน",
                        lambda _prompt, changes=changes: _proposal(**changes),
                        proposal_kind="query_structure",
                    )


class NLFlexPublicPathTests(unittest.TestCase):
    def test_deterministic_complete_course_list_makes_no_interpreter_call(self):
        calls = []
        response = ask(
            DB_PATH,
            "IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
            intent_model_callable=lambda prompt: calls.append(prompt) or _proposal(),
            context=QueryContext(program="IT", catalog_key="it-2565"),
        )["result"]

        status = response.get("status") if isinstance(response, dict) else response.status
        self.assertEqual(status, "answer")
        self.assertEqual(calls, [])

    def test_unscoped_prerequisite_collection_clarifies_without_interpretation(self):
        calls = []
        response = ask(
            DB_PATH,
            "วิชาที่มีวิชาบังคับก่อนมีอะไรบ้าง",
            intent_model_callable=lambda prompt: calls.append(prompt) or _proposal(),
        )["result"]

        status = response.get("status") if isinstance(response, dict) else response.status
        self.assertEqual(status, "clarify_program")
        self.assertEqual(calls, [])

    def test_scoped_prerequisite_collection_fails_closed_with_unknown_candidates(self):
        calls = []
        response = ask(
            DB_PATH,
            "วิชาที่มีวิชาบังคับก่อนมีอะไรบ้าง",
            intent_model_callable=lambda prompt: calls.append(prompt) or _proposal(),
            context=QueryContext(
                program="IT",
                catalog_key="it-2565",
                years=(3,),
                semesters=(1,),
            ),
        )["result"]

        self.assertEqual(response.status, "insufficient_evidence")
        self.assertFalse(response.provenance)
        self.assertTrue(
            all(claim.status == "insufficient_evidence" and claim.value is None
                for claim in response.claims)
        )
        self.assertEqual(calls, [])

    def test_surface_complete_prerequisite_list_fails_closed_with_unknown_candidates(self):
        calls = []
        response = ask(
            DB_PATH,
            "IT ปี 3 เทอม 1 มีวิชาอะไรบ้างที่มีวิชาบังคับก่อน",
            intent_model_callable=lambda prompt: calls.append(prompt) or _proposal(),
            context=QueryContext(program="IT", catalog_key="it-2565"),
        )["result"]

        self.assertEqual(response.status, "insufficient_evidence")
        self.assertFalse(response.provenance)
        self.assertTrue(
            all(claim.status == "insufficient_evidence" and claim.value is None
                for claim in response.claims)
        )
        self.assertEqual(calls, [])

    def test_flexible_exact_course_placement_uses_literal_title_then_canonical_evidence(self):
        calls = []
        response = ask(
            DB_PATH,
            "ปีไหนเรียน Calculus 2 ใน DSBA",
            intent_model_callable=lambda prompt: calls.append(prompt)
            or _proposal(
                operations=["placement"],
                predicate=None,
                course_name_span="Calculus 2",
            ),
            synthesize_answer=True,
            answer_model_callable=lambda _prompt: self.fail(
                "interpreted factual turns must not synthesize"
            ),
        )["result"]

        self.assertEqual(response["status"], "clarify_catalog")
        self.assertEqual(response["action"], "clarify_catalog")
        self.assertEqual(response["catalog_keys"], ["dsba-2560", "dsba-2565"])
        self.assertEqual(calls, [])

    def test_invalid_proposal_and_provider_failure_fail_closed(self):
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        for model, expected_calls in (
            (lambda _prompt: '{"operations":["list"]}', 1),
            (lambda _prompt: (_ for _ in ()).throw(RuntimeError("provider")), 1),
        ):
            calls = []

            def counted(prompt, model=model):
                calls.append(prompt)
                return model(prompt)

            result = ask(
                DB_PATH,
                "Calculus 2 หนักกี่เครดิต",
                context=context,
                intent_model_callable=counted,
            )["result"]
            status = result.get("status") if isinstance(result, dict) else result.status
            self.assertEqual(status, "insufficient_evidence")
            self.assertEqual(len(calls), expected_calls)

    def test_student_exact_course_phrasing_matrix_uses_canonical_answers(self):
        cases = (
            ("ใน DSBA Calculus 2 รหัสอะไร", "identity", "answer", 1),
            ("Calculus 2 ของ DSBA รหัสวิชาอะไร", "identity", "answer", 0),
            ("ปีไหนเรียน Calculus 2 ใน DSBA", "placement", "answer", 0),
            ("Calculus 2 ใน DSBA กี่หน่วยกิต", "sum_credits", "answer", 0),
            ("Calculus 2 ต้องผ่านอะไรบ้าง", "prerequisite", "clarify_program", 0),
        )
        for question, operation, expected_status, expected_calls in cases:
            with self.subTest(question=question):
                calls = []
                response = ask(
                    DB_PATH,
                    question,
                    conversation_context=(
                        QueryContext(program="DSBA", catalog_key="dsba-2565")
                        if parse_query_spec(question).program == "DSBA"
                        else None
                    ),
                    intent_model_callable=lambda prompt, operation=operation, calls=calls: (
                        calls.append(prompt)
                        or _proposal(
                            operations=[operation],
                            predicate=None,
                            course_name_span=(
                                None
                                if parse_query_spec(question).course_name
                                else "Calculus 2"
                            ),
                        )
                    ),
                )["result"]
                status = response.get("status") if isinstance(response, dict) else response.status
                self.assertEqual(
                    status,
                    expected_status,
                    msg=f"model calls: {len(calls)}; spec={parse_query_spec(question)!r}; result={response!r}",
                )
                self.assertEqual(len(calls), expected_calls)
                if expected_status == "answer":
                    self.assertTrue(response.claims)
                    self.assertTrue(any(claim.provenance for claim in response.claims))

    def test_proposal_cannot_replace_explicit_deterministic_operation(self):
        base = parse_query_spec("IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง")
        interpretation = interpret_question_intent(
            "IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง",
            lambda _prompt: _proposal(
                operations=["placement"], predicate=None
            ),
            proposal_kind="query_structure",
        )
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(base, interpretation)

    def test_prerequisite_collection_compiles_to_existing_operations(self):
        base = parse_query_spec("IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง")
        interpretation = interpret_question_intent(
            "IT ปี 3 เทอม 1 มีวิชาอะไรบ้างที่มีวิชาบังคับก่อน",
            lambda _prompt: _proposal(),
            proposal_kind="query_structure",
        )

        compiled = compile_intent_to_query_spec(base, interpretation)

        self.assertEqual(compiled.program, "IT")
        self.assertEqual(compiled.years, (3,))
        self.assertEqual(compiled.semesters, (1,))
        self.assertEqual(compiled.operations, ("list", "prerequisite"))

    def test_predicate_rejects_non_list_operation_combinations(self):
        with self.assertRaises(IntentValidationError):
            interpret_question_intent(
                "IT มีกี่วิชาที่มีวิชาบังคับก่อน",
                lambda _prompt: _proposal(operations=["count"]),
                proposal_kind="query_structure",
            )

    def test_course_name_span_must_be_literal_bounded_and_unparsed(self):
        question = "DSBA ช่วยดู Calculus 2 ตอนเรียนปีไหน"
        parsed = interpret_question_intent(
            question,
            lambda _prompt: _proposal(
                operations=["placement"],
                predicate=None,
                course_name_span="Calculus 2",
            ),
            proposal_kind="query_structure",
        )
        base = parse_query_spec(question)
        compiled = compile_intent_to_query_spec(base, parsed)
        self.assertEqual(compiled.program, "DSBA")
        self.assertEqual(compiled.course_name, "Calculus 2")
        self.assertEqual(compiled.operations, ("placement",))

        for question_text, span in (
            (question, "Calculus 3"),
            (question, "ช่วยดู Calculus 2"),
            ("DSBA วิชา Calculus 2 เรียนปีไหน", "Calculus 2"),
        ):
            with self.subTest(question=question_text, span=span):
                with self.assertRaises(IntentValidationError):
                    interpret_question_intent(
                        question_text,
                        lambda _prompt, span=span: _proposal(
                            operations=["placement"],
                            predicate=None,
                            course_name_span=span,
                        ),
                        proposal_kind="query_structure",
                    )


class NL2PreGuardRecoveryTests(unittest.TestCase):
    """NL-2: pre-guard bounded exact-course language recovery.

    The recovery seam only fires for requests that would otherwise fail
    closed for lack of an exact-course target, and only the literal title
    may come from the model. Every other field stays deterministic.
    """

    maxDiff = None

    def _ask(self, question, proposal, context=None, model=None):
        calls = []

        def counted(prompt):
            calls.append(prompt)
            if model is not None:
                return model(prompt)
            return json.dumps(
                {
                    "operations": proposal[0],
                    "predicate": proposal[1],
                    "course_name_span": proposal[2],
                },
                ensure_ascii=False,
            )

        result = ask(
            DB_PATH,
            question,
            context=context,
            intent_model_callable=counted,
        )["result"]
        return result, calls

    @staticmethod
    def _status(result):
        return result.get("status") if isinstance(result, dict) else result.status

    def test_b_b6_credit_recovers_title_keeping_deterministic_operation(self):
        """B6: Calculus 2 หนักกี่เครดิต answers canonical course credits."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "Calculus 2 หนักกี่เครดิต",
            (["sum_credits"], None, "Calculus 2"),
            context=context,
        )
        status = self._status(result)
        self.assertEqual(status, "answer")
        self.assertTrue(result.provenance)
        self.assertEqual(len(calls), 1)
        self.assertIn("3 หน่วยกิต", result.final_answer)
        code_control = ask(
            DB_PATH,
            "06046401 กี่หน่วยกิต",
            context=context,
        )["result"]
        self.assertEqual(self._status(code_control), "answer")
        self.assertIn("3 หน่วยกิต", code_control.final_answer)

    def test_d_supported_exact_course_wording_keeps_accepted_behavior(self):
        """DSBA multi-edition wording without catalog stays safely closed."""
        result, calls = self._ask(
            "ปีไหนเรียน Calculus 2 ใน DSBA",
            (["placement"], None, "Calculus 2"),
        )
        status = self._status(result)
        self.assertEqual(status, "clarify_catalog")
        self.assertEqual(calls, [])
        self.assertEqual(result["catalog_keys"], ["dsba-2560", "dsba-2565"])

    def test_e_operation_conflict_fails_closed_without_execution(self):
        """Deterministic sum_credits must not become a placement answer."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "Calculus 2 หนักกี่เครดิต",
            (["placement"], None, "Calculus 2"),
            context=context,
        )
        status = self._status(result)
        self.assertEqual(status, "insufficient_evidence")
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(result.claims), ())
        self.assertEqual(tuple(result.provenance), ())

    def test_f_invented_title_is_rejected(self):
        """A title that is not in the question must never answer."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "Calculus 2 หนักกี่เครดิต",
            (["sum_credits"], None, "Calculus 3"),
            context=context,
        )
        status = self._status(result)
        self.assertEqual(status, "insufficient_evidence")
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(result.claims), ())
        self.assertEqual(tuple(result.provenance), ())

    def test_g_expanded_title_is_rejected(self):
        """A non-literal expanded span must never answer."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "ช่วยดู Calculus 2 หนักกี่เครดิต",
            (["sum_credits"], None, "ช่วยดู Calculus 2"),
            context=context,
        )
        status = self._status(result)
        self.assertEqual(status, "insufficient_evidence")
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(result.claims), ())
        self.assertEqual(tuple(result.provenance), ())

    def test_h_provider_failure_fails_closed_without_retrieval(self):
        """A raising interpreter must yield a safe failure, not a dump."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        calls = []

        def raising(prompt):
            calls.append(prompt)
            raise RuntimeError("provider unavailable")

        result = ask(
            DB_PATH,
            "Calculus 2 หนักกี่เครดิต",
            context=context,
            intent_model_callable=raising,
        )["result"]
        status = self._status(result)
        self.assertEqual(status, "insufficient_evidence")
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(result.claims), ())
        self.assertEqual(tuple(result.provenance), ())

    def test_i_d1_without_course_anchor_stays_closed(self):
        """วิชานี้ดีไหม has no usable anchor and must not be rescued."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "วิชานี้ดีไหม",
            (["describe"], None, "วิชานี้"),
            context=context,
        )
        status = self._status(result)
        self.assertEqual(status, "insufficient_evidence")
        self.assertEqual(calls, [])

    def test_j_d5_broad_request_stays_closed_without_leakage(self):
        """ขอทุกอย่างในฐานข้อมูล must not become a broad dump."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, _calls = self._ask(
            "ขอทุกอย่างในฐานข้อมูล",
            (["list"], None, None),
            context=context,
        )
        status = self._status(result)
        self.assertEqual(status, "insufficient_evidence")
        self.assertEqual(tuple(result.provenance), ())

    def test_k_b5_nickname_stays_closed_without_alias_inference(self):
        """แคลสองกี่หน่วย must not gain nickname resolution in NL-2."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "แคลสองกี่หน่วย",
            (["sum_credits"], None, "แคลสอง"),
            context=context,
        )
        status = self._status(result)
        self.assertEqual(status, "insufficient_evidence")
        self.assertEqual(calls, [])

    def test_l_b3_policy_eligibility_keeps_safe_behavior(self):
        """GPA honors eligibility must stay policy-routed, never certified."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "ถ้า GPA 3.6 มีสิทธิ์ได้เกียรตินิยมไหม",
            (["sum_credits"], None, "GPA 3.6"),
            context=context,
        )
        status = self._status(result)
        self.assertEqual(status, "unsupported")
        self.assertEqual(calls, [])

    def test_m_ambiguous_program_stays_closed(self):
        """A recovered title must not resolve an ambiguous program scope."""
        result, calls = self._ask(
            "ปีไหนเรียน Calculus 2 ใน DSBA",
            (["placement"], None, "Calculus 2"),
        )
        status = self._status(result)
        self.assertEqual(status, "clarify_catalog")
        self.assertEqual(calls, [])
        self.assertEqual(result["action"], "clarify_catalog")

    def test_n_model_cannot_widen_factual_scope(self):
        """The query-structure schema admits no program/scope fields."""
        for changes in (
            {"years": [3]},
            {"semesters": [1]},
            {"plans": ["coop"]},
            {"course_codes": ["06046401"]},
        ):
            payload = {
                "operations": ["placement"],
                "predicate": None,
                "course_name_span": "Calculus 2",
            }
            payload.update(changes)
            with self.subTest(changes=changes):
                with self.assertRaises(IntentValidationError):
                    interpret_question_intent(
                        "ปีไหนเรียน Calculus 2 ใน DSBA",
                        lambda _prompt, payload=payload: json.dumps(
                            payload, ensure_ascii=False
                        ),
                        proposal_kind="query_structure",
                    )


class NL2BExactCourseRecoveryTests(unittest.TestCase):
    """NL-2B: bounded linguistic-operation authority for exact-course recovery.

    A heuristic deterministic operation on a request with NO exact course
    target is linguistic signal, not factual authority: the bounded
    query-structure interpreter may replace it with a validated reading.
    A parsed exact target keeps the deterministic operation authoritative.
    """

    def _ask(self, question, proposal, context=None, model=None,
             synthesize_answer=False, answer_model_callable=None):
        calls = []

        def counted(prompt):
            calls.append(prompt)
            if model is not None:
                return model(prompt)
            return json.dumps(
                {
                    "operations": proposal[0],
                    "predicate": proposal[1],
                    "course_name_span": proposal[2],
                },
                ensure_ascii=False,
            )

        result = ask(
            DB_PATH,
            question,
            context=context,
            intent_model_callable=counted,
            synthesize_answer=synthesize_answer,
            answer_model_callable=answer_model_callable,
        )["result"]
        return result, calls

    @staticmethod
    def _status(result):
        return result.get("status") if isinstance(result, dict) else result.status

    def test_p1_prerequisite_recovers_title_and_canonical_evidence(self):
        """P1 PRIMARY: describe-base yields to prerequisite + literal title."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "ก่อนเรียน Calculus 2 ต้องเรียนอะไร",
            (["prerequisite"], None, "Calculus 2"),
            context=context,
            synthesize_answer=True,
            answer_model_callable=lambda _prompt: self.fail(
                "interpreted factual turns must not synthesize"
            ),
        )
        self.assertEqual(self._status(result), "answer")
        self.assertTrue(result.provenance)
        self.assertEqual(len(calls), 1)
        control = ask(
            DB_PATH,
            "06046401 ต้องผ่านอะไรบ้าง",
            context=context,
        )["result"]
        self.assertEqual(self._status(control), "answer")
        self.assertIn("06046400", result.final_answer)
        self.assertIn("06046400", control.final_answer)

    def test_p1_placement_proposal_executes_canonical_placement(self):
        """A bounded placement reading still yields canonical facts only."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "ก่อนเรียน Calculus 2 ต้องเรียนอะไร",
            (["placement"], None, "Calculus 2"),
            context=context,
        )
        self.assertEqual(self._status(result), "answer")
        self.assertEqual(len(calls), 1)
        self.assertTrue(
            any(claim.operation == "placement" for claim in result.claims)
        )
        self.assertTrue(result.provenance)

    def test_exact_target_conflict_stays_strict_at_compiler(self):
        """Parsed target + sum_credits vs placement proposal must raise."""
        base = parse_query_spec("06046401 กี่หน่วยกิต")
        self.assertEqual(tuple(base.course_codes), ("06046401",))
        interpretation = interpret_question_intent(
            "06046401 กี่หน่วยกิต",
            lambda _prompt: json.dumps(
                {
                    "operations": ["placement"],
                    "predicate": None,
                    "course_name_span": None,
                }
            ),
            proposal_kind="query_structure",
        )
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(base, interpretation)

    def test_exact_target_request_answers_without_model_override(self):
        """A fully parsed request keeps deterministic semantics, zero calls."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "06046401 กี่หน่วยกิต",
            (["placement"], None, None),
            context=context,
        )
        self.assertEqual(self._status(result), "answer")
        self.assertEqual(calls, [])
        self.assertIn("3 หน่วยกิต", result.final_answer)
        self.assertTrue(result.provenance)

    def test_describe_base_rejects_collection_operation(self):
        """list/count/compare proposals cannot ride the linguistic override."""
        context = QueryContext(program="AIT", catalog_key="ait-2566")
        result, calls = self._ask(
            "ก่อนเรียน Calculus 2 ต้องเรียนอะไร",
            (["count"], None, "Calculus 2"),
            context=context,
        )
        self.assertEqual(self._status(result), "insufficient_evidence")
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(result.claims), ())
        self.assertEqual(tuple(result.provenance), ())


if __name__ == "__main__":
    unittest.main()
