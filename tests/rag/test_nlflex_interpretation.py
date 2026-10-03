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
        )["result"]

        status = response.get("status") if isinstance(response, dict) else response.status
        self.assertEqual(status, "answer")
        self.assertEqual(calls, [])

    def test_unscoped_prerequisite_collection_interprets_then_clarifies(self):
        calls = []
        response = ask(
            DB_PATH,
            "วิชาที่มีวิชาบังคับก่อนมีอะไรบ้าง",
            intent_model_callable=lambda prompt: calls.append(prompt) or _proposal(),
        )["result"]

        status = response.get("status") if isinstance(response, dict) else response.status
        self.assertEqual(status, "clarify_program")
        self.assertEqual(len(calls), 1)

    def test_scoped_interpreted_prerequisite_list_fails_closed_on_unknown_candidate(self):
        calls = []
        response = ask(
            DB_PATH,
            "วิชาที่มีวิชาบังคับก่อนมีอะไรบ้าง",
            intent_model_callable=lambda prompt: calls.append(prompt) or _proposal(),
            context=QueryContext(program="IT", years=(3,), semesters=(1,)),
        )["result"]

        self.assertEqual(response.status, "insufficient_evidence")
        self.assertEqual(len(calls), 1)

    def test_surface_complete_scoped_prerequisite_list_stays_deterministic(self):
        calls = []
        response = ask(
            DB_PATH,
            "IT ปี 3 เทอม 1 มีวิชาอะไรบ้างที่มีวิชาบังคับก่อน",
            intent_model_callable=lambda prompt: calls.append(prompt) or _proposal(),
        )["result"]

        self.assertEqual(response.status, "insufficient_evidence")
        self.assertFalse(any(claim.status == "complete" for claim in response.claims))
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

        self.assertEqual(response.status, "answer")
        self.assertTrue(response.claims)
        self.assertTrue(any(claim.provenance for claim in response.claims))
        self.assertEqual(len(calls), 1)

    def test_invalid_proposal_and_provider_failure_fail_closed(self):
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
                "วิชาที่มีวิชาบังคับก่อนมีอะไรบ้าง",
                intent_model_callable=counted,
            )["result"]
            status = result.get("status") if isinstance(result, dict) else result.status
            self.assertEqual(status, "insufficient_evidence")
            self.assertEqual(len(calls), expected_calls)

    def test_student_exact_course_phrasing_matrix_uses_canonical_answers(self):
        cases = (
            ("ใน DSBA Calculus 2 รหัสอะไร", "identity", "answer", 1),
            ("Calculus 2 ของ DSBA รหัสวิชาอะไร", "identity", "answer", 1),
            ("ปีไหนเรียน Calculus 2 ใน DSBA", "placement", "answer", 1),
            ("Calculus 2 ใน DSBA กี่หน่วยกิต", "sum_credits", "answer", 0),
            ("Calculus 2 ต้องผ่านอะไรบ้าง", "prerequisite", "clarify_program", 1),
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


if __name__ == "__main__":
    unittest.main()
