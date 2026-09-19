import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag.grounded_answer import GroundedAnswerResult, GroundedClaim
from rag.hybrid_demo import (
    DEFAULT_CURRICULUM_DB_PATH,
    answer_question_once,
    main,
    run_hybrid_demo,
)


def _typed_response(final_answer="คำตอบภาษาไทย"):
    provenance = ({"program": "IT", "source_page": 12},)
    claim = GroundedClaim(
        "claim_001",
        "count",
        value=1,
        provenance=provenance,
    )
    result = GroundedAnswerResult(
        "answer",
        "deterministic",
        final_answer,
        (claim,),
        provenance,
    )
    return {"route": None, "result": result}


class RagHybridDemoTest(unittest.TestCase):
    def test_prints_typed_final_answer_and_forwards_callable(self):
        structured_model_callable = lambda _prompt: "SELECT 1"
        answer_model_callable = lambda _prompt: "คำตอบภาษาไทย"
        response = _typed_response()
        output = io.StringIO()

        with patch("rag.hybrid_demo.ask", return_value=response) as ask_mock:
            with redirect_stdout(output):
                result = run_hybrid_demo(
                    "curriculum.db",
                    "How many credits?",
                    structured_model_callable,
                    2,
                    answer_model_callable,
                )

        self.assertIsInstance(result["result"], GroundedAnswerResult)
        self.assertEqual(result["final_answer"], "คำตอบภาษาไทย")
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["provenance"], response["result"].provenance)
        ask_mock.assert_called_once_with(
            "curriculum.db",
            "How many credits?",
            structured_model_callable=structured_model_callable,
            top_k=2,
            answer_model_callable=answer_model_callable,
            intent_model_callable=None,
        )
        printed = output.getvalue()
        self.assertIn("Question: How many credits?", printed)
        self.assertIn("Final Answer: คำตอบภาษาไทย", printed)

    def test_typed_route_prints_result_answer_without_second_call(self):
        response = _typed_response("คำตอบจากหลักฐาน")
        output = io.StringIO()
        answer_model_callable = lambda _prompt: "คำตอบจาก Gemini หน้า 12, 13"

        with patch("rag.hybrid_demo.ask", return_value=response) as ask_mock:
            with redirect_stdout(output):
                run_hybrid_demo(
                    "curriculum.db",
                    "What topics?",
                    top_k=1,
                    answer_model_callable=answer_model_callable,
                )

        ask_mock.assert_called_once_with(
            "curriculum.db",
            "What topics?",
            structured_model_callable=None,
            top_k=1,
            answer_model_callable=answer_model_callable,
            intent_model_callable=None,
        )
        printed = output.getvalue()
        self.assertIn("Question: What topics?", printed)
        self.assertIn("Final Answer: คำตอบจากหลักฐาน", printed)

    def test_cli_uses_one_gemini_callable_for_sql_and_final_answer(self):
        provider = lambda _prompt: "SELECT 1"
        with patch("rag.hybrid_demo.load_dotenv") as load_dotenv, patch(
            "rag.hybrid_demo.make_gemini_callable", return_value=provider
        ) as factory:
            with patch("rag.hybrid_demo.run_hybrid_demo") as run_demo:
                main(["curriculum.db", "How many credits?", "--structured-provider", "gemini"])

        load_dotenv.assert_called_once_with()
        factory.assert_called_once_with()
        run_demo.assert_called_once_with(
            Path("curriculum.db"),
            "How many credits?",
            structured_model_callable=provider,
            top_k=10,
            answer_model_callable=provider,
            intent_model_callable=provider,
        )

    def test_cli_accepts_question_without_structured_db_path(self):
        structured_model_callable = lambda _prompt: "SELECT 1"
        answer_model_callable = lambda _prompt: "คำตอบ"
        question = "มีวิชาไหนเกี่ยวกับฐานข้อมูลบ้าง"

        sources = [Path("outputs/consolidated/it/coop/full/curriculum.json")]
        with patch("rag.hybrid_demo.run_hybrid_demo") as run_demo:
            main(
                [question],
                structured_model_callable=structured_model_callable,
                answer_model_callable=answer_model_callable,
            )

        run_demo.assert_called_once_with(
            DEFAULT_CURRICULUM_DB_PATH,
            question,
            structured_model_callable=structured_model_callable,
            top_k=10,
            answer_model_callable=answer_model_callable,
            intent_model_callable=None,
        )

    def test_answer_question_once_is_non_printing_and_returns_final_answer(self):
        response = _typed_response("คำตอบ")
        with patch("rag.hybrid_demo.ask", return_value=response) as ask_mock:
            with patch("builtins.print") as print_mock:
                result = answer_question_once(
                    "curriculum.db",
                    "คำถาม",
                    top_k=3,
                )

        self.assertEqual(result["final_answer"], "คำตอบ")
        ask_mock.assert_called_once_with(
            "curriculum.db",
            "คำถาม",
            structured_model_callable=None,
            top_k=3,
            answer_model_callable=None,
            intent_model_callable=None,
        )
        print_mock.assert_not_called()

    def test_answer_question_once_returns_blocked_result_without_synthesis(self):
        response = {
            "route": None,
            "result": {
                "status": "clarify_program",
                "action": "clarify_program",
                "blocking_ambiguity": ("program",),
                "resolved_program": None,
                "course_references": [],
            },
        }

        with patch("rag.hybrid_demo.ask", return_value=response):
            result = answer_question_once("curriculum.db", "วิชา NOSQL")

        self.assertIs(result, response)
        self.assertNotIn("final_answer", result)

    def test_cli_default_does_not_rebuild_and_forwards_answer_callable(self):
        question = "มีวิชาไหนเกี่ยวกับฐานข้อมูลบ้าง"
        database_path = DEFAULT_CURRICULUM_DB_PATH
        structured_model_callable = lambda _prompt: "SELECT 1"
        answer_model_callable = lambda _prompt: "คำตอบ"
        response = _typed_response("คำตอบ")

        with patch("rag.hybrid_demo.load_dotenv"), patch(
            "rag.hybrid_demo.ensure_index", return_value=database_path
        ) as ensure, patch(
            "rag.hybrid_demo.ask",
            return_value=response,
        ) as ask_mock:
            with redirect_stdout(io.StringIO()):
                main(
                    [question],
                    structured_model_callable=structured_model_callable,
                    answer_model_callable=answer_model_callable,
                )

        ask_mock.assert_called_once_with(
            database_path,
            question,
            structured_model_callable=structured_model_callable,
            top_k=10,
            answer_model_callable=answer_model_callable,
            intent_model_callable=None,
        )
        ensure.assert_not_called()

    def test_cli_automatically_wires_gemini_to_final_answer(self):
        provider = lambda _prompt: "คำตอบจาก Gemini"
        with patch("rag.hybrid_demo.make_gemini_callable", return_value=provider) as factory:
            with patch("rag.hybrid_demo.run_hybrid_demo") as run_demo:
                main(["curriculum.db", "What topics?"])

        factory.assert_called_once_with()
        run_demo.assert_called_once_with(
            Path("curriculum.db"),
            "What topics?",
            structured_model_callable=provider,
            top_k=10,
            answer_model_callable=provider,
            intent_model_callable=provider,
        )

    def test_cli_fails_clearly_when_gemini_api_key_is_missing(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                main(["curriculum.db", "How many credits?"])

    def test_injected_callables_are_not_replaced(self):
        structured_model_callable = lambda _prompt: "SELECT 1"
        answer_model_callable = lambda _prompt: "คำตอบ"
        with patch("rag.hybrid_demo.make_gemini_callable") as factory:
            with patch("rag.hybrid_demo.run_hybrid_demo") as run_demo:
                main(
                    ["curriculum.db", "How many credits?", "--structured-provider", "gemini"],
                    structured_model_callable=structured_model_callable,
                    answer_model_callable=answer_model_callable,
                )

        factory.assert_not_called()
        run_demo.assert_called_once_with(
            Path("curriculum.db"),
            "How many credits?",
            structured_model_callable=structured_model_callable,
            top_k=10,
            answer_model_callable=answer_model_callable,
            intent_model_callable=None,
        )
    def test_explicit_gemini_api_key_is_not_replaced_by_dotenv(self):
        provider = lambda _prompt: "SELECT 1"
        with patch.dict(os.environ, {"GEMINI_API_KEY": "explicit-key"}):
            with patch("rag.hybrid_demo.make_gemini_callable", return_value=provider):
                with patch("rag.hybrid_demo.run_hybrid_demo"):
                    main(
                        [
                            "curriculum.db",
                            "How many credits?",
                            "--structured-provider",
                            "gemini",
                        ]
                    )
            self.assertEqual(os.environ["GEMINI_API_KEY"], "explicit-key")

    def test_answer_question_once_forwards_intent_model_callable(self):
        intent_model_callable = lambda _prompt: "{}"
        response = _typed_response()
        with patch("rag.hybrid_demo.ask", return_value=response) as ask_mock:
            answer_question_once(
                "curriculum.db",
                "คำถาม",
                structured_model_callable=None,
                top_k=3,
                answer_model_callable=None,
                intent_model_callable=intent_model_callable,
            )

        ask_mock.assert_called_once_with(
            "curriculum.db",
            "คำถาม",
            structured_model_callable=None,
            top_k=3,
            answer_model_callable=None,
            intent_model_callable=intent_model_callable,
        )

    def test_run_hybrid_demo_forwards_intent_model_callable(self):
        intent_model_callable = lambda _prompt: "{}"
        response = _typed_response()
        with patch("rag.hybrid_demo.ask", return_value=response) as ask_mock:
            with redirect_stdout(io.StringIO()):
                run_hybrid_demo(
                    "curriculum.db",
                    "คำถาม",
                    intent_model_callable=intent_model_callable,
                )

        ask_mock.assert_called_once_with(
            "curriculum.db",
            "คำถาม",
            structured_model_callable=None,
            top_k=5,
            answer_model_callable=None,
            intent_model_callable=intent_model_callable,
        )


@unittest.skipUnless(
    DEFAULT_CURRICULUM_DB_PATH.is_file(), "runtime curriculum.db missing"
)
class RagHybridDemoIntentCallCountsTest(unittest.TestCase):
    def test_deterministic_query_invokes_no_intent_or_structured_models(self):
        intent_calls = []
        structured_calls = []

        def intent_model(prompt):
            intent_calls.append(prompt)
            return "{}"

        def structured_model(prompt):
            structured_calls.append(prompt)
            return "SELECT 1"

        result = answer_question_once(
            DEFAULT_CURRICULUM_DB_PATH,
            "IT ปี 2 เทอม 1 เรียนอะไรบ้าง",
            structured_model_callable=structured_model,
            top_k=5,
            answer_model_callable=lambda _prompt: "คำตอบ",
            intent_model_callable=intent_model,
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(intent_calls, [])
        self.assertEqual(structured_calls, [])

    def test_long_tail_query_uses_intent_once_and_no_structured_call(self):
        intent_calls = []
        structured_calls = []

        def intent_model(prompt):
            intent_calls.append(prompt)
            return json.dumps(
                {
                    "intent": "placement_query",
                    "proposed_program": "IT",
                    "course_codes": ["06016414"],
                    "requested_facts": ["placement"],
                },
                ensure_ascii=False,
            )

        def structured_model(prompt):
            structured_calls.append(prompt)
            return "SELECT 1"

        result = answer_question_once(
            DEFAULT_CURRICULUM_DB_PATH,
            "IT 06016414",
            structured_model_callable=structured_model,
            top_k=5,
            answer_model_callable=None,
            intent_model_callable=intent_model,
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(intent_calls), 1)
        self.assertEqual(structured_calls, [])
        operations = [
            claim.operation for claim in result["result"].claims
        ]
        self.assertIn("placement", operations)

    def test_sql_fallback_uses_structured_only(self):
        intent_calls = []
        structured_calls = []

        def intent_model(prompt):
            intent_calls.append(prompt)
            return "{}"

        def structured_model(prompt):
            structured_calls.append(prompt)
            return (
                "SELECT DISTINCT p.course_id AS course_id "
                "FROM v_plan_courses p WHERE p.program = 'IT' "
                "AND p.year = 2 AND p.semester = 1 LIMIT 3"
            )

        result = answer_question_once(
            DEFAULT_CURRICULUM_DB_PATH,
            "IT ปี 2 เทอม 1 วิชาบังคับมีอะไรบ้าง",
            structured_model_callable=structured_model,
            top_k=5,
            answer_model_callable=None,
            intent_model_callable=intent_model,
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(structured_calls), 1)
        self.assertEqual(intent_calls, [])

    def test_malformed_and_failing_intent_fail_closed_without_retry(self):
        for stub, failure in (
            (lambda _prompt: "{not json", "malformed"),
            (_failing_intent_model, "provider"),
        ):
            with self.subTest(failure=failure):
                calls = []

                def intent_model(prompt, _stub=stub, _calls=calls):
                    _calls.append(prompt)
                    return _stub(prompt)

                result = answer_question_once(
                    DEFAULT_CURRICULUM_DB_PATH,
                    "IT 06016414",
                    structured_model_callable=lambda _prompt: "SELECT 1",
                    top_k=5,
                    answer_model_callable=None,
                    intent_model_callable=intent_model,
                )

                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertEqual(len(calls), 1)


def _failing_intent_model(prompt):
    raise RuntimeError("provider unavailable")


if __name__ == "__main__":
    unittest.main()
