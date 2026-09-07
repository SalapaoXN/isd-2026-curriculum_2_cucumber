import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag.hybrid_demo import main, run_hybrid_demo


class RagHybridDemoTest(unittest.TestCase):
    def test_prints_structured_route_and_final_answer(self):
        structured_model_callable = lambda _prompt: "SELECT 1"
        answer_model_callable = lambda _prompt: "คำตอบภาษาไทย"
        response = {
            "route": "structured",
            "result": {
                "sql": "SELECT course_code FROM courses",
                "columns": ["course_code"],
                "rows": [("CS101",)],
            },
        }
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

        self.assertIs(result, response)
        self.assertEqual(result["final_answer"], "คำตอบภาษาไทย")
        ask_mock.assert_called_once_with(
            "curriculum.db",
            "How many credits?",
            structured_model_callable=structured_model_callable,
            top_k=2,
        )
        printed = output.getvalue()
        self.assertIn("Question: How many credits?", printed)
        self.assertIn("Route: structured", printed)
        self.assertIn("Final Answer: คำตอบภาษาไทย", printed)

    def test_semantic_route_uses_persistent_index_and_final_answer(self):
        response = {
            "route": "semantic",
            "result": [
                {
                    "chunk_id": "course-1-description",
                    "distance": 0.125,
                    "text": "course description",
                    "source_page": [12, 13],
                }
            ],
        }
        output = io.StringIO()
        answer_model_callable = lambda _prompt: "คำตอบจาก Gemini หน้า 12, 13"

        with patch("rag.hybrid_demo.ask") as ask_mock, patch(
            "rag.hybrid_demo.search_index", return_value=response["result"]
        ) as query_index_mock:
            with redirect_stdout(output):
                run_hybrid_demo(
                    "curriculum.db",
                    "What topics?",
                    top_k=1,
                    answer_model_callable=answer_model_callable,
                )

        ask_mock.assert_not_called()
        query_index_mock.assert_called_once_with(
            Path("rag_artifacts/semantic.db"), "What topics?", top_k=1
        )
        printed = output.getvalue()
        self.assertIn("Question: What topics?", printed)
        self.assertIn("Route: semantic", printed)
        self.assertIn("Final Answer: คำตอบจาก Gemini หน้า 12, 13", printed)

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
            top_k=5,
            answer_model_callable=provider,
        )

    def test_cli_accepts_question_without_structured_db_path(self):
        structured_model_callable = lambda _prompt: "SELECT 1"
        answer_model_callable = lambda _prompt: "คำตอบ"
        question = "มีวิชาไหนเกี่ยวกับฐานข้อมูลบ้าง"

        with patch("rag.hybrid_demo.run_hybrid_demo") as run_demo:
            main(
                [question],
                structured_model_callable=structured_model_callable,
                answer_model_callable=answer_model_callable,
            )

        run_demo.assert_called_once_with(
            Path("demo.db"),
            question,
            structured_model_callable=structured_model_callable,
            top_k=5,
            answer_model_callable=answer_model_callable,
        )

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
            top_k=5,
            answer_model_callable=provider,
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
            top_k=5,
            answer_model_callable=answer_model_callable,
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


if __name__ == "__main__":
    unittest.main()
