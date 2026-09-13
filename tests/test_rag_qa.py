import unittest
from pathlib import Path
from unittest.mock import patch

from rag.qa import ask


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

    def test_structured_route_without_callable_fails(self):
        with self.assertRaises(ValueError):
            ask("curriculum.db", "What are the prerequisites?")


if __name__ == "__main__":
    unittest.main()
