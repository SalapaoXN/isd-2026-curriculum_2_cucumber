import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag.hybrid_demo import main, run_hybrid_demo


class RagHybridDemoTest(unittest.TestCase):
    def test_prints_structured_route_sql_and_results(self):
        model_callable = lambda _prompt: "SELECT 1"
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
                result = run_hybrid_demo("curriculum.db", "How many credits?", model_callable, 2)

        self.assertIs(result, response)
        ask_mock.assert_called_once_with(
            "curriculum.db",
            "How many credits?",
            structured_model_callable=model_callable,
            top_k=2,
        )
        printed = output.getvalue()
        self.assertIn("selected route: structured", printed)
        self.assertIn("sql: SELECT course_code FROM courses", printed)
        self.assertIn("rows: [('CS101',)]", printed)

    def test_prints_semantic_evidence_and_source_page(self):
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

        with patch("rag.hybrid_demo.ask", return_value=response):
            with redirect_stdout(output):
                run_hybrid_demo("curriculum.db", "What topics?", top_k=1)

        printed = output.getvalue()
        self.assertIn("selected route: semantic", printed)
        self.assertIn("course description", printed)
        self.assertIn("source_page: 12, 13", printed)

    def test_cli_can_create_callable_through_existing_provider_boundary(self):
        provider = lambda _prompt: "SELECT 1"
        with patch("rag.hybrid_demo.make_gemini_callable", return_value=provider) as factory:
            with patch("rag.hybrid_demo.run_hybrid_demo") as run_demo:
                main(["curriculum.db", "How many credits?", "--structured-provider", "gemini"])

        factory.assert_called_once_with()
        run_demo.assert_called_once_with(
            Path("curriculum.db"),
            "How many credits?",
            structured_model_callable=provider,
            top_k=5,
        )


if __name__ == "__main__":
    unittest.main()
