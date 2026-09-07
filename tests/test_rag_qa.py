import unittest
from unittest.mock import patch

from rag.qa import ask


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

    def test_structured_route_without_callable_fails(self):
        with self.assertRaises(ValueError):
            ask("curriculum.db", "What are the prerequisites?")


if __name__ == "__main__":
    unittest.main()
