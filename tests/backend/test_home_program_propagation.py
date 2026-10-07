"""Focused API-to-semantic transport tests for home_program."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from backend.main import ask
from backend.schemas import AskRequest
from rag.grounded_answer import GroundedAnswerResult
from rag.semantic.modes import QA_MODE_LEGACY, QA_MODE_SEMANTIC, QA_MODE_SHADOW


class HomeProgramPropagationTests(unittest.TestCase):
    @staticmethod
    def _outcome(context=None):
        return SimpleNamespace(
            result=GroundedAnswerResult(status="answer", answer_mode="deterministic"),
            next_context=context,
            trace=None,
        )

    def _ask_semantic(self, request):
        observed = []

        def semantic_answer(db_path, question, context, **kwargs):
            observed.append((db_path, question, context, kwargs))
            return self._outcome(context)

        with (
            patch("backend.main.active_qa_mode", return_value=QA_MODE_SEMANTIC),
            patch("backend.main._curriculum_db", return_value="curriculum.db"),
            patch("backend.main._semantic_providers", return_value={
                "interpret_callable": lambda prompt: "unused",
                "answer_callable": lambda prompt: "unused",
                "sql_callable": lambda prompt: "unused",
            }),
            patch("rag.semantic.pipeline.semantic_answer", side_effect=semantic_answer),
        ):
            response = ask(request)
        return response, observed

    def test_explicit_home_program_reaches_pipeline_with_context_unchanged(self):
        context = {"program": "IT", "catalog_key": "it-2565", "plan": "coop"}
        request = AskRequest(
            question="IT มีวิชาอะไรบ้าง",
            home_program="IT",
            conversation_context=context,
        )
        _, observed = self._ask_semantic(request)
        self.assertEqual(len(observed), 1)
        _, _, passed_context, kwargs = observed[0]
        self.assertEqual(kwargs["home_program"], "IT")
        self.assertEqual(passed_context, context)

    def test_explicit_null_reaches_pipeline_as_none(self):
        _, observed = self._ask_semantic(
            AskRequest(question="IT มีวิชาอะไรบ้าง", home_program=None))
        self.assertIsNone(observed[0][3]["home_program"])

    def test_omitted_home_program_is_backward_compatible_none(self):
        request = AskRequest(question="IT มีวิชาอะไรบ้าง")
        self.assertIsNone(request.home_program)
        _, observed = self._ask_semantic(request)
        self.assertIsNone(observed[0][3]["home_program"])

    def test_invalid_home_program_structural_types_rejected(self):
        for value in (123, True, [], {}, ("IT",)):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AskRequest.model_validate({
                    "question": "IT มีวิชาอะไรบ้าง",
                    "home_program": value,
                })

    def test_pinned_program_is_explicitly_unsupported_in_legacy_and_shadow(self):
        request = AskRequest(question="IT มีวิชาอะไรบ้าง", home_program="IT")
        for mode in (QA_MODE_LEGACY, QA_MODE_SHADOW):
            with self.subTest(mode=mode), patch(
                "backend.main.active_qa_mode", return_value=mode
            ), patch("backend.main.semantic_ask_response") as semantic, patch(
                "backend.main._capture_shadow_semantic"
            ) as shadow:
                response = ask(request)
            self.assertEqual(response["status"], "unsupported")
            self.assertIn("semantic", response["answer"])
            semantic.assert_not_called()
            shadow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
