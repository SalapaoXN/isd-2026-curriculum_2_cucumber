"""P1.1 red tests: operational failure must not masquerade as evidence failure.

Contract under test (via the real ``backend.main.ask`` semantic route,
real pipeline, real runtime DB; only the model provider is stubbed):

- provider construction failure / provider timeout -> ``provider_unavailable``
  (NOT ``insufficient_evidence``), no fabricated answer, no internals.
- unexpected pipeline exception -> ``error`` (NOT ``insufficient_evidence``).
- operational failure preserves the last validated safe context for retry:
  no forged fields, no comparison-operand leakage, no home mutation.
- genuine evidence insufficiency stays ``insufficient_evidence``.
- answerer failure after VerifiedResult still returns the verified answer
  through the existing deterministic fallback.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

from backend.main import ask
from backend.schemas import AskRequest
from rag.semantic.modes import QA_MODE_SEMANTIC


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"

IT_CONTEXT = {"program": "IT", "catalog_key": "it-2565", "plan": "coop"}
DSBA_CONTEXT = {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "no_coop"}

OPERATIONAL_STATUSES = ("provider_unavailable", "error")


def _intent(**overrides):
    data = {
        "task": "lookup", "subject": "course", "relation": "credits",
        "target": {"kind": "literal", "raw_text": "CALCULUS 2",
                   "normalized_hint": None, "ordinal": None},
        "scope": {"program": "AIT", "catalog": "ait-2566", "plan": None,
                  "plan_hint": None, "year": None, "semester": None},
        "filters": [], "aggregation": None, "ranking": None,
        "comparison": None, "requested_fields": [],
        "clarification": None, "policy_topic": None, "observed_value": None,
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def _empty_scope_list_intent():
    data = {
        "task": "list", "subject": "course", "relation": None,
        "target": {"kind": "none", "raw_text": None,
                   "normalized_hint": None, "ordinal": None},
        "scope": {"program": "IT", "catalog": "it-2565", "plan": None,
                  "plan_hint": None, "year": 5, "semester": 1},
        "filters": [], "aggregation": None, "ranking": None,
        "comparison": None, "requested_fields": [],
        "clarification": None, "policy_topic": None, "observed_value": None,
    }
    return json.dumps(data, ensure_ascii=False)


class SemanticOperationalFailureTests(unittest.TestCase):
    def _ask(self, request, providers):
        with (
            patch("backend.main.active_qa_mode", return_value=QA_MODE_SEMANTIC),
            patch("backend.main._curriculum_db", return_value=DB),
            patch("backend.main._semantic_providers", return_value=providers),
        ):
            return ask(request)

    def _providers(self, interpret=None, answer=None):
        def _interpret(prompt):
            if isinstance(interpret, Exception):
                raise interpret
            if callable(interpret):
                return interpret(prompt)
            return interpret

        def _answer(prompt):
            if isinstance(answer, Exception):
                raise answer
            if callable(answer):
                return answer(prompt)
            return answer

        return {
            "interpret_callable": _interpret,
            "answer_callable": _answer,
            "sql_callable": _interpret,
        }

    def test_provider_construction_failure_is_operational(self):
        response = self._ask(
            AskRequest(question="AIT ait-2566 CALCULUS 2 กี่หน่วยกิต",
                       conversation_context=dict(IT_CONTEXT)),
            {},
        )
        self.assertIn(response["status"], OPERATIONAL_STATUSES)
        self.assertNotEqual(response["status"], "insufficient_evidence")
        self.assertIn("กรุณาลองใหม่อีกครั้ง", response["answer"])
        self.assertEqual(response["provenance"], [])

    def test_interpreter_timeout_preserves_it_context(self):
        response = self._ask(
            AskRequest(question="AIT ait-2566 CALCULUS 2 กี่หน่วยกิต",
                       conversation_context=dict(IT_CONTEXT)),
            self._providers(
                interpret=TimeoutError("request timed out after 20000ms"),
                answer="unused",
            ),
        )
        self.assertEqual(response["status"], "provider_unavailable")
        self.assertIn("กรุณาลองใหม่อีกครั้ง", response["answer"])
        context = response["next_context"] or {}
        self.assertEqual(context.get("program"), "IT")
        self.assertEqual(context.get("catalog_key"), "it-2565")
        self.assertEqual(context.get("plan"), "coop")

    def test_unexpected_pipeline_exception_is_system_error(self):
        with patch(
            "rag.semantic.pipeline.resolve_semantic_intent",
            side_effect=RuntimeError("kaboom /tmp/secret trace"),
        ):
            response = self._ask(
                AskRequest(question="AIT ait-2566 CALCULUS 2 กี่หน่วยกิต",
                           conversation_context=dict(IT_CONTEXT)),
                self._providers(interpret=_intent(), answer="unused"),
            )
        self.assertEqual(response["status"], "error")
        self.assertIn("กรุณาลองใหม่อีกครั้ง", response["answer"])
        dumped = json.dumps(response, ensure_ascii=False)
        for leaked in ("kaboom", "/tmp/secret", "Traceback", "RuntimeError"):
            self.assertNotIn(leaked, dumped)

    def test_true_insufficient_evidence_is_unchanged(self):
        response = self._ask(
            AskRequest(question="IT it-2565 ปี 5 เทอม 1 มีวิชาอะไรบ้าง",
                       conversation_context=dict(IT_CONTEXT)),
            self._providers(interpret=_empty_scope_list_intent(), answer="unused"),
        )
        self.assertEqual(response["status"], "insufficient_evidence")

    def test_dsba_scope_survives_provider_failure(self):
        response = self._ask(
            AskRequest(question="วิชาบังคับก่อนของ Calculus คืออะไร",
                       home_program="DSBA",
                       conversation_context=dict(DSBA_CONTEXT)),
            self._providers(
                interpret=ConnectionError("connection reset by peer"),
                answer="unused",
            ),
        )
        self.assertEqual(response["status"], "provider_unavailable")
        context = response["next_context"] or {}
        self.assertEqual(context.get("program"), "DSBA")
        self.assertEqual(context.get("catalog_key"), "dsba-2565")
        self.assertEqual(context.get("plan"), "no_coop")

    def test_comparison_failure_leaks_no_operands(self):
        response = self._ask(
            AskRequest(question="IT it-2565 เปรียบเทียบกับ DSBA dsba-2565",
                       conversation_context=dict(IT_CONTEXT)),
            self._providers(
                interpret=TimeoutError("deadline exceeded"),
                answer="unused",
            ),
        )
        self.assertIn(response["status"], OPERATIONAL_STATUSES)
        context = response["next_context"] or {}
        for key in ("left", "right", "operand", "operands", "comparison",
                    "comparison_sides"):
            self.assertNotIn(key, context)
        self.assertEqual(context.get("program"), "IT")

    def test_clarification_retry_state_not_fabricated(self):
        request = AskRequest(
            question="IT it-2565 เปรียบเทียบกับ DSBA dsba-2565",
            conversation_context=dict(IT_CONTEXT),
            clarification_resolution={
                "dimension": "catalog", "program": "IT",
                "operand": "left", "value": "it-2565",
            },
        )
        response = self._ask(
            request,
            self._providers(
                interpret=TimeoutError("service unavailable"),
                answer="unused",
            ),
        )
        self.assertIn(response["status"], OPERATIONAL_STATUSES)
        self.assertNotIn("clarification_target", response)
        context = response["next_context"] or {}
        self.assertEqual(context.get("program"), "IT")

    def test_no_secret_or_internals_leak(self):
        secret = "AIzaSECRET123"
        providers = self._providers(
            interpret=RuntimeError(
                f"GEMINI_API_KEY={secret} SELECT * FROM courses "
                "prompt=[system prompt] Traceback File \"/app/x.py\""
            ),
            answer="unused",
        )
        response = self._ask(
            AskRequest(question="AIT ait-2566 CALCULUS 2 กี่หน่วยกิต",
                       conversation_context=dict(IT_CONTEXT)),
            providers,
        )
        self.assertIn(response["status"], OPERATIONAL_STATUSES)
        dumped = json.dumps(response, ensure_ascii=False)
        for leaked in (secret, "AIza", "SELECT", "Traceback", ".py",
                       "system prompt", "RuntimeError"):
            self.assertNotIn(leaked, dumped)

    def test_forged_context_fields_are_not_preserved(self):
        forged = dict(IT_CONTEXT)
        forged["injected"] = {"nested": "x" * 100}
        forged["left"] = {"program": "DSBA"}
        with self.assertRaises(ValidationError):
            AskRequest(question="AIT ait-2566 CALCULUS 2 กี่หน่วยกิต",
                       conversation_context=forged)

    def test_answerer_failure_still_returns_verified_answer(self):
        response = self._ask(
            AskRequest(question="AIT ait-2566 CALCULUS 2 กี่หน่วยกิต"),
            self._providers(
                interpret=_intent(),
                answer=RuntimeError("answerer exploded"),
            ),
        )
        self.assertEqual(response["status"], "answer")
        self.assertIn("06046401", response["answer"])
        self.assertTrue(response["provenance"])


if __name__ == "__main__":
    unittest.main()
