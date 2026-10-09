"""P1.2 provider-call budget pins (derived from code, not comments).

Per-turn application-level maximums:
- interpretation: 1 provider call (no repair loop)
- SQL route: <=2 generation calls (initial + one repair) + 1 answer call
- answer presentation: 1 provider call (deterministic fallback, no repair)
- adapter: <= MAX_PROVIDER_ATTEMPTS HTTP attempts per call, SDK retry off

Worst case per semantic turn: 5 app calls (SQL route) -> <=10 HTTP
attempts; non-SQL turns: 2 app calls -> <=4 HTTP attempts.
"""

import json
import unittest
from pathlib import Path

from backend.llm_sql_qa import ask_sql
from rag.providers import gemini as provider
from rag.semantic.pipeline import semantic_answer


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"

QUESTION = "AIT ait-2566 CALCULUS 2 กี่หน่วยกิต"


def _credit_intent():
    return json.dumps({
        "task": "lookup", "subject": "course", "relation": "credits",
        "target": {"kind": "literal", "raw_text": "CALCULUS 2",
                   "normalized_hint": None, "ordinal": None},
        "scope": {"program": "AIT", "catalog": "ait-2566", "plan": None,
                  "plan_hint": None, "year": None, "semester": None},
        "filters": [], "aggregation": None, "ranking": None,
        "comparison": None, "requested_fields": [],
        "clarification": None, "policy_topic": None, "observed_value": None,
    }, ensure_ascii=False)


class ProviderCallBudgetTests(unittest.TestCase):
    def test_adapter_bounds_are_pinned(self):
        self.assertEqual(provider.MAX_PROVIDER_ATTEMPTS, 2)
        self.assertEqual(provider.REQUEST_TIMEOUT_MS, 20_000)
        self.assertLessEqual(provider.RETRY_BACKOFF_SECONDS, 1.0)

    def test_transient_classification(self):
        from google.genai.errors import APIError

        for exc in (TimeoutError("t"), ConnectionError("c"),
                    APIError(429, {}), APIError(500, {}), APIError(503, {})):
            self.assertTrue(provider._is_transient_provider_error(exc),
                            type(exc).__name__)
        for exc in (APIError(400, {}), APIError(401, {}),
                    APIError(403, {}), APIError(404, {}),
                    ValueError("bad"), RuntimeError("no text")):
            self.assertFalse(provider._is_transient_provider_error(exc),
                             getattr(exc, "code", type(exc).__name__))

    def test_interpretation_and_answer_called_once_each(self):
        calls = {"interpret": 0, "answer": 0}

        def _interpret(prompt):
            calls["interpret"] += 1
            return _credit_intent()

        def _answer(prompt):
            calls["answer"] += 1
            return "synthesis"

        outcome = semantic_answer(
            DB, QUESTION, None,
            interpret_callable=_interpret, answer_callable=_answer,
        )
        self.assertEqual(outcome.result.status, "answer")
        self.assertEqual(calls, {"interpret": 1, "answer": 1})

    def test_sql_generation_repairs_at_most_once(self):
        calls = {"sql": 0}

        def _bad_literal_sql(prompt):
            calls["sql"] += 1
            return "SELECT course_code FROM courses WHERE course_code = '99999999'"

        def _answer(prompt):
            self.fail("invalid SQL must never reach the answer model")

        result = ask_sql(
            DB, "IT มีวิชาอะไรบ้าง", "IT", _bad_literal_sql, _answer,
            conversation_context={"program": "IT", "catalog_key": "it-2565"},
        )
        self.assertEqual(calls["sql"], 2)
        self.assertEqual(result["status"], "error")
        self.assertIn(result["error"]["code"],
                      ("unsupported_course_code_literal", "invalid_sql"))


if __name__ == "__main__":
    unittest.main()
