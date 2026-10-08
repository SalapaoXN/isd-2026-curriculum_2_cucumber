"""Transport-only tests for one-call structured semantic interpretation."""

import json
from pathlib import Path
import unittest

from rag.semantic.errors import SemanticOperationalError
from rag.semantic.interpreter import (
    SemanticSchemaError,
    interpret_semantic_intent,
    parse_semantic_intent_payload,
)
from rag.semantic.pipeline import semantic_answer


QUESTION = "IT 06016405 กี่หน่วยกิต"
DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


def _payload():
    return json.dumps(
        {
            "task": "lookup",
            "subject": "course",
            "relation": "credits",
            "target": {
                "kind": "literal",
                "raw_text": "06016405",
                "normalized_hint": None,
                "ordinal": None,
            },
            "scope": {
                "program": "IT",
                "catalog": None,
                "plan": None,
                "year": None,
                "semester": None,
            },
            "filters": [],
            "aggregation": None,
            "ranking": None,
            "comparison": None,
            "requested_fields": ["credits"],
            "clarification": None,
            "policy_topic": None,
            "observed_value": None,
        }
    )


class SemanticInterpreterStructuredTransportTests(unittest.TestCase):
    def test_structured_capable_callable_receives_json_mime_and_schema_once(self):
        calls = []

        def provider(prompt, *, response_mime_type=None, response_json_schema=None):
            calls.append((prompt, response_mime_type, response_json_schema))
            return _payload()

        intent, _ = interpret_semantic_intent(QUESTION, provider)

        self.assertEqual(intent.task, "lookup")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "application/json")
        schema = calls[0][2]
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {
            "task", "subject", "relation", "target", "scope", "filters",
            "aggregation", "ranking", "comparison", "requested_fields",
            "clarification", "policy_topic", "observed_value",
        })
        self.assertEqual(set(schema["properties"]["task"]["enum"]), {
            "lookup", "compose", "list", "search", "aggregate", "compare", "rank",
            "policy", "requirement", "unknown",
        })

    def test_one_argument_callable_is_called_once_without_structured_kwargs(self):
        calls = []

        def provider(prompt):
            calls.append(prompt)
            return _payload()

        intent, _ = interpret_semantic_intent(QUESTION, provider)
        self.assertEqual(intent.target.raw_text, "06016405")
        self.assertEqual(len(calls), 1)

    def test_one_argument_callable_stays_compatible_through_pipeline_counter(self):
        calls = []

        def provider(prompt):
            calls.append(prompt)
            return _payload()

        outcome = semantic_answer(
            DB,
            QUESTION,
            {"program": "IT", "catalog_key": "it-2565"},
            home_program="IT",
            interpret_callable=provider,
        )
        self.assertEqual(outcome.result.status, "answer")
        self.assertEqual(len(calls), 1)

    def test_structured_capability_survives_pipeline_counter_wrapper(self):
        calls = []

        def provider(prompt, *, response_mime_type=None, response_json_schema=None):
            calls.append((prompt, response_mime_type, response_json_schema))
            return _payload()

        outcome = semantic_answer(
            DB,
            QUESTION,
            {"program": "IT", "catalog_key": "it-2565"},
            home_program="IT",
            interpret_callable=provider,
            answer_callable=None,
        )
        self.assertEqual(outcome.result.status, "answer")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "application/json")
        self.assertIn("properties", calls[0][2])

    def test_structured_callable_internal_typeerror_is_not_retried(self):
        calls = []

        def provider(prompt, **kwargs):
            calls.append((prompt, kwargs))
            raise TypeError("provider internal type error")

        with self.assertRaisesRegex(TypeError, "provider internal"):
            interpret_semantic_intent(QUESTION, provider)
        self.assertEqual(len(calls), 1)

    def test_malformed_structured_output_still_fails_closed_once(self):
        calls = []

        def provider(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return '{"task": "list",'

        with self.assertRaises(SemanticSchemaError):
            interpret_semantic_intent(QUESTION, provider)
        self.assertEqual(len(calls), 1)

    def test_plain_and_whole_fenced_json_remain_supported(self):
        body = _payload()
        fenced = "```json\n" + body + "\n```"
        for output in (body, fenced):
            with self.subTest(fenced=output.startswith("```")):
                intent, _ = interpret_semantic_intent(QUESTION, lambda _prompt: output)
                self.assertEqual(intent.task, "lookup")

    def test_unknown_enum_and_unknown_field_remain_closed(self):
        base = json.loads(_payload())
        invalid_task = dict(base, task="invented")
        extra_field = dict(base, extra="not allowed")
        invalid_scope = json.loads(_payload())
        invalid_scope["scope"]["extra_dimension"] = "not allowed"
        for invalid in (invalid_task, extra_field, invalid_scope):
            with self.subTest(payload=invalid), self.assertRaises(SemanticSchemaError):
                parse_semantic_intent_payload(json.dumps(invalid))

    def test_provider_timeout_keeps_provider_unavailable_typing(self):
        def provider(prompt, **kwargs):
            raise TimeoutError("transient provider failure")

        with self.assertRaises(SemanticOperationalError) as raised:
            semantic_answer(
                DB,
                QUESTION,
                {"program": "IT", "catalog_key": "it-2565"},
                home_program="IT",
                interpret_callable=provider,
            )
        self.assertEqual(raised.exception.status, "provider_unavailable")


if __name__ == "__main__":
    unittest.main()
