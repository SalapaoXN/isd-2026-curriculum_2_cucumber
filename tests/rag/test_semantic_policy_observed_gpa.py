"""Regression coverage for LLM-interpreted observed GPA policy comparisons."""

import json
from pathlib import Path
import unittest

from rag.semantic.modes import semantic_ask_response


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


def _payload(observed_value):
    return json.dumps(
        {
            "task": "policy",
            "subject": "policy",
            "relation": None,
            "target": {"kind": "none", "raw_text": None, "normalized_hint": None, "ordinal": None},
            "scope": {"program": None, "catalog": None, "plan": None, "plan_hint": None, "year": None, "semester": None},
            "filters": [],
            "aggregation": None,
            "ranking": None,
            "comparison": None,
            "requested_fields": [],
            "clarification": None,
            "policy_topic": "probation",
            "observed_value": observed_value,
        },
        ensure_ascii=False,
    )


class SemanticPolicyObservedGpaTests(unittest.TestCase):
    def test_student_observed_gpa_reaches_canonical_probation_comparison(self):
        question = "ถ้าได้เกรดเฉลี่ย 1.9 จะติดโปรมั้ย"
        calls = []

        def interpreter(*_args, **_kwargs):
            calls.append(1)
            return _payload("1.9")

        response = semantic_ask_response(
            DB,
            question,
            None,
            interpret_callable=interpreter,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(response["status"], "answer")
        self.assertIn("GPA สะสม 1.9", response["answer"])
        self.assertIn("เข้าข่ายภาคทัณฑ์", response["answer"])
        self.assertTrue(response["provenance"])

    def test_interpreter_cannot_invent_observed_gpa_not_in_question(self):
        response = semantic_ask_response(
            DB,
            "ถ้าได้เกรดเฉลี่ย 1.9 จะติดโปรมั้ย",
            None,
            interpret_callable=lambda *_args, **_kwargs: _payload("1.8"),
        )

        self.assertNotEqual(response["status"], "answer")
        self.assertFalse(response["provenance"])


if __name__ == "__main__":
    unittest.main()
