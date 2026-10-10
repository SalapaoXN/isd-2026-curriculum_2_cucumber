"""Regression: process logging must never change deterministic policy answers."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.policy.query import parse_policy_question
from rag.semantic.modes import semantic_ask_response


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


class SemanticPolicyLogIndependenceTests(unittest.TestCase):
    def test_probation_student_wording_is_deterministic_with_or_without_logging(self):
        question = "ติดโปรต้องเกรดเท่าไหร่"
        parsed = parse_policy_question(question)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.kind, "probation_entry")

        def forbidden_provider(*_args, **_kwargs):
            return json.dumps({"task":"policy","subject":"policy","relation":None,"target":{"kind":"none","raw_text":None,"normalized_hint":None,"ordinal":None},"scope":{"program":None,"catalog":None,"plan":None,"plan_hint":None,"year":None,"semester":None},"filters":[],"aggregation":None,"ranking":None,"comparison":None,"requested_fields":[],"clarification":None,"policy_topic":"probation","observed_value":None}, ensure_ascii=False)

        responses = []
        for flag in ("", "1"):
            output = StringIO()
            with patch.dict(os.environ, {"CUCUMBER_PROCESS_LOG": flag}, clear=False):
                with redirect_stdout(output):
                    response = semantic_ask_response(
                        DB,
                        question,
                        None,
                        interpret_callable=forbidden_provider,
                    )
            responses.append(response)

        self.assertEqual([item["status"] for item in responses], ["answer", "answer"])
        self.assertEqual(responses[0]["answer"], responses[1]["answer"])
        self.assertIn("GPA ต่ำกว่า", responses[0]["answer"])


if __name__ == "__main__":
    unittest.main()
