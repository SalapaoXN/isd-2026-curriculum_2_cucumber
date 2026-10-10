"""Provider-bounded regressions for tiered semantic topic relevance."""

import json
from pathlib import Path
import re
import unittest

from rag.retrieval.retrieve import (
    CONSTRAINED_TOPIC_DISTANCE_THRESHOLD,
    TOPIC_RELEVANCE_JUDGE_PROMPT_VERSION,
    TOPIC_TIER2_MAX_COURSES,
    retrieve_constrained_topic_evidence,
)
from rag.semantic.modes import semantic_ask_response
from rag.structured.queries import scoped_course_set
from tests.rag.test_semantic_topic_realdata_regression import topic_payload


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"
CONTEXT = {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "coop"}
QUESTION = "วิชาที่เกี่ยวข้องกับการคำนวณมีอะไรบ้าง"
TOPIC = "การคำนวณ"


def select_course_code(code, calls):
    def judge(prompt):
        calls.append(prompt)
        match = re.search(rf"^\[(\d+)\]\s+{re.escape(code)}\s+\|", prompt, re.MULTILINE)
        if match is None:
            return json.dumps({"relevant": []})
        return json.dumps({"relevant": [int(match.group(1))]})

    return judge


class SemanticTopicTieredRerankTests(unittest.TestCase):
    def candidates(self):
        structural = scoped_course_set(
            DB, "DSBA", ("coop",), catalog_key="dsba-2565"
        )
        self.assertEqual(structural["status"], "ok")
        return structural["courses"]

    def test_borderline_course_can_be_selected_in_one_bounded_judge_call(self):
        calls = []
        result = retrieve_constrained_topic_evidence(
            DB,
            TOPIC,
            self.candidates(),
            relevance_judge=select_course_code("06066000", calls),
        )
        self.assertEqual(result.status, "scored")
        self.assertEqual(len(calls), 1)
        self.assertIn(TOPIC_RELEVANCE_JUDGE_PROMPT_VERSION, calls[0])
        self.assertLessEqual(len(re.findall(r"^\[\d+\]", calls[0], re.MULTILINE)),
                             TOPIC_TIER2_MAX_COURSES)
        selected = {candidate.get("course_code") for candidate in result.scored_candidates}
        self.assertIn("06066000", selected)
        distances = [
            evidence.get("distance")
            for candidate in result.candidates
            if candidate.get("course_code") == "06066000"
            for evidence in candidate.get("description_evidence", ())
            if evidence.get("distance") is not None
        ]
        self.assertTrue(distances)
        self.assertGreater(min(distances), CONSTRAINED_TOPIC_DISTANCE_THRESHOLD)

    def test_structured_topic_judge_requests_provider_enforced_json(self):
        calls = []

        def structured_judge(prompt, *, response_mime_type=None, response_json_schema=None):
            calls.append((prompt, response_mime_type, response_json_schema))
            match = re.search(r"^\[(\d+)\]\s+06066000\s+\|", prompt, re.MULTILINE)
            return json.dumps({"relevant": [int(match.group(1))] if match else []})

        result = retrieve_constrained_topic_evidence(
            DB,
            TOPIC,
            self.candidates(),
            relevance_judge=structured_judge,
        )
        self.assertEqual(result.status, "scored")
        self.assertEqual(len(calls), 1)
        _prompt, mime_type, schema = calls[0]
        self.assertEqual(mime_type, "application/json")
        self.assertEqual(schema["required"], ["relevant"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["relevant"]["items"]["minimum"], 1)

    def test_semantic_pipeline_uses_topic_judge_but_deterministic_presentation(self):
        calls = []
        response = semantic_ask_response(
            DB,
            QUESTION,
            CONTEXT,
            home_program="DSBA",
            interpret_callable=lambda prompt: topic_payload(TOPIC),
            answer_callable=select_course_code("06066000", calls),
        )
        self.assertEqual(response["status"], "answer")
        self.assertIn("06066000", response["answer"])
        self.assertEqual(len(calls), 1)
        self.assertIn(TOPIC_RELEVANCE_JUDGE_PROMPT_VERSION, calls[0])

    def test_unselected_plan_uses_one_relevance_judge_call(self):
        calls = []
        context = {"program": "DSBA", "catalog_key": "dsba-2565"}
        response = semantic_ask_response(
            DB,
            QUESTION,
            context,
            home_program="DSBA",
            interpret_callable=lambda prompt: topic_payload(TOPIC),
            answer_callable=select_course_code("06066000", calls),
        )
        self.assertEqual(response["status"], "answer")
        self.assertEqual(len(calls), 1)

    def test_malformed_judge_output_fails_closed(self):
        calls = []

        def malformed(prompt):
            calls.append(prompt)
            return "06066000"

        response = semantic_ask_response(
            DB,
            QUESTION,
            CONTEXT,
            home_program="DSBA",
            interpret_callable=lambda prompt: topic_payload(TOPIC),
            answer_callable=malformed,
        )
        self.assertEqual(response["status"], "insufficient_evidence")
        self.assertEqual(response["answer"], "")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
