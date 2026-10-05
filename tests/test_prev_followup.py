"""Focused regression tests for malformed-provenance hardening in follow-ups."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.prev_followup import answer_previous_followup
from rag.policy.answer import PolicyAnswer


VALID_REF = {
    "source_filename": "rule_page_011.png",
    "source_page": 11,
}

POLICY_REF = {
    "route": "policy",
    "policy_kind": "sanction_appeal_deadline",
    "evidence_ids": ["rule:43"],
}

COURSE_REF = {
    "route": "course",
    "course_code": "06016414",
    "operations": ["sum_credits"],
    "program": "IT",
    "catalog_key": "it-2565",
}


def _policy_answer(provenance):
    return PolicyAnswer(
        status="complete",
        query_type="sanction_appeal_deadline",
        provenance=tuple(provenance),
        rendered_answer="ต้องยื่นภายใน 30 วัน",
    )


class MalformedProvenanceFollowupTests(unittest.TestCase):
    def test_policy_explain_mixed_provenance_fails_closed(self):
        with patch(
            "backend.prev_followup.answer_policy_query",
            return_value=_policy_answer([VALID_REF, "junk", 42]),
        ):
            result = answer_previous_followup(
                db_path="unused.db",
                question="ขยายความหน่อย",
                followup_kind="explain",
                last_answer=dict(POLICY_REF),
                program=None,
                catalog_key=None,
                model_callable=lambda prompt: "unreachable",
            )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])

    def test_policy_source_mixed_provenance_fails_closed(self):
        with patch(
            "backend.prev_followup.answer_policy_query",
            return_value=_policy_answer([VALID_REF, None]),
        ):
            result = answer_previous_followup(
                db_path="unused.db",
                question="อยู่ในข้อไหน",
                followup_kind="source",
                last_answer=dict(POLICY_REF),
                program=None,
                catalog_key=None,
            )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])

    def test_policy_all_invalid_provenance_fails_closed(self):
        with patch(
            "backend.prev_followup.answer_policy_query",
            return_value=_policy_answer(["junk"]),
        ):
            result = answer_previous_followup(
                db_path="unused.db",
                question="ขยายความหน่อย",
                followup_kind="explain",
                last_answer=dict(POLICY_REF),
                program=None,
                catalog_key=None,
                model_callable=lambda prompt: "unreachable",
            )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])

    def test_course_mixed_provenance_fails_closed(self):
        with patch(
            "backend.prev_followup.answer_question_once",
            return_value={
                "route": None,
                "result": None,
                "status": "answer",
                "final_answer": "วิชา 06016414 มี 3 หน่วยกิต",
                "provenance": [VALID_REF, {"source_filename": "", "source_page": 1}],
            },
        ):
            result = answer_previous_followup(
                db_path="unused.db",
                question="ขยายความหน่อย",
                followup_kind="explain",
                last_answer=dict(COURSE_REF),
                program="IT",
                catalog_key="it-2565",
                model_callable=lambda prompt: "unreachable",
            )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])

    def test_valid_provenance_behavior_unchanged(self):
        with patch(
            "backend.prev_followup.answer_policy_query",
            return_value=_policy_answer([dict(VALID_REF)]),
        ):
            result = answer_previous_followup(
                db_path="unused.db",
                question="ขยายความหน่อย",
                followup_kind="explain",
                last_answer=dict(POLICY_REF),
                program=None,
                catalog_key=None,
                model_callable=None,
            )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["provenance"], [VALID_REF])

    def test_api_never_500_on_malformed_provenance(self):
        client = TestClient(main.app)
        with (
            patch.object(main, "answer_hard_question", return_value=None),
            patch(
                "backend.prev_followup.answer_policy_query",
                return_value=_policy_answer([VALID_REF, "junk"]),
            ),
        ):
            response = client.post(
                "/api/ask",
                json={
                    "question": "ขยายความหน่อย",
                    "conversation_context": {"last_answer": POLICY_REF},
                },
            )
        self.assertEqual(response.status_code, 200, response.json())
        payload = response.json()
        self.assertEqual(payload["status"], "insufficient_evidence")
        self.assertEqual(payload["provenance"], [])


if __name__ == "__main__":
    unittest.main()
