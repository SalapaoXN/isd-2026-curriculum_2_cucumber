"""H17: public helper + Lab10 multi-turn conversation E2E (real DB)."""

import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import rag.qa as qa_module
from lab10_fastapi.curriculum_app import main
from rag.grounded_answer import GroundedAnswerResult
from rag.hybrid_demo import answer_question_once
from rag.resolution import QueryContext


DB_PATH = Path(__file__).resolve().parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"

FORBIDDEN_KEYS = (
    "answer", "count", "credits", "existence", "prerequisites",
    "description", "claims", "evidence", "provenance", "sql", "rows",
)


class H17PublicHelperE2E(unittest.TestCase):
    def test_count_followup_via_public_helper(self):
        t1 = answer_question_once(DB_PATH, "IT ปี 3 เทอม 1 มีทั้งหมดกี่วิชา")
        nc = t1.get("next_context")
        self.assertIsInstance(nc, dict)
        self.assertEqual(nc.get("program"), "IT")
        with patch.object(
            qa_module, "execute_evidence_plan", wraps=qa_module.execute_evidence_plan
        ) as execute:
            t2 = answer_question_once(DB_PATH, "แล้วเทอม 2 ล่ะ", conversation_context=nc)
        self.assertEqual(execute.call_count, 1)
        self.assertIsInstance(t2["result"], GroundedAnswerResult)
        self.assertEqual(t2["result"].status, "answer")
        for claim in t2["result"].claims:
            self.assertEqual(claim.effective_scope.years, (3,))
            self.assertEqual(claim.effective_scope.semesters, (2,))
        self.assertNotEqual(t1["result"].provenance, t2["result"].provenance)
        self.assertTrue(t2["result"].provenance)

    def test_exact_course_credit_followup(self):
        t1 = answer_question_once(DB_PATH, "IT 06016454 คือวิชาอะไร")
        self.assertEqual(t1["next_context"].get("course_code"), "06016454")
        t2 = answer_question_once(
            DB_PATH, "แล้วกี่หน่วยกิต", conversation_context=t1["next_context"]
        )
        self.assertEqual(t2["result"].status, "answer")
        self.assertIn("sum_credits", [c.operation for c in t2["result"].claims])
        self.assertTrue(t2["result"].provenance)

    def test_prerequisite_followup(self):
        t1 = answer_question_once(DB_PATH, "IT 06016454 คือวิชาอะไร")
        t2 = answer_question_once(
            DB_PATH, "มี prerequisite ไหม", conversation_context=t1["next_context"]
        )
        self.assertEqual(t2["result"].status, "answer")
        self.assertIn("prerequisite", [c.operation for c in t2["result"].claims])
        self.assertTrue(t2["result"].provenance)

    def test_program_switch_overrides(self):
        t1 = answer_question_once(DB_PATH, "IT มีกี่วิชา")
        t2 = answer_question_once(
            DB_PATH, "แล้ว DSBA ล่ะ", conversation_context=t1["next_context"]
        )
        self.assertTrue(
            all(c.effective_scope.program == "DSBA" for c in t2["result"].claims)
        )

    def test_next_context_has_no_factual_values(self):
        t1 = answer_question_once(DB_PATH, "IT ปี 3 เทอม 1 มีทั้งหมดกี่วิชา")
        nc = t1.get("next_context")
        for key in FORBIDDEN_KEYS:
            self.assertNotIn(key, nc)
        # next_context is separate from the answer object
        self.assertIsInstance(t1["result"], GroundedAnswerResult)
        self.assertNotIsInstance(nc, GroundedAnswerResult)

    def test_backward_compat_question_only(self):
        r = answer_question_once(DB_PATH, "IT วิชา 06016454 มีกี่หน่วยกิต")
        self.assertEqual(r["result"].status, "answer")
        self.assertIn("3", r["result"].final_answer)

    def test_malformed_dict_rejected(self):
        for bad in (
            {"program": "IT", "count": 5},
            {"program": "IT", "sql": "SELECT 1"},
            {"program": "", "years": [3]},
            {"years": "3"},
            {"bogus": 1},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    answer_question_once(DB_PATH, "แล้วเทอม 2 ล่ะ", conversation_context=bad)

    def test_object_input_still_accepted(self):
        ctx = QueryContext(program="IT", years=(3,), operations=("count",))
        r = answer_question_once(DB_PATH, "แล้วเทอม 1 ล่ะ", conversation_context=ctx)
        self.assertEqual(r["result"].status, "answer")


class H17Lab10E2E(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.db_patch = patch.object(main, "DEFAULT_CURRICULUM_DB_PATH", DB_PATH)
        self.db_patch.start()
        main._provider = None

    def tearDown(self):
        main._provider = None
        self.db_patch.stop()

    def test_two_turn_lab10(self):
        r1 = self.client.post(
            "/api/ask", json={"question": "IT ปี 3 เทอม 1 มีทั้งหมดกี่วิชา"}
        )
        self.assertEqual(r1.status_code, 200)
        nc = r1.json().get("next_context")
        self.assertIsInstance(nc, dict)
        r2 = self.client.post(
            "/api/ask",
            json={"question": "แล้วเทอม 2 ล่ะ", "conversation_context": nc},
        )
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["status"], "answer")
        self.assertTrue(r2.json()["provenance"])

    def test_no_hidden_memory(self):
        r = self.client.post("/api/ask", json={"question": "แล้วเทอม 2 ล่ะ"})
        self.assertEqual(r.status_code, 200)
        self.assertNotEqual(r.json().get("status"), "answer")

    def test_malformed_is_4xx(self):
        r = self.client.post(
            "/api/ask",
            json={
                "question": "แล้วเทอม 2 ล่ะ",
                "conversation_context": {"program": "IT", "count": 5},
            },
        )
        self.assertIn(r.status_code, (400, 422))

    def test_no_trace_leak_and_clean_context(self):
        r1 = self.client.post(
            "/api/ask", json={"question": "IT ปี 3 เทอม 1 มีทั้งหมดกี่วิชา"}
        )
        payload = r1.json()
        for key in ("sql", "rows", "model_trace"):
            self.assertNotIn(key, payload)
        for key in FORBIDDEN_KEYS:
            self.assertNotIn(key, payload.get("next_context") or {})


if __name__ == "__main__":
    unittest.main()
