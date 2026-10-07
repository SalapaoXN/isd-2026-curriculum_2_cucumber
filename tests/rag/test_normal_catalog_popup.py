"""Normal catalog targets use normal context, never operand resolutions."""

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from tests.rag.test_semantic_clarification_transport import DB, payload


QUESTION = "DSBA ปี 2 เทอม 1 มีวิชาอะไรบ้าง"
TARGET = {"dimension": "catalog", "program": "DSBA", "operand": None}


class NormalCatalogPopupTests(unittest.TestCase):
    def request(self, context=None, home=None, program="DSBA", extra=None):
        data = payload(None)
        data.update(task="list", subject="course", comparison=None)
        data["scope"].update(program=program, catalog=None, year=2, semester=1)
        body = {"question": QUESTION.replace("DSBA", program), "home_program": home,
                "conversation_context": context, **(extra or {})}
        with patch("backend.main.active_qa_mode", return_value="semantic"), \
             patch("backend.main._curriculum_db", return_value=DB), \
             patch("backend.main._semantic_providers", return_value={
                 "interpret_callable": lambda prompt: json.dumps(data),
                 "answer_callable": None, "sql_callable": None,
             }):
            return TestClient(app).post("/api/ask", json=body)

    def test_normal_catalog_required_has_nullable_operand_and_canonical_choices(self):
        response = self.request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "catalog_required")
        self.assertEqual(response.json()["clarification_target"], TARGET)
        self.assertEqual(response.json()["provenance"], [])
        with patch("backend.main._curriculum_db", return_value=DB):
            programs = TestClient(app).get("/api/programs").json()["programs"]
        editions = next(p["editions"] for p in programs if p["program_code"] == TARGET["program"])
        self.assertTrue(editions)
        self.assertTrue(all(e["catalog_key"].startswith("dsba-") for e in editions))
        self.assertIn("dsba-2565", [e["catalog_key"] for e in editions])

    def test_retry_uses_selected_normal_context_without_operand_resolution(self):
        for home in (None, "DSBA"):
            with self.subTest(home=home):
                response = self.request({"program": "DSBA", "catalog_key": "dsba-2565"}, home)
                self.assertEqual(response.json()["status"], "answer")
                self.assertEqual(response.json()["next_context"]["program"], "DSBA")
                self.assertEqual(response.json()["next_context"]["catalog_key"], "dsba-2565")
                self.assertNotIn("clarification_target", response.json())

    def test_unscoped_explicit_switch_does_not_inherit_selected_dsba_catalog(self):
        dsba = self.request({"program": "DSBA", "catalog_key": "dsba-2565"}).json()
        switched = self.request(dsba["next_context"], program="IT").json()
        self.assertEqual(switched["action"], "catalog_required")
        self.assertEqual(switched["clarification_target"], {"dimension": "catalog", "program": "IT", "operand": None})
        self.assertEqual(switched["provenance"], [])

    def test_wrong_program_catalog_cannot_produce_factual_answer(self):
        response = self.request({"program": "DSBA", "catalog_key": "it-2565"})
        self.assertNotEqual(response.json()["status"], "answer")
        self.assertEqual(response.json()["provenance"], [])

    def test_normal_target_is_not_an_accepted_operand_resolution(self):
        for extra in ({"clarification_resolution": {**TARGET, "value": "dsba-2565"}},
                      {"clarification_resolutions": [{**TARGET, "value": "dsba-2565"}]}):
            with self.subTest(extra=extra):
                self.assertEqual(self.request(extra=extra).status_code, 422)
