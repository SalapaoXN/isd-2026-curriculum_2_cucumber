"""Catalog choices and one-shot comparison retry through the real API."""

from copy import deepcopy
import json
import sqlite3
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import AskRequest
from rag.semantic.executor import execute_comparison
from tests.rag.test_semantic_clarification_transport import DB, payload


QUESTION = "เปรียบเทียบหน่วยกิตหลักสูตร DSBA กับ IT"
CONTEXT = {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "coop"}
TARGET = {"dimension": "catalog", "program": "IT", "operand": "right"}
RESOLUTION = {**TARGET, "value": "it-2565"}


def catalog_payload():
    data = payload(None, "difference")
    data["comparison"]["left"]["program"] = "DSBA"
    data["comparison"]["right"]["program"] = "IT"
    for side in ("left", "right"):
        data["comparison"][side]["catalog"] = None
    return data


class OperandCatalogRetryTests(unittest.TestCase):
    def request(self, resolution=None, data=None, question=QUESTION, context=CONTEXT):
        before = deepcopy(context)
        body = {"question": question, "home_program": context["program"], "conversation_context": before}
        if resolution is not None:
            body["clarification_resolution"] = resolution
        raw = json.dumps(data if data is not None else catalog_payload())
        with patch("backend.main.active_qa_mode", return_value="semantic"), \
             patch("backend.main._curriculum_db", return_value=DB), \
             patch("backend.main._semantic_providers", return_value={
                 "interpret_callable": lambda prompt: raw, "answer_callable": None, "sql_callable": None,
             }), patch("rag.semantic.pipeline.execute_comparison", wraps=execute_comparison) as execution:
            response = TestClient(app).post("/api/ask", json=body)
        self.assertEqual(before, context)
        return response, execution

    def test_missing_right_catalog_exposes_deterministic_target(self):
        response, execution = self.request()
        self.assertEqual(response.json()["action"], "catalog_required")
        self.assertEqual(response.json()["clarification_target"], TARGET)
        self.assertEqual(response.json()["next_context"], CONTEXT)
        execution.assert_not_called()

    def test_program_options_are_exactly_canonical_editions(self):
        with patch("backend.main._curriculum_db", return_value=DB):
            response = TestClient(app).get("/api/programs")
        editions = next(p["editions"] for p in response.json()["programs"] if p["program_code"] == "IT")
        with sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            expected = set(connection.execute(
                "SELECT DISTINCT c.catalog_key,c.academic_year FROM catalogs c JOIN programs p ON p.catalog_id=c.catalog_id WHERE p.program_code='IT'"
            ))
        self.assertEqual({(e["catalog_key"], e["academic_year"]) for e in editions}, expected)
        self.assertTrue(editions)

    def test_selected_catalog_is_transient_and_normal_dsba_query_does_not_leak(self):
        response, execution = self.request(RESOLUTION)
        result = response.json()
        self.assertEqual(result["status"], "answer")
        self.assertTrue(result["provenance"])
        sides = execution.call_args.args[1].comparison_sides
        self.assertEqual((sides[0].scope.program, sides[0].scope.catalog_key), ("DSBA", "dsba-2565"))
        self.assertEqual((sides[1].scope.program, sides[1].scope.catalog_key), ("IT", "it-2565"))
        self.assertEqual(result["next_context"], CONTEXT)
        self.assertNotIn("clarification_target", result)
        data = payload(None)
        data.update(task="list", subject="course", comparison=None)
        data["scope"].update(program="DSBA", catalog=None, year=2, semester=1)
        normal, _ = self.request(data=data, question="DSBA ปี 2 เทอม 1 มีวิชาอะไรบ้าง", context=result["next_context"])
        self.assertEqual(normal.json()["status"], "answer")
        self.assertEqual(normal.json()["next_context"]["catalog_key"], "dsba-2565")
        self.assertTrue(all(r["program"] == "DSBA" for r in normal.json()["next_context"]["result_courses"]))

    def test_wrong_program_catalog_and_mismatched_target_fail_closed(self):
        for resolution in ({**RESOLUTION, "value": "dsba-2565"},
                           {**RESOLUTION, "value": "it-9999"},
                           {**RESOLUTION, "program": "DSBA"},
                           {**RESOLUTION, "operand": "left"}):
            with self.subTest(resolution=resolution):
                response, execution = self.request(resolution)
                self.assertEqual(response.json()["status"], "insufficient_evidence")
                self.assertEqual(response.json()["next_context"], CONTEXT)
                self.assertEqual(response.json()["provenance"], [])
                execution.assert_not_called()

    def test_explicit_catalog_cannot_be_overwritten(self):
        data = catalog_payload()
        data["comparison"]["right"]["catalog"] = "it-2560"
        response, execution = self.request(RESOLUTION, data, QUESTION + " หลักสูตร it-2560")
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertEqual(response.json()["next_context"], CONTEXT)
        execution.assert_not_called()

    def test_left_operand_catalog_uses_same_contract(self):
        context = {"program": "IT", "catalog_key": "it-2565", "plan": "coop"}
        target = {"dimension": "catalog", "program": "DSBA", "operand": "left"}
        initial, _ = self.request(context=context)
        self.assertEqual(initial.json()["clarification_target"], target)
        answer, execution = self.request({**target, "value": "dsba-2565"}, context=context)
        self.assertEqual(answer.json()["status"], "answer")
        self.assertEqual(execution.call_args.args[1].comparison_sides[0].scope.catalog_key, "dsba-2565")
        self.assertEqual(answer.json()["next_context"], context)

    def test_malformed_resolution_and_backward_compatibility(self):
        self.assertIsNone(AskRequest(question=QUESTION).clarification_resolution)
        for value in ([], {**RESOLUTION, "dimension": "year"}, {**RESOLUTION, "value": 2565},
                      {**RESOLUTION, "operand": "middle"}, {**RESOLUTION, "extra": True}):
            with self.subTest(value=value):
                response, execution = self.request(value)
                self.assertEqual(response.status_code, 422)
                execution.assert_not_called()
