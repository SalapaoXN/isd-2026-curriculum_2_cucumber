"""One-shot operand plan transport through the real API and semantic pipeline."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import AskRequest
from rag.semantic.executor import execute_comparison
from tests.rag.test_semantic_clarification_transport import DB, CONTEXT, payload


QUESTION = "เปรียบเทียบหน่วยกิต IT หลักสูตร it-2565 แผน coop ปี 1 เทอม 1 กับ DSBA หลักสูตร dsba-2565 ปี 1 เทอม 1"
TARGET = {"dimension": "plan", "program": "DSBA", "operand": "right"}
RESOLUTION = {**TARGET, "value": "no_coop"}


def comparison_payload(left_plan="coop", right_plan=None):
    data = payload(None, "difference")
    data["subject"] = "semester"
    for side, plan in (("left", left_plan), ("right", right_plan)):
        data["comparison"][side].update(plan=plan, plan_hint=plan, year=1, semester=1)
    return data


class OperandPlanRetryTests(unittest.TestCase):
    def request(self, data=None, resolution=None, question=QUESTION):
        context = deepcopy(CONTEXT)
        body = {"question": question, "home_program": "IT", "conversation_context": context}
        if resolution is not None:
            body["clarification_resolution"] = resolution
        raw = json.dumps(data if data is not None else comparison_payload())
        with patch("backend.main.active_qa_mode", return_value="semantic"), \
             patch("backend.main._curriculum_db", return_value=DB), \
             patch("backend.main._semantic_providers", return_value={
                 "interpret_callable": lambda prompt: raw,
                 "answer_callable": None, "sql_callable": None,
             }), patch("rag.semantic.pipeline.execute_comparison", wraps=execute_comparison) as execution:
            response = TestClient(app).post("/api/ask", json=body)
        self.assertEqual(context, CONTEXT)
        return response, execution

    def test_response_metadata_comes_from_right_side_preflight(self):
        response, execution = self.request()
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["action"], "plan_required")
        self.assertEqual(result["clarification_target"], TARGET)
        self.assertEqual(result["next_context"], CONTEXT)
        self.assertEqual(result["provenance"], [])
        execution.assert_not_called()

    def test_valid_right_retry_is_canonical_and_transient(self):
        response, execution = self.request(resolution=RESOLUTION)
        result = response.json()
        self.assertEqual(result["status"], "answer")
        self.assertTrue(result["provenance"])
        self.assertIn("IT แผนสหกิจ", result["answer"])
        self.assertIn("DSBA แผนไม่สหกิจ", result["answer"])
        self.assertNotIn("coop", result["answer"])
        self.assertEqual(result["answer"].count(": 18 หน่วยกิต"), 2)
        self.assertIn("มีผลต่าง 0 หน่วยกิต", result["answer"])
        execution.assert_called_once()
        sides = execution.call_args.args[1].comparison_sides
        self.assertEqual((sides[0].scope.program, sides[0].scope.plan), ("IT", "coop"))
        self.assertEqual((sides[1].scope.program, sides[1].scope.plan), ("DSBA", "no_coop"))
        self.assertEqual(result["next_context"], CONTEXT)
        self.assertNotIn("clarification_resolution", result["next_context"])
        self.assertNotIn("clarification_target", result)

    def test_left_retry_targets_only_missing_left_operand(self):
        question = "เปรียบเทียบหน่วยกิต IT หลักสูตร it-2565 ปี 1 เทอม 1 กับ DSBA หลักสูตร dsba-2565 แผน coop ปี 1 เทอม 1"
        data = comparison_payload(None, "coop")
        target = {"dimension": "plan", "program": "IT", "operand": "left"}
        initial, _ = self.request(data, question=question)
        self.assertEqual(initial.json()["clarification_target"], target)
        response, execution = self.request(data, {**target, "value": "no_coop"}, question)
        self.assertEqual(response.json()["status"], "answer")
        sides = execution.call_args.args[1].comparison_sides
        self.assertEqual((sides[0].scope.plan, sides[1].scope.plan), ("no_coop", "coop"))
        self.assertEqual(response.json()["next_context"], CONTEXT)

    def test_explicit_operand_plan_cannot_be_overridden(self):
        question = QUESTION.replace("dsba-2565 ปี", "dsba-2565 แผน coop ปี")
        response, execution = self.request(comparison_payload("coop", "coop"), RESOLUTION, question)
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertEqual(response.json()["provenance"], [])
        self.assertEqual(response.json()["next_context"], CONTEXT)
        execution.assert_not_called()

    def test_wrong_target_and_noncanonical_value_fail_closed(self):
        for resolution in (
            {**RESOLUTION, "program": "IT", "operand": "left"},
            {**RESOLUTION, "program": "IT"},
            {**RESOLUTION, "operand": "left"},
            {**RESOLUTION, "program": "NOT_A_PROGRAM"},
            {**RESOLUTION, "value": "NOT_A_PLAN"},
        ):
            with self.subTest(resolution=resolution):
                response, execution = self.request(resolution=resolution)
                self.assertEqual(response.json()["status"], "insufficient_evidence")
                self.assertEqual(response.json()["provenance"], [])
                self.assertEqual(response.json()["next_context"], CONTEXT)
                execution.assert_not_called()

    def test_malformed_resolution_rejected_at_http_boundary(self):
        for resolution in (
            [], "right", {**RESOLUTION, "dimension": "year"},
            {**RESOLUTION, "operand": "middle"}, {**RESOLUTION, "value": 3},
            {**RESOLUTION, "program": True}, {**RESOLUTION, "extra": "ignored"},
            {**RESOLUTION, "value": "x" * 81},
            {key: value for key, value in RESOLUTION.items() if key != "value"},
        ):
            with self.subTest(resolution=resolution):
                response, execution = self.request(resolution=resolution)
                self.assertEqual(response.status_code, 422)
                execution.assert_not_called()

    def test_normal_request_cannot_consume_operand_resolution(self):
        data = payload(None)
        data.update(task="list", subject="course", comparison=None)
        data["scope"].update(program="IT", catalog="it-2565", year=2, semester=1)
        response, execution = self.request(data, RESOLUTION, "IT หลักสูตร it-2565 ปี 2 เทอม 1 มีวิชาอะไรบ้าง")
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        execution.assert_not_called()

    def test_old_request_and_next_normal_it_query_are_unchanged(self):
        self.assertIsNone(AskRequest(question="IT courses").clarification_resolution)
        retry, _ = self.request(resolution=RESOLUTION)
        self.assertEqual(retry.json()["next_context"], CONTEXT)
        data = payload(None)
        data.update(task="list", subject="course", comparison=None)
        data["scope"].update(program="IT", catalog="it-2565", year=2, semester=1)
        response, _ = self.request(data, question="IT หลักสูตร it-2565 ปี 2 เทอม 1 มีวิชาอะไรบ้าง")
        self.assertEqual(response.json()["status"], "answer")
        self.assertEqual(response.json()["next_context"]["plan"], "coop")
        self.assertNotIn("clarification_target", response.json())
