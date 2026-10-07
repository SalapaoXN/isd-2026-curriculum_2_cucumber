"""Bounded accumulated resolutions for one original comparison request."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.main import app
from rag.semantic.executor import execute_comparison
from tests.rag.test_semantic_clarification_transport import DB, payload


QUESTION = "เปรียบเทียบหน่วยกิตหลักสูตร DSBA กับ IT"
LEFT = {"dimension": "catalog", "program": "DSBA", "operand": "left", "value": "dsba-2565"}
RIGHT = {"dimension": "catalog", "program": "IT", "operand": "right", "value": "it-2565"}


def intent_payload(term=False):
    data = payload(None, "difference")
    for name, program in (("left", "DSBA"), ("right", "IT")):
        data["comparison"][name].update(program=program, catalog=None, plan=None, plan_hint=None,
                                        year=1 if term else None, semester=1 if term else None)
    if term:
        data["subject"] = "semester"
    return data


class ClarificationChainTests(unittest.TestCase):
    def request(self, resolutions=None, data=None, question=QUESTION, singular=None):
        body = {"question": question, "home_program": None, "conversation_context": None}
        if resolutions is not None:
            body["clarification_resolutions"] = deepcopy(resolutions)
        if singular is not None:
            body["clarification_resolution"] = singular
        raw = json.dumps(data if data is not None else intent_payload())
        with patch("backend.main.active_qa_mode", return_value="semantic"), \
             patch("backend.main._curriculum_db", return_value=DB), \
             patch("backend.main._semantic_providers", return_value={
                 "interpret_callable": lambda prompt: raw, "answer_callable": None, "sql_callable": None,
             }), patch("rag.semantic.pipeline.execute_comparison", wraps=execute_comparison) as execution:
            response = TestClient(app).post("/api/ask", json=body)
        return response, execution

    def test_two_catalog_chain_preserves_first_choice_and_unscoped_context(self):
        initial, execution = self.request()
        self.assertEqual(initial.json()["clarification_target"], {key: value for key, value in LEFT.items() if key != "value"})
        execution.assert_not_called()
        first, execution = self.request([LEFT])
        self.assertEqual(first.json()["clarification_target"], {key: value for key, value in RIGHT.items() if key != "value"})
        execution.assert_not_called()
        final, execution = self.request([LEFT, RIGHT])
        self.assertEqual(final.json()["status"], "answer")
        self.assertTrue(final.json()["provenance"])
        sides = execution.call_args.args[1].comparison_sides
        self.assertEqual((sides[0].scope.catalog_key, sides[1].scope.catalog_key), ("dsba-2565", "it-2565"))
        self.assertIsNone(final.json()["next_context"])
        self.assertNotIn("clarification_target", final.json())

    def test_catalogs_then_plans_preserve_all_four_selections(self):
        question = "เปรียบเทียบหน่วยกิต DSBA ปี 1 เทอม 1 กับ IT ปี 1 เทอม 1"
        chain = [LEFT, RIGHT,
                 {"dimension": "plan", "program": "DSBA", "operand": "left", "value": "coop"},
                 {"dimension": "plan", "program": "IT", "operand": "right", "value": "no_coop"}]
        for count in range(4):
            response, execution = self.request(chain[:count], intent_payload(True), question)
            self.assertEqual(response.json()["clarification_target"], {k: v for k, v in chain[count].items() if k != "value"})
            execution.assert_not_called()
        final, execution = self.request(chain, intent_payload(True), question)
        self.assertEqual(final.json()["status"], "answer")
        sides = execution.call_args.args[1].comparison_sides
        self.assertEqual((sides[0].scope.plan, sides[1].scope.plan), ("coop", "no_coop"))
        self.assertEqual((sides[0].scope.catalog_key, sides[1].scope.catalog_key), ("dsba-2565", "it-2565"))
        self.assertIsNone(final.json()["next_context"])

    def test_input_order_does_not_reorder_deterministic_preflight(self):
        final, execution = self.request([RIGHT, LEFT])
        self.assertEqual(final.json()["status"], "answer")
        self.assertEqual(execution.call_args.args[1].comparison_sides[0].scope.catalog_key, "dsba-2565")
        duplicate, _ = self.request([LEFT, LEFT, RIGHT])
        self.assertEqual(duplicate.json()["status"], "answer")

    def test_conflicts_wrong_ownership_and_unneeded_plans_fail_closed(self):
        for chain in (
            [LEFT, {**LEFT, "value": "dsba-2560"}, RIGHT],
            [{**LEFT, "value": "it-2565"}, RIGHT],
            [{**LEFT, "program": "IT"}, RIGHT],
            [LEFT, RIGHT, {"dimension": "plan", "program": "DSBA", "operand": "left", "value": "coop"}],
        ):
            with self.subTest(chain=chain):
                response, execution = self.request(chain)
                self.assertEqual(response.json()["status"], "insufficient_evidence")
                self.assertEqual(response.json()["provenance"], [])
                execution.assert_not_called()

    def test_explicit_catalog_and_plan_cannot_be_overridden(self):
        data = intent_payload()
        data["comparison"]["left"]["catalog"] = "dsba-2560"
        response, execution = self.request([LEFT, RIGHT], data,
            "เปรียบเทียบหน่วยกิต DSBA หลักสูตร dsba-2560 กับ IT")
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        execution.assert_not_called()
        data = intent_payload(True)
        data["comparison"]["left"].update(plan="coop", plan_hint="coop")
        resolutions = [LEFT, RIGHT, {"dimension": "plan", "program": "DSBA", "operand": "left", "value": "no_coop"}]
        response, execution = self.request(resolutions, data,
            "เปรียบเทียบหน่วยกิต DSBA แผน coop ปี 1 เทอม 1 กับ IT ปี 1 เทอม 1")
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        execution.assert_not_called()

    def test_malformed_and_ambiguous_transport_rejected(self):
        for resolutions in ("catalogs", [None], [{**LEFT, "operand": "middle"}],
                            [{**LEFT, "dimension": "year"}], [{**LEFT, "value": 2565}],
                            [{**LEFT, "extra": True}], [LEFT] * 5):
            with self.subTest(resolutions=resolutions):
                response, execution = self.request(resolutions)
                self.assertEqual(response.status_code, 422)
                execution.assert_not_called()
        response, execution = self.request([LEFT], singular=LEFT)
        self.assertEqual(response.status_code, 422)
        execution.assert_not_called()

    def test_singular_retry_is_still_supported(self):
        response, execution = self.request(singular=LEFT)
        self.assertEqual(response.json()["clarification_target"]["program"], "IT")
        self.assertEqual(response.json()["clarification_target"]["operand"], "right")
        execution.assert_not_called()
