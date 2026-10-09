"""Replay the live planning omission at the interpretation/validation boundary."""

import json
import unittest
from unittest.mock import patch

from rag.semantic.interpreter import parse_semantic_intent_payload
from rag.semantic.modes import semantic_ask_response
from rag.semantic.prompts import build_semantic_interpreter_prompt
from rag.semantic.validation import validate_semantic_intent
from rag.semantic.executor import execute_mixed_scope
from tests.rag.test_semantic_mixed_scope_composition import DB, payload

QUESTION = "ถ้าจะลง INFRASTRUCTURE SYSTEMS AND SERVICES (06016420) ใน IT แบบไม่สหกิจปี 2 เทอม 2 ต้องเตรียมผ่านวิชาอะไรในเทอมก่อนหน้า และเทอมนี้มีหน่วยกิตรวมเท่าไร?"
CONTEXT = {"program": "IT", "catalog_key": "it-2565", "plan": "no_coop"}


def captured_payload():
    data = payload(("prerequisites", "credits"), "prerequisite")
    data["scope"].update(plan="แบบไม่สหกิจ", plan_hint="no_coop")
    return data


def parsed(data):
    return parse_semantic_intent_payload(json.dumps(data))


class MixedScopePlanningContractTests(unittest.TestCase):
    def test_exact_live_interpretation_cannot_pass_as_complete_planning(self):
        validation = validate_semantic_intent(parsed(captured_payload()), QUESTION)
        self.assertFalse(validation.valid)
        self.assertIn("planning", validation.reason)

    def test_incomplete_live_replay_fails_before_execution(self):
        with patch("rag.semantic.pipeline.execute_mixed_scope", wraps=execute_mixed_scope) as execute:
            response = semantic_ask_response(DB, QUESTION, CONTEXT,
                interpret_callable=lambda prompt: json.dumps(captured_payload()))
        self.assertNotEqual(response["status"], "answer")
        execute.assert_not_called()

    def test_dependency_placement_requires_explicit_target_placement(self):
        data = payload(("prerequisites", "prerequisite_placement"))
        validation = validate_semantic_intent(parsed(data), "IT no_coop 06016420 year 2 semester 2 prerequisites and term credits?")
        self.assertFalse(validation.valid)

    def test_prior_term_planning_requires_both_placement_roles(self):
        for fields in (("prerequisites", "placement"), ("prerequisites", "prerequisite_placement"), ("prerequisites",)):
            data = captured_payload()
            data["requested_fields"] = list(fields)
            with self.subTest(fields=fields):
                self.assertFalse(validate_semantic_intent(parsed(data), QUESTION).valid)

    def test_invented_planning_wordings_use_same_contract(self):
        questions = (
            "IT no_coop 06016420 year 2 semester 2: prerequisites in the preceding semester and term total credits?",
            "IT no_coop 06016420 year 2 semester 2: prerequisites in a previous term and term total credits?",
            "IT no_coop 06016420 ปี 2 เทอม 2 ต้องผ่านวิชาในภาคก่อน แล้วรวมหน่วยกิตเท่าไร?",
        )
        data = payload(("prerequisites", "credits"))
        for question in questions:
            with self.subTest(question=question):
                self.assertFalse(validate_semantic_intent(parsed(data), question).valid)

    def test_plain_prerequisite_and_total_request_keeps_existing_shape(self):
        data = payload(("prerequisites",))
        self.assertTrue(validate_semantic_intent(parsed(data),
            "IT no_coop 06016420 year 2 semester 2: prerequisites and term total credits?").valid)

    def test_complete_planning_interpretation_is_valid(self):
        data = captured_payload()
        data["requested_fields"] = ["prerequisites", "prerequisite_placement", "placement"]
        self.assertTrue(validate_semantic_intent(parsed(data), QUESTION).valid)

    def test_complete_replay_public_answer_keeps_every_owned_fact(self):
        data = captured_payload()
        data["requested_fields"] = ["prerequisites", "prerequisite_placement", "placement"]
        def forbidden(prompt):
            self.fail("mixed-scope answer presentation called a provider")
        response = semantic_ask_response(DB, QUESTION, CONTEXT,
            interpret_callable=lambda prompt: json.dumps(data), answer_callable=forbidden)
        self.assertEqual(response["status"], "answer")
        lines = response["answer"].splitlines()
        self.assertTrue(any("06016420" in line and "มีวิชาบังคับก่อน: 06016413" in line for line in lines))
        self.assertTrue(any("06016420" in line and "ชั้นปีที่ 2" in line and "ภาคการศึกษาที่ 2" in line for line in lines))
        self.assertTrue(any("06016413" in line and "ชั้นปีที่ 2" in line and "ภาคการศึกษาที่ 1" in line for line in lines))
        self.assertIn("หน่วยกิตรวมทั้งเทอม: 30", response["answer"])
        self.assertTrue(response["provenance"])

    def test_prompt_teaches_planning_roles_without_benchmark_wording(self):
        prompt = build_semantic_interpreter_prompt("invented planning request")
        self.assertIn("PREREQUISITE PLANNING", prompt)
        self.assertIn("target placement is a required verified fact", prompt)
        self.assertIn("previous/preceding term", prompt)
        self.assertNotIn(QUESTION, prompt)
