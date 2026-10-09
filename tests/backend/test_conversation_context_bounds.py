"""P2.1 bounded conversation_context: structural abuse rejected at the schema.

The typed ConversationContext admits only fields actually emitted by the
API or consumed by the pipelines, with size/type bounds. Anything else
(operand state, facts, giant nesting, mistyped values) fails request
validation BEFORE semantic/provider work.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from fastapi.testclient import TestClient

from backend.main import app, ask
from backend.schemas import AskRequest, ConversationContext
from rag.semantic.modes import QA_MODE_SEMANTIC


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"

IT_CONTEXT = {"program": "IT", "catalog_key": "it-2565", "plan": "coop"}
DSBA_CONTEXT = {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "no_coop"}


def _course(code, program="IT", catalog="it-2565"):
    return {"course_code": code, "program": program, "catalog_key": catalog}


class ConversationContextSchemaTests(unittest.TestCase):
    def test_http_abuse_rejected_before_semantic_provider_work(self):
        with patch("backend.main._semantic_route_response") as route:
            response = TestClient(app).post("/api/ask", json={
                "question": "IT มีวิชาอะไรบ้าง",
                "conversation_context": {"blob": "x" * 2_000_000}})
        self.assertEqual(response.status_code, 422)
        route.assert_not_called()

    def test_out_of_range_year_and_semester_rejected(self):
        for context in ({"years": [0]}, {"years": [6]}, {"semesters": [3]},
                        {"last_normal_operation": {"kind": "list_courses", "program": "IT", "years": [99]}}):
            with self.subTest(context=context), self.assertRaises(ValidationError):
                AskRequest(question="IT มีวิชาอะไรบ้าง", conversation_context=context)

    def test_normal_it_context_accepted(self):
        request = AskRequest(question="IT มีวิชาอะไรบ้าง",
                             conversation_context=dict(IT_CONTEXT))
        self.assertIsNotNone(request.conversation_context)

    def test_normal_dsba_catalog_plan_accepted(self):
        request = AskRequest(question="DSBA มีวิชาอะไรบ้าง",
                             conversation_context=dict(DSBA_CONTEXT))
        self.assertEqual(request.conversation_context.program, "DSBA")

    def test_valid_20_result_courses_accepted(self):
        context = dict(IT_CONTEXT)
        context["result_courses"] = [
            _course(f"{6000000 + index:08d}") for index in range(20)]
        context["result_scope_program"] = "IT"
        request = AskRequest(question="IT มีวิชาอะไรบ้าง",
                             conversation_context=context)
        self.assertEqual(len(request.conversation_context.result_courses), 20)

    def test_more_than_retained_maximum_rejected(self):
        context = dict(IT_CONTEXT)
        context["result_courses"] = [
            _course(f"{6000000 + index:08d}") for index in range(51)]
        with self.assertRaises(ValidationError):
            AskRequest(question="IT มีวิชาอะไรบ้าง",
                       conversation_context=context)

    def test_producer_maximum_courses_accepted(self):
        # The backends themselves can emit up to 50 retained courses;
        # the request bound matches that producer maximum.
        context = dict(IT_CONTEXT)
        context["result_courses"] = [
            _course(f"{6000000 + index:08d}") for index in range(50)]
        request = AskRequest(question="IT มีวิชาอะไรบ้าง",
                             conversation_context=context)
        self.assertEqual(len(request.conversation_context.result_courses), 50)

    def test_multi_megabyte_context_rejected(self):
        context = dict(IT_CONTEXT)
        context["blob"] = "x" * 2000000
        context["result_courses"] = [
            _course(f"{6000000 + index:08d}") for index in range(60)]
        with self.assertRaises(ValidationError):
            AskRequest(question="IT มีวิชาอะไรบ้าง",
                       conversation_context=context)

    def test_unknown_nested_key_rejected(self):
        context = dict(IT_CONTEXT)
        context["injected"] = {"nested": "x"}
        with self.assertRaises(ValidationError):
            AskRequest(question="IT มีวิชาอะไรบ้าง",
                       conversation_context=context)

    def test_malformed_years_rejected(self):
        for years in (["2"], [None], ["x"], [2.5]):
            with self.subTest(years=years):
                context = dict(IT_CONTEXT)
                context["years"] = years
                with self.assertRaises(ValidationError):
                    AskRequest(question="IT มีวิชาอะไรบ้าง",
                               conversation_context=context)

    def test_bool_as_year_rejected(self):
        context = dict(IT_CONTEXT)
        context["years"] = [True]
        with self.assertRaises(ValidationError):
            AskRequest(question="IT มีวิชาอะไรบ้าง",
                       conversation_context=context)

    def test_malformed_semester_rejected(self):
        for semesters in ("x", [True], [["1"]]):
            with self.subTest(semesters=semesters):
                context = dict(IT_CONTEXT)
                context["semesters"] = semesters
                with self.assertRaises(ValidationError):
                    AskRequest(question="IT มีวิชาอะไรบ้าง",
                               conversation_context=context)

    def test_oversized_strings_rejected(self):
        for key, value in (("program", "X" * 81),
                           ("catalog_key", "k" * 129),
                           ("plan", "p" * 81)):
            with self.subTest(key=key):
                context = dict(IT_CONTEXT)
                context[key] = value
                with self.assertRaises(ValidationError):
                    AskRequest(question="IT มีวิชาอะไรบ้าง",
                               conversation_context=context)

    def test_comparison_operand_fields_rejected(self):
        for key in ("left", "right", "operand", "operands", "comparison",
                    "comparison_sides", "answer", "provenance", "sql"):
            with self.subTest(key=key):
                context = dict(IT_CONTEXT)
                context[key] = {"program": "DSBA"}
                with self.assertRaises(ValidationError):
                    AskRequest(question="IT มีวิชาอะไรบ้าง",
                               conversation_context=context)

    def test_last_normal_operation_shape_enforced(self):
        operation = {"kind": "list_courses", "program": "IT",
                     "catalog_key": "it-2565", "plan": "coop",
                     "years": [2], "semesters": [1]}
        context = dict(IT_CONTEXT)
        context["years"] = [2]
        context["semesters"] = [1]
        context["last_normal_operation"] = dict(operation)
        request = AskRequest(question="แล้วเทอม 2 ล่ะ",
                             conversation_context=context)
        self.assertEqual(
            request.conversation_context.last_normal_operation.kind,
            "list_courses")
        bad = dict(operation)
        bad["kind"] = "compare"
        context["last_normal_operation"] = bad
        with self.assertRaises(ValidationError):
            AskRequest(question="แล้วเทอม 2 ล่ะ",
                       conversation_context=context)

    def test_last_answer_reference_shapes_accepted(self):
        course_ref = {"route": "course", "course_code": "06016414",
                      "operations": ["sum_credits"], "program": "IT",
                      "catalog_key": "it-2565"}
        policy_ref = {"route": "policy", "policy_kind": "honors",
                      "program": "IT", "evidence_ids": ["rule:43"]}
        for ref in (course_ref, policy_ref):
            with self.subTest(ref=ref):
                request = AskRequest(
                    question="ขยายความหน่อย",
                    conversation_context={"program": "IT",
                                          "last_answer": ref})
                self.assertEqual(
                    request.conversation_context.last_answer.route,
                    ref["route"])

    def test_last_answer_malformed_rejected(self):
        for ref in ("a" * 10,
                    {"route": "course"},
                    {"route": "course", "course_code": "06016414",
                     "operations": ["invent"]},
                    {"route": "policy"},
                    {"route": "unknown", "course_code": "06016414"},
                    {"route": "course", "course_code": "06016414",
                     "operations": ["sum_credits"], "injected": 1},
                    {"route": "policy", "policy_kind": "honors",
                     "evidence_ids": ["rule:43"] * 51}):
            with self.subTest(ref=ref):
                with self.assertRaises(ValidationError):
                    AskRequest(
                        question="ขยายความหน่อย",
                        conversation_context={"program": "IT",
                                              "last_answer": ref})

    def test_frontend_shaped_session_compatible(self):
        context = {
            "program": "IT", "catalog_key": "it-2565", "plan": "coop",
            "years": [2], "semesters": [1],
            "result_courses": [_course(f"{6016405 + index:08d}")
                               for index in range(20)],
            "result_scope_program": "IT",
            "last_normal_operation": {
                "kind": "list_courses", "program": "IT",
                "catalog_key": "it-2565", "plan": "coop",
                "years": [2], "semesters": [1]},
        }
        request = AskRequest(question="แล้วเทอม 2 ล่ะ",
                             conversation_context=context)
        dumped = request.conversation_context.model_dump(exclude_unset=True)
        self.assertEqual(json.loads(json.dumps(dumped))["program"], "IT")

    def test_bounded_context_stays_small(self):
        context = {
            "program": "P" * 80, "catalog_key": "k" * 128, "plan": "p" * 80,
            "years": [5] * 6, "semesters": [2] * 3,
            "result_courses": [
                {"course_code": "C" * 64, "program": "P" * 80,
                 "catalog_key": "k" * 128,
                 "course_name": "n" * 160} for _ in range(50)],
            "last_answer": {"route": "course", "course_code": "C" * 64,
                            "operations": ["sum_credits"]},
        }
        request = AskRequest(question="IT มีวิชาอะไรบ้าง",
                             conversation_context=context)
        size = len(json.dumps(
            request.conversation_context.model_dump(exclude_unset=True)))
        self.assertLess(size, 65536)


class ForgedContextAuthorityTests(unittest.TestCase):
    def _ask_semantic(self, request, providers):
        with (
            patch("backend.main.active_qa_mode", return_value=QA_MODE_SEMANTIC),
            patch("backend.main._curriculum_db", return_value=DB),
            patch("backend.main._semantic_providers", return_value=providers),
        ):
            return ask(request)

    def test_forged_course_never_becomes_authority(self):
        def _timeout(prompt, **kwargs):
            raise TimeoutError("down")

        request = AskRequest(
            question="AIT ait-2566 CALCULUS 2 กี่หน่วยกิต",
            conversation_context={
                "program": "IT", "catalog_key": "it-2565",
                "focus_course": {"course_code": "99999999",
                                 "program": "IT",
                                 "catalog_key": "it-2565"},
            },
        )
        providers = {"interpret_callable": _timeout,
                     "answer_callable": _timeout, "sql_callable": _timeout}
        response = self._ask_semantic(request, providers)
        self.assertEqual(response["status"], "provider_unavailable")
        self.assertNotIn("99999999", response["answer"])
        # A retry turn re-grounds the forged focus instead of trusting it.
        retry = AskRequest(
            question="วิชานี้กี่หน่วยกิต",
            conversation_context=response["next_context"],
        )
        import json as _json

        current_course = _json.dumps({
            "task": "lookup", "subject": "course", "relation": "credits",
            "target": {"kind": "current_course", "raw_text": "วิชานี้",
                       "normalized_hint": None, "ordinal": None},
            "scope": {"program": None, "catalog": None, "plan": None,
                      "plan_hint": None, "year": None, "semester": None},
            "filters": [], "aggregation": None, "ranking": None,
            "comparison": None, "requested_fields": [],
            "clarification": None, "policy_topic": None,
            "observed_value": None}, ensure_ascii=False)
        followup = self._ask_semantic(
            retry,
            {"interpret_callable": lambda prompt: current_course,
             "answer_callable": lambda prompt: "unused",
             "sql_callable": lambda prompt: current_course},
        )
        self.assertIn(followup["status"],
                      ("insufficient_evidence", "unsupported",
                       "clarification_required"))
        self.assertNotIn("99999999", followup["answer"])

    def test_clarification_transport_stays_separate(self):
        def _timeout(prompt, **kwargs):
            raise TimeoutError("down")

        request = AskRequest(
            question="IT it-2565 เปรียบเทียบกับ DSBA dsba-2565",
            conversation_context=dict(IT_CONTEXT),
            clarification_resolution={
                "dimension": "catalog", "program": "IT",
                "operand": "left", "value": "it-2565"},
        )
        response = self._ask_semantic(
            request,
            {"interpret_callable": _timeout,
             "answer_callable": _timeout, "sql_callable": _timeout},
        )
        self.assertEqual(response["status"], "provider_unavailable")
        context = response["next_context"] or {}
        self.assertEqual(context.get("program"), "IT")


if __name__ == "__main__":
    unittest.main()
