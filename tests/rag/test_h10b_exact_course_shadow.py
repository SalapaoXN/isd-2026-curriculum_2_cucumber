import json
import unittest
from dataclasses import replace
from pathlib import Path

from rag.intent_gate import run_exact_course_shadow
from rag.intent_interpreter import parse_intent_payload
from rag.qa import _classify_structured_parse_completeness, ask
from rag.query_spec import parse_query_spec
from rag.resolution import resolve_query_spec


DB_PATH = Path(__file__).resolve().parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"


def _payload(intent, requested_facts, *, course_codes=None, **overrides):
    value = {
        "intent": intent,
        "proposed_program": "DSBA",
        "proposed_plans": [],
        "proposed_years": [],
        "proposed_semesters": [],
        "course_codes": list(course_codes or ("06026212",)),
        "topic": None,
        "requested_facts": list(requested_facts),
        "judgement_dimension": None,
        "unresolved": [],
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


def _incomplete_exact_course(question):
    spec = parse_query_spec("DSBA 06026212 มีหน่วยกิตเท่าไร")
    spec = replace(spec, operations=(), normalized_question=question)
    resolution = resolve_query_spec(spec, DB_PATH)
    completeness = _classify_structured_parse_completeness(spec, resolution)
    completeness = replace(completeness, classification="unrecognized_structured")
    return spec, resolution, completeness


class ExactCourseShadowTest(unittest.TestCase):
    def test_prerequisite_shadow_validates_one_call_and_never_executes(self):
        spec, resolution, completeness = _incomplete_exact_course(
            "DSBA 06026212 ขอข้อมูลวิชาบังคับก่อน"
        )
        calls = []
        result = run_exact_course_shadow(
            "DSBA 06026212 ขอข้อมูลวิชาบังคับก่อน",
            spec,
            completeness,
            resolution,
            None,
            lambda prompt: calls.append(prompt) or _payload(
                "prerequisite_query", ("prerequisite",)
            ),
            interpret_callable=lambda question, model: parse_intent_payload(
                model(question)
            ),
            family="prerequisite_query",
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.status, "validated")
        self.assertEqual(result.compiled_spec.operations, ("prerequisite",))
        self.assertFalse(result.executed)

    def test_description_shadow_validates_one_call_and_never_executes(self):
        spec, resolution, completeness = _incomplete_exact_course(
            "DSBA 06026212 รายละเอียดวิชา"
        )
        calls = []
        result = run_exact_course_shadow(
            "DSBA 06026212 รายละเอียดวิชา",
            spec,
            completeness,
            resolution,
            None,
            lambda prompt: calls.append(prompt) or _payload(
                "course_description", ("course_description",)
            ),
            interpret_callable=lambda question, model: parse_intent_payload(
                model(question)
            ),
            family="course_description",
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.status, "validated")
        self.assertEqual(result.compiled_spec.operations, ("describe",))
        self.assertFalse(result.executed)

    def test_wrong_family_target_conflict_and_malformed_payload_fail_closed(self):
        spec, resolution, completeness = _incomplete_exact_course(
            "DSBA 06026212 ขอข้อมูลวิชาบังคับก่อน"
        )
        proposals = (
            _payload("course_description", ("course_description",)),
            _payload("prerequisite_query", ("prerequisite",), course_codes=["06000000"]),
            "not json",
        )
        for proposal in proposals:
            result = run_exact_course_shadow(
                "DSBA 06026212 ขอข้อมูลวิชาบังคับก่อน",
                spec,
                completeness,
                resolution,
                None,
                lambda prompt, proposal=proposal: proposal,
                interpret_callable=lambda question, model: parse_intent_payload(
                    model(question)
                ),
                family="prerequisite_query",
            )
            self.assertNotEqual(result.status, "validated")
            self.assertFalse(result.executed)

    def test_complete_or_semantic_queries_make_zero_shadow_calls(self):
        cases = (
            "DSBA วิชา 06026212 มีวิชาบังคับก่อนคืออะไร",
            "DSBA มีวิชาเกี่ยวกับ database อะไรบ้าง",
            "DSBA 06026212 เรียนช่วงไหน",
        )
        for question in cases:
            spec = parse_query_spec(question)
            resolution = resolve_query_spec(spec, DB_PATH)
            completeness = _classify_structured_parse_completeness(spec, resolution)
            calls = []
            result = run_exact_course_shadow(
                question,
                spec,
                completeness,
                resolution,
                None,
                lambda prompt: calls.append(prompt),
                family="prerequisite_query",
            )
            self.assertEqual(calls, [], question)
            self.assertFalse(result.attempted, question)

    def test_shadow_on_preserves_final_result_shape(self):
        question = "DSBA 06026212 รายละเอียดวิชา"
        off = ask(str(DB_PATH), question, shadow_intent=False)
        on = ask(
            str(DB_PATH),
            question,
            shadow_intent=True,
            intent_model_callable=lambda prompt: _payload(
                "course_description", ("course_description",)
            ),
        )
        self.assertEqual(off["result"], on["result"])


if __name__ == "__main__":
    unittest.main()
