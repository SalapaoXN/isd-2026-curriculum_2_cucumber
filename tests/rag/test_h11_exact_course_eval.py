import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import rag.qa as qa_module
from rag.intent_gate import run_exact_course_shadow
from rag.intent_interpreter import parse_intent_payload
from rag.query_spec import parse_query_spec
from rag.qa import _classify_structured_parse_completeness, ask
from rag.resolution import resolve_query_spec


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


def _payload(intent, facts, *, code="06026212", **overrides):
    value = {
        "intent": intent,
        "proposed_program": "DSBA",
        "proposed_plans": [],
        "proposed_years": [],
        "proposed_semesters": [],
        "course_codes": [code],
        "topic": None,
        "requested_facts": list(facts),
        "judgement_dimension": None,
        "unresolved": [],
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


class H11ExactCourseEvaluationTest(unittest.TestCase):
    def _shadow_inputs(self, question):
        parsed = parse_query_spec("DSBA 06026212 มีหน่วยกิตเท่าไร")
        spec = replace(parsed, operations=(), normalized_question=question)
        resolution = resolve_query_spec(spec, DB_PATH)
        completeness = _classify_structured_parse_completeness(spec, resolution)
        completeness = replace(completeness, classification="unrecognized_structured")
        return spec, resolution, completeness

    def test_bounded_families_have_perfect_offline_eligibility_and_one_call(self):
        cases = (
            ("DSBA 06026212 ขอข้อมูลวิชาบังคับก่อน", "prerequisite_query", ("prerequisite",), ("prerequisite",)),
            ("DSBA 06026212 prerequisite details", "prerequisite_query", ("prerequisite",), ("prerequisite",)),
            ("DSBA 06026212 รายละเอียดวิชา", "course_description", ("course_description",), ("describe",)),
            ("DSBA 06026212 course description", "course_description", ("course_description",), ("describe",)),
        )
        for question, family, facts, operations in cases:
            with self.subTest(question=question):
                spec, resolution, completeness = self._shadow_inputs(question)
                calls = []
                result = run_exact_course_shadow(
                    question,
                    spec,
                    completeness,
                    resolution,
                    None,
                    lambda prompt: calls.append(prompt) or _payload(family, facts),
                    interpret_callable=lambda q, model: parse_intent_payload(model(q)),
                    family=family,
                )
                self.assertTrue(result.eligible)
                self.assertEqual(result.status, "validated")
                self.assertEqual(result.compiled_spec.operations, operations)
                self.assertEqual(len(calls), 1)
                self.assertEqual(result.comparison, "compatible_extension")

    def test_zero_call_negative_fixture_set(self):
        negatives = (
            "DSBA 06026212 มีวิชาบังคับก่อนคืออะไร",
            "DSBA 06026212 รายละเอียดวิชาเรียนเกี่ยวกับอะไร",
            "DSBA 06026212 เรียนช่วงไหน",
            "IT ปี 3 มีรายวิชาทั้งหมดเท่าไหร่",
            "DSBA มีวิชาเกี่ยวกับ database อะไรบ้าง",
            "IT 06016414 กับ 06016419 ตัวไหนเรียนก่อน",
            "IT 06026212 ยากไหม",
            "IT จบการศึกษาต้องมีเกรดเท่าไร",
            "DSBA 06026212 กับ 06026213 ขอวิชาบังคับก่อน",
            "06026212 มีวิชาบังคับก่อนอะไรบ้าง",
            "ทุกวิชาที่มีวิชาบังคับก่อนมีอะไรบ้าง",
            "IT ขอข้อมูลวิชาแบบไม่ระบุรหัส",
        )
        for question in negatives:
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                resolution = resolve_query_spec(spec, DB_PATH)
                completeness = _classify_structured_parse_completeness(spec, resolution)
                calls = []
                for family in ("prerequisite_query", "course_description"):
                    result = run_exact_course_shadow(
                        question,
                        spec,
                        completeness,
                        resolution,
                        None,
                        lambda prompt: calls.append(prompt),
                        family=family,
                    )
                    self.assertFalse(result.attempted)
                self.assertEqual(calls, [])

    def test_bad_proposals_fail_closed_without_extra_calls(self):
        question = "DSBA 06026212 ขอข้อมูลวิชาบังคับก่อน"
        spec, resolution, completeness = self._shadow_inputs(question)
        proposals = (
            _payload("course_description", ("course_description",)),
            _payload("prerequisite_query", ("prerequisite",), code="06000000"),
            _payload("prerequisite_query", ("prerequisite",), proposed_program="IT"),
            _payload("prerequisite_query", ("course_list",)),
            _payload("prerequisite_query", ("prerequisite",), unresolved=["topic"]),
            "not json",
        )
        for proposal in proposals:
            with self.subTest(proposal=proposal):
                calls = []
                result = run_exact_course_shadow(
                    question,
                    spec,
                    completeness,
                    resolution,
                    None,
                    lambda prompt, proposal=proposal: calls.append(prompt) or proposal,
                    interpret_callable=lambda q, model: parse_intent_payload(model(q)),
                    family="prerequisite_query",
                )
                self.assertNotEqual(result.status, "validated")
                self.assertEqual(len(calls), 1)

    def test_gated_prerequisite_execution_uses_canonical_path_once(self):
        question = "DSBA 06026212 prerequisite"
        original = parse_query_spec("DSBA 06026212 มีหน่วยกิตเท่าไร")
        shadow_spec = replace(original, operations=(), normalized_question=question)
        answer_calls = []
        with patch.object(qa_module, "parse_query_spec", return_value=shadow_spec), \
             patch.object(
                 qa_module,
                 "_classify_structured_parse_completeness",
                 side_effect=lambda spec, resolution, context=None: replace(
                     _classify_structured_parse_completeness(spec, resolution, context),
                     classification=(
                         "unrecognized_structured"
                         if not tuple(getattr(spec, "operations", ()))
                         else "complete"
                     ),
                 ),
             ):
            result = ask(
                DB_PATH,
                question,
                shadow_intent=True,
                intent_model_callable=lambda prompt: _payload(
                    "prerequisite_query", ("prerequisite",)
                ),
                answer_model_callable=lambda prompt: answer_calls.append(prompt),
            )
        self.assertEqual(result["result"].status, "answer")
        self.assertEqual(len(answer_calls), 0)
        self.assertTrue(result["prerequisite_shadow"].executed)
        self.assertTrue(result["result"].provenance)

    def test_gated_description_execution_is_canonical_and_shadow_off_is_unchanged(self):
        question = "DSBA 06026212 course description"
        original = parse_query_spec("DSBA 06026212 มีหน่วยกิตเท่าไร")
        shadow_spec = replace(original, operations=(), normalized_question=question)
        with patch.object(qa_module, "parse_query_spec", return_value=shadow_spec), \
             patch.object(
                 qa_module,
                 "_classify_structured_parse_completeness",
                 side_effect=lambda spec, resolution, context=None: replace(
                     _classify_structured_parse_completeness(spec, resolution, context),
                     classification=(
                         "unrecognized_structured"
                         if not tuple(getattr(spec, "operations", ()))
                         else "complete"
                     ),
                 ),
             ):
            on = ask(
                DB_PATH,
                question,
                shadow_intent=True,
                intent_model_callable=lambda prompt: _payload(
                    "course_description", ("course_description",)
                ),
            )
        self.assertEqual(on["result"].status, "answer")
        self.assertTrue(on["description_shadow"].executed)
        self.assertTrue(on["result"].provenance)
        native = ask(DB_PATH, "DSBA 06026212 เรียนเกี่ยวกับอะไร")
        self.assertEqual(native["result"].status, "answer")
        self.assertEqual(on["result"].final_answer, native["result"].final_answer)


if __name__ == "__main__":
    unittest.main()
