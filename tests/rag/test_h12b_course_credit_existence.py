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


DB_PATH = Path(__file__).resolve().parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"


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


class H12BExactCourseTest(unittest.TestCase):
    def _inputs(self, question):
        spec = replace(
            parse_query_spec("DSBA 06026212 มีหน่วยกิตเท่าไร"),
            operations=(),
            normalized_question=question,
        )
        resolution = resolve_query_spec(spec, DB_PATH)
        completeness = replace(
            _classify_structured_parse_completeness(spec, resolution),
            classification="unrecognized_structured",
        )
        return spec, resolution, completeness

    def test_credit_and_existence_contracts_validate_once(self):
        cases = (
            ("DSBA 06026212 กี่หน่วยอะ", "course_credit_query", ("course_credit",), ("sum_credits",)),
            ("DSBA 06026212 มีไหม", "existence_query", ("course_list",), ("existence",)),
            ("DSBA 06026212 credits?", "course_credit_query", ("course_credit",), ("sum_credits",)),
        )
        for question, family, facts, operations in cases:
            with self.subTest(question=question):
                spec, resolution, completeness = self._inputs(question)
                calls = []
                result = run_exact_course_shadow(
                    question, spec, completeness, resolution, None,
                    lambda prompt: calls.append(prompt) or _payload(family, facts),
                    interpret_callable=lambda q, model: parse_intent_payload(model(q)),
                    family=family,
                )
                self.assertTrue(result.eligible)
                self.assertEqual(result.status, "validated")
                self.assertEqual(result.compiled_spec.operations, operations)
                self.assertEqual(len(calls), 1)
                self.assertFalse(result.executed)

    def test_wrong_scope_target_or_payload_fails_closed(self):
        spec, resolution, completeness = self._inputs("DSBA 06026212 กี่หน่วย")
        proposals = (
            _payload("existence_query", ("course_list",)),
            _payload("course_credit_query", ("course_credit",), code="06000000"),
            _payload("course_credit_query", ("course_credit",), proposed_program="IT"),
            _payload("course_credit_query", ("course_credit",), topic="data"),
            _payload("course_credit_query", ("course_credit",), unresolved=["requirement_type"]),
            "not json",
        )
        for proposal in proposals:
            with self.subTest(proposal=proposal):
                result = run_exact_course_shadow(
                    "DSBA 06026212 กี่หน่วย", spec, completeness, resolution, None,
                    lambda prompt, proposal=proposal: proposal,
                    interpret_callable=lambda q, model: parse_intent_payload(model(q)),
                    family="course_credit_query",
                )
                self.assertNotEqual(result.status, "validated")

    def test_gated_credit_and_existence_use_canonical_paths_without_polish(self):
        cases = (
            ("DSBA 06026212 กี่หน่วย", "course_credit_query", ("course_credit",), "sum_credits"),
            ("DSBA 06026212 มีไหม", "existence_query", ("course_list",), "existence"),
        )
        for question, family, facts, operation in cases:
            with self.subTest(question=question):
                shadow_spec = replace(
                    parse_query_spec("DSBA 06026212 มีหน่วยกิตเท่าไร"),
                    operations=(), normalized_question=question,
                )
                answer_calls = []
                with patch.object(qa_module, "parse_query_spec", return_value=shadow_spec), \
                     patch.object(
                         qa_module,
                         "_classify_structured_parse_completeness",
                         side_effect=lambda spec, resolution, context=None: replace(
                             _classify_structured_parse_completeness(spec, resolution, context),
                             classification=("unrecognized_structured" if not spec.operations else "complete"),
                         ),
                     ):
                    result = ask(
                        DB_PATH, question, shadow_intent=True,
                        intent_model_callable=lambda prompt, family=family, facts=facts: _payload(family, facts),
                        answer_model_callable=lambda prompt: answer_calls.append(prompt),
                    )
                self.assertEqual(result["result"].status, "answer")
                self.assertEqual(len(answer_calls), 0)
                key = "credit_shadow" if family == "course_credit_query" else "existence_shadow"
                self.assertTrue(result[key].executed)
                self.assertTrue(result["result"].provenance)
                self.assertIn(operation, {claim.operation for claim in result["result"].claims})

    def test_complete_and_semantic_controls_make_zero_new_calls(self):
        for question in (
            "DSBA 06026212 มีกี่หน่วยกิต",
            "DSBA มีวิชาเกี่ยวกับ database อะไรบ้าง",
        ):
            calls = []
            result = ask(DB_PATH, question, shadow_intent=True, intent_model_callable=lambda p: calls.append(p))
            self.assertEqual(calls, [], question)


if __name__ == "__main__":
    unittest.main()
