"""H21: course_list_query interpreter family (bounded LIST recovery)."""

import json
import unittest
from pathlib import Path

from rag.intent_compiler import IntentCompilerError, compile_intent_to_query_spec
from rag.intent_interpreter import (
    IntentInterpretation,
    IntentValidationError,
    parse_intent_payload,
)
from rag.qa import ask
from rag.query_spec import parse_query_spec


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)

ACCEPTANCE_QUESTION = "ขอวิชาเลือกของ DSBA เทอม 2"
DETERMINISTIC_EQUIVALENT = "DSBA เทอม 2 มีวิชาเลือกอะไรบ้าง"


def _payload(**changes):
    data = {
        "intent": "course_list_query",
        "proposed_program": None,
        "proposed_plans": [],
        "proposed_years": [],
        "proposed_semesters": [],
        "course_codes": [],
        "topic": None,
        "requested_facts": ["course_list"],
        "judgement_dimension": None,
        "unresolved": [],
    }
    data.update(changes)
    return json.dumps(data, ensure_ascii=False)


def _scripted_model(text=None, calls=None, error=None):
    def _call(_prompt):
        if calls is not None:
            calls.append(_prompt)
        if error is not None:
            raise error
        return text

    return _call


def _list_codes(result):
    codes = set()
    for claim in result["result"].claims:
        value = claim.value
        if isinstance(value, dict):
            value = (value,)
        for entry in value or ():
            code = entry.get("course_code")
            if isinstance(code, str) and code.strip():
                codes.add(code)
            for member in entry.get("alternative_courses") or ():
                member_code = member.get("course_code")
                if isinstance(member_code, str) and member_code.strip():
                    codes.add(member_code)
    return codes


class H21InterpreterContractTests(unittest.TestCase):
    def test_course_list_payload_validates(self):
        interpretation = parse_intent_payload(_payload())
        self.assertEqual(interpretation.intent, "course_list_query")
        self.assertEqual(interpretation.requested_facts, ("course_list",))

    def test_wrong_requested_facts_rejected_at_compile(self):
        spec = parse_query_spec(ACCEPTANCE_QUESTION)
        interpretation = parse_intent_payload(
            _payload(requested_facts=["course_credit"])
        )
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                spec,
                interpretation,
                authoritative_program="DSBA",
                authoritative_years=(),
                authoritative_semesters=(2,),
                allowed_course_codes=(),
            )

    def test_scope_widening_proposal_rejected(self):
        spec = parse_query_spec(ACCEPTANCE_QUESTION)
        interpretation = parse_intent_payload(
            _payload(proposed_semesters=[1, 2])
        )
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                spec,
                interpretation,
                authoritative_program="DSBA",
                authoritative_semesters=(2,),
                allowed_course_codes=(),
            )

    def test_scope_narrowing_proposal_preserves_authority(self):
        spec = parse_query_spec("ขอวิชาเลือกของ DSBA เทอม 1 เทอม 2")
        interpretation = parse_intent_payload(_payload(proposed_semesters=[2]))
        compiled = compile_intent_to_query_spec(
            spec,
            interpretation,
            authoritative_program="DSBA",
            authoritative_semesters=(1, 2),
            allowed_course_codes=(),
        )
        self.assertEqual(compiled.semesters, (1, 2))
        self.assertEqual(compiled.operations, ("list",))

    def test_category_scope_preserved_not_replaced(self):
        spec = parse_query_spec(ACCEPTANCE_QUESTION)
        self.assertEqual(spec.category, "วิชาเลือก")
        interpretation = parse_intent_payload(_payload())
        compiled = compile_intent_to_query_spec(
            spec,
            interpretation,
            authoritative_program="DSBA",
            authoritative_semesters=(2,),
            allowed_course_codes=(),
        )
        self.assertEqual(compiled.category, "วิชาเลือก")
        self.assertEqual(compiled.operations, ("list",))
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(_payload(proposed_category="วิชาเลือก"))

    def test_conflicting_program_rejected(self):
        spec = parse_query_spec(ACCEPTANCE_QUESTION)
        interpretation = parse_intent_payload(_payload(proposed_program="IT"))
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                spec,
                interpretation,
                authoritative_program="DSBA",
                authoritative_semesters=(2,),
                allowed_course_codes=(),
            )

    def test_topic_or_judgement_addition_rejected(self):
        spec = parse_query_spec(ACCEPTANCE_QUESTION)
        # Topic reaches the compiler guard; a judgement dimension is
        # rejected even earlier at payload validation. Both fail closed.
        with self.assertRaises(IntentCompilerError):
            compile_intent_to_query_spec(
                spec,
                parse_intent_payload(_payload(topic="data")),
                authoritative_program="DSBA",
                authoritative_semesters=(2,),
                allowed_course_codes=(),
            )
        with self.assertRaises(IntentValidationError):
            parse_intent_payload(_payload(judgement_dimension="workload"))


class H21LiveGateTests(unittest.TestCase):
    def test_deterministic_complete_list_makes_zero_calls(self):
        question = "IT ปี 3 มีวิชาอะไรบ้าง"
        plain = ask(DB_PATH, question)
        calls = {"structured": [], "intent": [], "answer": []}
        response = ask(
            DB_PATH,
            question,
            structured_model_callable=lambda prompt: calls["structured"].append(prompt) or "x",
            intent_model_callable=lambda prompt: calls["intent"].append(prompt) or "x",
            answer_model_callable=lambda prompt: calls["answer"].append(prompt) or "x",
        )
        self.assertEqual(response["result"].status, "answer")
        self.assertEqual(calls["intent"], [])
        self.assertEqual(
            response["result"].final_answer, plain["result"].final_answer
        )

    def test_eligible_long_tail_recovers_list_with_one_call(self):
        intent_calls = []
        response = ask(
            DB_PATH,
            ACCEPTANCE_QUESTION,
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
        )
        self.assertEqual(len(intent_calls), 1)
        self.assertEqual(response["result"].status, "answer")
        self.assertTrue(response["result"].provenance)
        expected = ask(DB_PATH, DETERMINISTIC_EQUIVALENT)
        self.assertEqual(expected["result"].status, "answer")
        self.assertEqual(
            _list_codes(response), _list_codes(expected)
        )
        self.assertEqual(
            response["result"].provenance, expected["result"].provenance
        )

    def test_credit_value_filter_makes_zero_calls_and_fails_closed(self):
        intent_calls = []
        response = ask(
            DB_PATH,
            "IT ปี 3 มีวิชาไหน 3 หน่วยกิตบ้าง",
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
        )
        self.assertEqual(intent_calls, [])
        self.assertEqual(response["result"].status, "insufficient_evidence")

    def test_wrong_intent_fails_closed(self):
        response = ask(
            DB_PATH,
            ACCEPTANCE_QUESTION,
            intent_model_callable=_scripted_model(
                _payload(intent="count_query")
            ),
        )
        self.assertEqual(response["result"].status, "insufficient_evidence")

    def test_malformed_or_unresolved_proposal_fails_closed(self):
        cases = (
            "{not json",
            _payload(intent="no_such_intent"),
            _payload(requested_facts=["course_list", "nope"]),
            _payload(unresolved=["3 หน่วยกิต"]),
        )
        for payload in cases:
            with self.subTest(payload=payload[:40]):
                response = ask(
                    DB_PATH,
                    ACCEPTANCE_QUESTION,
                    intent_model_callable=_scripted_model(payload),
                )
                self.assertEqual(response["result"].status, "insufficient_evidence")

    def test_provider_exception_fails_closed(self):
        response = ask(
            DB_PATH,
            ACCEPTANCE_QUESTION,
            intent_model_callable=_scripted_model(
                error=RuntimeError("provider down")
            ),
        )
        self.assertEqual(response["result"].status, "insufficient_evidence")

    def test_elective_alias_preserved_through_h21(self):
        question = "DSBA เทอม 2 มี elective อะไร"
        self.assertEqual(parse_query_spec(question).category, "วิชาเลือก")
        intent_calls = []
        response = ask(
            DB_PATH,
            question,
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
        )
        self.assertEqual(len(intent_calls), 1)
        self.assertEqual(response["result"].status, "answer")
        expected = ask(DB_PATH, DETERMINISTIC_EQUIVALENT)
        self.assertEqual(expected["result"].status, "answer")
        self.assertEqual(
            _list_codes(response), _list_codes(expected)
        )
        self.assertEqual(
            response["result"].provenance, expected["result"].provenance
        )

    def test_unsupported_modifier_creates_no_new_filter(self):
        question = "IT ปี 3 ตอนเช้า มีอะไร"
        spec = parse_query_spec(question)
        self.assertIsNone(spec.category)
        self.assertIsNone(spec.topic)
        self.assertEqual(spec.course_codes, ())
        intent_calls = []
        response = ask(
            DB_PATH,
            question,
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
        )
        self.assertEqual(response["result"].status, "answer")
        expected = ask(DB_PATH, "IT ปี 3 มีวิชาอะไรบ้าง")
        self.assertEqual(
            _list_codes(response), _list_codes(expected)
        )

    def test_successful_turn_skips_answer_synthesis(self):
        intent_calls, answer_calls = [], []
        response = ask(
            DB_PATH,
            ACCEPTANCE_QUESTION,
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
            answer_model_callable=lambda prompt: answer_calls.append(prompt) or "x",
            synthesize_answer=True,
        )
        self.assertEqual(response["result"].status, "answer")
        self.assertEqual(len(intent_calls), 1)
        self.assertEqual(answer_calls, [])


class H22CategoryOnlyTests(unittest.TestCase):
    def test_category_only_recovers_list_with_one_call(self):
        question = "ขอหมวดศึกษาทั่วไปของ IT"
        self.assertEqual(
            parse_query_spec(question).category, "หมวดวิชาศึกษาทั่วไป"
        )
        intent_calls = []
        response = ask(
            DB_PATH,
            question,
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
        )
        self.assertEqual(len(intent_calls), 1)
        self.assertEqual(response["result"].status, "answer")
        self.assertTrue(response["result"].provenance)
        for claim in response["result"].claims:
            self.assertEqual(claim.effective_scope.category, "หมวดวิชาศึกษาทั่วไป")
        expected = ask(DB_PATH, "IT ลงเรียนวิชาศึกษาทั่วไปอะไรได้บ้าง")
        self.assertEqual(expected["result"].status, "answer")
        self.assertEqual(
            _list_codes(response), _list_codes(expected)
        )
        self.assertEqual(
            response["result"].provenance, expected["result"].provenance
        )

    def test_elective_category_only_answers_filtered_list(self):
        question = "IT มี elective อะไร"
        self.assertEqual(parse_query_spec(question).category, "วิชาเลือก")
        intent_calls = []
        response = ask(
            DB_PATH,
            question,
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
        )
        self.assertLessEqual(len(intent_calls), 1)
        self.assertEqual(response["result"].status, "answer")
        expected = ask(DB_PATH, "IT มีวิชาเลือกอะไรบ้าง")
        self.assertEqual(expected["result"].status, "answer")
        self.assertEqual(
            _list_codes(response), _list_codes(expected)
        )
        self.assertEqual(
            response["result"].provenance, expected["result"].provenance
        )

    def test_bare_it_mi_arai_stays_ineligible(self):
        intent_calls = []
        response = ask(
            DB_PATH,
            "IT มีอะไร",
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
        )
        self.assertEqual(intent_calls, [])
        self.assertEqual(response["result"].status, "insufficient_evidence")

    def test_thai_modifier_gains_no_filter(self):
        question = "IT ตอนเช้ามีอะไร"
        spec = parse_query_spec(question)
        self.assertIsNone(spec.category)
        self.assertIsNone(spec.topic)
        self.assertEqual(spec.course_codes, ())
        intent_calls = []
        response = ask(
            DB_PATH,
            question,
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
        )
        self.assertEqual(intent_calls, [])
        self.assertEqual(response["result"].status, "insufficient_evidence")

    def test_topic_and_exact_course_shapes_unchanged(self):
        intent_calls = []
        response = ask(
            DB_PATH,
            "วิชาเลือกของ IT ที่เกี่ยวกับ AI",
            intent_model_callable=_scripted_model(_payload(), calls=intent_calls),
        )
        self.assertEqual(response["result"].status, "insufficient_evidence")
        response = ask(DB_PATH, "IT 06016454 มี prerequisite ไหม")
        self.assertEqual(response["result"].status, "answer")
        self.assertIn(
            "prerequisite", [claim.operation for claim in response["result"].claims]
        )

    def test_wrong_family_category_only_fails_closed(self):
        response = ask(
            DB_PATH,
            "ขอหมวดศึกษาทั่วไปของ IT",
            intent_model_callable=_scripted_model(
                _payload(intent="count_query")
            ),
        )
        self.assertEqual(response["result"].status, "insufficient_evidence")

    def test_provider_failure_category_only_fails_closed(self):
        response = ask(
            DB_PATH,
            "ขอหมวดศึกษาทั่วไปของ IT",
            intent_model_callable=_scripted_model(
                error=RuntimeError("provider down")
            ),
        )
        self.assertEqual(response["result"].status, "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
