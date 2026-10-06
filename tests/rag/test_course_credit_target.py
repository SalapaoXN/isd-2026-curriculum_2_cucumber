"""CORRECTNESS-1: exact-course credit questions must not collapse into program totals.

Generic invariant: in a credit-interrogative question, a Latin alphanumeric
title run immediately preceding the credit cue denotes the exact course,
independent of script (Thai-script titles were already extracted); a leading
program qualifier is scope, never part of the title. Program scope without an
exact target still routes to the program-total authority.
"""

import unittest
from pathlib import Path

from rag.query_spec import parse_query_spec
from rag.qa import ask
from rag.resolution import QueryContext


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)

AIT_CONTEXT = QueryContext(program="AIT", catalog_key="ait-2566")


def _parse(question):
    return parse_query_spec(question, has_validated_context_scope=False)


class CourseCreditTitleParseTests(unittest.TestCase):
    def test_credit_title_shapes_extract_course_name_generically(self):
        cases = (
            ("AIT Calculus 2 กี่หน่วยกิต", "AIT", "Calculus 2"),
            ("DSBA Calculus 2 กี่หน่วยกิต", "DSBA", "Calculus 2"),
            ("BIT Data Warehousing กี่หน่วยกิต", "BIT", "Data Warehousing"),
            ("IT Calculus 2 กี่เครดิต", "IT", "Calculus 2"),
            ("AIT Calculus 2 กี่เครดิต", "AIT", "Calculus 2"),
            ("Calculus 2 กี่หน่วยกิต", None, "Calculus 2"),
            ("วิชา Calculus 2 กี่หน่วยกิต", None, "Calculus 2"),
            ("AIT วิชา Calculus 2 กี่หน่วยกิต", "AIT", "Calculus 2"),
        )
        for question, program, title in cases:
            with self.subTest(question=question):
                spec = _parse(question)
                self.assertEqual(spec.program, program)
                self.assertEqual(spec.course_name, title)
                self.assertEqual(tuple(spec.operations), ("sum_credits",))

    def test_program_total_shapes_extract_no_course_name(self):
        for question in (
            "AIT มีกี่หน่วยกิต",
            "AIT กี่หน่วยกิต",
            "หลักสูตร AIT ต้องเรียนทั้งหมดกี่หน่วยกิต",
        ):
            with self.subTest(question=question):
                spec = _parse(question)
                self.assertIsNone(spec.course_name)
                self.assertEqual(tuple(spec.course_codes), ())


class CourseCreditPrecedenceTests(unittest.TestCase):
    @staticmethod
    def _status(result):
        return result.get("status") if isinstance(result, dict) else result.status

    def _ask(self, question, context=None):
        calls = []
        result = ask(
            DB_PATH,
            question,
            context=context,
            intent_model_callable=lambda prompt: calls.append(prompt)
            or (_ for _ in ()).throw(AssertionError("must be deterministic")),
        )["result"]
        return result, calls

    def test_primary_program_title_answers_course_credits(self):
        result, calls = self._ask("AIT Calculus 2 กี่หน่วยกิต")
        self.assertEqual(self._status(result), "answer")
        self.assertEqual(calls, [])
        self.assertIn("3 หน่วยกิต", result.final_answer)
        self.assertNotIn("120", result.final_answer)
        self.assertTrue(
            any(claim.operation == "sum_credits" for claim in result.claims)
        )
        self.assertTrue(result.provenance)

    def test_credit_wording_answers_course_credits(self):
        result, calls = self._ask("AIT Calculus 2 กี่เครดิต")
        self.assertEqual(self._status(result), "answer")
        self.assertEqual(calls, [])
        self.assertIn("3 หน่วยกิต", result.final_answer)
        self.assertTrue(
            any(claim.operation == "sum_credits" for claim in result.claims)
        )
        self.assertTrue(result.provenance)

    def test_context_scoped_bare_title_still_answers_course_credits(self):
        result, calls = self._ask("Calculus 2 กี่หน่วยกิต", context=AIT_CONTEXT)
        self.assertEqual(self._status(result), "answer")
        self.assertEqual(calls, [])
        self.assertIn("3 หน่วยกิต", result.final_answer)
        self.assertTrue(result.provenance)

    def test_course_code_control_answers_course_credits(self):
        result, calls = self._ask("AIT 06046401 กี่หน่วยกิต")
        self.assertEqual(self._status(result), "answer")
        self.assertEqual(calls, [])
        self.assertIn("3 หน่วยกิต", result.final_answer)
        self.assertNotIn("120", result.final_answer)

    def test_program_totals_without_target_stay_program_totals(self):
        for question in (
            "AIT มีกี่หน่วยกิต",
            "หลักสูตร AIT ต้องเรียนทั้งหมดกี่หน่วยกิต",
        ):
            with self.subTest(question=question):
                result, calls = self._ask(question)
                self.assertEqual(self._status(result), "answer")
                self.assertEqual(calls, [])
                self.assertIn("120", result.final_answer)
                self.assertEqual(tuple(result.claims), ())

    def test_prerequisite_semantics_preserved(self):
        result, calls = self._ask("AIT Calculus 2 ต้องผ่านอะไรบ้าง")
        self.assertEqual(self._status(result), "answer")
        self.assertIn("06046400", result.final_answer)

    def test_bare_title_without_scope_stays_closed(self):
        result, calls = self._ask("Calculus 2 กี่หน่วยกิต")
        self.assertNotEqual(self._status(result), "answer")
        provenance = (
            result.get("provenance", ())
            if isinstance(result, dict)
            else result.provenance
        )
        self.assertEqual(tuple(provenance), ())


if __name__ == "__main__":
    unittest.main()
