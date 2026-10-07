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
    def test_thai_exact_credit_titles_preserve_program_qualifiers(self):
        for question in (
            "DSBA แคลคูลัส 2 กี่หน่วยกิต",
            "แคลคูลัส 2 ของ DSBA กี่หน่วยกิต",
            "แคลคูลัส 2 ใน DSBA กี่เครดิต",
            "วิชา แคลคูลัส 2 กี่หน่วยกิต",
            "DSBA วิชา แคลคูลัส 2 กี่หน่วยกิต",
        ):
            with self.subTest(question=question):
                spec = _parse(question)
                self.assertEqual(spec.course_name, "แคลคูลัส 2")
                self.assertEqual(spec.operations, ("sum_credits",))
                self.assertIsNone(spec.credit_units)

    def test_credit_filter_collections_have_no_exact_title(self):
        cases = (
            ("IT ปี 3 เทอม 1 มีวิชาไหน 3 หน่วยกิตบ้าง", ("list",)),
            ("ปี 3 เทอม 1 มีวิชา 3 หน่วยกิตกี่วิชา", ("count",)),
            ("IT มีวิชา 3 หน่วยกิตไหม", ("existence",)),
            ("IT ปี 3 เทอม 1 3 หน่วยกิตเรียนตอนไหน", ("placement",)),
            ("IT ปี 3 เทอม 1 3 หน่วยกิตรวมกี่หน่วยกิต", ("sum_credits",)),
            ("IT วิชา 3 หน่วยกิตมีอะไรบ้าง", ("list",)),
            ("IT วิชาเลือก 3 หน่วยกิตมีอะไรบ้าง", ("list",)),
        )
        for question, operations in cases:
            with self.subTest(question=question):
                spec = _parse(question)
                self.assertIsNone(spec.course_name)
                self.assertEqual(spec.course_codes, ())
                self.assertEqual(spec.credit_units, 3)
                self.assertEqual(spec.operations, operations)
        scoped = _parse(cases[0][0])
        self.assertEqual(scoped.program, "IT")
        self.assertEqual(scoped.years, (3,))
        self.assertEqual(scoped.semesters, (1,))

    def test_scoped_credit_totals_have_no_exact_title(self):
        for question in (
            "BIT ปี 1 เทอม 1 หน่วยกิตรวมเท่าไหร่",
            "IT ปี 3 รวมกี่หน่วยกิต",
            "DSBA ปี 2 เทอม 1 กี่หน่วยกิต",
        ):
            with self.subTest(question=question):
                spec = _parse(question)
                self.assertIsNone(spec.course_name)
                self.assertEqual(spec.operations, ("sum_credits",))
                self.assertTrue(spec.years)

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

    def test_exact_course_titles_survive_relation_and_scope_orderings(self):
        cases = (
            ("AIT Calculus 2 ต้องผ่านอะไรบ้าง", "AIT", "Calculus 2", ("prerequisite",)),
            ("DSBA Calculus 2 กี่หน่วยกิต", "DSBA", "Calculus 2", ("sum_credits",)),
            ("Calculus 2 ใน DSBA กี่หน่วยกิต", "DSBA", "Calculus 2", ("sum_credits",)),
            ("Calculus 2 ต้องผ่านอะไรบ้าง", None, "Calculus 2", ("prerequisite",)),
            ("ปีไหนเรียน Calculus 2 ใน DSBA", "DSBA", "Calculus 2", ("placement",)),
        )
        for question, program, course_name, operations in cases:
            with self.subTest(question=question):
                spec = _parse(question)
                self.assertEqual(spec.program, program)
                self.assertEqual(spec.course_name, course_name)
                self.assertEqual(spec.operations, operations)


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

    def test_dsba_exact_course_credit_uses_course_target_not_program_total(self):
        result, calls = self._ask(
            "DSBA Calculus 2 กี่หน่วยกิต",
            context=QueryContext(program="DSBA", catalog_key="dsba-2565"),
        )
        self.assertEqual(self._status(result), "answer")
        self.assertEqual(calls, [])
        self.assertIn("3 หน่วยกิต", result.final_answer)
        self.assertTrue(
            any(claim.operation == "sum_credits" for claim in result.claims)
        )
        self.assertTrue(
            any(
                target.get("course_code") == "06026201"
                for claim in result.claims
                for target in getattr(claim.effective_scope, "course_targets", ())
            )
        )
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
        self.assertTrue(
            any(
                target.get("course_code") == "06046401"
                for claim in result.claims
                for target in getattr(claim.effective_scope, "course_targets", ())
            )
        )

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
