import shutil
import sqlite3
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from rag.grounded_answer import GroundedAnswerResult
from rag.policy.query import parse_policy_question
from rag.policy.routing import route_policy_question
from rag.qa import ask
from rag.query_spec import parse_query_spec


DB_PATH = Path(__file__).parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"


def _no_model(prompt: str) -> str:
    raise AssertionError("policy route must not call any model")


def _ask(question: str, db_path: Path = DB_PATH) -> dict:
    return ask(
        db_path,
        question,
        structured_model_callable=_no_model,
        answer_model_callable=_no_model,
        intent_model_callable=_no_model,
    )


def _has_policy_provenance(result: GroundedAnswerResult) -> bool:
    return any(
        isinstance(reference, Mapping)
        and reference.get("document_category") in ("rule", "program_requirement")
        for reference in result.provenance
    )


class H25P3ProgramTotalTests(unittest.TestCase):
    def test_axis_free_totals_route_to_policy_requirement(self):
        expected = {"AIT": "120", "BIT": "126", "DSBA": "132", "IT": "129"}
        for program, value in expected.items():
            with self.subTest(program=program):
                response = _ask(f"{program} ต้องเรียนกี่หน่วยกิต")
                self.assertIsNone(response["route"])
                result = response["result"]
                self.assertIsInstance(result, GroundedAnswerResult)
                self.assertEqual(result.status, "answer")
                self.assertIn(value, result.final_answer)
                self.assertTrue(result.provenance)
                for reference in result.provenance:
                    self.assertIsInstance(reference, Mapping)
                    self.assertEqual(
                        reference.get("document_category"), "program_requirement"
                    )

    def test_graduation_wording_dual_routes_to_policy(self):
        result = _ask("IT ต้องเรียนกี่หน่วยกิตถึงจบ")["result"]
        self.assertIsInstance(result, GroundedAnswerResult)
        self.assertEqual(result.status, "answer")
        self.assertIn("129", result.final_answer)
        self.assertTrue(_has_policy_provenance(result))

    def test_scoped_totals_keep_existing_curriculum_behavior(self):
        for question in (
            "IT ปี 3 รวมทั้งหมดกี่หน่วยกิต",
            "IT ปี 3 เทอม 1 รวมทั้งหมดกี่หน่วยกิต",
            "IT แผนสหกิจ รวมทั้งหมดกี่หน่วยกิต",
            "IT ปี 3 ต้องเรียนกี่หน่วยกิตถึงจบ",
        ):
            with self.subTest(question=question):
                result = _ask(question)["result"]
                self.assertIsInstance(result, GroundedAnswerResult)
                self.assertEqual(result.status, "answer")
                self.assertFalse(_has_policy_provenance(result))

    def test_category_program_only_total_fails_closed(self):
        # H27-B supersedes the former "IT วิชาเลือก รวมทั้งหมดกี่หน่วยกิต"
        # answer pin: category + sum without an explicit single term now
        # fails closed (no default-scope all-category total), still routed
        # to the curriculum path (zero policy provenance).
        result = _ask("IT วิชาเลือก รวมทั้งหมดกี่หน่วยกิต")["result"]
        self.assertIsInstance(result, GroundedAnswerResult)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertFalse(_has_policy_provenance(result))

    def test_neither_and_multi_program_shapes_unchanged(self):
        # H28-B supersedes the former bare-max pins ("IT ลงทะเบียนได้สูงสุด
        # กี่หน่วยกิต" / "ลงทะเบียนได้สูงสุดกี่หน่วยกิต"): bare regular-max
        # wording now routes to the registration policy authority (see
        # test_h28_bare_max). The neither/multi-program shapes below keep
        # existing behavior.
        result = _ask("ต้องเรียนกี่หน่วยกิตถึงจบ")["result"]
        self.assertIsInstance(result, GroundedAnswerResult)
        self.assertEqual(result.status, "insufficient_evidence")
        result = _ask("IT กับ DSBA ต้องเรียนกี่หน่วยกิต")["result"]
        self.assertIsInstance(result, GroundedAnswerResult)
        self.assertEqual(result.status, "answer")
        self.assertFalse(_has_policy_provenance(result))

    def test_missing_program_requirement_fails_closed_through_public_ask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.db"
            shutil.copy2(DB_PATH, path)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute(
                    "DELETE FROM program_requirements "
                    "WHERE program_code = 'IT' "
                    "AND requirement_type = 'total_program_credits'"
                )
                connection.commit()
            result = _ask("IT ต้องเรียนกี่หน่วยกิต", db_path=path)["result"]
            self.assertIsInstance(result, GroundedAnswerResult)
            self.assertEqual(result.status, "insufficient_evidence")


class H25P3CompareTests(unittest.TestCase):
    def test_comparisons_route_to_policy_verdicts(self):
        cases = (
            ("ลง 24 หน่วยกิตได้ไหม", "อาจทำได้เฉพาะกรณี"),
            ("IT ลง 24 หน่วยกิตได้ไหม", "อาจทำได้เฉพาะกรณี"),
            ("IT ปี 3 เทอม 1 ลง 24 หน่วยกิตได้ไหม", "อาจทำได้เฉพาะกรณี"),
            ("ลงทะเบียน 30 หน่วยกิตได้หรือไม่", "ไม่ได้"),
            ("เรียน 9 หน่วยกิตได้หรือเปล่า", "ได้ตามเพดานปกติ"),
            ("IT ปี 3 เทอม 1 มีวิชาเรียน 3 หน่วยกิตได้ไหม", "ได้ตามเพดานปกติ"),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                response = _ask(question)
                self.assertIsNone(response["route"])
                result = response["result"]
                self.assertIsInstance(result, GroundedAnswerResult)
                self.assertEqual(result.status, "answer")
                self.assertIn(expected, result.final_answer)
                self.assertTrue(result.provenance)
                for reference in result.provenance:
                    self.assertIsInstance(reference, Mapping)
                    self.assertEqual(reference.get("document_category"), "rule")

    def test_unsupported_comparison_unchanged(self):
        result = _ask("IT ลงทะเบียนได้ไหมโดยไม่บอกจำนวน")["result"]
        self.assertIsInstance(result, GroundedAnswerResult)
        self.assertEqual(result.status, "insufficient_evidence")

    def test_list_compare_dual_guard_trigger_absent_in_practice(self):
        # Structural pin: no constructible shape parses as compare while the
        # curriculum parse yields exactly ("list",), so the R2 fail-closed
        # guard is defensive-only (verified by read, not reachable here).
        near_misses = (
            "มีวิชาให้ลงเรียน 3 หน่วยกิตบ้างได้ไหม",
            "IT มีวิชาเรียน 3 หน่วยกิตบ้างได้ไหม",
            "IT ปี 3 เทอม 1 มีวิชาเรียน 3 หน่วยกิตบ้างได้ไหม",
            "ลง 24 หน่วยกิตได้ไหม",
        )
        for question in near_misses:
            with self.subTest(question=question):
                policy = parse_policy_question(question)
                spec = parse_query_spec(question)
                self.assertFalse(
                    policy is not None
                    and policy.kind == "registration_compare"
                    and tuple(spec.operations) == ("list",)
                )

    def test_non_routed_shapes_still_return_none(self):
        self.assertIsNone(
            route_policy_question(DB_PATH, "BIT ปี 4 เทอม 1 รวมกี่หน่วยกิต")
        )
        self.assertIsNone(route_policy_question(DB_PATH, "ฝึกงานต้องผ่านอะไรบ้าง"))


if __name__ == "__main__":
    unittest.main()
