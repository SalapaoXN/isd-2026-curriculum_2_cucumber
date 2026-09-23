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


DB_PATH = Path(__file__).parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"

BARE_MAX_QUESTIONS = (
    "ลงทะเบียนได้สูงสุดกี่หน่วยกิต",
    "ลงได้สูงสุดกี่หน่วยกิต",
    "IT ลงทะเบียนได้สูงสุดกี่หน่วยกิต",
    "ลงทะเบียนได้มากสุดกี่หน่วยกิต",
    "ปกติลงได้มากสุดกี่หน่วยกิต",
    "ลงทะเบียนได้ไม่เกินกี่หน่วยกิต",
)


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


class H28BareMaxTests(unittest.TestCase):
    def test_bare_max_synonyms_parse_to_regular_max(self):
        for question in BARE_MAX_QUESTIONS:
            with self.subTest(question=question):
                query = parse_policy_question(question)
                self.assertIsNotNone(query)
                self.assertEqual(query.kind, "registration_regular_max")

    def test_bare_max_answers_through_public_ask_without_model_calls(self):
        for question in BARE_MAX_QUESTIONS:
            with self.subTest(question=question):
                response = _ask(question)
                self.assertIsNone(response["route"])
                result = response["result"]
                self.assertIsInstance(result, GroundedAnswerResult)
                self.assertEqual(result.status, "answer")
                self.assertEqual(result.answer_mode, "deterministic")
                self.assertIn("22", result.final_answer)
                self.assertTrue(result.provenance)
                for reference in result.provenance:
                    self.assertIsInstance(reference, Mapping)
                    self.assertEqual(reference.get("document_category"), "rule")

    def test_axis_bare_max_keeps_curriculum_path(self):
        question = "IT ปี 3 ลงทะเบียนได้สูงสุดกี่หน่วยกิต"
        self.assertIsNone(route_policy_question(DB_PATH, question))
        result = _ask(question)["result"]
        self.assertIsInstance(result, GroundedAnswerResult)
        self.assertEqual(result.status, "answer")
        self.assertFalse(_has_policy_provenance(result))

    def test_existing_routes_unchanged(self):
        parse_cases = {
            "ปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต": "registration_regular_max",
            "ลงทะเบียนขั้นต่ำกี่หน่วยกิต": "registration_regular_min",
            "ลงทะเบียนขั้นต่ำและสูงสุดกี่หน่วยกิต": "registration_regular_min",
            "กรณีพิเศษลงได้สูงสุดเท่าไร": "registration_exception_max",
            "ซัมเมอร์ลงได้กี่หน่วยกิต": "registration_special_max",
            "ลง 24 หน่วยกิตได้ไหม": "registration_compare",
            "IT ต้องเรียนกี่หน่วยกิต": "program_total_credits",
            "IT ต้องเรียนสูงสุดกี่หน่วยกิต": "program_total_credits",
        }
        for question, kind in parse_cases.items():
            with self.subTest(question=question):
                query = parse_policy_question(question)
                self.assertIsNotNone(query)
                self.assertEqual(query.kind, kind)
        public_cases = (
            ("ปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต", "answer", "22", True),
            ("ลงทะเบียนขั้นต่ำกี่หน่วยกิต", "answer", "9", True),
            ("กรณีพิเศษลงได้สูงสุดเท่าไร", "answer", "27", True),
            ("ลง 24 หน่วยกิตได้ไหม", "answer", "อาจทำได้เฉพาะกรณี", True),
            ("IT ต้องเรียนกี่หน่วยกิต", "answer", "129", True),
            ("IT ปี 3 เทอม 1 รวมทั้งหมดกี่หน่วยกิต", "answer", None, False),
            ("IT วิชาเลือก รวมทั้งหมดกี่หน่วยกิต", "insufficient_evidence", None, False),
        )
        for question, status, text, policy in public_cases:
            with self.subTest(question=question):
                result = _ask(question)["result"]
                self.assertIsInstance(result, GroundedAnswerResult)
                self.assertEqual(result.status, status)
                if text is not None:
                    self.assertIn(text, result.final_answer)
                self.assertEqual(_has_policy_provenance(result), policy)

    def test_non_registration_max_wording_stays_unrouted(self):
        for question in (
            "maximum credits per semester",
            "เทอมหนึ่งลงได้ไม่เกินกี่หน่วย",
            "ลงเกินได้สูงสุดกี่หน่วยกิต",
            "ลงได้มากสุดเท่าไหร่",
            "หลักสูตรไหนให้หน่วยกิตสูงสุด",
        ):
            with self.subTest(question=question):
                self.assertIsNone(parse_policy_question(question))
                self.assertIsNone(route_policy_question(DB_PATH, question))
                result = _ask(question)["result"]
                self.assertIsInstance(result, GroundedAnswerResult)
                self.assertEqual(result.status, "insufficient_evidence")
                self.assertFalse(_has_policy_provenance(result))

    def test_missing_regular_maximum_fails_closed_through_public_ask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.db"
            shutil.copy2(DB_PATH, path)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("DELETE FROM policy_facts WHERE fact_id = 28")
                connection.commit()
            result = _ask("IT ลงทะเบียนได้สูงสุดกี่หน่วยกิต", db_path=path)["result"]
            self.assertIsInstance(result, GroundedAnswerResult)
            self.assertEqual(result.status, "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
