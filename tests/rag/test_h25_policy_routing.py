import shutil
import sqlite3
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from rag.grounded_answer import GroundedAnswerResult
from rag.policy.answer import PolicyAnswer
from rag.policy.routing import (
    POLICY_ROUTE_ALLOWLIST,
    adapt_policy_answer,
    route_policy_question,
)
from rag.qa import ask


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


class H25PolicyRoutingTests(unittest.TestCase):
    def test_allowlist_contains_only_program_independent_limit_kinds(self):
        self.assertEqual(
            POLICY_ROUTE_ALLOWLIST,
            frozenset(
                {
                    "registration_regular_max",
                    "registration_regular_min",
                    "registration_exception_max",
                    "registration_special_max",
                    "probation_entry",
                    "probation_cleared",
                    "honors_first",
                    "honors_second",
                    "reentry_limit",
                }
            ),
        )
        self.assertNotIn("program_total_credits", POLICY_ROUTE_ALLOWLIST)
        self.assertNotIn("registration_compare", POLICY_ROUTE_ALLOWLIST)

    def test_whitelisted_kinds_answer_through_public_ask_without_model_calls(self):
        cases = (
            ("ปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต", ("22",)),
            ("ขั้นต่ำกี่หน่วยกิต", ("9",)),
            ("กรณีพิเศษลงได้สูงสุดเท่าไร", ("27",)),
            ("ซัมเมอร์ลงได้กี่หน่วยกิต", ("9",)),
            ("GPA เท่าไรถึงติดโปร", ("2",)),
            ("GPA เท่าไรถึงพ้นโปร", ("2",)),
            ("เกียรตินิยมอันดับหนึ่ง GPA เท่าไร", ("3.75", "3.5")),
            ("เกียรตินิยมอันดับสอง GPA เท่าไร", ("3.25",)),
            ("กลับเข้าศึกษาได้ภายในกี่ปี", ("1",)),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                response = _ask(question)
                self.assertIsNone(response["route"])
                result = response["result"]
                self.assertIsInstance(result, GroundedAnswerResult)
                self.assertEqual(result.status, "answer")
                for text in expected:
                    self.assertIn(text, result.final_answer)
                self.assertTrue(result.provenance)
                for reference in result.provenance:
                    self.assertIsInstance(reference, Mapping)
                    self.assertEqual(reference.get("document_category"), "rule")

    def test_dual_whitelisted_shape_routes_to_policy_authority(self):
        response = _ask("IT ปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต")
        result = response["result"]
        self.assertIsInstance(result, GroundedAnswerResult)
        self.assertEqual(result.status, "answer")
        self.assertIn("22", result.final_answer)
        self.assertTrue(result.provenance)

    def test_excluded_program_total_keeps_existing_behavior(self):
        # H25-P3 supersedes the axis-free pin below: unscoped totals now route
        # to policy (see test_h25p3). Scoped totals keep existing behavior.
        response = _ask("IT ปี 3 เทอม 1 รวมทั้งหมดกี่หน่วยกิต")
        result = response["result"]
        self.assertIsInstance(result, GroundedAnswerResult)
        self.assertEqual(result.status, "answer")
        for reference in result.provenance:
            if isinstance(reference, Mapping):
                self.assertNotIn(
                    reference.get("document_category"),
                    ("rule", "program_requirement"),
                )

    def test_excluded_compare_and_unsupported_keep_existing_behavior(self):
        # H25-P3 supersedes the compare pin: comparisons now route to policy
        # (see test_h25p3). The unsupported shape below keeps existing behavior.
        result = _ask("ลง 24 หน่วยกิตได้ไหม")["result"]
        self.assertIsInstance(result, GroundedAnswerResult)
        self.assertEqual(result.status, "answer")
        for question in ("ฝึกงานต้องผ่านอะไรบ้าง",):
            with self.subTest(question=question):
                result = _ask(question)["result"]
                self.assertIsInstance(result, GroundedAnswerResult)
                self.assertEqual(result.status, "insufficient_evidence")

    def test_corpus_plan_comparison_with_normal_wording_unchanged(self):
        result = _ask("แผนสหกิจมีวิชาเกี่ยวกับ data มากกว่าแผนปกติไหม")["result"]
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "clarify_program")

    def test_missing_policy_evidence_fails_closed_through_public_ask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.db"
            shutil.copy2(DB_PATH, path)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("DELETE FROM policy_facts WHERE fact_id = 28")
                connection.commit()
            result = _ask("ปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต", db_path=path)["result"]
            self.assertIsInstance(result, GroundedAnswerResult)
            self.assertEqual(result.status, "insufficient_evidence")

    def test_route_returns_none_for_non_whitelisted_shapes(self):
        self.assertIsNone(route_policy_question(DB_PATH, "IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง"))
        # H25-P3: axis-free totals and comparisons now route (see test_h25p3);
        # scoped totals and unsupported shapes still return None.
        self.assertIsNone(
            route_policy_question(DB_PATH, "IT ปี 3 เทอม 1 รวมทั้งหมดกี่หน่วยกิต")
        )
        self.assertIsNone(route_policy_question(DB_PATH, "ฝึกงานต้องผ่านอะไรบ้าง"))

    def test_adapter_status_mapping(self):
        complete = adapt_policy_answer(
            PolicyAnswer(status="complete", rendered_answer="ok", provenance=({"a": 1},))
        )
        self.assertEqual(complete.status, "answer")
        self.assertEqual(complete.answer_mode, "deterministic")
        self.assertEqual(complete.final_answer, "ok")
        self.assertEqual(complete.claims, ())
        self.assertEqual(len(complete.provenance), 1)
        self.assertEqual(
            adapt_policy_answer(PolicyAnswer(status="unsupported")).status, "unsupported"
        )
        self.assertEqual(
            adapt_policy_answer(PolicyAnswer(status="insufficient_evidence")).status,
            "insufficient_evidence",
        )
        self.assertEqual(
            adapt_policy_answer(PolicyAnswer(status="invalid_query")).status,
            "unsupported",
        )
        with self.assertRaises(TypeError):
            adapt_policy_answer("not-an-answer")


if __name__ == "__main__":
    unittest.main()
