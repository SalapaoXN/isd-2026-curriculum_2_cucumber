import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from rag.policy import answer_combined_question, answer_policy_question


DB_PATH = Path(__file__).parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"


class RagCombinedPolicyTest(unittest.TestCase):
    def test_within_normal_limit_uses_canonical_curriculum_load(self):
        result = answer_combined_question(
            DB_PATH,
            "IT แผนไม่สหกิจ ปี 1 เทอม 1 มี 18 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.current_credits, 18)
        self.assertEqual(result.added_credits, 3)
        self.assertEqual(result.resulting_total, 21)
        self.assertEqual(result.normal_maximum, 22)
        self.assertEqual(result.decision, "within_normal_limit")

    def test_above_normal_below_exception_is_conditional(self):
        result = answer_combined_question(
            DB_PATH,
            "DSBA แผนไม่สหกิจ ปี 2 เทอม 2 มี 21 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.current_credits, 21)
        self.assertEqual(result.resulting_total, 24)
        self.assertEqual(result.decision, "conditional_exception")
        self.assertIn("เงื่อนไข", result.conditional_note)
        self.assertNotIn("อนุมัติ", result.rendered_answer)

    def test_above_exception_is_deterministically_rejected(self):
        result = answer_combined_question(
            DB_PATH,
            "DSBA แผนไม่สหกิจ ปี 2 เทอม 2 มี 21 หน่วยกิต ถ้าลงเพิ่มอีก 7 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.resulting_total, 28)
        self.assertEqual(result.decision, "exceeds_exception_maximum")

    def test_second_program_and_semester_are_not_hardcoded(self):
        result = answer_combined_question(
            DB_PATH,
            "AIT ปี 3 เทอม 1 มี 18 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.program, "AIT")
        self.assertEqual(result.current_credits, 18)
        self.assertEqual(result.resulting_total, 21)
        self.assertEqual(result.curriculum_evidence[0].plan_key, "default")

    def test_unconstrained_plans_are_retained_when_their_load_agrees(self):
        result = answer_combined_question(
            DB_PATH,
            "DSBA ปี 2 เทอม 2 มี 21 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual({load.plan_key for load in result.curriculum_evidence}, {"coop", "no_coop"})

    def test_stated_load_conflict_fails_closed(self):
        result = answer_combined_question(
            DB_PATH,
            "IT ปี 3 เทอม 1 มี 21 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.resulting_total)

    def test_missing_scope_and_student_specific_eligibility_fail_closed(self):
        missing_scope = answer_combined_question(
            DB_PATH,
            "IT มี 18 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        student_specific = answer_combined_question(
            DB_PATH,
            "IT แผนไม่สหกิจ ปี 1 เทอม 1 มี 18 หน่วยกิต ถ้าฉันมีสิทธิ์ลงเพิ่มอีก 3 หน่วยกิตไหม",
        )
        self.assertEqual(missing_scope.status, "insufficient_evidence")
        self.assertEqual(student_specific.status, "insufficient_evidence")

    def test_missing_policy_fact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.db"
            shutil.copy2(DB_PATH, path)
            connection = sqlite3.connect(path)
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("DELETE FROM policy_facts WHERE fact_id = 28")
                connection.commit()
            finally:
                connection.close()
            result = answer_combined_question(
                path,
                "IT แผนไม่สหกิจ ปี 1 เทอม 1 มี 18 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
            )
            self.assertEqual(result.status, "insufficient_evidence")

    def test_dual_provenance_and_r5_regression_survive(self):
        result = answer_combined_question(
            DB_PATH,
            "DSBA แผนไม่สหกิจ ปี 2 เทอม 2 มี 21 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertTrue(result.curriculum_provenance)
        self.assertTrue(result.policy_provenance)
        self.assertEqual(
            {ref["document_category"] for ref in result.curriculum_provenance},
            {"plan", "description"},
        )
        self.assertEqual(
            {ref["document_category"] for ref in result.policy_provenance},
            {"rule"},
        )
        self.assertEqual(
            answer_policy_question(DB_PATH, "IT ต้องเรียนกี่หน่วยกิต").value,
            129,
        )

    def test_no_llm_authority_is_required(self):
        result = answer_combined_question(
            DB_PATH,
            "IT แผนไม่สหกิจ ปี 1 เทอม 1 มี 18 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.normal_maximum, 22)


if __name__ == "__main__":
    unittest.main()
