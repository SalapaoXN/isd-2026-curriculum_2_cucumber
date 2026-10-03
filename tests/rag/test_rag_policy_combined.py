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

    def test_unscoped_multiedition_dsba_above_normal_fails_closed(self):
        result = answer_combined_question(
            DB_PATH,
            "DSBA แผนไม่สหกิจ ปี 2 เทอม 2 มี 21 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.resulting_total)
        self.assertFalse(result.curriculum_evidence)

    def test_unscoped_multiedition_dsba_above_exception_fails_closed(self):
        result = answer_combined_question(
            DB_PATH,
            "DSBA แผนไม่สหกิจ ปี 2 เทอม 2 มี 21 หน่วยกิต ถ้าลงเพิ่มอีก 7 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.resulting_total)
        self.assertFalse(result.curriculum_evidence)

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

    def test_unscoped_multiedition_dsba_without_plan_fails_closed(self):
        result = answer_combined_question(
            DB_PATH,
            "DSBA ปี 2 เทอม 2 มี 21 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.resulting_total)
        self.assertFalse(result.curriculum_evidence)

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

    def test_unscoped_multiedition_dsba_does_not_claim_dual_provenance(self):
        result = answer_combined_question(
            DB_PATH,
            "DSBA แผนไม่สหกิจ ปี 2 เทอม 2 มี 21 หน่วยกิต ถ้าลงเพิ่มอีก 3 หน่วยกิตได้ไหม",
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertFalse(result.curriculum_provenance)
        self.assertFalse(result.policy_provenance)
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
