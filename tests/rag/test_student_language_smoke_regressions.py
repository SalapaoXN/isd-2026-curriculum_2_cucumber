import unittest
from pathlib import Path

from rag.hybrid_demo import answer_question_once


DB_PATH = Path(__file__).parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"


def ask_it(question: str):
    return answer_question_once(
        DB_PATH,
        question,
        conversation_context={"program": "IT", "catalog_key": "it-2565"},
    )


class StudentLanguageSmokeRegressionTest(unittest.TestCase):
    def test_targetless_followups_fail_closed_instead_of_widening(self):
        for question in ("กี่หน่วยอะ", "ตัวนี้เรียนตอนไหน"):
            with self.subTest(question=question):
                result = ask_it(question)
                self.assertEqual(result["status"], "insufficient_evidence", result)
                self.assertEqual(result["provenance"], ())

    def test_policy_wording_cannot_fall_through_to_curriculum_lists(self):
        result = ask_it("โอนหน่วยกิตจากที่อื่นได้มั้ย")
        self.assertEqual(result["status"], "answer", result)
        self.assertIn("ข้อ 28", result["final_answer"])
        self.assertNotIn("ปี 1 ภาคเรียน", result["final_answer"])
        self.assertTrue(result["provenance"])

        personal = ask_it("กูติดหนี้100บาทยังจบได้มั้ย")
        self.assertEqual(personal["status"], "unsupported", personal)
        self.assertEqual(personal["provenance"], ())

    def test_colloquial_plan_totals_use_program_requirement(self):
        cases = (
            ("IT สหกิจนี่รวมกี่หน่วยนะ", 129, "แผนสหกิจ"),
            ("IT ไม่ coop ต้องเรียนกี่หน่วยอะ", 129, "แผนไม่สหกิจ"),
        )
        for question, total, plan_label in cases:
            with self.subTest(question=question):
                result = ask_it(question)
                self.assertEqual(result["status"], "answer", result)
                self.assertIn(str(total), result["final_answer"])
                self.assertIn(plan_label, result["final_answer"])
                self.assertNotIn("ปี 1 ภาคเรียน", result["final_answer"])
                self.assertTrue(result["provenance"])

    def test_topic_placement_is_restricted_to_semantic_topic_matches(self):
        result = ask_it("วิชา database ตัวไหนอยู่ปีไหนบ้าง")
        self.assertEqual(result["status"], "answer", result)
        self.assertTrue(result["provenance"])
        self.assertLess(len(result["final_answer"].splitlines()), 80)
        self.assertTrue(
            any(
                code in result["final_answer"]
                for code in ("06016414", "06016457", "06016458", "06066300")
            ),
            result["final_answer"],
        )

    def test_compound_topics_do_not_truncate_to_legacy_single_tokens(self):
        cases = (
            ("มีวิชาเกี่ยวกับ network security ไหม", "060164"),
            ("มีวิชาเกี่ยวกับ data center ปะ", "06016465"),
            ("หา subject เกี่ยวกับ big data ให้หน่อย", "06016471"),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                result = ask_it(question)
                self.assertEqual(result["status"], "answer", result)
                self.assertTrue(result["provenance"])
                self.assertIn(expected, result["final_answer"])
                self.assertLess(len(result["final_answer"].splitlines()), 80)


if __name__ == "__main__":
    unittest.main()
