import unittest

from rag.router import route_question


class RagRouterTest(unittest.TestCase):
    def test_structured_rules(self):
        questions = (
            "How many courses are in the program?",
            "Which courses are offered in year 2 semester 1?",
            "What are the credits for this course?",
            "What is the prerequisite for C101?",
            "Compare the credits of C101 and C102.",
            "มีวิชาทั้งหมดกี่วิชาในภาคเรียนที่ 1",
            "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
            "วิชานี้เรียนช่วงไหนของหลักสูตร",
            "หลักสูตรนี้กำหนดแน่นอนหรือยืดหยุ่น",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(route_question(question), "structured")

    def test_semantic_rules_and_safe_default(self):
        questions = (
            "What topics does this course cover?",
            "Explain the meaning of this course.",
            "Which courses are similar to C101?",
            "เนื้อหาของวิชานี้เกี่ยวกับอะไร",
            "Tell me about this course.",
            "ช่วยแนะนำวิชา",
            "วิชานี้เรียนเกี่ยวกับอะไร",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(route_question(question), "semantic")

    def test_combined_rules_use_both_retrieval_capabilities(self):
        self.assertEqual(
            route_question("What database topics are offered in year 1 semester 1?"),
            "hybrid",
        )
        self.assertEqual(
            route_question(
                "สำหรับ IT แบบสหกิจ วิชา 06016418 เรียนช่วงไหนของหลักสูตร "
                "และเนื้อหาครอบคลุมเรื่องใดเกี่ยวกับฐานข้อมูลบ้าง?"
            ),
            "hybrid",
        )

    def test_unmatched_question_keeps_semantic_default(self):
        self.assertEqual(route_question("ข้อมูลนี้มีรายละเอียดอย่างไร"), "semantic")


if __name__ == "__main__":
    unittest.main()
