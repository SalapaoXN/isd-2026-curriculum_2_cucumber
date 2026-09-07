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
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(route_question(question), "semantic")


if __name__ == "__main__":
    unittest.main()
