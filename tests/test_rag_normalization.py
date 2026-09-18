import unittest

from rag.normalization import normalize_thai_surface


class ThaiSurfaceNormalizationTests(unittest.TestCase):
    def test_direct_mappings(self):
        cases = {
            "ปีสอง": "ปี 2",
            "ปีสาม": "ปี 3",
            "ตอนปีสาม": "ปี 3",
            "เทอมสอง": "เทอม 2",
            "เทอมปลาย": "เทอม 2",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(normalize_thai_surface(question), expected)

    def test_combined_question_preserves_meaning(self):
        self.assertEqual(
            normalize_thai_surface("IT ปีสามเทอมปลายเรียนหนักไหม"),
            "IT ปี 3 เทอม 2 เรียนหนักไหม",
        )
        self.assertEqual(
            normalize_thai_surface("ปีสองเทอมสองของ IT มีวิชาอะไร"),
            "ปี 2 เทอม 2 ของ IT มีวิชาอะไร",
        )

    def test_semantic_terms_are_not_rewritten(self):
        question = "คอม data หนัก"
        self.assertEqual(normalize_thai_surface(question), question)

    def test_already_normalized_input_is_unchanged(self):
        question = "IT ปี 3 เทอม 2"
        self.assertEqual(normalize_thai_surface(question), question)

    def test_unrelated_text_and_course_code_are_preserved(self):
        question = "IT 06016420 TCP/IP MODEL data คอม หนัก"
        self.assertEqual(normalize_thai_surface(question), question)


if __name__ == "__main__":
    unittest.main()
