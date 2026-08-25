import unittest
from pathlib import Path

from src.extractor import CurriculumExtractor


def extract_codes(lines):
    result = CurriculumExtractor(program="AIT").extract_from_lines(lines)
    return [course["code"] for course in result["courses"]]


def course_lines(code, title="TEST COURSE"):
    return [code, title, "3(3-0-6)", "TEST COURSE"]


class CourseCodeValidationTests(unittest.TestCase):
    def test_seven_digit_standalone_numeric_code_is_rejected(self):
        self.assertEqual(extract_codes(course_lines("0604640")), [])

    def test_ait_page_24_shape_keeps_following_valid_row(self):
        lines = [
            "ปีที่ 2 ภาคการศึกษาที่ 2",
            "0604640",
            "พื้นฐานวิทยาการข้อมูล",
            "3(3-0-6)",
            "FUNDAMENTALS OF DATA SCIENCE",
            "06046408",
            "การแสดงข้อมูลด้วยแผนภาพ",
            "3(2-2-5)",
            "DATA VISUALIZATION",
        ]

        self.assertEqual(extract_codes(lines), ["06046408"])

    def test_eight_digit_numeric_code_is_accepted(self):
        self.assertEqual(extract_codes(course_lines("06046407")), ["06046407"])

    def test_stored_ait_ocr_keeps_valid_codes_without_repair_mapping(self):
        outputs = Path(__file__).resolve().parents[1] / "outputs"
        page_24 = CurriculumExtractor(program="AIT").process_file(
            outputs / "ait_page_024_ocr.json"
        )
        page_292 = CurriculumExtractor(program="AIT").process_file(
            outputs / "ait_page_292_ocr.json"
        )

        page_24_codes = [course["code"] for course in page_24["courses"]]
        page_292_codes = [course["code"] for course in page_292["courses"]]
        self.assertNotIn("0604640", page_24_codes)
        self.assertIn("06046408", page_24_codes)
        self.assertIn("06046407", page_292_codes)

    def test_nine_digit_numeric_code_is_rejected(self):
        self.assertEqual(extract_codes(course_lines("060464070")), [])

    def test_embedded_short_numeric_code_does_not_start_a_course(self):
        self.assertEqual(
            extract_codes(course_lines("COURSE 0604640")),
            [],
        )

    def test_placeholder_codes_remain_supported(self):
        cases = {
            "06026XX": "06026xxx",
            "06026xxx": "06026xxx",
            "060464xx": "060464xx",
            "9064xxxx": "9064xxxx",
            "xxxxxxxx": "xxxxxxxx",
        }

        for candidate, expected in cases.items():
            with self.subTest(candidate=candidate):
                self.assertEqual(extract_codes(course_lines(candidate)), [expected])


if __name__ == "__main__":
    unittest.main()
