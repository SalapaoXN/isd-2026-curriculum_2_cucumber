import unittest

from src.extractor import CurriculumExtractor


def description_context(page=100):
    return {
        "program": "IT",
        "source_filename": f"it_page_{page:03d}.png",
        "source_page": page,
        "document_category": "description",
    }


class CourseFieldQualityTests(unittest.TestCase):
    def test_two_option_credit_connector_is_canonicalized(self):
        result = CurriculumExtractor(program="DSBA", plan="coop").extract_from_lines(
            [
                "06000001",
                "วิชาทดสอบ",
                "3(3-0-6)หรือ3(2-2-5)",
                "TEST COURSE",
            ]
        )

        self.assertEqual(
            result["courses"][0]["credits"], "3(3-0-6) หรือ 3(2-2-5)"
        )

    def test_exact_duplicate_credit_value_is_collapsed(self):
        result = CurriculumExtractor(program="DSBA", plan="coop").extract_from_lines(
            [
                "06000002",
                "วิชาทดสอบ",
                "3(3-0-6)",
                "3(3-0-6)",
                "TEST COURSE",
            ]
        )

        self.assertEqual(result["courses"][0]["credits"], "3(3-0-6)")

    def test_post_process_collapses_only_exact_duplicate_credit_values(self):
        courses = [
            {"code": "06000003", "credits": "3(3-0-6)3(3-0-6)"},
            {"code": "06000004", "credits": "3(3-0-6)3(2-2-5)"},
        ]

        result = CurriculumExtractor(program="DSBA", plan="coop").post_process(
            courses
        )

        self.assertEqual(
            [course["credits"] for course in result],
            ["3(3-0-6)", "3(3-0-6)3(2-2-5)"],
        )

    def test_three_option_credit_connector_preserves_all_groups(self):
        result = CurriculumExtractor(program="IT", plan="no_coop").extract_from_lines(
            [
                "06000002",
                "วิชาทดสอบ",
                "3(3-0-6)หรือ3(2-2-5)",
                "TEST COURSE",
                "หรือ 3(0-6-3)",
            ]
        )

        self.assertEqual(
            result["courses"][0]["credits"],
            "3(3-0-6) หรือ 3(2-2-5) หรือ 3(0-6-3)",
        )

    def test_plan_suffixes_follow_the_title_language(self):
        result = CurriculumExtractor(program="DSBA", plan="coop").extract_from_lines(
            [
                "06000003",
                "ชื่อไทย",
                "1",
                "3(3-0-6)",
                "ENGLISH TITLE",
                "6",
            ]
        )

        course = result["courses"][0]
        self.assertEqual(course["name_th"], "ชื่อไทย 1")
        self.assertEqual(course["name_en"], "ENGLISH TITLE 6")

    def test_plan_suffix_does_not_repeat_split_credit_number(self):
        result = CurriculumExtractor(program="GENED", plan="gened").extract_from_lines(
            [
                "90644001",
                "ชื่อไทย",
                "1",
                "1",
                "(0-2-1)",
                "ENGLISH TITLE",
                "1",
            ]
        )

        course = result["courses"][0]
        self.assertEqual(course["name_th"], "ชื่อไทย 1")
        self.assertEqual(course["name_en"], "ENGLISH TITLE 1")

    def test_contextual_l_i_and_ii_are_normalized(self):
        result = CurriculumExtractor(program="DSBA", plan="coop").extract_from_lines(
            [
                "06000004",
                "ชื่อไทย",
                "L",
                "3(3-0-6)",
                "ENGLISH TITLE",
                "06000005",
                "ENGLISH TITLE",
                "I",
                "3(3-0-6)",
                "06000006",
                "ชื่อไทย",
                "II",
                "3(3-0-6)",
            ]
        )

        courses = result["courses"]
        self.assertEqual(courses[0]["name_th"], "ชื่อไทย 1")
        self.assertEqual(courses[1]["name_en"], "ENGLISH TITLE 1")
        self.assertEqual(courses[2]["name_th"], "ชื่อไทย 2")

    def test_ambiguous_l_is_rejected(self):
        result = CurriculumExtractor(program="DSBA", plan="coop").extract_from_lines(
            [
                "06000007",
                "ชื่อไทย",
                "L",
                "ENGLISH TITLE",
                "3(3-0-6)",
            ]
        )

        course = result["courses"][0]
        self.assertEqual(course["name_th"], "ชื่อไทย")
        self.assertEqual(course["name_en"], "ENGLISH TITLE")

    def test_description_preserves_four_and_six_without_cross_language_copying(self):
        result = CurriculumExtractor(program="IT", plan="no_coop").extract_descriptions(
            [
                "คำอธิบายรายวิชา",
                "06000008",
                "ชื่อไทย",
                "4",
                "3(3-0-6)",
                "ENGLISH TITLE",
                "6",
                "PREREQUISITE",
                "NONE",
            ],
            description_context(),
        )

        course = result["courses"][0]
        self.assertEqual(course["name_th"], "ชื่อไทย 4")
        self.assertEqual(course["name_en"], "ENGLISH TITLE 6")

    def test_twenty_first_century_number_is_not_a_course_suffix(self):
        result = CurriculumExtractor(program="GENED", plan="gened").extract_descriptions(
            [
                "คำอธิบายรายวิชา",
                "06000009",
                "การบริหารงานในศตวรรษที่",
                "21",
                "3(3-0-6)",
                "PUBLIC ADMINISTRATION IN THE 21st CENTURY",
                "PREREQUISITE",
                "NONE",
            ],
            {
                "program": "GENED",
                "source_filename": "gened_page_100.png",
                "source_page": 100,
                "document_category": "description",
            },
        )

        course = result["courses"][0]
        self.assertEqual(course["name_th"], "การบริหารงานในศตวรรษที่")
        self.assertNotIn("21", course["name_th"])

    def test_missing_raw_suffix_is_not_invented(self):
        result = CurriculumExtractor(program="DSBA", plan="coop").extract_from_lines(
            [
                "06000010",
                "ชื่อไทย",
                "3(3-0-6)",
                "ENGLISH TITLE",
            ]
        )

        course = result["courses"][0]
        self.assertEqual(course["name_th"], "ชื่อไทย")
        self.assertEqual(course["name_en"], "ENGLISH TITLE")


if __name__ == "__main__":
    unittest.main()
