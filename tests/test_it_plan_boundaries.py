import unittest

from src.extractor import CurriculumExtractor


def extract_plan(lines, program="IT", plan="no_coop"):
    return CurriculumExtractor(program=program, plan=plan).extract_from_lines(lines)


class ItPlanBoundaryTests(unittest.TestCase):
    def test_section_heading_stops_the_preceding_it_title(self):
        result = extract_plan(
            [
                "ปีที่ 2",
                "06066302",
                "การเขียนโปรแกรมเจ็",
                "3(2-2-5)",
                "FUNDAMENTAL WEB PROGRAMMING",
                "กลุ่ม วิชาด้านการพัฒนาซอฟต์แวร์",
                "06016420",
                "ระบบโครงสร้างพื้นฐานและการบริการ",
                "3(2-2-5)",
                "INFRASTRUCTURE SYSTEMS AND SERVICES",
                "กลุ่ม วิชาด้านสื่อประสมสำหรับการพัฒนาสื่อเชิงโต้ตอบ",
                "ว็บ และ เกม",
                "06016418",
                "การพัฒนาเว็บฝังเซิร์ฟเวอร์",
                "3(2-2-5)",
                "SERVER SIDE WEB DEVELOPMENT",
                "กลุ่ม วิชาด้านโครงสร้างพื้นฐานเทคโนโลยีสารสนเทศ",
                "06016423",
                "การออโตเมชันและโครงสร้างพื้นฐานที่สามารถโปรแกรมได้",
                "3(2-2-5)",
                "INFRASTRUCTURE PROGRAMMABILITY AND AUTOMATION",
                "กลุ่ม วิชาด้านสื่อประสมสำหรับการพัฒนาสื่อเชิงโต้ตอบ",
                "06016424",
                "การออกแบบส่วนต่อประสานกับมนุษย์",
                "3(3-0-6)",
                "HUMAN INTERFACE DESIGN",
            ]
        )

        courses = {course["code"]: course for course in result["courses"]}
        self.assertEqual(courses["06066302"]["name_th"], "การเขียนโปรแกรมเจ็")
        self.assertEqual(
            courses["06016420"]["name_th"], "ระบบโครงสร้างพื้นฐานและการบริการ"
        )
        self.assertEqual(
            courses["06016418"]["name_th"], "การพัฒนาเว็บฝังเซิร์ฟเวอร์"
        )
        self.assertEqual(
            courses["06016423"]["name_th"],
            "การออโตเมชันและโครงสร้างพื้นฐานที่สามารถโปรแกรมได้",
        )
        self.assertNotIn("กลุ่ม", courses["06066302"]["name_th"])
        self.assertNotIn("สื่อประสม", courses["06016420"]["name_th"])

    def test_thai_heading_fragment_after_complete_row_is_not_appended(self):
        result = extract_plan(
            [
                "ปีที่ 2",
                "06016415",
                "การเขียนโปรแกรมเชิงฟังก์ชัน",
                "3(2-2-5)",
                "FUNCTIONAL PROGRAMMING",
                "ด้านโครงสร้างพื้นฐานเทคโนโลยีสารสนเทศ",
                "06016419",
                "โครงสร้างพื้นฐานเครือข่ายการสื่อสาร",
                "3(2-2-5)",
                "COMMUNICATION NETWORK INFRASTRUCTURE",
            ]
        )

        courses = {course["code"]: course for course in result["courses"]}
        self.assertEqual(
            courses["06016415"]["name_th"], "การเขียนโปรแกรมเชิงฟังก์ชัน"
        )
        self.assertEqual(courses["06016415"]["name_en"], "FUNCTIONAL PROGRAMMING")

    def test_multiline_thai_and_english_titles_remain_intact(self):
        result = extract_plan(
            [
                "ปีที่ 1",
                "06000001",
                "การวิเคราะห์ระบบ",
                "และการออกแบบ",
                "3(3-0-6)",
                "SYSTEM ANALYSIS",
                "AND DESIGN",
            ]
        )

        course = result["courses"][0]
        self.assertEqual(course["name_th"], "การวิเคราะห์ระบบ และการออกแบบ")
        self.assertEqual(course["name_en"], "SYSTEM ANALYSIS AND DESIGN")

    def test_other_programs_keep_their_existing_basic_plan_semantics(self):
        cases = [
            ("DSBA", "coop", "06000001", 1, 1, "บังคับ", "ไม่มี"),
            ("AIT", None, "06000002", 1, 1, "บังคับ", "ไม่มี"),
            ("GENED", "gened", "90600001", None, None, "เลือก", None),
        ]

        for program, plan, code, year, semester, course_type, prerequisite in cases:
            with self.subTest(program=program):
                result = extract_plan(
                    [
                        "ปีที่ 1",
                        code,
                        "การทดสอบวิชา",
                        "3(3-0-6)",
                        "TEST COURSE",
                    ],
                    program=program,
                    plan=plan,
                )
                course = result["courses"][0]
                self.assertEqual(course["name_th"], "การทดสอบวิชา")
                self.assertEqual(course["name_en"], "TEST COURSE")
                
                if program == "GENED":
                    self.assertNotIn("year", course)
                    self.assertNotIn("semester", course)
                else:
                    self.assertEqual(course["year"], year)
                    self.assertEqual(course["semester"], semester)
                    
                self.assertEqual(course["type"], course_type)
                self.assertEqual(course["prerequisite"], prerequisite)


if __name__ == "__main__":
    unittest.main()
