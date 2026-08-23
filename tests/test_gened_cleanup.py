import unittest

from merge_consecutive import merge_plan_with_description
from src.extractor import CurriculumExtractor


def context(page, category):
    return {
        "program": "GENED",
        "source_filename": f"gened_page_{page:03d}.png",
        "source_page": page,
        "document_category": category,
    }


def catalog_course(code, page):
    return {
        "code": code,
        "name_th": "CATALOG THAI",
        "name_en": "CATALOG ENGLISH",
        "credits": "3(3-0-6)",
        "year": 0,
        "semester": 0,
        "category": "หมวดวิชาศึกษาทั่วไป",
        "type": "เลือก",
        "prerequisite": None,
        "source_provenance": [context(page, "plan")],
    }


def description_course(code, page):
    return {
        "code": code,
        "prerequisite": "ไม่มี",
        "desc_th": "DESCRIPTION THAI",
        "desc_en": "DESCRIPTION ENGLISH",
        "source_provenance": [context(page, "description")],
    }


class GenEdCleanupTests(unittest.TestCase):
    def test_numeric_section_heading_does_not_bleed_into_gened_catalog_course(self):
        extractor = CurriculumExtractor(program="GENED", plan="gened")
        result = extractor.extract_from_lines(
            [
                "90641003",
                "กีฬาและนันทนาการ",
                "1",
                "(0-3-2)",
                "SPORTS AND RECREATIONAL ACTIVITIES",
                "3.5.2 กลุ่มทักษะด้านบุคคลและทักษะส่งเสริมวิชาชีพ (PERSONAL AND PROFESSIONAL",
                "SKILLS)",
                "90642001",
                "ปฏิบัติงานตามทักษะด้านบุคคล",
                "1 (0-2-1)",
                "PRACTICE UNDER PERSONAL SKILLS",
            ],
            context(16, "plan"),
        )

        courses = {course["code"]: course for course in result["courses"]}
        course = courses["90641003"]
        self.assertEqual(course["name_th"], "กีฬาและนันทนาการ 1")
        self.assertEqual(course["name_en"], "SPORTS AND RECREATIONAL ACTIVITIES")
        self.assertNotIn("3.5.2", course["name_th"])
        self.assertNotIn("SKILLS)", course["name_en"])
        self.assertEqual(
            course["source_provenance"][0]["document_category"], "plan"
        )
        self.assertIn("90642001", courses)

    def test_gened_description_fields_prerequisite_and_provenance(self):
        extractor = CurriculumExtractor(program="GENED", plan="gened")
        result = extractor.extract_descriptions(
            [
                "คำอธิบายรายวิชา",
                "90600001",
                "วิชาทดสอบ",
                "3 (3-0-6)",
                "TEST COURSE",
                "รายวิชาบังคับก่อน",
                "ไม่มี",
                "PREREQUISITE",
                "NONE",
                "คำอธิบายภาษาไทย",
                "DESCRIPTION ENGLISH",
            ],
            context(44, "description"),
        )

        course = result["courses"][0]
        self.assertEqual(course["prerequisite"], "ไม่มี")
        self.assertEqual(course["desc_th"], "คำอธิบายภาษาไทย")
        self.assertEqual(course["desc_en"], "DESCRIPTION ENGLISH")
        self.assertEqual(
            course["source_provenance"], [context(44, "description")]
        )

    def test_gened_cross_page_continuation_preserves_description_provenance(self):
        extractor = CurriculumExtractor(program="GENED", plan="gened")
        result = extractor.extract_descriptions_from_pages(
            [
                (
                    [
                        "คำอธิบายรายวิชา",
                        "90600001",
                        "วิชาแรก",
                        "3 (3-0-6)",
                        "FIRST COURSE",
                        "PREREQUISITE",
                        "NONE",
                        "เนื้อหาภาษาไทย",
                        "ENGLISH BODY",
                    ],
                    context(44, "description"),
                ),
                (
                    [
                        "CONTINUATION",
                        "90600002",
                        "วิชาที่สอง",
                        "3 (3-0-6)",
                        "SECOND COURSE",
                        "PREREQUISITE",
                        "NONE",
                    ],
                    context(45, "description"),
                ),
            ]
        )

        first = result["courses"][0]
        self.assertEqual(first["desc_en"], "ENGLISH BODY\nCONTINUATION")
        self.assertEqual(
            [entry["source_page"] for entry in first["source_provenance"]],
            [44, 45],
        )

    def test_gened_unique_merge_keeps_catalog_fields_and_adds_description_fields(self):
        plan = catalog_course("90600001", 16)
        description = description_course("90600001", 44)

        result = merge_plan_with_description(
            [plan], [description], {"program": "GENED", "plan": "gened"}
        )

        merged = result["courses"][0]
        self.assertEqual(merged["name_th"], "CATALOG THAI")
        self.assertEqual(merged["name_en"], "CATALOG ENGLISH")
        self.assertEqual(merged["prerequisite"], "ไม่มี")
        self.assertEqual(merged["desc_th"], "DESCRIPTION THAI")
        self.assertEqual(merged["desc_en"], "DESCRIPTION ENGLISH")
        self.assertEqual(
            [entry["document_category"] for entry in merged["source_provenance"]],
            ["plan", "description"],
        )

    def test_gened_repeated_catalog_code_is_not_occurrence_paired(self):
        plans = [catalog_course("90644004", 26), catalog_course("90644004", 30)]
        description = description_course("90644004", 101)

        result = merge_plan_with_description(
            plans, [description], {"program": "GENED", "plan": "gened"}
        )

        self.assertEqual(len(result["courses"]), 2)
        self.assertEqual([course.get("desc_en") for course in result["courses"]], [None, None])
        self.assertEqual(
            [item["code"] for item in result["unresolved_descriptions"]],
            ["90644004"],
        )

    def test_gened_unmatched_description_is_preserved_as_unresolved(self):
        plan = catalog_course("90600001", 16)
        description = description_course("90699999", 117)

        result = merge_plan_with_description(
            [plan], [description], {"program": "GENED", "plan": "gened"}
        )

        self.assertEqual([course["code"] for course in result["courses"]], ["90600001"])
        self.assertEqual(
            [item["code"] for item in result["unresolved_descriptions"]],
            ["90699999"],
        )
        self.assertEqual(
            result["unresolved_descriptions"][0]["source_provenance"],
            [context(117, "description")],
        )


if __name__ == "__main__":
    unittest.main()
