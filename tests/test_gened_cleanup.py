import unittest

from merge_consecutive import merge_plan_with_description
from merge_consecutive import _apply_gened_audit_credit
from merge_consecutive import _recover_credit_from_matching_description
from merge_consecutive import _recover_gened_structural_credit
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
        self.assertIsNone(course["prerequisite"])
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
            [plan],
            [description],
            {"program": "GENED", "plan": "gened"},
        )

        merged = result["courses"][0]

        self.assertEqual(merged["name_th"], "CATALOG THAI")
        self.assertEqual(merged["name_en"], "CATALOG ENGLISH")
        self.assertEqual(merged["prerequisite"], "ไม่มี")
        self.assertEqual(merged["desc_th"], "DESCRIPTION THAI")
        self.assertEqual(merged["desc_en"], "DESCRIPTION ENGLISH")
        self.assertEqual(
            {
                item["document_category"]
                for item in merged["source_provenance"]
            },
            {"plan", "description"},
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

    def test_gened_split_zero_credit_and_no_plan_placement_fields(self):
        extractor = CurriculumExtractor(program="GENED", plan="gened")

        result = extractor.extract_from_lines(
            [
                "90644005",
                "การอ่านและการเขียนเชิงวิชาการ",
                "ACADEMIC READING AND WRITING",
                "0",
                "(4-0-8)",
            ],
            context(26, "plan"),
        )

        course = result["courses"][0]

        self.assertEqual(course["code"], "90644005")
        self.assertEqual(course["credits"], "0(4-0-8)")
        self.assertNotIn("year", course)
        self.assertNotIn("semester", course)
    
    def test_gened_audit_footnote_is_metadata_not_duplicate_courses(self):
        extractor = CurriculumExtractor(program="GENED", plan="gened")

        lines = [
            "90644066",
            "ภาษามาเลย์เพื่อการท่องเที่ยว",
            "3 (3-0-6)",
            "MALAY FOR TRAVEL",
            "90644004 การฟังและการพูดเชิงวิชาการ",
            "ACADEMIC LISTENING AND SPEAKING",
            "90644005 การอ่านและการเขียนเชิงวิชาการ ACADEMIC READING AND WRITING",
            "90644006",
            "ภาษาอังกฤษเพื่อปรับพื้นฐาน",
            "PREPARATORY ENGLISH",
            "รายวิชาดังกล่าวเป็นรายวิชาที่ไม่เก็บหน่วยกิต (AUDIT)",
        ]

        result = extractor.extract_from_lines(lines)

        self.assertEqual(
            [course["code"] for course in result["courses"]],
            ["90644066"],
        )
        self.assertEqual(
            result["audit_course_codes"],
            ["90644004", "90644005", "90644006"],
        )
    
    def test_gened_audit_evidence_recovers_only_missing_leading_zero(self):
        audit_codes = {"90644004", "90644005", "90644006"}

        recovered = _apply_gened_audit_credit(
            {"code": "90644005", "credits": "(4-0-8)"},
            audit_codes,
        )
        self.assertEqual(recovered["credits"], "0(4-0-8)")

        complete = _apply_gened_audit_credit(
            {"code": "90644004", "credits": "0(4-0-8)"},
            audit_codes,
        )
        self.assertEqual(complete["credits"], "0(4-0-8)")

        normal = _apply_gened_audit_credit(
            {"code": "90644066", "credits": "3(3-0-6)"},
            audit_codes,
        )
        self.assertEqual(normal["credits"], "3(3-0-6)")
    
    def test_gened_none_prerequisite_ocr_variants_become_none(self):
        extractor = CurriculumExtractor(program="GENED", plan="gened")

        for value in ("NONE", "NON=", "NOVE"):
            self.assertIsNotNone(
                extractor.GENED_NONE_PREREQ_RE.fullmatch(value)
            )

        self.assertIsNone(
            None if extractor.GENED_NONE_PREREQ_RE.fullmatch("NON=") else "NON="
        )
        self.assertIsNone(
            None if extractor.GENED_NONE_PREREQ_RE.fullmatch("NOVE") else "NOVE"
        )

        self.assertIsNone(
            extractor.GENED_NONE_PREREQ_RE.fullmatch("FOUNDATION ENGLISH")
        )

    def test_credit_recovery_requires_matching_inner_tuple(self):
        self.assertEqual(
            _recover_credit_from_matching_description(
                "(0-3-2)",
                "1(0-3-2)",
            ),
            "1(0-3-2)",
        )

        self.assertEqual(
            _recover_credit_from_matching_description(
                "(0-2-1)",
                "1(0-2-1)",
            ),
            "1(0-2-1)",
        )

        # Conflicting evidence must not be guessed.
        self.assertIsNone(
            _recover_credit_from_matching_description(
                "(4-0-8)",
                "3(3-0-6)",
            )
        )

        # Complete plan credits must never be overwritten.
        self.assertIsNone(
            _recover_credit_from_matching_description(
                "3(3-0-6)",
                "4(4-0-8)",
            )
        )

    def test_gened_structural_credit_recovery_is_conservative(self):
        audit_codes = {
            "90644004",
            "90644005",
            "90644006",
        }

        recovered = _recover_gened_structural_credit(
            {
                "code": "90644038",
                "credits": "(4-0-8)",
                "source_provenance": [
                    {"document_category": "plan"}
                ],
            },
            audit_codes,
        )
        self.assertEqual(
            recovered["credits"],
            "4(4-0-8)",
        )

        # Audit course must never be inferred as 4 credits.
        audit = _recover_gened_structural_credit(
            {
                "code": "90644005",
                "credits": "(4-0-8)",
                "source_provenance": [
                    {"document_category": "plan"}
                ],
            },
            audit_codes,
        )
        self.assertEqual(
            audit["credits"],
            "(4-0-8)",
        )

        # Different workload structure: do not guess.
        unusual = _recover_gened_structural_credit(
            {
                "code": "90641003",
                "credits": "(0-3-2)",
                "source_provenance": [
                    {"document_category": "plan"}
                ],
            },
            audit_codes,
        )
        self.assertEqual(
            unusual["credits"],
            "(0-3-2)",
        )

        # Without Audit context, abstain instead of guessing.
        no_context = _recover_gened_structural_credit(
            {
                "code": "90644038",
                "credits": "(4-0-8)",
                "source_provenance": [
                    {"document_category": "plan"}
                ],
            },
            set(),
        )
        self.assertEqual(
            no_context["credits"],
            "(4-0-8)",
        )

if __name__ == "__main__":
    unittest.main()
