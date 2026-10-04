import unittest

from src.pipeline.tools.extraction.program_requirements import (
    ProgramRequirementExtractionError,
    extract_explicit_total_credits,
    extract_program_requirement_from_lines,
    extract_program_requirements,
    extract_total_credit_requirement,
    merge_plan_requirements,
)


class FakeOCREngine:
    def __init__(self, pages):
        self.pages = pages

    def extract_text(self, image_path, detail=0):
        return self.pages[image_path.name]


class ProgramRequirementsTests(unittest.TestCase):
    def test_extracts_bounded_credit_value_with_provenance(self):
        result = extract_program_requirement_from_lines(
            "IT",
            [
                "หมวดที่ 1 ข้อมูลทั่วไป",
                "4. จำนวนหน่วยกิตที่เรียนตลอดหลักสูตร",
                "129 หน่วยกิต",
            ],
            source_filename="it_page_006.png",
            source_page=6,
            document_page=1,
        )

        self.assertEqual(result["program"], "IT")
        self.assertEqual(result["requirement_type"], "total_program_credits")
        self.assertEqual(result["operator"], "=")
        self.assertEqual(result["value"], 129)
        self.assertEqual(result["unit"], "credits")
        self.assertEqual(result["source_provenance"][0]["source_page"], 6)
        self.assertEqual(result["source_provenance"][0]["document_page"], 1)
        self.assertEqual(
            result["source_provenance"][0]["document_category"],
            "program_requirement",
        )

    def test_extracts_thai_digit_value(self):
        result = extract_program_requirement_from_lines(
            "DSBA",
            ["จำนวนหน่วยกิตที่เรียนตลอดหลักสูตร", "๑๓๒ หน่วยกิต"],
            source_filename="dsba_page_006.png",
            source_page=6,
        )
        self.assertEqual(result["value"], 132)

    def test_zero_candidates_fail_closed(self):
        with self.assertRaises(ProgramRequirementExtractionError):
            extract_program_requirement_from_lines(
                "AIT",
                ["รายละเอียดหลักสูตร", "120 หน่วยกิต"],
                source_filename="ait_page_005.png",
                source_page=5,
            )

    def test_multiple_bounded_values_fail_closed(self):
        with self.assertRaises(ProgramRequirementExtractionError):
            extract_program_requirement_from_lines(
                "BIT",
                [
                    "จำนวนหน่วยกิตที่เรียนตลอดหลักสูตร",
                    "126 หน่วยกิต",
                    "129 หน่วยกิต",
                ],
                source_filename="bit_page_006.png",
                source_page=6,
            )

    def test_all_configured_sources_are_extracted_independently(self):
        pages = {
            "ait_page_005.png": ["จำนวนหน่วยกิตที่เรียนตลอดหลักสูตร", "120 หน่วยกิต"],
            "bit_page_006.png": ["จำนวนหน่วยกิตที่เรียนตลอดหลักสูตร", "126 หน่วยกิต"],
            "dsba_page_006.png": ["จำนวนหน่วยกิตที่เรียนตลอดหลักสูตร", "132 หน่วยกิต"],
            "it_page_006.png": ["จำนวนหน่วยกิตที่เรียนตลอดหลักสูตร", "129 หน่วยกิต"],
        }
        results = extract_program_requirements(
            "data/input/rule",
            ocr_engine=FakeOCREngine(pages),
        )

        self.assertEqual(
            {item["program"]: item["value"] for item in results},
            {"AIT": 120, "BIT": 126, "DSBA": 132, "IT": 129},
        )
        self.assertEqual(
            [item["source_provenance"][0]["source_filename"] for item in results],
            [
                "ait_page_005.png",
                "bit_page_006.png",
                "dsba_page_006.png",
                "it_page_006.png",
            ],
        )


class PlanTotalRequirementTests(unittest.TestCase):
    @staticmethod
    def provenance(plan, source_filename="dsba_page_029.png", source_page=29):
        return [{
            "program": "DSBA",
            "source_filename": source_filename,
            "source_page": source_page,
            "document_category": "plan",
            "plan": plan,
        }]

    def requirement(self, value, plan, source_page):
        return extract_total_credit_requirement(
            f"รวมตลอดหลักสูตร\n{value}\nหน่วยกิต",
            requirement_type="total_program_credits",
            catalog_key="dsba-2560",
            program="DSBA",
            source_provenance=self.provenance(
                plan, f"dsba2560_page_{source_page:03d}.png", source_page
            ),
        )

    def test_accepts_single_authoritative_plan_page(self):
        result = extract_total_credit_requirement(
            "รวมตลอดหลักสูตร 126 หน่วยกิต",
            requirement_type="total_program_credits",
            catalog_key="dsba-2560",
            program="DSBA",
            source_provenance=self.provenance("no_coop"),
        )
        self.assertEqual(result["value"], 126)
        self.assertEqual(result["catalog_key"], "dsba-2560")

    def test_accepts_multiline_total_and_thai_digits(self):
        self.assertEqual(
            extract_explicit_total_credits("รวมตลอดหลักสูตร\n ๑๒๖ \nหน่วยกิต"),
            126,
        )

    def test_accepts_casefolded_english_total_anchor(self):
        self.assertEqual(
            extract_explicit_total_credits("TOTAL CREDITS FOR THE PROGRAM\n129 credits"),
            129,
        )

    def test_equal_plan_totals_merge_and_retain_both_provenance_entries(self):
        result = merge_plan_requirements([
            self.requirement(126, "no_coop", 29),
            self.requirement(126, "coop", 34),
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["value"], 126)
        self.assertEqual(
            [entry["plan"] for entry in result[0]["source_provenance"]],
            ["no_coop", "coop"],
        )

    def test_conflicting_plan_totals_fail_closed(self):
        with self.assertRaises(ProgramRequirementExtractionError):
            merge_plan_requirements([
                self.requirement(126, "no_coop", 29),
                self.requirement(132, "coop", 34),
            ])

    def test_missing_anchor_emits_no_requirement(self):
        result = extract_total_credit_requirement(
            "หลักสูตรนี้มีหน่วยกิต 126 หน่วยกิต",
            requirement_type="total_program_credits",
            catalog_key="dsba-2560",
            program="DSBA",
            source_provenance=self.provenance("no_coop"),
        )
        self.assertIsNone(result)

    def test_bare_number_is_not_a_requirement(self):
        self.assertIsNone(extract_explicit_total_credits("126 หน่วยกิต"))

    def test_audited_terminal_page_excerpts_have_expected_totals(self):
        sources = {
            "DSBA2560 no_coop": (
                "รวมตลอดหลักสูตร\n126\nหน่วยกิต", 126
            ),
            "DSBA2560 coop": (
                "06026130\nสหกิจศึกษา\nรวมตลอดหลักสูตร 126 หน่วยกิต", 126
            ),
            "DSBA2565 no_coop": (
                "ปีที่ 4 ภาคการศึกษาที่ 2\nรวมตลอดหลักสูตร\n132\nหน่วยกิต", 132
            ),
            "DSBA2565 coop": (
                "06026259 สหกิจศึกษา\nรวมตลอดหลักสูตร 132 หน่วยกิต", 132
            ),
            "IT no_coop": ("รวมตลอดหลักสูตร 129 หน่วยกิต", 129),
            "IT coop": ("รวมตลอดหลักสูตร\n129\nหน่วยกิต", 129),
            "BIT no_coop": ("รวมตลอดหลักสูตร 126 หน่วยกิต", 126),
            "BIT coop": ("สหกิจศึกษา\nรวมตลอดหลักสูตร 126 หน่วยกิต", 126),
            "AIT": ("รวมตลอดหลักสูตร\n120\nหน่วยกิต", 120),
        }
        for label, (ocr_excerpt, expected) in sources.items():
            with self.subTest(source=label):
                self.assertEqual(
                    extract_explicit_total_credits(ocr_excerpt), expected
                )

    def test_audited_dsba_editions_remain_separate_with_plan_provenance(self):
        cases = [
            ("dsba-2560", "no_coop", 126, "dsba2560_page_029.png", 29, None,
             "รวมตลอดหลักสูตร 126 หน่วยกิต"),
            ("dsba-2560", "coop", 126, "dsba2560_page_034.png", 34, 29,
             "รวมตลอดหลักสูตร 126 หน่วยกิต"),
            ("dsba-2565", "no_coop", 132, "dsba_page_032.png", 32, 27,
             "รวมตลอดหลักสูตร 132 หน่วยกิต"),
            ("dsba-2565", "coop", 132, "dsba_page_039.png", 39, 34,
             "รวมตลอดหลักสูตร 132 หน่วยกิต"),
        ]
        records = []
        for catalog_key, plan, expected, filename, source_page, document_page, text in cases:
            provenance = [{
                "program": "DSBA",
                "source_filename": filename,
                "source_page": source_page,
                "document_page": document_page,
                "document_category": "plan",
                "plan": plan,
            }]
            record = extract_total_credit_requirement(
                text,
                requirement_type="total_program_credits",
                catalog_key=catalog_key,
                program="DSBA",
                source_provenance=provenance,
            )
            self.assertEqual(record["value"], expected)
            records.append(record)

        merged = merge_plan_requirements(records)
        self.assertEqual(
            {record["catalog_key"]: record["value"] for record in merged},
            {"dsba-2560": 126, "dsba-2565": 132},
        )
        self.assertEqual(
            [len(record["source_provenance"]) for record in merged], [2, 2]
        )

    def test_gened_requires_an_explicit_supported_total(self):
        explicit = extract_total_credit_requirement(
            "รวมหน่วยกิตหมวดวิชาศึกษาทั่วไป 30 หน่วยกิต",
            requirement_type="general_education_total_credits",
            scope="GENED",
            source_provenance=[{
                "program": "GENED",
                "source_filename": "gened_page_030.png",
                "source_page": 30,
                "document_category": "plan",
                "plan": "gened",
            }],
        )
        self.assertEqual(explicit["scope"], "GENED")
        self.assertEqual(explicit["requirement_type"], "general_education_total_credits")
        self.assertEqual(explicit["value"], 30)

        record = extract_total_credit_requirement(
            "รายวิชาดังกล่าวเป็นรายวิชาที่ไม่เก็บหน่วยกิต (AUDIT)",
            requirement_type="general_education_total_credits",
            scope="GENED",
            source_provenance=[{
                "program": "GENED",
                "source_filename": "gened_page_030.png",
                "source_page": 30,
                "document_category": "plan",
                "plan": "gened",
            }],
        )
        self.assertIsNone(record)


if __name__ == "__main__":
    unittest.main()
