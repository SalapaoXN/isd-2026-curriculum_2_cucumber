import unittest

from src.pipeline.tools.extraction.program_requirements import (
    ProgramRequirementExtractionError,
    extract_program_requirement_from_lines,
    extract_program_requirements,
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


if __name__ == "__main__":
    unittest.main()
