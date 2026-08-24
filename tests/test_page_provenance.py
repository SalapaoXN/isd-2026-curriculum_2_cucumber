import json
import tempfile
import unittest
from pathlib import Path

from merge_consecutive import merge_plan_with_description
from src.extractor import (
    CurriculumExtractor,
    document_page_from_lines,
    merge_source_provenance,
    parse_document_page,
)
from src.file_handler import save_ocr_results


def context(program, page, document_page, category):
    return {
        "program": program,
        "source_filename": f"{program.lower()}_page_{page:03d}.png",
        "source_page": page,
        "document_page": document_page,
        "document_category": category,
    }


class PageProvenanceTests(unittest.TestCase):
    @staticmethod
    def source_context(program, page, category, lines, document_page=None):
        extractor = CurriculumExtractor(
            program=program,
            plan="gened" if program == "GENED" else "no_coop",
        )
        metadata = {
            "program": program,
            "source_filename": f"{program.lower()}_page_{page:03d}.png",
            "source_page": page,
        }
        if document_page is not None:
            metadata["document_page"] = document_page
        return extractor._source_context(
            metadata=metadata,
            document_category=category,
            text_lines=lines,
        )

    def test_numeric_first_line_is_document_page(self):
        self.assertEqual(document_page_from_lines(["", "27", "content"]), 27)

    def test_page_number_in_first_boundary_lines_is_document_page(self):
        self.assertEqual(
            document_page_from_lines(
                ["header", "section", "หน้า 27", "content", "more"]
            ),
            27,
        )

    def test_page_number_in_last_boundary_lines_is_document_page(self):
        self.assertEqual(
            document_page_from_lines(
                ["90641001", "course", "3(3-0-6)", "description", "16"]
            ),
            16,
        )

    def test_thai_prefix_is_document_page(self):
        self.assertEqual(parse_document_page("หน้า 27"), 27)

    def test_thai_digits_are_document_page(self):
        self.assertEqual(parse_document_page("หน้า ๒๗"), 27)

    def test_malformed_or_non_page_first_line_is_unresolved(self):
        self.assertIsNone(document_page_from_lines(["06046400", "course name"]))
        self.assertIsNone(document_page_from_lines(["Page 27"]))
        self.assertIsNone(parse_document_page("หน้า 27 extra"))

    def test_course_code_in_middle_of_page_is_not_document_page(self):
        self.assertIsNone(
            document_page_from_lines(
                ["header", "06046400", "course", "credits", "footer"]
            )
        )

    def test_conflicting_boundary_page_numbers_are_unresolved(self):
        self.assertIsNone(
            document_page_from_lines(["27", "header", "content", "footer", "28"])
        )

    def test_duplicate_boundary_page_numbers_are_accepted(self):
        self.assertEqual(
            document_page_from_lines(["27", "header", "content", "footer", "หน้า ๒๗"]),
            27,
        )

    def test_malformed_boundary_text_is_unresolved(self):
        self.assertIsNone(
            document_page_from_lines(
                ["หน้า 27 extra", "content", "footer text", "page twenty"]
            )
        )

    def test_totals_and_year_markers_are_not_document_pages(self):
        self.assertEqual(
            document_page_from_lines(["27", "ปีที่", "1", "รวม", "18", "footer"]),
            27,
        )

    def test_credit_values_in_boundary_are_not_document_pages(self):
        self.assertEqual(
            document_page_from_lines(
                [
                    "continuation",
                    "90642062",
                    "course name",
                    "3",
                    "(3-0-6)",
                    "description",
                    "footer",
                    "59",
                ]
            ),
            59,
        )

    def test_context_words_in_normal_text_do_not_hide_footer(self):
        self.assertEqual(
            document_page_from_lines(
                ["header", "course name", "description with หน่วยกิต", "26"]
            ),
            26,
        )

    def test_gened_description_verified_offset_fallback_for_audited_pages(self):
        cases = {
            74: ["7O", "YOUR", "GRAPHICS"],
            81: ["course body without a footer"],
            86: [
                "leading 1",
                "leading 2",
                "leading 3",
                "leading 4",
                "leading 5",
                "82",
                "trailing 1",
                "trailing 2",
                "trailing 3",
                "trailing 4",
                "trailing 5",
            ],
            104: ["course body", "1OO"],
            111: [
                "leading 1",
                "leading 2",
                "leading 3",
                "leading 4",
                "leading 5",
                "107",
                "trailing 1",
                "trailing 2",
                "trailing 3",
                "trailing 4",
                "trailing 5",
                "trailing 6",
            ],
        }
        expected = {74: 70, 81: 77, 86: 82, 104: 100, 111: 107}

        for source_page, lines in cases.items():
            with self.subTest(source_page=source_page):
                result = self.source_context("GENED", source_page, "description", lines)
                self.assertEqual(result["document_page"], expected[source_page])
                self.assertEqual(result["source_page"], source_page)

    def test_valid_gened_ocr_page_is_not_overridden(self):
        result = self.source_context("GENED", 74, "description", ["70", "content"])

        self.assertEqual(result["document_page"], 70)
        self.assertEqual(result["source_page"], 74)

    def test_conflicting_gened_ocr_pages_remain_unresolved(self):
        result = self.source_context(
            "GENED",
            74,
            "description",
            ["70", "header", "content", "footer", "71"],
        )

        self.assertIsNone(result["document_page"])
        self.assertEqual(result["source_page"], 74)

    def test_gened_catalog_page_does_not_use_description_fallback(self):
        result = self.source_context("GENED", 74, "plan", ["7O", "content"])

        self.assertIsNone(result["document_page"])
        self.assertEqual(result["source_page"], 74)

    def test_non_gened_description_page_does_not_use_fallback(self):
        result = self.source_context("IT", 74, "description", ["7O", "content"])

        self.assertIsNone(result["document_page"])
        self.assertEqual(result["source_page"], 74)

    def test_valid_gened_metadata_page_is_not_overridden(self):
        result = self.source_context(
            "GENED", 74, "description", ["71", "content"], document_page=70
        )

        self.assertEqual(result["document_page"], 70)
        self.assertEqual(result["source_page"], 74)

    def test_ocr_metadata_and_course_context_preserve_source_page(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_ocr_results(
                ["27", "ปีที่ 1", "06000001", "ชื่อวิชา", "3(3-0-6)"],
                output_dir,
                "it_page_032",
                source_filename="it_page_032.png",
                source_page=32,
                program="IT",
            )

            raw = json.loads(
                (output_dir / "it_page_032_ocr.json").read_text(encoding="utf-8")
            )
            result = CurriculumExtractor(program="IT", plan="no_coop").process_file(
                output_dir / "it_page_032_ocr.json"
            )

        self.assertEqual(raw["source_page"], 32)
        self.assertEqual(raw["document_page"], 27)
        self.assertEqual(
            result["courses"][0]["source_provenance"][0]["source_page"], 32
        )
        self.assertEqual(
            result["courses"][0]["source_provenance"][0]["document_page"], 27
        )

    def test_document_page_survives_plan_description_merge(self):
        plan = {
            "code": "06000001",
            "name_th": "PLAN",
            "name_en": "PLAN",
            "credits": "3(3-0-6)",
            "year": 1,
            "semester": 1,
            "prerequisite": "ไม่มี",
            "source_provenance": [context("IT", 32, 27, "plan")],
        }
        description = {
            "code": "06000001",
            "desc_th": "คำอธิบาย",
            "desc_en": "DESCRIPTION",
            "prerequisite": "ไม่มี",
            "source_provenance": [context("IT", 328, 323, "description")],
        }

        result = merge_plan_with_description(
            [plan], [description], {"program": "IT", "plan": "no_coop"}
        )

        self.assertEqual(
            [
                (entry["source_page"], entry["document_page"])
                for entry in result["courses"][0]["source_provenance"]
            ],
            [(32, 27), (328, 323)],
        )

    def test_cross_page_description_keeps_each_document_page(self):
        extractor = CurriculumExtractor(program="IT", plan="no_coop")
        result = extractor.extract_descriptions_from_pages(
            [
                (
                    [
                        "323",
                        "คำอธิบายรายวิชา",
                        "06000001",
                        "ชื่อวิชา",
                        "3(3-0-6)",
                        "COURSE NAME",
                        "PREREQUISITE",
                        "NONE",
                        "คำอธิบายภาษาไทย",
                        "ENGLISH BODY",
                    ],
                    context("IT", 328, 323, "description"),
                ),
                (
                    [
                        "324",
                        "มคอ.",
                        "2",
                        "ENGLISH BODY",
                        "CONTINUATION",
                        "06000002",
                        "วิชาที่สอง",
                        "3(3-0-6)",
                        "SECOND COURSE",
                        "PREREQUISITE",
                        "NONE",
                    ],
                    context("IT", 329, 324, "description"),
                ),
            ]
        )

        first = result["courses"][0]
        self.assertEqual(first["desc_en"], "ENGLISH BODY\nCONTINUATION")
        self.assertEqual(
            [
                (entry["source_page"], entry["document_page"])
                for entry in first["source_provenance"]
            ],
            [(328, 323), (329, 324)],
        )

    def test_old_provenance_without_document_page_remains_valid(self):
        old_entry = {
            "program": "IT",
            "source_filename": "it_page_032.png",
            "source_page": 32,
            "document_category": "plan",
        }

        self.assertEqual(
            merge_source_provenance({"source_provenance": [old_entry]}),
            [old_entry],
        )

    def test_provenance_order_and_deduplication_are_unchanged(self):
        first = context("IT", 328, 323, "description")
        second = context("IT", 329, 324, "description")
        duplicate_without_page = dict(first)
        duplicate_without_page.pop("document_page")

        merged = merge_source_provenance(
            {"source_provenance": [first, second, duplicate_without_page]}
        )

        self.assertEqual(
            [(entry["source_page"], entry["document_page"]) for entry in merged],
            [(328, 323), (329, 324)],
        )


if __name__ == "__main__":
    unittest.main()
