import json
import tempfile
import unittest
from pathlib import Path

from extract_rules import main
from src.rule_extractor import RuleExtractor, RulePage, discover_rule_ocr_files


def page(number, lines, filename=None, metadata=None):
    return RulePage(
        lines=lines,
        source_filename=filename or f"rule_page_{number:03d}_ocr.txt",
        source_page=number,
        metadata=metadata or {},
    )


class RuleExtractorTests(unittest.TestCase):
    def test_thai_and_arabic_anchors_normalize_identifiers(self):
        result = RuleExtractor().extract_from_pages(
            [
                page(
                    1,
                    [
                        "หมวด ๖",
                        "การวัดและประมวลผลการศึกษา",
                        "ข้อ ๑๙ การวัดผลการศึกษา",
                        "๑๙.๔ ให้ใช้ค่าระดับคะแนนตามข้อกำหนด",
                        "19.5 การให้ค่าระดับคะแนน I",
                    ],
                )
            ]
        )

        self.assertEqual(
            [rule["rule_id"] for rule in result["rules"]],
            ["rule:19", "rule:19.4", "rule:19.5"],
        )
        self.assertEqual(result["rules"][0]["category"], "หมวด 6 การวัดและประมวลผลการศึกษา")

    def test_nested_hierarchy_derives_paths_and_parents(self):
        result = RuleExtractor().extract_from_lines(
            [
                "ข้อ ๓๗ วินัยนักศึกษา",
                "๓๗.๖ ความผิดวินัยอย่างร้ายแรง มีดังนี้",
                "๓๗.๖.๑ การกลั่นแกล้งจนเป็นเหตุให้ผู้อื่นได้รับความเสียหาย",
            ],
            source_page=10,
        )

        records = {rule["section_number"]: rule for rule in result["rules"]}
        self.assertEqual(records["37"]["section_path"], ["37"])
        self.assertIsNone(records["37"]["parent_rule_id"])
        self.assertEqual(records["37.6"]["section_path"], ["37", "37.6"])
        self.assertEqual(records["37.6"]["parent_rule_id"], "rule:37")
        self.assertEqual(
            records["37.6.1"]["section_path"], ["37", "37.6", "37.6.1"]
        )
        self.assertEqual(records["37.6.1"]["parent_rule_id"], "rule:37.6")

    def test_chapter_and_special_headings_are_metadata_not_rules(self):
        result = RuleExtractor().extract_from_lines(
            [
                "หมวด ๑",
                "บททั่วไป",
                "ข้อ ๕ ในข้อบังคับนี้",
                "หมวด ๒ การจัดการศึกษา",
                "ข้อ ๖ ระบบการจัดการศึกษา",
                "บทเฉพาะกาล",
                "ข้อ ๕๒ กรณีเฉพาะ",
            ]
        )

        self.assertEqual(result["total_rules"], 3)
        self.assertEqual(result["rules"][0]["category"], "หมวด 1 บททั่วไป")
        self.assertEqual(result["rules"][1]["category"], "หมวด 2 การจัดการศึกษา")
        self.assertEqual(result["rules"][2]["category"], "บทเฉพาะกาล")

    def test_cross_page_continuation_preserves_all_rule_pages(self):
        result = RuleExtractor().extract_from_pages(
            [
                page(5, ["ข้อ ๑๙ การวัดผลการศึกษา", "ข้อความหน้าแรก"]),
                page(6, ["๖", "ข้อความต่อเนื่องหน้าใหม่", "19.1 รายการย่อย"]),
            ]
        )

        rule_19, rule_19_1 = result["rules"]
        self.assertEqual(
            [entry["source_page"] for entry in rule_19["source_provenance"]],
            [5, 6],
        )
        self.assertIn("ข้อความต่อเนื่องหน้าใหม่", rule_19["rule_text"])
        self.assertEqual(
            [entry["source_page"] for entry in rule_19_1["source_provenance"]],
            [6],
        )

    def test_new_page_new_rule_does_not_merge_with_previous_rule(self):
        result = RuleExtractor().extract_from_pages(
            [
                page(1, ["ข้อ ๑ กฎหนึ่ง", "ข้อความหนึ่ง"]),
                page(2, ["๒", "ข้อ ๒ กฎสอง", "ข้อความสอง"]),
            ]
        )

        self.assertEqual([rule["section_number"] for rule in result["rules"]], ["1", "2"])
        self.assertNotIn("ข้อความสอง", result["rules"][0]["rule_text"])

    def test_table_text_stays_in_the_active_rule(self):
        result = RuleExtractor().extract_from_lines(
            [
                "ข้อ ๑๙.๓ การให้ค่าระดับคะแนน",
                "ค่าระดับคะแนน  แต้ม  ผลการศึกษา",
                "A  ๔.๐๐  ดีเลิศ (Excellent)",
                "B+ ๓.๕๐  ดีมาก (Very Good)",
            ],
            source_page=5,
        )

        self.assertEqual(result["total_rules"], 1)
        self.assertIn("A  ๔.๐๐  ดีเลิศ (Excellent)", result["rules"][0]["rule_text"])
        self.assertIn("B+ ๓.๕๐  ดีมาก (Very Good)", result["rules"][0]["rule_text"])

    def test_obvious_page_separator_and_signature_noise_is_filtered(self):
        result = RuleExtractor().extract_from_lines(
            [
                "ข้อ ๕๓ การใช้บังคับ",
                "ข้อความของบทเฉพาะกาล",
                ".................",
                "๑๓",
                "ประกาศ ณ วันที่ ๒๐ สิงหาคม พ.ศ. ๒๕๖๔",
                "(ศาสตราจารย์พิเศษ ผู้ลงนาม)",
            ],
            source_page=13,
        )

        self.assertEqual(result["total_rules"], 1)
        self.assertNotIn("ประกาศ", result["rules"][0]["rule_text"])
        self.assertNotIn("ศาสตราจารย์", result["rules"][0]["rule_text"])

    def test_explicit_rule_references_are_normalized_and_deduplicated(self):
        result = RuleExtractor().extract_from_lines(
            [
                "ข้อ ๔๐ การลงโทษ",
                "ให้เป็นไปตามข้อ ๓๗.๖ และข้อ ๓๗.๖ รวมถึงข้อ ๒๐ วรรคสอง",
            ],
            source_page=11,
        )

        self.assertEqual(result["rules"][0]["references"], ["37.6", "20"])

    def test_provenance_contains_source_filename_page_and_category(self):
        result = RuleExtractor().extract_from_lines(
            ["ข้อ ๑ กฎหนึ่ง"],
            source_filename="rule_page_001_ocr.json",
            source_page=1,
            metadata={"program": "RULE"},
        )

        self.assertEqual(
            result["rules"][0]["source_provenance"],
            [
                {
                    "source_filename": "rule_page_001_ocr.json",
                    "source_page": 1,
                    "document_category": "rule",
                    "program": "RULE",
                }
            ],
        )

    def test_ids_are_deterministic(self):
        pages = [page(2, ["ข้อ ๒ กฎสอง", "ข้อความ"])]
        first = RuleExtractor().extract_from_pages(pages)
        second = RuleExtractor().extract_from_pages(pages)
        self.assertEqual(first, second)

    def test_cli_prefers_json_and_sorts_pages_numerically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "ocr"
            input_dir.mkdir()
            (input_dir / "rule_page_010_ocr.txt").write_text(
                "ข้อ ๑๐ TXT ควรถูกแทนที่", encoding="utf-8"
            )
            (input_dir / "rule_page_010_ocr.json").write_text(
                json.dumps({"text_lines": ["ข้อ ๑๐ JSON ที่ถูกเลือก"]}),
                encoding="utf-8",
            )
            (input_dir / "rule_page_002_ocr.json").write_text(
                json.dumps({"text_lines": ["ข้อ ๒ หน้าแรก"]}),
                encoding="utf-8",
            )
            output_path = Path(temp_dir) / "rules.json"

            files = discover_rule_ocr_files(input_dir)
            self.assertEqual(
                [file.name for file in files],
                ["rule_page_002_ocr.json", "rule_page_010_ocr.json"],
            )
            self.assertEqual(main([str(input_dir), "--output", str(output_path)]), 0)
            result = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [rule["section_number"] for rule in result["rules"]], ["2", "10"]
        )
        self.assertIn("JSON ที่ถูกเลือก", result["rules"][1]["rule_text"])
        self.assertNotIn("TXT ควรถูกแทนที่", result["rules"][1]["rule_text"])

    def test_missing_rule_ocr_files_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "No Rule OCR"):
                discover_rule_ocr_files(temp_dir)


if __name__ == "__main__":
    unittest.main()
