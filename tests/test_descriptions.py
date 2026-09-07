import json
import tempfile
import unittest
from pathlib import Path

from merge_consecutive import merge_consecutive_files, merge_plan_with_description
from src.extractor import CurriculumExtractor


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def source_context(program, page):
    return {
        "program": program,
        "source_filename": f"{program.lower()}_page_{page:03d}.png",
        "source_page": page,
        "document_category": "description",
    }


class DescriptionExtractionTests(unittest.TestCase):
    def test_stored_dsba_it_and_ait_pages_preserve_both_description_languages(self):
        cases = [
            ("DSBA", "coop", "dsba_page_317_ocr.json", "06026200"),
            ("IT", "coop", "it_page_328_ocr.json", "06016401"),
            ("AIT", None, "ait_page_287_ocr.json", "06046400"),
        ]

        for program, plan, filename, code in cases:
            with self.subTest(program=program):
                result = CurriculumExtractor(program=program, plan=plan).process_file(
                    OUTPUTS / filename
                )
                course = next(item for item in result["courses"] if item["code"] == code)
                self.assertTrue(course.get("desc_th"))
                self.assertTrue(course.get("desc_en"))
                self.assertEqual(
                    course["source_provenance"][0]["document_category"],
                    "description",
                )

    def test_description_capture_stops_at_section_boundary(self):
        extractor = CurriculumExtractor(program="DSBA", plan="coop")
        result = extractor.extract_descriptions(
            [
                "คำอธิบายรายวิชา",
                "06000001",
                "ชื่อวิชา",
                "3(3-0-6)",
                "TEST COURSE",
                "PREREQUISITE",
                "NONE",
                "คำอธิบายภาษาไทย",
                "ENGLISH BODY",
                "กลุ่ม วิชาถัดไป",
                "06000002",
                "วิชาถัดไป",
                "3(3-0-6)",
                "NEXT COURSE",
                "PREREQUISITE",
                "NONE",
            ],
            source_context("DSBA", 317),
        )

        self.assertEqual([item["code"] for item in result["courses"]], [
            "06000001",
            "06000002",
        ])
        first = result["courses"][0]
        self.assertEqual(first["desc_th"], "คำอธิบายภาษาไทย")
        self.assertEqual(first["desc_en"], "ENGLISH BODY")
        self.assertNotIn("กลุ่ม", first["desc_th"])
        self.assertNotIn("NEXT COURSE", first["desc_en"])

    def test_ordered_pages_join_continuation_and_deduplicate_overlap(self):
        extractor = CurriculumExtractor(program="DSBA", plan="coop")
        result = extractor.extract_descriptions_from_pages(
            [
                (
                    [
                        "312",
                        "มคอ.",
                        "06000001",
                        "วิชาแรก",
                        "3(3-0-6)",
                        "FIRST COURSE",
                        "PREREQUISITE",
                        "NONE",
                        "เนื้อหาภาษาไทย",
                        "ENGLISH BODY",
                    ],
                    source_context("DSBA", 317),
                ),
                (
                    [
                        "313",
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
                    source_context("DSBA", 318),
                ),
            ]
        )

        first = result["courses"][0]
        self.assertEqual(first["desc_en"], "ENGLISH BODY\nCONTINUATION")
        self.assertEqual(
            [entry["source_page"] for entry in first["source_provenance"]],
            [317, 318],
        )

    def test_repeated_it_code_uses_unique_description_for_all_placements(self):
        extractor = CurriculumExtractor(program="IT", plan="coop")
        description = next(
            item
            for item in extractor.process_file(OUTPUTS / "it_page_336_ocr.json")["courses"]
            if item["code"] == "06016418"
        )
        plans = [
            {
                "code": "06016418",
                "name_th": "PLAN FIRST",
                "name_en": "SERVER SIDE WEB DEVELOPMENT",
                "credits": "3(2-2-5)",
                "year": 3,
                "semester": 1,
                "prerequisite": "06016408",
                "source_provenance": [source_context("IT", 43)],
            },
            {
                "code": "06016418",
                "name_th": "PLAN SECOND",
                "name_en": "SERVER SIDE WEB DEVELOPMENT",
                "credits": "3(2-2-5)",
                "year": 3,
                "semester": 1,
                "prerequisite": "06016408",
                "source_provenance": [source_context("IT", 43)],
            },
        ]

        result = merge_plan_with_description(
            plans, [description], {"program": "IT", "plan": "coop"}
        )

        self.assertEqual(result["total_courses"], 2)
        self.assertTrue(all(item.get("desc_en") for item in result["courses"]))
        self.assertTrue(all(item.get("desc_th") for item in result["courses"]))
        self.assertEqual(len(result.get("unresolved_descriptions", [])), 0)

    def test_unique_merge_keeps_plan_identity_and_adds_description(self):
        extractor = CurriculumExtractor(program="DSBA", plan="coop")
        description = next(
            item
            for item in extractor.process_file(OUTPUTS / "dsba_page_317_ocr.json")["courses"]
            if item["code"] == "06026200"
        )
        plan = {
            "code": "06026200",
            "name_th": "PLAN NAME",
            "name_en": "PLAN ENGLISH NAME",
            "credits": "3(3-0-6)",
            "year": 1,
            "semester": 1,
            "category": "หมวดวิชาเฉพาะ",
            "type": "บังคับ",
            "prerequisite": "ไม่มี",
            "source_provenance": [source_context("DSBA", 33)],
        }

        result = merge_plan_with_description(
            [plan], [description], {"program": "DSBA", "plan": "coop"}
        )
        merged = result["courses"][0]
        self.assertEqual(merged["name_th"], "PLAN NAME")
        self.assertEqual(merged["name_en"], "PLAN ENGLISH NAME")
        self.assertEqual((merged["year"], merged["semester"]), (1, 1))
        self.assertTrue(merged["desc_th"])
        self.assertTrue(merged["desc_en"])
        self.assertEqual(
            [entry["source_page"] for entry in merged["source_provenance"]],
            [33, 317],
        )

    def test_file_merge_replays_ordered_description_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "outputs"
            output_dir = Path(temp_dir) / "consolidated"
            input_dir.mkdir()

            plan = {
                "program": "DSBA",
                "plan": "coop",
                "courses": [
                    {
                        "code": "06000001",
                        "name_th": "PLAN NAME",
                        "name_en": "PLAN NAME",
                        "credits": "3(3-0-6)",
                        "year": 1,
                        "semester": 1,
                        "prerequisite": "ไม่มี",
                        "source_provenance": [source_context("DSBA", 33)],
                    }
                ],
            }
            (input_dir / "dsba_page_033_ocr_extracted.json").write_text(
                json.dumps(plan, ensure_ascii=False), encoding="utf-8"
            )

            page_lines = {
                317: [
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
                318: [
                    "มคอ.",
                    "2",
                    "ENGLISH BODY",
                    "CONTINUATION",
                ],
            }
            for page, lines in page_lines.items():
                stem = f"dsba_page_{page:03d}_ocr"
                raw = {
                    "text_lines": lines,
                    "source_filename": f"dsba_page_{page:03d}.png",
                    "source_page": page,
                    "program": "DSBA",
                }
                extracted = {
                    "program": "DSBA",
                    "plan": "coop",
                    "courses": (
                        [
                            {
                                "code": "06000001",
                                "prerequisite": "ไม่มี",
                                "source_provenance": [source_context("DSBA", page)],
                            }
                        ]
                        if page == 317
                        else []
                    ),
                }
                (input_dir / f"{stem}.json").write_text(
                    json.dumps(raw, ensure_ascii=False), encoding="utf-8"
                )
                (input_dir / f"{stem}_extracted.json").write_text(
                    json.dumps(extracted, ensure_ascii=False), encoding="utf-8"
                )

            merge_consecutive_files(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                plan_filter="coop",
                prefix="dsba",
                desc_pages="317-318",
            )
            result = json.loads(
                (output_dir / "merged_dsba_coop_full.json").read_text(encoding="utf-8")
            )

        merged = result["courses"][0]
        self.assertEqual(merged["desc_en"], "ENGLISH BODY\nCONTINUATION")
        self.assertEqual(
            [entry["source_page"] for entry in merged["source_provenance"]],
            [33, 317, 318],
        )


if __name__ == "__main__":
    unittest.main()
