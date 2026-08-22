import copy
import json
import tempfile
import unittest
from pathlib import Path

from merge_consecutive import merge_plan_with_description
from src.extractor import CurriculumExtractor
from src.file_handler import save_ocr_results


def plan_lines():
    return [
        "ปีที่ 1",
        "06000001",
        "การวิเคราะห์ข้อมูล",
        "3(3-0-6)",
        "DATA ANALYSIS",
    ]


def description_lines():
    return [
        "คำอธิบายรายวิชา",
        "06000001",
        "การวิเคราะห์ข้อมูล",
        "3(3-0-6)",
        "DATA ANALYSIS",
        "PREREQUISITE NONE",
    ]


def course(code, marker, source_page):
    return {
        "code": code,
        "name_th": f"TH {marker}",
        "name_en": f"EN {marker}",
        "credits": "3(3-0-6)",
        "prerequisite": "NONE",
        "source_provenance": [
            {
                "program": "DSBA",
                "source_filename": f"dsba_page_{source_page:03d}.png",
                "source_page": source_page,
                "document_category": "plan",
            }
        ],
    }


class ProvenanceTests(unittest.TestCase):
    def test_image_metadata_becomes_filename_and_page_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_ocr_results(
                plan_lines(),
                output_dir,
                "dsba_page_026",
                source_filename="dsba_page_026.png",
                source_page=26,
                program="DSBA",
            )

            result = CurriculumExtractor(program="DSBA").process_file(
                output_dir / "dsba_page_026_ocr.json"
            )

        self.assertEqual(
            result["courses"][0]["source_provenance"],
            [
                {
                    "program": "DSBA",
                    "source_filename": "dsba_page_026.png",
                    "source_page": 26,
                    "document_category": "plan",
                }
            ],
        )

    def test_explicit_txt_and_json_inputs_use_filename_page_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            txt_path = input_dir / "it_page_041_ocr.txt"
            txt_path.write_text("\n".join(plan_lines()), encoding="utf-8")
            json_path = input_dir / "ait_page_042_ocr.json"
            json_path.write_text(
                json.dumps({"text_lines": plan_lines()}), encoding="utf-8"
            )

            txt_result = CurriculumExtractor(program="IT").process_file(txt_path)
            json_result = CurriculumExtractor(program="AIT").process_file(json_path)

        self.assertEqual(
            txt_result["courses"][0]["source_provenance"][0],
            {
                "program": "IT",
                "source_filename": "it_page_041_ocr.txt",
                "source_page": 41,
                "document_category": "plan",
            },
        )
        self.assertEqual(
            json_result["courses"][0]["source_provenance"][0],
            {
                "program": "AIT",
                "source_filename": "ait_page_042_ocr.json",
                "source_page": 42,
                "document_category": "plan",
            },
        )

    def test_plan_and_description_categories_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            plan_path = input_dir / "dsba_page_026_ocr.txt"
            plan_path.write_text("\n".join(plan_lines()), encoding="utf-8")
            desc_path = input_dir / "dsba_page_317_ocr.txt"
            desc_path.write_text("\n".join(description_lines()), encoding="utf-8")

            extractor = CurriculumExtractor(program="DSBA")
            plan_result = extractor.process_file(plan_path)
            desc_result = extractor.process_file(desc_path)

        self.assertEqual(
            plan_result["courses"][0]["source_provenance"][0]["document_category"],
            "plan",
        )
        self.assertEqual(
            desc_result["courses"][0]["source_provenance"][0]["document_category"],
            "description",
        )

    def test_legacy_ocr_json_without_metadata_still_extracts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = Path(temp_dir) / "dsba_page_009_ocr.json"
            legacy_path.write_text(
                json.dumps({"filename": "dsba_page_009", "text_lines": plan_lines()}),
                encoding="utf-8",
            )

            result = CurriculumExtractor(program="DSBA").process_file(legacy_path)

        self.assertEqual(result["courses"][0]["code"], "06000001")
        self.assertEqual(
            result["courses"][0]["source_provenance"][0]["source_filename"],
            "dsba_page_009_ocr.json",
        )
        self.assertEqual(
            result["courses"][0]["source_provenance"][0]["source_page"], 9
        )

    def test_plan_and_description_provenance_are_unioned_in_source_order(self):
        plan = course("06000001", "PLAN", 26)
        description = course("06000001", "DESCRIPTION", 317)
        description.update(
            {
                "desc_th": "DESCRIPTION TH",
                "desc_en": "DESCRIPTION EN",
            }
        )

        result = merge_plan_with_description(
            [plan], [description], {"program": "DSBA", "plan": "coop"}
        )

        self.assertEqual(
            result["courses"][0]["source_provenance"],
            plan["source_provenance"] + description["source_provenance"],
        )
        self.assertEqual(result["courses"][0]["desc_th"], "DESCRIPTION TH")
        self.assertEqual(result["courses"][0]["desc_en"], "DESCRIPTION EN")

    def test_alternative_course_unions_both_contributing_sources(self):
        extractor = CurriculumExtractor(
            program="DSBA", coop_pairs=[("06000001", "06000002", "6(0-35-0)")]
        )
        first = course("06000001", "FIRST", 33)
        second = course("06000002", "SECOND", 34)

        result = extractor.post_process([first, second])

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["source_provenance"],
            first["source_provenance"] + second["source_provenance"],
        )
        self.assertEqual(result[0]["code"], "06000001 หรือ 06000002")

    def test_unmatched_description_keeps_its_provenance(self):
        description = course("06000009", "DESCRIPTION", 318)

        result = merge_plan_with_description(
            [], [description], {"program": "DSBA", "plan": "coop"}
        )

        self.assertEqual(result["courses"][0]["source_provenance"], description["source_provenance"])

    def test_duplicate_records_keep_order_fields_and_independent_sources(self):
        first = course("06000001", "FIRST", 26)
        second = course("06000001", "SECOND", 27)
        first_before = copy.deepcopy(first)
        second_before = copy.deepcopy(second)

        result = merge_plan_with_description([first, second], [], {})

        self.assertEqual(
            [(item["code"], item["name_en"]) for item in result["courses"]],
            [(first_before["code"], first_before["name_en"]),
             (second_before["code"], second_before["name_en"])],
        )
        self.assertEqual(
            [item["source_provenance"] for item in result["courses"]],
            [first_before["source_provenance"], second_before["source_provenance"]],
        )

    def test_duplicate_description_occurrences_get_consumed_provenance(self):
        plans = [course("06000001", "FIRST", 26), course("06000001", "SECOND", 27)]
        descriptions = [course("06000001", "DESC1", 317), course("06000001", "DESC2", 318)]
        descriptions[0].update({"desc_th": "TH 1", "desc_en": "EN 1"})
        descriptions[1].update({"desc_th": "TH 2", "desc_en": "EN 2"})

        result = merge_plan_with_description(plans, descriptions, {})

        self.assertEqual(
            [(item["code"], item["name_en"]) for item in result["courses"]],
            [("06000001", "EN FIRST"), ("06000001", "EN SECOND")],
        )
        self.assertEqual(
            [item["desc_en"] for item in result["courses"]], ["EN 2", "EN 2"]
        )
        self.assertEqual(
            [
                [entry["source_page"] for entry in item["source_provenance"]]
                for item in result["courses"]
            ],
            [[26, 317], [27, 318]],
        )


if __name__ == "__main__":
    unittest.main()
