import json
import tempfile
import unittest
from pathlib import Path

from merge_consecutive import merge_consecutive_files, merge_plan_with_description
from src.extractor import CurriculumExtractor


def provenance(program, page, category):
    return [
        {
            "program": program,
            "source_filename": f"{program.lower()}_page_{page:03d}.png",
            "source_page": page,
            "document_category": category,
        }
    ]


def alternative_course(code, page, category, marker):
    return {
        "code": code,
        "name_th": f"Thai {marker}",
        "name_en": f"ENGLISH {marker}",
        "credits": "6(0-36-0)",
        "year": 0,
        "semester": 0,
        "category": "หมวดวิชาเฉพาะ",
        "type": "เลือก",
        "prerequisite": "ไม่มี",
        "flexible_year_semester": "4/1",
        "note": "CO-OP NOTE",
        "desc_th": f"DESC TH {marker}",
        "desc_en": f"DESC EN {marker}",
        "source_provenance": provenance("IT", page, category),
    }


class CourseCleanupTests(unittest.TestCase):
    def test_it_plan_pair_merges_and_preserves_fields(self):
        first = alternative_course("06016481", 44, "plan", "LOCAL")
        second = alternative_course("06016482", 44, "plan", "OVERSEAS")

        result = CurriculumExtractor(program="IT", plan="coop").post_process(
            [first, second]
        )

        self.assertEqual(len(result), 1)
        merged = result[0]
        self.assertEqual(merged["code"], "06016481 หรือ 06016482")
        self.assertEqual(merged["credits"], "6(0-36-0)")
        self.assertEqual(merged["name_th"], "Thai LOCAL\nThai OVERSEAS")
        self.assertEqual(merged["name_en"], "ENGLISH LOCAL\nENGLISH OVERSEAS")
        self.assertEqual(merged["desc_th"], "DESC TH LOCAL\nDESC TH OVERSEAS")
        self.assertEqual(merged["desc_en"], "DESC EN LOCAL\nDESC EN OVERSEAS")
        self.assertEqual(merged["prerequisite"], "ไม่มี")
        self.assertEqual(merged["note"], "CO-OP NOTE")
        self.assertEqual(
            [entry["source_page"] for entry in merged["source_provenance"]],
            [44],
        )

    def test_unconfigured_adjacent_it_codes_remain_separate(self):
        first = alternative_course("06016479", 44, "plan", "FIRST")
        second = alternative_course("06016480", 44, "plan", "SECOND")

        result = CurriculumExtractor(program="IT", plan="coop").post_process(
            [first, second]
        )

        self.assertEqual([course["code"] for course in result], ["06016479", "06016480"])

    def test_bit_plan_table_assigns_thai_names_before_codes(self):
        lines = [
            "หน่วยกิต",
            "(บรรยาย-ปฏิบัติ-ศึกษา",
            "ด้วยตนเอง)",
            "พื้นฐานทางด้านเทคโนโลยีสารสนเทศ",
            "06036100",
            "3(2-2-5)",
            "INFORMATION TECHNOLOGY FUNDAMENTALS",
            "การแก",
            "้ปัญหาทางด้านเทคโนโลยีสารสนเทศ",
            "06036118",
            "3(2-2-5)",
            "PROBLEM SOLVING IN INFORMATION TECHNOLOGY",
            "โรงเรียนสร้างเสน่ห์",
            "96641001",
            "2(1-2-3)",
            "CHARM SCHOOL",
        ]
        extractor = CurriculumExtractor(program="BIT", plan="no_coop")
        courses = [
            extractor.parse_single_block(block)
            for block in extractor.split_into_blocks(lines)
        ]
        names = {course["code"]: course["name_th"] for course in courses}

        self.assertEqual(names["06036100"], "พื้นฐานทางด้านเทคโนโลยีสารสนเทศ")
        self.assertEqual(
            names["06036118"], "การแก้ปัญหาทางด้านเทคโนโลยีสารสนเทศ"
        )
        self.assertEqual(names["96641001"], "โรงเรียนสร้างเสน่ห์")

    def test_bit_plan_pair_merges_as_one_alternative_slot(self):
        first = alternative_course("06036147", 35, "plan", "LOCAL")
        second = alternative_course("06036148", 36, "plan", "OVERSEAS")
        first["credits"] = second["credits"] = "6(0-35-0)"

        result = CurriculumExtractor(program="BIT", plan="coop").post_process(
            [first, second]
        )

        self.assertEqual(len(result), 1)
        merged = result[0]
        self.assertEqual(merged["code"], "06036147 หรือ 06036148")
        self.assertEqual(merged["credits"], "6(0-35-0)")
        self.assertEqual(merged["name_th"], "Thai LOCAL\nThai OVERSEAS")
        self.assertEqual(merged["name_en"], "ENGLISH LOCAL\nENGLISH OVERSEAS")
        self.assertEqual(merged["desc_th"], "DESC TH LOCAL\nDESC TH OVERSEAS")
        self.assertEqual(merged["desc_en"], "DESC EN LOCAL\nDESC EN OVERSEAS")
        self.assertEqual(
            merged["source_provenance"],
            first["source_provenance"] + second["source_provenance"],
        )

    def test_merge_consecutive_files_post_processes_bit_table_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "consolidated"

            for plan, page, table_courses in (
                (
                    "no_coop",
                    26,
                    [alternative_course("06000001", 26, "plan", "OTHER")],
                ),
                (
                    "coop",
                    31,
                    [
                        alternative_course("06036147", 35, "plan", "LOCAL"),
                        alternative_course("06036148", 36, "plan", "OVERSEAS"),
                    ],
                ),
            ):
                input_dir = Path(temp_dir) / plan
                input_dir.mkdir()
                description_courses = [
                    alternative_course("06036147", 366, "description", "LOCAL"),
                    alternative_course("06036148", 367, "description", "OVERSEAS"),
                ]
                for page_num, courses in (
                    (page, table_courses),
                    (366, description_courses),
                ):
                    (input_dir / f"bit_{plan}_page_{page_num:03d}_ocr_extracted.json").write_text(
                        json.dumps(
                            {
                                "program": "BIT",
                                "plan": plan,
                                "courses": courses,
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

                merge_consecutive_files(
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    plan_filter=plan,
                    prefix="bit",
                    desc_pages="366",
                )

                result = json.loads(
                    (output_dir / f"merged_bit_{plan}_full.json").read_text(
                        encoding="utf-8"
                    )
                )
                combined_code = "06036147 \u0e2b\u0e23\u0e37\u0e2d 06036148"
                pair_codes = [
                    course["code"]
                    for course in result["courses"]
                    if course["code"] in {"06036147", "06036148"}
                    or course["code"] == combined_code
                ]
                self.assertEqual(pair_codes, [combined_code])


    def test_it_description_only_pair_merges_for_no_coop(self):
        first = alternative_course("06016481", 366, "description", "LOCAL")
        second = alternative_course("06016482", 367, "description", "OVERSEAS")

        result = merge_plan_with_description(
            [],
            [first, second],
            {
                "source": "GT_Template-2.xlsx / Academic Plan GT - IT no_coop",
                "description": "Ground Truth curriculum",
                "program": "IT",
                "plan": "no_coop",
            },
        )

        self.assertEqual(result["plan"], "no_coop")
        self.assertEqual(len(result["courses"]), 1)
        self.assertEqual(result["courses"][0]["code"], "06016481 หรือ 06016482")
        self.assertEqual(result["courses"][0]["credits"], "6(0-36-0)")
        self.assertEqual(
            [entry["source_page"] for entry in result["courses"][0]["source_provenance"]],
            [366, 367],
        )
        self.assertNotIn("Ground Truth", result["source"])
        self.assertNotIn("GT_Template", result["source"])
        self.assertNotIn("Ground Truth", result["description"])

    def test_it_plan_pair_does_not_append_description_alternatives_again(self):
        plan_pair = CurriculumExtractor(program="IT", plan="coop").post_process(
            [
                alternative_course("06016481", 44, "plan", "LOCAL"),
                alternative_course("06016482", 44, "plan", "OVERSEAS"),
            ]
        )
        descriptions = [
            alternative_course("06016481", 366, "description", "LOCAL"),
            alternative_course("06016482", 367, "description", "OVERSEAS"),
        ]

        result = merge_plan_with_description(
            plan_pair, descriptions, {"program": "IT", "plan": "coop"}
        )

        self.assertEqual(len(result["courses"]), 1)
        self.assertEqual(result["courses"][0]["code"], "06016481 หรือ 06016482")
        self.assertEqual(
            [entry["source_page"] for entry in result["courses"][0]["source_provenance"]],
            [44, 366, 367],
        )

    def test_dsba_no_coop_description_pair_behavior_is_unchanged(self):
        first = alternative_course("06026259", 317, "description", "FIRST")
        second = alternative_course("06026260", 318, "description", "SECOND")

        result = merge_plan_with_description(
            [],
            [first, second],
            {"program": "DSBA", "plan": "no_coop"},
        )

        self.assertEqual(
            [course["code"] for course in result["courses"]],
            ["06026259", "06026260"],
        )

    def test_dsba_composite_keeps_existing_description_occurrences(self):
        plan = CurriculumExtractor(program="DSBA", plan="coop").post_process(
            [
                alternative_course("06026259", 33, "plan", "FIRST"),
                alternative_course("06026260", 33, "plan", "SECOND"),
            ]
        )
        descriptions = [
            alternative_course("06026259", 317, "description", "FIRST"),
            alternative_course("06026260", 318, "description", "SECOND"),
        ]

        result = merge_plan_with_description(
            plan, descriptions, {"program": "DSBA", "plan": "coop"}
        )

        self.assertEqual(
            [course["code"] for course in result["courses"]],
            ["06026259 หรือ 06026260", "06026259", "06026260"],
        )

    def test_description_replay_uses_requested_plan_and_neutral_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "outputs"
            output_dir = Path(temp_dir) / "consolidated"
            input_dir.mkdir()

            plan = {
                "source": "GT_Template-2.xlsx / Academic Plan GT - IT no_coop",
                "description": "Ground Truth curriculum",
                "program": "IT",
                "plan": "no_coop",
                "courses": [
                    {
                        "code": "06000001",
                        "name_th": "PLAN NAME",
                        "name_en": "PLAN NAME",
                        "credits": "3(3-0-6)",
                        "year": 1,
                        "semester": 1,
                        "category": "หมวดวิชาเฉพาะ",
                        "type": "บังคับ",
                        "prerequisite": "ไม่มี",
                        "source_provenance": provenance("IT", 38, "plan"),
                    }
                ],
            }
            (input_dir / "it_page_038_ocr_extracted.json").write_text(
                json.dumps(plan, ensure_ascii=False), encoding="utf-8"
            )

            raw_description = {
                "text_lines": [
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
                "source_filename": "it_page_366.png",
                "source_page": 366,
                "program": "IT",
            }
            (input_dir / "it_page_366_ocr.json").write_text(
                json.dumps(raw_description, ensure_ascii=False), encoding="utf-8"
            )
            (input_dir / "it_page_366_ocr_extracted.json").write_text(
                json.dumps(
                    {
                        "source": "GT_Template-2.xlsx / Academic Plan GT - IT coop",
                        "description": "Ground Truth curriculum",
                        "program": "IT",
                        "plan": "coop",
                        "courses": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            merge_consecutive_files(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                plan_filter="no_coop",
                prefix="it",
                desc_pages="366",
            )

            result = json.loads(
                (output_dir / "merged_it_no_coop_full.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result["plan"], "no_coop")
        self.assertEqual(result["courses"][0]["desc_th"], "คำอธิบายภาษาไทย")
        self.assertEqual(result["courses"][0]["desc_en"], "ENGLISH BODY")
        self.assertEqual(
            [entry["source_page"] for entry in result["courses"][0]["source_provenance"]],
            [38, 366],
        )
        self.assertNotIn("Ground Truth", result["source"])
        self.assertNotIn("GT_Template", result["source"])


if __name__ == "__main__":
    unittest.main()
