import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.tools.merge.consolidator import (
    _preserve_authoritative_extracted_credit,
    _preserve_exact_description_terminal_suffix,
    merge_consecutive_files,
    merge_plan_with_description,
)
from src.pipeline.tools.extraction.engine import CourseBlock, CurriculumExtractor


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
    def test_exact_description_terminal_suffix_preservation_rules(self):
        cases = (
            ("ชื่อโครงงาน", "ชื่อโครงงาน 1", "ชื่อโครงงาน 1"),
            ("PROJECT", "PROJECT 2", "PROJECT 2"),
            ("COURSE 1", "COURSE 1", "COURSE 1"),
            ("COURSE 1", "COURSE 2", "COURSE 1"),
            ("COURSE", "COURSE", "COURSE"),
            ("COURSE", "DIFFERENT COURSE 1", "COURSE"),
            ("COURSE NAME", "COURSE  NAME 1", "COURSE NAME"),
            ("COURSE", "COURSE 12", "COURSE"),
            ("COURSE", "COURSE I", "COURSE"),
            ("", "COURSE 1", ""),
            ("COURSE", "", "COURSE"),
        )
        for plan_title, description_title, expected in cases:
            with self.subTest(plan_title=plan_title, description_title=description_title):
                self.assertEqual(
                    _preserve_exact_description_terminal_suffix(
                        plan_title, description_title
                    ),
                    expected,
                )

    def test_unique_description_merge_preserves_exact_suffix_titles_and_fields(self):
        plan = alternative_course("SUFFIX01", 1, "plan", "PLAN")
        plan["name_th"] = "ชื่อโครงงาน"
        plan["name_en"] = "BUSINESS PROJECT"
        description = alternative_course("SUFFIX01", 2, "description", "DESC")
        description["name_th"] = "ชื่อโครงงาน 1"
        description["name_en"] = "BUSINESS PROJECT 2"

        result = merge_plan_with_description(
            [plan], [description], {"program": "IT", "plan": "no_coop"}
        )
        merged = result["courses"][0]

        self.assertEqual(merged["name_th"], "ชื่อโครงงาน 1")
        self.assertEqual(merged["name_en"], "BUSINESS PROJECT 2")
        self.assertEqual(merged["prerequisite"], description["prerequisite"])
        self.assertEqual(merged["desc_th"], description["desc_th"])
        self.assertEqual(merged["desc_en"], description["desc_en"])
        self.assertEqual(
            [entry["source_page"] for entry in merged["source_provenance"]],
            [1, 2],
        )

    def test_bilingual_terminal_suffix_reconciliation_rules(self):
        cases = (
            ({"name_th": "ชื่อไทย 1", "name_en": "ENGLISH"}, "ชื่อไทย 1", "ENGLISH 1"),
            ({"name_th": "ชื่อไทย", "name_en": "ENGLISH 2"}, "ชื่อไทย 2", "ENGLISH 2"),
            ({"name_th": "ชื่อไทย 3", "name_en": "ENGLISH 3"}, "ชื่อไทย 3", "ENGLISH 3"),
            ({"name_th": "ชื่อไทย 1", "name_en": "ENGLISH 2"}, "ชื่อไทย 1", "ENGLISH 2"),
            ({"name_th": "ชื่อไทย", "name_en": "ENGLISH"}, "ชื่อไทย", "ENGLISH"),
            ({"name_th": "", "name_en": "ENGLISH 2"}, "", "ENGLISH 2"),
            ({"name_th": "ชื่อไทย 1", "name_en": ""}, "ชื่อไทย 1", ""),
            ({"code": "06016409", "name_th": "ชื่อไทย", "name_en": "ENGLISH"}, "ชื่อไทย", "ENGLISH"),
        )
        for raw, expected_th, expected_en in cases:
            with self.subTest(raw=raw):
                result = CurriculumExtractor._reconcile_bilingual_terminal_suffix(dict(raw))
                self.assertEqual(result.get("name_th"), expected_th)
                self.assertEqual(result.get("name_en"), expected_en)

    def test_gened_bilingual_terminal_suffix_reconciliation_is_disabled(self):
        result = CurriculumExtractor._reconcile_bilingual_terminal_suffix(
            {"name_th": "ชื่อไทย 1", "name_en": "ENGLISH"}, "GENED"
        )
        self.assertEqual(result["name_th"], "ชื่อไทย 1")
        self.assertEqual(result["name_en"], "ENGLISH")

    def test_source_backed_title_repair_requires_exact_identity_and_before_value(self):
        before = "ปฏิบัติงานตามทักษะด้านการจัดการ"
        after = "ปฏิบัติงานตามทักษะด้านการจัดการ 1"
        cases = (
            (24, "90643001", before, after),
            (89, "90643001", before, after),
            (25, "90643001", before, before),
            (24, "90643002", before, before),
            (24, "90643001", after, after),
            (24, "90643001", "ชื่ออื่น", "ชื่ออื่น"),
        )
        for page, code, current, expected in cases:
            with self.subTest(page=page, code=code, current=current):
                provenance_data = provenance("GENED", page, "plan")
                course = {
                    "code": code,
                    "name_th": current,
                    "name_en": "PRACTICE UNDER MANAGEMENT SKILLS",
                    "source_provenance": provenance_data,
                }
                result = CurriculumExtractor._apply_source_backed_title_repairs(
                    course
                )
                self.assertEqual(result["name_th"], expected)
                self.assertEqual(
                    result["name_en"], "PRACTICE UNDER MANAGEMENT SKILLS"
                )
                self.assertEqual(result["source_provenance"], provenance_data)

    def test_source_backed_ait_title_repairs_require_exact_identity_and_before_value(self):
        cases = (
            (23, "90641004", "TEAM PR0ECT 1", "TEAM-PROJECT 1"),
            (24, "90641005", "TEAM PR0JECT 2", "TEAM-PROJECT 2"),
            (25, "90641006", "TEAM PROECT 3", "TEAM-PROJECT 3"),
        )
        for page, code, before, after in cases:
            with self.subTest(page=page, code=code):
                provenance_data = provenance("AIT", page, "plan")
                course = {
                    "code": code,
                    "name_th": "AIT Thai",
                    "name_en": before,
                    "source_provenance": provenance_data,
                }
                result = CurriculumExtractor._apply_source_backed_title_repairs(
                    course
                )
                self.assertEqual(result["name_en"], after)

        fail_closed_cases = (
            (24, "90641004", "TEAM PR0ECT 1"),
            (23, "90649999", "TEAM PR0ECT 1"),
            (23, "90641004", "TEAM PROJECT 1"),
        )
        for page, code, current in fail_closed_cases:
            with self.subTest(page=page, code=code, current=current):
                course = {
                    "code": code,
                    "name_th": "AIT Thai",
                    "name_en": current,
                    "source_provenance": provenance("AIT", page, "plan"),
                }
                result = CurriculumExtractor._apply_source_backed_title_repairs(
                    course
                )
                self.assertEqual(result["name_en"], current)

    def test_source_backed_title_repair_covers_both_persisted_gened_occurrences(self):
        extractor = CurriculumExtractor(program="GENED", plan="gened")
        plan = extractor.extract_from_lines(
            [
                "90643001",
                "ปฏิบัติงานตามทักษะด้านการจัดการ",
                "1 (0-2-1)",
                "PRACTICE UNDER MANAGEMENT SKILLS",
            ],
            provenance("GENED", 24, "plan")[0],
        )["courses"][0]
        description = extractor.extract_descriptions(
            [
                "คำอธิบายรายวิชา",
                "90643001",
                "ปฏิบัติงานตามทักษะด้านการจัดการ",
                "(0-2-1)",
                "PRACTICE UNDER MANAGEMENT SKILLS 1",
                "รายวิชาบังคับก่อน",
                "ไม่มี",
            ],
            provenance("GENED", 89, "description")[0],
        )["courses"][0]

        self.assertEqual(plan["name_th"], "ปฏิบัติงานตามทักษะด้านการจัดการ 1")
        self.assertEqual(
            description["name_th"], "ปฏิบัติงานตามทักษะด้านการจัดการ 1"
        )
        self.assertEqual(plan["name_en"], "PRACTICE UNDER MANAGEMENT SKILLS")
        self.assertEqual(
            description["name_en"], "PRACTICE UNDER MANAGEMENT SKILLS 1"
        )
        self.assertEqual(
            plan["source_provenance"], [provenance("GENED", 24, "plan")[0]]
        )
        self.assertEqual(
            description["source_provenance"],
            [provenance("GENED", 89, "description")[0]],
        )

    def test_plan_parser_reconciles_one_sided_terminal_suffix(self):
        extractor = CurriculumExtractor(program="BIT", plan="no_coop")
        result = extractor.parse_single_block(
            CourseBlock(
                code="06000001",
                lines=["ชื่อไทย 1", "ENGLISH TITLE", "3(3-0-6)"],
            )
        )
        self.assertEqual(result["name_th"], "ชื่อไทย 1")
        self.assertEqual(result["name_en"], "ENGLISH TITLE 1")

    def test_bit_pending_numeric_suffix_before_code_is_reconciled(self):
        extractor = CurriculumExtractor(program="BIT", plan="no_coop")
        result = extractor.extract_from_lines(
            ["ชื่อไทย", "1", "06000011", "3(3-0-6)", "ENGLISH TITLE"]
        )["courses"][0]

        self.assertEqual(result["name_th"], "ชื่อไทย 1")
        self.assertEqual(result["name_en"], "ENGLISH TITLE 1")
        self.assertEqual(result["credits"], "3(3-0-6)")

    def test_bit_pending_numeric_suffix_requires_pending_title_and_code(self):
        extractor = CurriculumExtractor(program="BIT", plan="no_coop")

        no_pending = extractor.extract_from_lines(
            ["1", "06000012", "3(3-0-6)", "ENGLISH TITLE"]
        )["courses"][0]
        self.assertEqual(no_pending["name_en"], "ENGLISH TITLE")

        no_following_code = extractor.extract_from_lines(
            ["ชื่อไทย", "1", "не код", "06000013", "3(3-0-6)", "ENGLISH TITLE"]
        )["courses"][0]
        self.assertEqual(no_following_code["name_th"], "ชื่อไทย")
        self.assertEqual(no_following_code["name_en"], "ENGLISH TITLE")

        multi_digit = extractor.extract_from_lines(
            ["ชื่อไทย", "12", "06000014", "3(3-0-6)", "ENGLISH TITLE"]
        )["courses"][0]
        self.assertEqual(multi_digit["name_th"], "ไม่ระบุ")
        self.assertEqual(multi_digit["name_en"], "ENGLISH TITLE")

    def test_bit_pending_numeric_suffix_boundary_clears_state(self):
        for boundary in ("ปีที่ 2", "รหัสวิชา", "หมวดวิชาเลือก"):
            with self.subTest(boundary=boundary):
                extractor = CurriculumExtractor(program="BIT", plan="no_coop")
                result = extractor.extract_from_lines(
                    [
                        "ชื่อไทย",
                        "1",
                        boundary,
                        "06000015",
                        "3(3-0-6)",
                        "ENGLISH TITLE",
                    ]
                )["courses"][0]
                self.assertEqual(result["name_th"], "ไม่ระบุ")
                self.assertEqual(result["name_en"], "ENGLISH TITLE")

    def test_bit_pending_numeric_suffix_does_not_duplicate_or_replace(self):
        extractor = CurriculumExtractor(program="BIT", plan="no_coop")

        same_suffix = extractor.extract_from_lines(
            ["ชื่อไทย 1", "1", "06000016", "3(3-0-6)", "ENGLISH TITLE"]
        )["courses"][0]
        self.assertEqual(same_suffix["name_th"], "ชื่อไทย 1")
        self.assertEqual(same_suffix["name_en"], "ENGLISH TITLE 1")

        conflicting_suffix = extractor.extract_from_lines(
            ["ชื่อไทย 2", "1", "06000017", "3(3-0-6)", "ENGLISH TITLE"]
        )["courses"][0]
        self.assertEqual(conflicting_suffix["name_th"], "ชื่อไทย 2")
        self.assertEqual(conflicting_suffix["name_en"], "ENGLISH TITLE 2")

    def test_description_parser_attaches_suffix_before_partial_credit_prefix(self):
        extractor = CurriculumExtractor(program="BIT", plan="no_coop")
        result = extractor.extract_descriptions(
            [
                "คำอธิบายรายวิชา",
                "06000001",
                "ชื่อวิชาทดลอง",
                "2",
                "3(2-2",
                "EXPERIMENTAL COURSE",
                "วิชาบังคับก่อน",
                "ไม่มี",
            ]
        )["courses"][0]

        self.assertEqual(result["name_th"], "ชื่อวิชาทดลอง 2")
        self.assertEqual(result["name_en"], "EXPERIMENTAL COURSE 2")

    def test_description_parser_discards_numeric_suffix_without_credit_boundary(self):
        extractor = CurriculumExtractor(program="BIT", plan="no_coop")
        result = extractor.extract_descriptions(
            [
                "คำอธิบายรายวิชา",
                "06000002",
                "ชื่อวิชาทดลอง",
                "2",
                "not a credit boundary",
                "EXPERIMENTAL COURSE",
                "วิชาบังคับก่อน",
                "ไม่มี",
            ]
        )["courses"][0]

        self.assertEqual(result["name_th"], "ชื่อวิชาทดลอง")
        self.assertEqual(
            result["name_en"], "NOT A CREDIT BOUNDARY EXPERIMENTAL COURSE"
        )

    def test_complete_persisted_credit_replaces_empty_or_parenthetical_raw_credit(self):
        persisted = alternative_course("CREDIT01", 366, "description", "PERSISTED")
        persisted["credits"] = "3(3-0-6)"
        for raw_credit in ("", "(3-0-6)"):
            raw = dict(persisted, credits=raw_credit)
            result = _preserve_authoritative_extracted_credit(raw, [persisted])
            self.assertEqual(result["credits"], "3(3-0-6)")
            self.assertEqual(result["name_th"], raw["name_th"])

    def test_complete_credit_and_identical_credit_are_unchanged(self):
        persisted = alternative_course("CREDIT02", 367, "description", "PERSISTED")
        persisted["credits"] = "3(3-0-6)"
        raw = dict(persisted, credits="3(3-0-6)")
        result = _preserve_authoritative_extracted_credit(raw, [persisted])
        self.assertIs(result, raw)

    def test_conflicting_complete_credits_fail_closed(self):
        persisted = alternative_course("CREDIT03", 368, "description", "PERSISTED")
        persisted["credits"] = "3(3-0-6)"
        raw = dict(persisted, credits="4(3-0-6)")
        with self.assertRaises(ValueError):
            _preserve_authoritative_extracted_credit(raw, [persisted])

    def test_credit_preservation_requires_exact_source_identity(self):
        persisted = alternative_course("CREDIT04", 369, "description", "PERSISTED")
        persisted["credits"] = "3(3-0-6)"
        wrong_source = dict(persisted, credits="3(3-0-6)")
        wrong_source["source_provenance"] = provenance("IT", 370, "description")
        raw = dict(persisted, credits="(3-0-6)")
        result = _preserve_authoritative_extracted_credit(raw, [wrong_source])
        self.assertIs(result, raw)
        self.assertEqual(result["credits"], "(3-0-6)")

    def test_alternative_and_wildcard_credit_forms_are_not_reordered(self):
        for credit in ("3(3-0-6) หรือ 3(2-2-5)", "3(X-X-X)"):
            persisted = alternative_course("CREDIT05", 371, "description", "PERSISTED")
            persisted["credits"] = credit
            raw = dict(persisted, credits="(3-0-6)")
            result = _preserve_authoritative_extracted_credit(raw, [persisted])
            self.assertIs(result, raw)

    def test_page_range_and_full_outputs_preserve_persisted_description_credit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "bit"
            output_dir = Path(temp_dir) / "consolidated"
            input_dir.mkdir()
            table = alternative_course("06000001", 26, "plan", "TABLE")
            desc = alternative_course("06036135", 366, "description", "DESC")
            desc["source_provenance"] = provenance("BIT", 366, "description")
            desc["credits"] = "3(3-0-6)"
            (input_dir / "bit_page_026_ocr_extracted.json").write_text(
                json.dumps({"program": "BIT", "plan": "no_coop", "courses": [table]}),
                encoding="utf-8",
            )
            (input_dir / "bit_page_366_ocr.json").write_text(
                json.dumps(
                    {
                        "text_lines": [
                            "คำอธิบายรายวิชา",
                            "06036135",
                            "ชื่อวิชา",
                            "(3-0-6)",
                            "COURSE NAME",
                            "PREREQUISITE",
                            "NONE",
                        ],
                        "source_filename": "bit_page_366.png",
                        "source_page": 366,
                        "program": "BIT",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (input_dir / "bit_page_366_ocr_extracted.json").write_text(
                json.dumps(
                    {"program": "BIT", "plan": "no_coop", "courses": [desc]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            merge_consecutive_files(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                plan_filter="no_coop",
                prefix="bit",
                pages="26,366",
                desc_pages="366",
            )
            page = json.loads(
                (
                    output_dir
                    / "bit"
                    / "no_coop"
                    / "page_ranges"
                    / "merged_bit_no_coop_page_366-366.json"
                ).read_text(encoding="utf-8")
            )
            full = json.loads(
                (
                    output_dir
                    / "bit"
                    / "no_coop"
                    / "full"
                    / "merged_bit_no_coop_full.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(page["courses"][0]["credits"], "3(3-0-6)")
            self.assertEqual(
                next(
                    course
                    for course in full["courses"]
                    if course["code"] == "06036135"
                )["credits"],
                "3(3-0-6)",
            )

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
                    (output_dir / "bit" / plan / "full" / f"merged_bit_{plan}_full.json").read_text(
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
                (
                    output_dir
                    / "it"
                    / "no_coop"
                    / "full"
                    / "merged_it_no_coop_full.json"
                ).read_text(encoding="utf-8")
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
