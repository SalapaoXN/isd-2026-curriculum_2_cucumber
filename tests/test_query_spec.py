import unittest
from dataclasses import FrozenInstanceError, fields
import json
from pathlib import Path
from unittest.mock import patch

from rag.normalization import normalize_thai_surface
from rag.query_spec import QuerySpec, parse_query_spec


EXPECTED_FIELDS = (
    "original_question",
    "normalized_question",
    "program",
    "plans",
    "years",
    "semesters",
    "course_codes",
    "course_name",
    "category",
    "topic",
    "operations",
    "group_by",
    "judgement",
)


class QuerySpecSkeletonTests(unittest.TestCase):
    def test_query_spec_is_immutable(self):
        spec = parse_query_spec("คำถาม")
        with self.assertRaises(FrozenInstanceError):
            spec.topic = "database"

    def test_query_spec_has_exact_architecture_fields(self):
        self.assertEqual(tuple(field.name for field in fields(QuerySpec)), EXPECTED_FIELDS)
        forbidden = {
            "action",
            "blocking_ambiguity",
            "capabilities",
            "resolved_ids",
            "route",
            "retrieval_route",
        }
        self.assertTrue(forbidden.isdisjoint(EXPECTED_FIELDS))

    def test_parser_returns_required_defaults(self):
        spec = parse_query_spec("คำถามทั่วไป")
        self.assertIsNone(spec.program)
        self.assertEqual(spec.plans, ())
        self.assertEqual(spec.years, ())
        self.assertEqual(spec.semesters, ())
        self.assertEqual(spec.course_codes, ())
        self.assertIsNone(spec.course_name)
        self.assertIsNone(spec.category)
        self.assertIsNone(spec.topic)
        self.assertEqual(spec.operations, ())
        self.assertEqual(spec.group_by, ())
        self.assertEqual(spec.judgement, "none")
        for name in (
            "plans",
            "years",
            "semesters",
            "course_codes",
            "operations",
            "group_by",
        ):
            self.assertIsInstance(getattr(spec, name), tuple)

    def test_original_question_is_exact_and_normalized_question_uses_phase_4a(self):
        question = "IT ปีสามเทอมปลายเรียนหนักไหม"
        spec = parse_query_spec(question)
        self.assertEqual(spec.original_question, question)
        self.assertEqual(spec.normalized_question, "IT ปี 3 เทอม 2 เรียนหนักไหม")

    def test_parser_calls_only_the_surface_normalizer(self):
        question = "IT ปีสาม"
        with patch(
            "rag.query_spec.normalize_thai_surface",
            return_value="IT ปี 3",
        ) as normalize:
            spec = parse_query_spec(question)
        normalize.assert_called_once_with(question)
        self.assertEqual(spec.normalized_question, "IT ปี 3")


class QuerySpecEntityTests(unittest.TestCase):
    def test_program_aliases_are_explicit_and_boundary_safe(self):
        for alias, expected in (
            ("ait", "AIT"),
            ("BIT", "BIT"),
            ("dsba", "DSBA"),
            ("GENED", "GENED"),
            ("it", "IT"),
        ):
            with self.subTest(alias=alias):
                self.assertEqual(parse_query_spec(alias).program, expected)
        self.assertIsNone(parse_query_spec("curriculum").program)
        self.assertIsNone(parse_query_spec("06016414").program)
        self.assertIsNone(parse_query_spec("ITX").program)

    def test_plans_use_explicit_aliases_and_avoid_nested_coop_match(self):
        self.assertEqual(
            parse_query_spec("IT สหกิจกับไม่สหกิจต่างกันยังไง").plans,
            ("coop", "no_coop"),
        )
        self.assertEqual(parse_query_spec("ไม่สหกิจ").plans, ("no_coop",))
        self.assertEqual(
            parse_query_spec("coop no_coop default gened").plans,
            ("coop", "no_coop", "default", "gened"),
        )
        self.assertEqual(parse_query_spec("IT").plans, ())

    def test_years_and_semesters_use_normalized_numeric_surface(self):
        self.assertEqual(
            parse_query_spec("IT ปี 2 กับปี 3 ปีไหนมีวิชา programming เยอะกว่า").years,
            (2, 3),
        )
        self.assertEqual(
            parse_query_spec("IT ปีสองเทอมปลาย").years,
            (2,),
        )
        self.assertEqual(parse_query_spec("IT ปีสองเทอมปลาย").semesters, (2,))
        self.assertEqual(parse_query_spec("IT เทอมไหนมีวิชา database").semesters, ())
        self.assertEqual(parse_query_spec("year 3 term 1").years, (3,))
        self.assertEqual(parse_query_spec("year 3 term 1").semesters, (1,))
        self.assertEqual(parse_query_spec("IT ปี 5 เทอม 1").years, (5,))
        self.assertEqual(parse_query_spec("IT ปี 5 เทอม 1").semesters, (1,))
        self.assertEqual(parse_query_spec("ปี 20 เทอม 12").years, ())
        self.assertEqual(parse_query_spec("ปี 20 เทอม 12").semesters, ())
        self.assertEqual(parse_query_spec("IT ปี 6 เทอม 1").years, ())

    def test_course_codes_are_bounded_and_preserve_order(self):
        self.assertEqual(
            parse_query_spec("06016419 กับ 06016414").course_codes,
            ("06016419", "06016414"),
        )
        self.assertEqual(parse_query_spec("06019999").course_codes, ("06019999",))
        self.assertEqual(parse_query_spec("6016414 123456789").course_codes, ())
        self.assertEqual(parse_query_spec("06016414 06016414").course_codes, ("06016414",))

    def test_course_name_is_conservative_and_topic_is_not_promoted(self):
        name_spec = parse_query_spec("วิชา NOSQL เรียนเรื่องอะไรบ้าง")
        self.assertEqual(name_spec.course_name, "NOSQL")
        self.assertIsNone(name_spec.topic)

        for topic in (
            "database",
            "AI",
            "คอม",
            "คอมพิวเตอร์",
            "data",
            "เว็บ",
            "network",
            "programming",
        ):
            with self.subTest(topic=topic):
                spec = parse_query_spec(f"มีวิชาเกี่ยวกับ {topic} อะไรบ้าง")
                self.assertIsNone(spec.course_name)
                self.assertEqual(spec.topic, topic)
        self.assertIsNone(parse_query_spec("06016414 เรียนเกี่ยวกับอะไร").course_name)

    def test_thai_database_alias_uses_canonical_topic_and_stays_narrow(self):
        for question in (
            "มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง",
            "วิชาเกี่ยวกับฐานข้อมูลใน IT มีอะไรบ้าง",
        ):
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                self.assertEqual(spec.topic, "database")

        self.assertEqual(
            parse_query_spec("มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง").operations,
            ("list",),
        )

        self.assertEqual(
            parse_query_spec("วิชาเกี่ยวกับฐานข้อมูลใน IT มีอะไรบ้าง").operations,
            ("list",),
        )
        self.assertIsNone(parse_query_spec("ข้อมูลทั่วไปของหลักสูตร").topic)
        self.assertEqual(parse_query_spec("มีวิชาเกี่ยวกับข้อมูลอะไรบ้าง").operations, ())

    def test_category_is_explicit_and_can_coexist_with_topic(self):
        spec = parse_query_spec("วิชาเลือกของ IT ที่เกี่ยวกับ AI")
        self.assertEqual(spec.category, "วิชาเลือก")
        self.assertEqual(spec.topic, "AI")
        self.assertIsNone(parse_query_spec("IT เรียนเกี่ยวกับ AI").category)

    def test_representative_frozen_questions_extract_entities_without_later_stages(self):
        fixture_path = Path(__file__).parents[0] / "fixtures" / "natural_qa_v1.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        questions = {case["id"]: case["question"] for case in fixture["cases"]}

        expected = {
            "nq_006": {"course_codes": ("06016414",)},
            "nq_009": {"program": "IT", "plans": ("coop", "no_coop")},
            "nq_012": {"course_name": "NOSQL", "topic": None},
            "nq_017": {"program": "IT", "years": (2,), "topic": "database"},
            "nq_018": {"program": "IT", "years": (3,), "topic": "AI"},
            "nq_022": {"program": "IT", "years": (2,), "topic": "คอม"},
            "nq_023": {"program": "IT", "years": (2, 3), "topic": "programming"},
            "nq_024": {"program": "IT", "semesters": (), "topic": "database"},
            "nq_026": {"program": "IT", "category": "วิชาเลือก", "topic": "AI"},
            "nq_027": {"years": (3,), "category": "วิชาเลือก", "topic": "เว็บ"},
            "nq_031": {"course_codes": ("06016414", "06016419")},
            "nq_032": {"course_codes": ("06016414", "06016419")},
            "nq_039": {"course_codes": ("06019999",)},
            "nq_040": {"program": "IT", "years": (3,), "topic": "network"},
        }
        for case_id, entity_expectations in expected.items():
            with self.subTest(case_id=case_id):
                spec = parse_query_spec(questions[case_id])
                for field, value in entity_expectations.items():
                    self.assertEqual(getattr(spec, field), value)

    def test_operations_grouping_and_judgements_for_frozen_cases(self):
        fixture_path = Path(__file__).parents[0] / "fixtures" / "natural_qa_v1.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        questions = {case["id"]: case["question"] for case in fixture["cases"]}
        expected = {
            "nq_005": (("sum_credits", "compare"), ("semester",), "none"),
            "nq_009": (("compare",), ("plan",), "none"),
            "nq_010": (("placement", "earliest", "compare"), ("plan",), "none"),
            "nq_021": (("count",), (), "quantity"),
            "nq_022": (("count", "sum_credits"), (), "workload"),
            "nq_023": (("count", "compare"), ("year",), "none"),
            "nq_024": (("count", "compare"), ("semester",), "none"),
            "nq_030": (("describe", "placement"), (), "none"),
            "nq_031": (("similarity",), ("course",), "none"),
            "nq_032": (("placement", "compare"), ("course",), "none"),
            "nq_036": (("count", "sum_credits"), (), "workload"),
            "nq_037": ((), (), "unsupported"),
            "nq_038": ((), (), "unsupported"),
        }
        for case_id, (operations, group_by, judgement) in expected.items():
            with self.subTest(case_id=case_id):
                spec = parse_query_spec(questions[case_id])
                self.assertEqual(spec.operations, operations)
                self.assertEqual(spec.group_by, group_by)
                self.assertEqual(spec.judgement, judgement)

    def test_explicit_course_content_comparison_uses_similarity(self):
        spec = parse_query_spec(
            "วิชา 06016402 ของ IT แบบไม่สหกิจ กับ 06026207 ของ DSBA แบบไม่สหกิจต่างก็เกี่ยวข้องกับข้อมูล แต่แต่ละวิชาเน้นเรื่องใดบ้าง?"
        )
        self.assertEqual(spec.operations, ("similarity",))
        self.assertEqual(spec.group_by, ("course",))

        placement_comparison = parse_query_spec(
            "06016414 กับ 06016419 ตัวไหนเรียนก่อน"
        )
        self.assertEqual(placement_comparison.operations, ("placement", "compare"))

    def test_identity_operation_uses_narrow_course_identity_cues(self):
        name_to_code = parse_query_spec("วิชา Calculus 1 รหัสวิชาอะไร")
        self.assertEqual(name_to_code.course_name, "Calculus 1")
        self.assertEqual(name_to_code.operations, ("identity",))

        code_to_name = parse_query_spec("06016414 ชื่อวิชาอะไร")
        self.assertEqual(code_to_name.course_codes, ("06016414",))
        self.assertEqual(code_to_name.operations, ("identity",))

        code_to_name_short = parse_query_spec("06016414 รหัสอะไร")
        self.assertEqual(code_to_name_short.operations, ())

    def test_describe_wording_does_not_become_identity(self):
        for wording in ("เรียนเรื่องอะไร", "เรียนเกี่ยวกับอะไร"):
            with self.subTest(wording=wording):
                spec = parse_query_spec(f"วิชา Calculus 1 {wording}")
                self.assertEqual(spec.operations, ("describe",))

    def test_course_targeted_semantic_detail_wording_enables_describe(self):
        questions = (
            "วิชา 06016406 ของ IT แบบไม่สหกิจเป็นโครงงานลักษณะไหน และผู้เรียนต้องทำหรือแสดงผลลัพธ์อะไรบ้าง?",
            "วิชา 06046413 ในหลักสูตร AIT นำเทคโนโลยีไปใช้กับงานด้านไหน และใช้เครื่องมืออะไรบ้าง?",
            "วิชา 06036111 ของ BIT แบบสหกิจพูดถึงเทคโนโลยีอะไร และส่งผลต่อการทำธุรกิจในด้านใดบ้าง?",
            "GENED 90641002 ช่วยเตรียมทักษะอะไรบ้าง และควรระวังความเสี่ยงด้านใดเมื่อใช้เทคโนโลยีดิจิทัล?",
            "GENED 90642011 ช่วยฝึกการคิดวิเคราะห์อย่างไร และนำไปใช้กับการเรียนหรือการทำงานได้แบบไหน?",
            "สำหรับ IT แบบสหกิจ วิชา 06016418 เรียนช่วงไหนของหลักสูตร และเนื้อหาครอบคลุมเรื่องใดเกี่ยวกับฐานข้อมูลบ้าง?",
            "วิชา 06036115 ใน BIT แบบสหกิจเรียนปีไหน เทอมไหน และเนื้อหาช่วยจัดการความปลอดภัยของระบบสารสนเทศเรื่องใดบ้าง?",
        )
        for question in questions:
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                self.assertIn("describe", spec.operations)
                self.assertTrue(spec.course_codes or spec.course_name)

    def test_course_targeted_describe_rule_keeps_non_describe_guards(self):
        expected = {
            "IT ปี 2 เรียนอะไรบ้าง": ("list",),
            "IT มีกี่วิชา": ("count",),
            "06016420 เรียนปีไหน": ("placement",),
            "06016420 ต้องผ่านวิชาอะไรก่อน": ("prerequisite",),
            "06016420 ต้องเตรียมผ่านวิชาอะไรในเทอมก่อนหน้า": ("prerequisite",),
            "มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง": ("list",),
            "IT มีวิชาเกี่ยวกับฐานข้อมูลอะไรบ้าง": ("list",),
        }
        for question, operations in expected.items():
            with self.subTest(question=question):
                self.assertEqual(parse_query_spec(question).operations, operations)

    def test_residual_course_targeted_wording_emits_each_operation_once(self):
        cases = (
            (
                "สำหรับ IT แบบสหกิจ วิชา 06016418 เรียนช่วงไหนของหลักสูตร และเนื้อหาครอบคลุมเรื่องใดเกี่ยวกับฐานข้อมูลบ้าง?",
                ("placement", "describe"),
            ),
            (
                "วิชา 06036115 ใน BIT แบบสหกิจเรียนปีไหน เทอมไหน และเนื้อหาช่วยจัดการความปลอดภัยของระบบสารสนเทศเรื่องใดบ้าง?",
                ("placement", "describe"),
            ),
        )
        for question, operations in cases:
            with self.subTest(question=question):
                spec = parse_query_spec(question)
                self.assertEqual(spec.operations, operations)
                self.assertEqual(spec.operations.count("placement"), 1)
                self.assertEqual(spec.operations.count("describe"), 1)

    def test_residual_wording_requires_exact_course_target_and_context(self):
        self.assertNotIn(
            "describe",
            parse_query_spec("เนื้อหาครอบคลุมเรื่องใดบ้าง").operations,
        )
        self.assertNotIn(
            "describe",
            parse_query_spec("IT เนื้อหาช่วยจัดการเรื่องใดบ้าง").operations,
        )
        self.assertNotIn("placement", parse_query_spec("ช่วงไหน").operations)

    def test_curriculum_placement_wording_does_not_duplicate_placement(self):
        spec = parse_query_spec(
            "วิชา 06016418 ของ IT แบบสหกิจเรียนช่วงไหนของหลักสูตร และเรียนปีไหน?"
        )
        self.assertEqual(spec.operations.count("placement"), 1)

    def test_equivalent_study_placement_wording_enables_placement(self):
        questions = (
            "วิชา 06016419 ของ IT แบบไม่สหกิจอยู่ปีไหน เทอมไหน กี่หน่วยกิต และเนื้อหาเกี่ยวกับเครือข่ายกับความปลอดภัยอย่างไร?",
            "ถ้าเรียน BIT แบบไม่สหกิจ วิชา 06036122 อยู่ปีไหน เทอมไหน กี่หน่วยกิต และเรียนเรื่องใดเกี่ยวกับมัลติมีเดียบ้าง?",
            "วิชา 06016404 ของ IT แบบไม่สหกิจอยู่ปีไหน เทอมไหน กี่หน่วยกิต และเนื้อหาเกี่ยวกับบริการคลาวด์ด้านใดบ้าง?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    parse_query_spec(question).operations,
                    ("placement", "sum_credits", "describe"),
                )

        dsba = parse_query_spec(
            "วิชา 06026259 ของ DSBA แบบสหกิจเรียนปีไหน เทอมไหน กี่หน่วยกิต และมีเนื้อหาที่ทำงานร่วมกับสถานประกอบการอย่างไร?"
        )
        self.assertEqual(dsba.operations, ("placement", "sum_credits", "describe"))

        comparison = parse_query_spec(
            "วิชา 06036103 ใน BIT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหน เทอมไหน และรายละเอียดการจัดวางต่างกันอย่างไร?"
        )
        self.assertEqual(comparison.operations, ("placement", "compare", "describe"))

    def test_year_semester_scope_detects_operations_without_program(self):
        spec = parse_query_spec("ปี 1 เทอม 1 มีวิชาอะไรบ้าง")

        self.assertIsNone(spec.program)
        self.assertEqual(spec.years, (1,))
        self.assertEqual(spec.semesters, (1,))
        self.assertEqual(spec.operations, ("list",))

    def test_calculus_roman_numeral_remains_unresolved_surface(self):
        spec = parse_query_spec("วิชา Calculus I รหัสวิชาอะไร")

        self.assertEqual(spec.course_name, "Calculus I")
        self.assertEqual(spec.operations, ("identity",))

    def test_parser_has_no_later_stage_dependencies(self):
        spec = parse_query_spec("IT ปีสามเกี่ยวกับ database")
        self.assertEqual(spec.program, "IT")
        self.assertEqual(spec.topic, "database")
        self.assertEqual(spec.operations, ())
        self.assertEqual(spec.group_by, ())
        self.assertEqual(spec.judgement, "none")

    def test_all_frozen_questions_match_query_spec_contract(self):
        fixture_path = Path(__file__).parents[0] / "fixtures" / "natural_qa_v1.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        expected_group_by = {
            "nq_005": ("semester",),
            "nq_009": ("plan",),
            "nq_010": ("plan",),
            "nq_023": ("year",),
            "nq_024": ("semester",),
            "nq_031": ("course",),
            "nq_032": ("course",),
        }
        entity_fields = (
            "program",
            "plans",
            "years",
            "semesters",
            "course_codes",
            "course_name",
            "category",
            "topic",
        )
        for case in fixture["cases"]:
            with self.subTest(case_id=case["id"]):
                question = case["question"]
                spec = parse_query_spec(question)
                expected_entities = case["expected"]["entities"]
                for field in entity_fields:
                    actual = getattr(spec, field)
                    if isinstance(actual, tuple):
                        actual = list(actual)
                    self.assertEqual(actual, expected_entities[field], field)
                self.assertEqual(spec.operations, tuple(case["expected"]["operations"]))
                self.assertEqual(spec.judgement, case["expected"]["judgement"])
                self.assertEqual(spec.original_question, question)
                self.assertEqual(spec.normalized_question, normalize_thai_surface(question))
                self.assertEqual(spec.group_by, expected_group_by.get(case["id"], ()))


if __name__ == "__main__":
    unittest.main()
