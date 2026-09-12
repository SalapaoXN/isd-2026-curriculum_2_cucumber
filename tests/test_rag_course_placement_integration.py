import unittest
from pathlib import Path
from unittest.mock import patch

from rag.answer import EMPTY_ANSWER, answer_question
from rag.qa import ask
from rag.structured.queries import (
    course_placement,
    earliest_year_semester,
    earliest_year_semester_from_choices,
    get_semester_credits,
    parse_flexible_year_semester,
    placement_year_semester_choices,
    semester_credits_and_prerequisites,
)


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


def _placement_rows(result):
    structured = result["result"]
    return [
        dict(zip(structured["columns"], row))
        for row in structured["rows"]
    ]


class CoursePlacementIntegrationTest(unittest.TestCase):
    def test_flexible_parser_handles_single_and_multiple_choices(self):
        self.assertEqual(parse_flexible_year_semester("4/1"), [(4, 1)])
        self.assertEqual(
            parse_flexible_year_semester("3/1, 3/2, 4/1"),
            [(3, 1), (3, 2), (4, 1)],
        )

    def test_earliest_comparison_uses_fixed_and_flexible_choices(self):
        self.assertEqual(
            earliest_year_semester(None, None, "3/1, 3/2, 4/1"),
            (3, 1),
        )
        self.assertEqual(
            placement_year_semester_choices(3, 1, "4/1"),
            [(3, 1)],
        )
        self.assertEqual(
            earliest_year_semester(3, 2, None),
            (3, 2),
        )

    def test_malformed_flexible_value_fails_safely(self):
        self.assertEqual(parse_flexible_year_semester("3/1, unknown"), [])
        self.assertEqual(parse_flexible_year_semester("5/1"), [])
        self.assertEqual(placement_year_semester_choices(None, None, None), [])

    def test_earliest_composition_handles_mixed_placements_without_guessing(self):
        mixed_placements = [
            {"year_semester_choices": [(4, 1)]},
            {"year_semester_choices": [(3, 2), (4, 1)]},
            {"year_semester_choices": []},
        ]
        self.assertEqual(
            [
                earliest_year_semester_from_choices(
                    placement["year_semester_choices"]
                )
                for placement in mixed_placements
            ],
            [(4, 1), (3, 2), None],
        )
        self.assertIsNone(
            earliest_year_semester_from_choices([(3, 1), ("4", 1)])
        )

    def test_course_placement_preserves_raw_and_exposes_choices(self):
        result = course_placement(DB_PATH, "IT", "06016481", ["coop", "no_coop"])
        by_plan = {placement["plan_key"]: placement for placement in result["placements"]}

        self.assertEqual(by_plan["coop"]["flexible_year_semester_raw"], None)
        self.assertEqual(by_plan["coop"]["year_semester_choices"], [(3, 2)])
        self.assertEqual(
            by_plan["no_coop"]["flexible_year_semester_raw"],
            "3/1, 3/2, 4/1",
        )
        self.assertEqual(
            by_plan["no_coop"]["year_semester_choices"],
            [(3, 1), (3, 2), (4, 1)],
        )

    def test_composed_result_uses_choices_for_earliest_timing(self):
        result = ask(
            DB_PATH,
            "วิชา 06016481 ใน IT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหน เทอมไหน?",
        )
        structured = result["result"]
        rows = [
            dict(zip(structured["columns"], row))
            for row in structured["rows"]
        ]
        by_plan = {row["plan_key"]: row for row in rows}

        self.assertEqual(by_plan["coop"]["year_semester_choices"], [(3, 2)])
        self.assertEqual(by_plan["coop"]["earliest_year_semester"], (3, 2))
        self.assertEqual(
            by_plan["no_coop"]["year_semester_choices"],
            [(3, 1), (3, 2), (4, 1)],
        )
        self.assertEqual(
            by_plan["no_coop"]["earliest_year_semester"],
            (3, 1),
        )

    def test_it_placement_uses_deterministic_operation(self):
        result = ask(
            DB_PATH,
            "วิชา 06016481 ใน IT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหน เทอมไหน "
            "และรายละเอียดการจัดวางต่างกันอย่างไร?",
        )

        self.assertEqual(result["route"], "structured")
        structured = result["result"]
        self.assertEqual(structured["operation"], "course_placement")
        self.assertEqual(structured["status"], "ok")
        rows = _placement_rows(result)
        by_plan = {row["plan_key"]: row for row in rows}
        self.assertEqual(
            (
                by_plan["coop"]["course_id"],
                by_plan["coop"]["year"],
                by_plan["coop"]["semester"],
            ),
            (649, 3, 2),
        )
        self.assertEqual(
            (
                by_plan["no_coop"]["course_id"],
                by_plan["no_coop"]["year"],
                by_plan["no_coop"]["semester"],
                by_plan["no_coop"]["flexible_year_semester_raw"],
            ),
            (815, None, None, "3/1, 3/2, 4/1"),
        )
        self.assertTrue(by_plan["coop"]["name_en"])
        self.assertTrue(by_plan["no_coop"]["name_en"])

    def test_cross_plan_earliest_placement_uses_deterministic_operation(self):
        calls = []

        def forbidden_model(_prompt):
            calls.append(True)
            raise AssertionError("structured model must not be called")

        result = ask(
            DB_PATH,
            "ถ้าอยากลง DATA CENTER DESIGN (06016465) ให้เร็วที่สุดใน IT "
            "ควรเลือกแผนไหน และแต่ละแผนเปิดให้ลงช่วงใดบ้าง?",
            structured_model_callable=forbidden_model,
        )

        self.assertEqual(result["route"], "structured")
        structured = result["result"]
        self.assertEqual(structured["operation"], "course_placement")
        self.assertEqual(structured["earliest_plan"], "no_coop")
        rows = _placement_rows(result)
        self.assertEqual({row["plan_key"] for row in rows}, {"coop", "no_coop"})
        self.assertEqual(
            {
                row["plan_key"]: row["year_semester_choices"]
                for row in rows
            },
            {
                "coop": [(4, 1)],
                "no_coop": [(3, 1), (3, 2), (4, 1)],
            },
        )
        self.assertTrue(structured["provenance"])
        self.assertEqual(calls, [])

    def test_course_name_from_placement_reaches_grounded_answer(self):
        question = (
            "วิชา 06016414 ของ IT แบบไม่สหกิจชื่อภาษาอังกฤษว่าอะไร "
            "และมีหน่วยกิตเท่าไร?"
        )
        result = ask(DB_PATH, question)
        structured = result["result"]
        rows = _placement_rows(result)
        self.assertEqual(structured["operation"], "course_placement")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["name_en"])
        self.assertTrue(rows[0]["provenance"])
        self.assertTrue(structured["provenance"])
        self.assertEqual(
            {item["provenance_id"] for item in structured["provenance"]},
            {item["provenance_id"] for item in rows[0]["provenance"]},
        )

        prompts = []
        answers = iter([EMPTY_ANSWER, "คำตอบจากข้อมูลวิชา"])

        def answer_model(prompt):
            prompts.append(prompt)
            return next(answers)

        answer = answer_question(
            question,
            "structured",
            structured_result=structured,
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "คำตอบจากข้อมูลวิชา")
        self.assertEqual(len(prompts), 2)
        self.assertIn("name_en", prompts[0])
        self.assertIn(rows[0]["name_en"], prompts[0])

    def test_bit_placement_preserves_both_independent_plan_rows(self):
        result = ask(
            DB_PATH,
            "วิชา 06036103 ใน BIT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหนและเทอมไหน?",
        )

        self.assertEqual(result["route"], "structured")
        rows = _placement_rows(result)
        self.assertEqual(
            {
                row["plan_key"]: (row["course_id"], row["year"], row["semester"])
                for row in rows
            },
            {"coop": (68, 2, 1), "no_coop": (129, 2, 1)},
        )

    def test_non_placement_structured_question_keeps_nl_to_sql_fallback(self):
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            return "SELECT 1"

        result = ask(
            DB_PATH,
            "รวมกี่หน่วยกิต?",
            structured_model_callable=fake_model,
        )

        self.assertEqual(result["route"], "structured")
        self.assertNotIn("operation", result["result"])
        self.assertEqual(result["result"]["sql"], "SELECT 1 LIMIT 100")
        self.assertEqual(result["result"]["rows"], [(1,)])
        self.assertEqual(len(calls), 1)

    def test_non_plan_sensitive_course_credit_question_keeps_fallback(self):
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            return "SELECT 1"

        result = ask(
            DB_PATH,
            "IT วิชา 06016465 มีกี่หน่วยกิต?",
            structured_model_callable=fake_model,
        )

        self.assertEqual(result["route"], "structured")
        self.assertNotIn("operation", result["result"])
        self.assertEqual(result["result"]["rows"], [(1,)])
        self.assertEqual(len(calls), 1)

    def test_prerequisite_only_falls_back_but_credits_only_is_deterministic(self):
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            return "SELECT 1"

        prerequisite_result = ask(
            DB_PATH,
            "IT แบบไม่สหกิจ วิชา 06016420 ต้องเรียนก่อนวิชาอะไร?",
            structured_model_callable=fake_model,
        )
        credits_result = ask(
            DB_PATH,
            "IT แบบไม่สหกิจ ในปี 2 เทอม 2 ลงทะเบียนรวมกี่หน่วยกิต",
            structured_model_callable=fake_model,
        )

        self.assertNotIn("operation", prerequisite_result["result"])
        self.assertEqual(credits_result["result"]["operation"], "semester_credits")
        self.assertEqual(credits_result["result"]["total_credits"], 30)
        self.assertEqual(len(calls), 1)

    def test_semester_credits_operation_resolves_plan_and_components(self):
        result = get_semester_credits(DB_PATH, "IT", "coop", 2, 2)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_credits"], 30)
        self.assertTrue(result["components"])
        self.assertTrue(
            all(component["plan_key"] == "coop" for component in result["components"])
        )
        self.assertTrue(
            all(component["year"] == 2 for component in result["components"])
        )
        self.assertTrue(
            all(component["semester"] == 2 for component in result["components"])
        )

        structured = ask(
            DB_PATH,
            "IT แบบสหกิจ ปี 2 เทอม 2 รวมกี่หน่วยกิต",
        )["result"]
        self.assertTrue(structured["provenance"])
        self.assertTrue(all(row[-1] for row in structured["rows"]))

    def test_semester_credits_alternative_group_counts_once(self):
        result = get_semester_credits(DB_PATH, "IT", "coop", 3, 2)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_credits"], 6)
        alternatives = [
            component
            for component in result["components"]
            if component["alternative_group_id"] is not None
        ]
        self.assertEqual(len(alternatives), 1)
        self.assertEqual(alternatives[0]["counted_credit_units"], 6)

        structured = ask(
            DB_PATH,
            "IT แบบสหกิจ ปี 3 เทอม 2 รวมกี่หน่วยกิต",
        )["result"]
        group_rows = [
            row
            for row in structured["rows"]
            if row[structured["columns"].index("alternative_group_id")] is not None
        ]
        self.assertEqual(len(group_rows), 1)
        self.assertTrue(group_rows[0][-1])
        self.assertTrue(structured["provenance"])

    def test_semester_credits_missing_term_is_no_data(self):
        result = get_semester_credits(DB_PATH, "IT", "default", 1, 1)

        self.assertEqual(result["status"], "no_data")
        self.assertIsNone(result["total_credits"])
        self.assertEqual(result["components"], [])

    def test_semester_credits_and_prerequisite_use_deterministic_operation(self):
        result = ask(
            DB_PATH,
            "IT แบบไม่สหกิจ ในปี 2 เทอม 2 ลงทะเบียนรวมกี่หน่วยกิต "
            "และวิชา 06016420 ต้องผ่านวิชาอะไรมาก่อน?"
        )

        self.assertEqual(result["route"], "structured")
        structured = result["result"]
        self.assertEqual(
            structured["operation"], "semester_credits_and_prerequisites"
        )
        self.assertEqual(structured["status"], "ok")
        rows = _placement_rows(result)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["total_credits"], 30)
        self.assertEqual(rows[0]["course_code"], "06016420")
        self.assertEqual(rows[0]["course_id"], 739)
        self.assertEqual(rows[0]["prerequisite_course_id"], 728)
        self.assertEqual(rows[0]["prerequisite_course_code"], "06016413")
        self.assertEqual(rows[0]["requirement_type"], "required")
        self.assertEqual(rows[0]["raw_text"], "06016413")
        self.assertTrue(rows[0]["provenance"])

    def test_placement_semantic_hybrid_keeps_both_evidence_paths(self):
        semantic_evidence = [{"chunk_id": "it-06016481-description"}]
        question = (
            "วิชา 06016481 ใน IT แบบสหกิจและแบบไม่สหกิจ อยู่ปีไหน เทอมไหน "
            "และเนื้อหาเกี่ยวข้องกับสถานประกอบการอย่างไร?"
        )

        with patch("rag.qa.retrieve", return_value=semantic_evidence):
            result = ask(DB_PATH, question)

        self.assertEqual(result["route"], "hybrid")
        self.assertEqual(
            result["result"]["structured"]["operation"], "course_placement"
        )
        self.assertEqual(result["result"]["semantic"], semantic_evidence)

    def test_mixed_operation_full_miss_is_no_data(self):
        result = semester_credits_and_prerequisites(
            DB_PATH,
            "PROGRAM_WITHOUT_THIS_PLAN",
            "no_coop",
            2,
            2,
            "99999999",
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["plans"], [])

    def test_mixed_operation_excludes_unrelated_plan_provenance(self):
        result = semester_credits_and_prerequisites(
            DB_PATH,
            "IT",
            "no_coop",
            2,
            2,
            "06016420",
        )

        references = result["plans"][0]["prerequisites"][0]["provenance"]
        filenames = {reference["source_filename"] for reference in references}
        self.assertEqual(
            filenames,
            {
                "it_page_034.png",
                "it_page_035.png",
                "it_page_333.png",
                "it_page_334.png",
                "it_page_338.png",
            },
        )
        self.assertNotIn("it_page_328.png", filenames)
        self.assertNotIn("it_page_371.png", filenames)


if __name__ == "__main__":
    unittest.main()
