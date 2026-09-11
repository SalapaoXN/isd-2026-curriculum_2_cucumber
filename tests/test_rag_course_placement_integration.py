import unittest
from pathlib import Path
from unittest.mock import patch

from rag.qa import ask


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

    def test_semester_credits_and_prerequisite_do_not_use_placement_operation(self):
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            return "SELECT 1"

        result = ask(
            DB_PATH,
            "IT แบบไม่สหกิจ ในปี 2 เทอม 2 ลงทะเบียนรวมกี่หน่วยกิต "
            "และวิชา 06016420 ต้องผ่านวิชาอะไรมาก่อน?",
            structured_model_callable=fake_model,
        )

        self.assertEqual(result["route"], "structured")
        self.assertNotIn("operation", result["result"])
        self.assertEqual(len(calls), 1)

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


if __name__ == "__main__":
    unittest.main()
