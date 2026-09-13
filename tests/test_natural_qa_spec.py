import json
import unittest
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "natural_qa_v1.json"

CANONICAL_QUESTIONS = [
    "IT ปี 2 เทอม 1 เรียนอะไรบ้าง",
    "ปีสองเทอมสองของ IT มีวิชาอะไร",
    "IT ปี 3 ต้องเรียนกี่วิชา",
    "ปี 2 ของ IT รวมกี่หน่วยกิต",
    "IT เทอมไหนหน่วยกิตเยอะสุด",
    "06016414 เรียนปีไหน",
    "06016414 เปิดให้ลงช่วงไหนได้บ้าง",
    "06016414 ต้องเรียนวิชาอะไรมาก่อน",
    "IT สหกิจกับไม่สหกิจต่างกันยังไง",
    "06016465 แผนไหนได้เรียนเร็วกว่า",
    "06016414 เรียนเกี่ยวกับอะไร",
    "วิชา NOSQL เรียนเรื่องอะไรบ้าง",
    "IT มีวิชาเกี่ยวกับ AI อะไรบ้าง",
    "มีวิชาเกี่ยวกับ database อะไรบ้างใน IT",
    "ถ้าชอบเขียนโปรแกรม IT มีวิชาอะไรน่าสนใจ",
    "มีวิชาเกี่ยวกับเว็บไหม",
    "IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง",
    "ปี 3 ของ IT มีวิชาเกี่ยวกับ AI กี่วิชา",
    "IT ปี 2 เทอม 1 มีวิชาเกี่ยวกับ programming ไหม",
    "ปี 2 มีวิชาเกี่ยวกับคอมพิวเตอร์เยอะมั้ย",
    "IT ปี 2 มีวิชาเกี่ยวกับคอมพิวเตอร์เยอะมั้ย",
    "ปีสองของ IT เรียนคอมหนักมั้ย",
    "IT ปี 2 กับปี 3 ปีไหนมีวิชา programming เยอะกว่า",
    "เทอมไหนของ IT มีวิชาเกี่ยวกับ database เยอะสุด",
    "แผนสหกิจมีวิชาเกี่ยวกับ data มากกว่าแผนปกติไหม",
    "วิชาเลือกของ IT ที่เกี่ยวกับ AI มีอะไรบ้าง",
    "ปี 3 มีวิชาเลือกเกี่ยวกับเว็บกี่ตัว",
    "ถ้าอยากเรียน AI เริ่มมีวิชาแนวนี้ตั้งแต่ปีไหน",
    "ถ้าอยากเรียน database ต้องผ่านวิชาอะไรมาก่อนบ้าง",
    "06016414 เรียนเรื่องอะไร แล้วเรียนปีไหน",
    "06016414 กับ 06016419 เนื้อหาคล้ายกันไหม",
    "06016414 กับ 06016419 ตัวไหนเรียนก่อน",
    "ปี 4 ต้องเรียนอะไรบ้าง",
    "IT ปี 4 ต้องเรียนอะไรบ้าง",
    "เทอมสองปีสามมีตัว database เยอะปะ",
    "IT ปีสามเทอมปลายเรียนหนักไหม",
    "วิชาไหนยากที่สุดใน IT",
    "เรียนวิชาไหนแล้วเงินเดือนสูงสุด",
    "06019999 เรียนอะไร",
    "ตอนปีสามของ IT มีวิชาเกี่ยวกับ network อะไรบ้าง",
]

ENTITY_KEYS = {
    "program",
    "plans",
    "years",
    "semesters",
    "course_codes",
    "course_name",
    "category",
    "topic",
}
ALLOWED_OPERATIONS = {
    "list",
    "describe",
    "count",
    "sum_credits",
    "existence",
    "compare",
    "earliest",
    "placement",
    "prerequisite",
    "similarity",
}
ALLOWED_JUDGEMENTS = {"none", "quantity", "workload", "preference", "unsupported"}
ALLOWED_ACTIONS = {"answer", "clarify_program", "no_data", "unsupported"}
ALLOWED_CAPABILITIES = {
    "normalize_thai",
    "exact_course_resolution",
    "exact_course_name_resolution",
    "structured_filter",
    "semantic_retrieval",
    "aggregation",
    "comparison",
    "plan_aware",
    "prerequisite_lookup",
    "composition",
    "ambiguity_guard",
    "no_data_guard",
    "unsupported_guard",
}
ALLOWED_ROUTE_HINTS = {"structured", "semantic", "hybrid", None}
CLARIFY_PROGRAM_IDS = {
    "nq_016",
    "nq_020",
    "nq_025",
    "nq_027",
    "nq_028",
    "nq_029",
    "nq_033",
    "nq_035",
}
PLAN_AWARE_IDS = {
    "nq_001",
    "nq_002",
    "nq_003",
    "nq_004",
    "nq_005",
    "nq_006",
    "nq_007",
    "nq_009",
    "nq_010",
    "nq_013",
    "nq_014",
    "nq_015",
    "nq_017",
    "nq_018",
    "nq_019",
    "nq_021",
    "nq_022",
    "nq_023",
    "nq_024",
    "nq_026",
    "nq_030",
    "nq_032",
    "nq_034",
    "nq_036",
    "nq_040",
}
FORBIDDEN_CONTEXT_FIELDS = {
    "previous_turn",
    "conversation_history",
    "session_state",
    "previous_question",
    "follow_up_context",
}


class NaturalQaSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.cases = cls.data["cases"]
        cls.by_id = {case["id"]: case for case in cls.cases}

    def test_top_level_contract_and_ids(self):
        self.assertEqual(self.data["version"], "natural_qa_v1")
        self.assertIsInstance(self.cases, list)
        self.assertEqual(len(self.cases), 40)
        ids = [case["id"] for case in self.cases]
        self.assertEqual(ids, [f"nq_{number:03d}" for number in range(1, 41)])
        self.assertEqual(len(ids), len(set(ids)))

    def test_canonical_questions_are_exact_and_nonempty(self):
        self.assertEqual([case["question"] for case in self.cases], CANONICAL_QUESTIONS)
        self.assertTrue(all(question.strip() for question in CANONICAL_QUESTIONS))

    def test_required_structure_and_types(self):
        expected_keys = {
            "entities",
            "operations",
            "judgement",
            "blocking_ambiguity",
            "action",
            "capabilities",
            "route_hint",
        }
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertIsInstance(case["id"], str)
                self.assertIsInstance(case["question"], str)
                expected = case["expected"]
                self.assertEqual(set(expected), expected_keys)
                self.assertEqual(set(expected["entities"]), ENTITY_KEYS)

                entities = expected["entities"]
                for key in ("plans", "years", "semesters", "course_codes"):
                    self.assertIsInstance(entities[key], list)
                for key in ("program", "course_name", "category", "topic"):
                    self.assertTrue(entities[key] is None or isinstance(entities[key], str))

                self.assertIsInstance(expected["operations"], list)
                self.assertIsInstance(expected["blocking_ambiguity"], list)
                self.assertIsInstance(expected["capabilities"], list)
                self.assertIsInstance(expected["judgement"], str)
                self.assertIsInstance(expected["action"], str)

                self.assertTrue(all(isinstance(value, str) for value in entities["plans"]))
                self.assertTrue(all(isinstance(value, int) for value in entities["years"]))
                self.assertTrue(all(isinstance(value, int) for value in entities["semesters"]))
                self.assertTrue(all(isinstance(value, str) for value in entities["course_codes"]))
                self.assertTrue(all(isinstance(value, str) for value in expected["operations"]))
                self.assertTrue(all(isinstance(value, str) for value in expected["blocking_ambiguity"]))
                self.assertTrue(all(isinstance(value, str) for value in expected["capabilities"]))

    def test_allowed_enums(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                expected = case["expected"]
                self.assertTrue(set(expected["operations"]).issubset(ALLOWED_OPERATIONS))
                self.assertIn(expected["judgement"], ALLOWED_JUDGEMENTS)
                self.assertIn(expected["action"], ALLOWED_ACTIONS)
                self.assertTrue(set(expected["capabilities"]).issubset(ALLOWED_CAPABILITIES))
                self.assertIn(expected["route_hint"], ALLOWED_ROUTE_HINTS)

    def test_clarify_program_policy(self):
        actual = {
            case["id"]
            for case in self.cases
            if case["expected"]["action"] == "clarify_program"
        }
        self.assertEqual(actual, CLARIFY_PROGRAM_IDS)

        for case_id in CLARIFY_PROGRAM_IDS:
            expected = self.by_id[case_id]["expected"]
            entities = expected["entities"]
            with self.subTest(case=case_id):
                self.assertIsNone(entities["program"])
                self.assertEqual(expected["blocking_ambiguity"], ["program"])
                self.assertEqual(expected["action"], "clarify_program")
                self.assertIn("ambiguity_guard", expected["capabilities"])

    def test_plan_aware_policy(self):
        actual = {
            case["id"]
            for case in self.cases
            if "plan_aware" in case["expected"]["capabilities"]
        }
        self.assertEqual(actual, PLAN_AWARE_IDS)

    def test_frozen_fixture_corrections(self):
        nq_012 = self.by_id["nq_012"]["expected"]["entities"]
        self.assertEqual(nq_012["course_name"], "NOSQL")
        self.assertIsNone(nq_012["topic"])
        for case_id in ("nq_023", "nq_024"):
            self.assertIn(
                "structured_filter",
                self.by_id[case_id]["expected"]["capabilities"],
            )

    def test_no_conversation_state_fields_recursively(self):
        def visit(value, path=()):
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertNotIn(key.casefold(), FORBIDDEN_CONTEXT_FIELDS, path + (key,))
                    visit(child, path + (key,))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, path + (index,))

        visit(self.data)


if __name__ == "__main__":
    unittest.main()
