import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.policy import answer_policy_question
from rag.policy.routing import route_policy_question
from rag.qa import ask
from rag.resolution import QueryContext


DB_PATH = Path(__file__).parents[2] / "cucumber_outputs" / "runtime" / "curriculum.db"


class RagPolicyTest(unittest.TestCase):
    def test_dsba_total_requirement_is_selected_by_catalog(self):
        expected = {"dsba-2560": 126, "dsba-2565": 132}
        for catalog_key, total in expected.items():
            answer = answer_policy_question(
                DB_PATH,
                "DSBA ต้องเรียนทั้งหมดกี่หน่วยกิต",
                catalog_key=catalog_key,
            )
            self.assertEqual(answer.status, "complete")
            self.assertEqual(answer.value, total)
            self.assertTrue(answer.provenance)
            self.assertTrue(
                all(reference["document_category"] == "plan" for reference in answer.provenance)
            )

            result = ask(
                DB_PATH,
                "DSBA ต้องเรียนทั้งหมดกี่หน่วยกิต",
                conversation_context=QueryContext(
                    program="DSBA", catalog_key=catalog_key
                ),
            )
            grounded = result["result"]
            self.assertEqual(grounded.status, "answer")
            self.assertIn(str(total), grounded.final_answer)

        unscoped = route_policy_question(
            DB_PATH, "DSBA ต้องเรียนทั้งหมดกี่หน่วยกิต"
        )
        self.assertEqual(unscoped.status, "insufficient_evidence")

    def test_registration_facts_are_canonical_and_provenanced(self):
        maximum = answer_policy_question(DB_PATH, "ปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต")
        minimum = answer_policy_question(DB_PATH, "ขั้นต่ำกี่หน่วยกิต")
        self.assertEqual(maximum.status, "complete")
        self.assertEqual(maximum.value, 22)
        self.assertEqual(maximum.source_rule_id, "rule:11")
        self.assertEqual(minimum.value, 9)
        self.assertTrue(maximum.provenance)
        self.assertEqual(maximum.provenance[0]["document_category"], "rule")

    def test_registration_exception_and_special_limits_remain_distinct(self):
        exception = answer_policy_question(DB_PATH, "กรณีพิเศษลงได้สูงสุดเท่าไร")
        special = answer_policy_question(DB_PATH, "ซัมเมอร์ลงได้กี่หน่วยกิต")
        self.assertEqual(exception.value, 27)
        self.assertEqual(exception.context, "graduation_exception")
        self.assertEqual(special.value, 9)
        self.assertEqual(special.context, "special_semester")
        self.assertNotEqual(exception.context, special.context)

    def test_registration_comparison_is_deterministic_and_conditional(self):
        answer = answer_policy_question(DB_PATH, "ลง 24 หน่วยกิตได้ไหม")
        self.assertEqual(answer.status, "complete")
        self.assertEqual(answer.value, 24)
        self.assertIn("เกินเพดานปกติ 22", answer.rendered_answer)
        self.assertIn("อาจทำได้เฉพาะกรณี", answer.rendered_answer)
        self.assertEqual({fact.value for fact in answer.facts}, {22, 27})
        self.assertEqual(len(answer.provenance), 4)

    def test_probation_entry_and_cleared_thresholds_are_separate(self):
        entry = answer_policy_question(DB_PATH, "GPA เท่าไรถึงติดโปร")
        cleared = answer_policy_question(DB_PATH, "GPA เท่าไรถึงพ้นโปร")
        self.assertEqual(entry.value, 2)
        self.assertEqual(entry.operator, "<")
        self.assertEqual(entry.condition, "below")
        self.assertEqual(cleared.value, 2)
        self.assertEqual(cleared.operator, ">=")
        self.assertEqual(cleared.condition, "at_least")
        self.assertNotEqual(entry.condition, cleared.condition)

    def test_program_totals_come_from_catalog_scoped_program_requirements(self):
        expected = {
            ("AIT", "ait-2566"): 120,
            ("BIT", "bit-2565"): 126,
            ("DSBA", "dsba-2560"): 126,
            ("DSBA", "dsba-2565"): 132,
            ("IT", "it-2565"): 129,
        }
        for (program, catalog_key), value in expected.items():
            answer = answer_policy_question(
                DB_PATH,
                f"{program} ต้องเรียนกี่หน่วยกิต",
                catalog_key=catalog_key,
            )
            self.assertEqual(answer.status, "complete")
            self.assertEqual(answer.value, value)
            self.assertEqual(answer.program, program)
            self.assertIsNone(answer.source_rule_id)
            self.assertTrue(answer.provenance)
            self.assertTrue(
                all(reference["document_category"] == "plan" for reference in answer.provenance)
            )

    def test_honors_and_reentry_use_source_supported_facts(self):
        first = answer_policy_question(DB_PATH, "เกียรตินิยมอันดับหนึ่ง GPA เท่าไร")
        second = answer_policy_question(DB_PATH, "เกียรตินิยมอันดับสอง GPA เท่าไร")
        reentry = answer_policy_question(DB_PATH, "กลับเข้าศึกษาได้ภายในกี่ปี")
        self.assertEqual(set(first.value), {3.75, 3.5})
        self.assertEqual(second.value, (3.25,))
        self.assertEqual(reentry.value, 1)
        self.assertTrue(all(f.source_rule_id for f in first.facts))
        self.assertTrue(all(f.provenance for f in (*first.facts, *second.facts, *reentry.facts)))

    def test_unsupported_and_unknown_program_fail_closed(self):
        self.assertEqual(
            answer_policy_question(DB_PATH, "นโยบายที่ไม่มีในขอบเขตคืออะไร").status,
            "unsupported",
        )
        self.assertEqual(
            answer_policy_question(DB_PATH, "ZZZ ต้องเรียนกี่หน่วยกิต").status,
            "unsupported",
        )
        self.assertEqual(
            answer_policy_question(DB_PATH, "IT ลงทะเบียนได้ไหมโดยไม่บอกจำนวน").status,
            "unsupported",
        )

    def test_missing_required_fact_is_insufficient_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.db"
            shutil.copy2(DB_PATH, path)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("DELETE FROM policy_facts WHERE fact_id = 28")
                connection.commit()
            answer = answer_policy_question(path, "ปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต")
            self.assertEqual(answer.status, "insufficient_evidence")
            self.assertEqual(answer.provenance, ())

    def test_no_provider_or_curriculum_planner_is_required(self):
        answer = answer_policy_question(DB_PATH, "IT ต้องเรียนกี่หน่วยกิต")
        self.assertEqual(answer.status, "complete")
        self.assertEqual(answer.rendered_answer, "หลักสูตร IT ต้องเรียนทั้งหมด 129 credits")


if __name__ == "__main__":
    unittest.main()
