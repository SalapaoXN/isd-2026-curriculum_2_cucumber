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

    def test_plan_scoped_program_total_uses_requirement_and_plan_provenance(self):
        cases = (
            ("IT", "it-2565", "สหกิจ", "coop", 129),
            ("IT", "it-2565", "ไม่สหกิจ", "no_coop", 129),
            ("DSBA", "dsba-2565", "สหกิจ", "coop", 132),
            ("DSBA", "dsba-2565", "ไม่สหกิจ", "no_coop", 132),
        )
        for program, catalog_key, wording, plan, total in cases:
            question = f"{program} แผน{wording} รวมทั้งหมดกี่หน่วยกิต"
            with self.subTest(question=question):
                answer = answer_policy_question(
                    DB_PATH,
                    question,
                    catalog_key=catalog_key,
                )
                self.assertEqual(answer.status, "complete")
                self.assertEqual(answer.value, total)
                self.assertIn(str(total), answer.rendered_answer)
                self.assertIn(f"แผน{wording}", answer.rendered_answer)
                self.assertTrue(answer.provenance)
                self.assertTrue(
                    all(
                        reference.get("document_category") == "plan"
                        for reference in answer.provenance
                    )
                )

                routed = route_policy_question(
                    DB_PATH,
                    question,
                    catalog_key=catalog_key,
                )
                self.assertIsNotNone(routed)
                self.assertEqual(routed.status, "answer")
                self.assertIn(str(total), routed.final_answer)
                self.assertTrue(routed.provenance)

        # A semester/year total remains a curriculum aggregation, not the
        # whole-program requirement.
        self.assertIsNone(
            route_policy_question(
                DB_PATH,
                "IT แผนสหกิจ ปี 1 เทอม 1 รวมทั้งหมดกี่หน่วยกิต",
                catalog_key="it-2565",
            )
        )

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
            ("BIT", "bit-2560"): 126,
            ("BIT", "bit-2565"): 126,
            ("DSBA", "dsba-2560"): 126,
            ("DSBA", "dsba-2565"): 132,
            ("IT", "it-2560"): 130,
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

    def test_text_policy_answers_use_exact_regulation_rules_and_provenance(self):
        cases = (
            (
                "ลาพักการศึกษาต้องทำอย่างไร",
                ("rule:31.1", "rule:31.2", "rule:31.3", "rule:31.4"),
                ("ข้อ 31.1", "ข้อ 31.4"),
            ),
            (
                "ลาออกต้องทำอย่างไร",
                ("rule:32",),
                ("ข้อ 32", "ไม่มีหนี้สิน"),
            ),
            (
                "เทียบโอนหน่วยกิตมีหลักเกณฑ์อะไรบ้าง",
                ("rule:28", "rule:29"),
                ("ข้อ 28", "ข้อ 29"),
            ),
        )
        for question, expected_rule_ids, expected_text in cases:
            with self.subTest(question=question):
                answer = answer_policy_question(DB_PATH, question)
                self.assertEqual(answer.status, "complete")
                self.assertEqual(
                    tuple(rule.rule_id for rule in answer.rules),
                    expected_rule_ids,
                )
                self.assertTrue(answer.provenance)
                self.assertTrue(
                    all(
                        reference["document_category"] == "rule"
                        for reference in answer.provenance
                    )
                )
                for text in expected_text:
                    self.assertIn(text, answer.rendered_answer)

    def test_phase_b_policy_answers_are_grounded_and_bounded(self):
        exam = answer_policy_question(DB_PATH, "ทุจริตในการสอบมีโทษอย่างไร")
        self.assertEqual(exam.status, "complete")
        self.assertEqual(tuple(rule.rule_id for rule in exam.rules), ("rule:20",))
        self.assertIn("ข้อ 20", exam.rendered_answer)
        self.assertTrue(exam.provenance)

        discipline = answer_policy_question(DB_PATH, "โทษทางวินัยมีอะไรบ้าง")
        self.assertEqual(discipline.status, "complete")
        self.assertEqual(
            tuple(rule.rule_id for rule in discipline.rules),
            (
                "rule:38",
                "rule:38.1",
                "rule:38.2",
                "rule:38.3",
                "rule:39",
                "rule:39.1",
                "rule:39.2",
                "rule:39.3",
            ),
        )
        for text in ("ว่ากล่าวตักเตือน", "ภาคทัณฑ์", "พักการเรียน", "ให้ออก", "ไล่ออก"):
            self.assertIn(text, discipline.rendered_answer)

        deadline = answer_policy_question(
            DB_PATH, "อุทธรณ์คำสั่งลงโทษต้องยื่นภายในกี่วัน"
        )
        self.assertEqual(deadline.status, "complete")
        self.assertEqual(deadline.value, 30)
        self.assertEqual(deadline.unit, "วัน")
        self.assertEqual(deadline.source_rule_id, "rule:43")
        self.assertIn("30 วัน", deadline.rendered_answer)
        self.assertTrue(deadline.provenance)

        procedure = answer_policy_question(
            DB_PATH, "อุทธรณ์คำสั่งลงโทษต้องทำอย่างไร"
        )
        self.assertEqual(procedure.status, "complete")
        self.assertEqual(tuple(rule.rule_id for rule in procedure.rules), ("rule:43",))
        self.assertIn("ข้อ 43", procedure.rendered_answer)
        self.assertTrue(procedure.provenance)

    def test_phase_b_ambiguous_questions_fail_closed(self):
        for question in (
            "อุทธรณ์ต้องยื่นภายในกี่วัน",
            "โดนลงโทษแล้วทำยังไง",
            "ทำผิดวินัยจะโดนอะไร",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    answer_policy_question(DB_PATH, question).status,
                    "unsupported",
                )

    def test_phase_c_graduation_policy_answers(self):
        gpa = answer_policy_question(DB_PATH, "สำเร็จการศึกษาต้องมี GPA เท่าไร")
        self.assertEqual(gpa.status, "complete")
        self.assertEqual(set(gpa.value), {2})
        self.assertIn("GPA ตามโครงสร้างหลักสูตร", gpa.rendered_answer)
        self.assertIn("GPA สะสม", gpa.rendered_answer)
        self.assertTrue(gpa.provenance)

        english = answer_policy_question(
            DB_PATH, "จบการศึกษาต้องสอบ English Exit Exam ไหม"
        )
        self.assertEqual(english.status, "complete")
        self.assertEqual(english.source_rule_id, "rule:25.2")
        self.assertIn("English Exit Exam", english.rendered_answer)
        self.assertTrue(english.provenance)

        debt = answer_policy_question(DB_PATH, "จบการศึกษาต้องไม่มีหนี้สินไหม")
        self.assertEqual(debt.status, "complete")
        self.assertEqual(debt.source_rule_id, "rule:25.4")
        self.assertIn("ไม่มีหนี้สิน", debt.rendered_answer)
        self.assertTrue(debt.provenance)

        requirements = answer_policy_question(
            DB_PATH, "เกณฑ์สำเร็จการศึกษามีอะไรบ้าง"
        )
        self.assertEqual(requirements.status, "complete")
        self.assertIn("2.0", requirements.rendered_answer)
        self.assertIn("English Exit Exam", requirements.rendered_answer)
        self.assertIn("ข้อ 25.3", requirements.rendered_answer)
        self.assertIn("ข้อ 25.5", requirements.rendered_answer)
        self.assertTrue(requirements.provenance)

    def test_phase_c_graduation_queries_remain_bounded(self):
        for question in (
            "จบการศึกษาต้องครบกี่หน่วยกิต",
            "English Exit Exam ผ่านกี่คะแนน",
            "จบได้ไหมถ้าผมติดหนี้อยู่ 100 บาท",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    answer_policy_question(DB_PATH, question).status,
                    "unsupported",
                )

    def test_remaining_gt_policy_families_are_grounded(self):
        termination = answer_policy_question(
            DB_PATH, "พ้นสภาพนักศึกษามีกรณีอะไรบ้าง"
        )
        self.assertEqual(termination.status, "complete")
        self.assertEqual(len(termination.rules), 12)
        self.assertIn("ข้อ 33.1", termination.rendered_answer)
        self.assertIn("ข้อ 33.12", termination.rendered_answer)
        self.assertIn("ไม่ยืนยันจำนวนครั้ง", termination.rendered_answer)
        self.assertTrue(termination.provenance)

        termination_gpa = answer_policy_question(
            DB_PATH, "GPA เท่าไรถึงพ้นสภาพนักศึกษา"
        )
        self.assertEqual(termination_gpa.status, "complete")
        self.assertEqual(termination_gpa.value, 1)
        self.assertEqual(termination_gpa.condition, "below")
        self.assertEqual(termination_gpa.source_rule_id, "rule:33.12")
        self.assertIn("GPA สะสมต่ำกว่า 1", termination_gpa.rendered_answer)

        gpa = answer_policy_question(DB_PATH, "การคิด GPA มีกี่ประเภท")
        self.assertEqual(gpa.status, "complete")
        self.assertIn("3 ประเภท", gpa.rendered_answer)
        self.assertIn("ค่าเฉลี่ยสะสมตามโครงสร้างหลักสูตร", gpa.rendered_answer)
        self.assertEqual(
            tuple(rule.rule_id for rule in gpa.rules),
            ("rule:21.1", "rule:21.2", "rule:21.2.1", "rule:21.2.2", "rule:21.2.3"),
        )

        assessment = answer_policy_question(DB_PATH, "การวัดผลการศึกษาทำได้อย่างไร")
        self.assertEqual(assessment.status, "complete")
        self.assertIn("การสอบหรือวิธีอื่น", assessment.rendered_answer)
        self.assertIn("คณะกรรมการ", assessment.rendered_answer)
        self.assertEqual(
            tuple(rule.rule_id for rule in assessment.rules),
            ("rule:19.1", "rule:19.2"),
        )

        conduct = answer_policy_question(
            DB_PATH, "นักศึกษาต้องปฏิบัติตัวอย่างไร"
        )
        self.assertEqual(conduct.status, "complete")
        self.assertIn("แต่งกายให้สุภาพ", conduct.rendered_answer)
        self.assertIn("ไม่เสพสุรา", conduct.rendered_answer)
        self.assertEqual(len(conduct.rules), 5)

        serious = answer_policy_question(
            DB_PATH, "ความผิดวินัยร้ายแรงมีอะไรบ้าง"
        )
        self.assertEqual(serious.status, "complete")
        self.assertIn("ทุจริตในการสอบ", serious.rendered_answer)
        self.assertIn("เล่นการพนัน", serious.rendered_answer)
        self.assertEqual(len(serious.rules), 11)
        self.assertTrue(serious.provenance)

    def test_remaining_gt_policy_families_stay_bounded(self):
        for question in (
            "GPA 0.95 จะพ้นสภาพวันไหน",
            "B+ ได้กี่แต้ม",
            "ผมทำแบบนี้จะโดนไล่ออกไหม",
            "สอบวิชานี้ใช้ข้อสอบแบบไหน",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    answer_policy_question(DB_PATH, question).status,
                    "unsupported",
                )

    def test_text_policy_queries_remain_bounded(self):
        for question in (
            "ลาออกแล้วได้เงินคืนไหม",
            "ลาพัก 2 เทอมได้ไหม",
            "เทียบโอนวิชานี้ได้ไหม",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    answer_policy_question(DB_PATH, question).status,
                    "unsupported",
                )

    def test_text_policy_missing_required_rule_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.db"
            shutil.copy2(DB_PATH, path)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("DELETE FROM regulation_rules WHERE rule_id = 'rule:31.2'")
                connection.commit()
            answer = answer_policy_question(path, "ลาพักการศึกษาต้องทำอย่างไร")
            self.assertEqual(answer.status, "insufficient_evidence")
            self.assertEqual(answer.provenance, ())

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
        answer = answer_policy_question(DB_PATH, "AIT ต้องเรียนกี่หน่วยกิต")
        self.assertEqual(answer.status, "complete")
        self.assertEqual(answer.rendered_answer, "หลักสูตร AIT ต้องเรียนทั้งหมด 120 credits")

    def test_multi_edition_program_total_without_catalog_fails_closed(self):
        # IT now has two catalog editions (it-2560: 130, it-2565: 129), so
        # an edition-ambiguous total must fail closed instead of guessing.
        answer = answer_policy_question(DB_PATH, "IT ต้องเรียนกี่หน่วยกิต")
        self.assertEqual(answer.status, "insufficient_evidence")
        legacy = answer_policy_question(
            DB_PATH, "IT ต้องเรียนกี่หน่วยกิต", catalog_key="it-2560"
        )
        self.assertEqual(legacy.status, "complete")
        self.assertEqual(legacy.value, 130)
        current = answer_policy_question(
            DB_PATH, "IT ต้องเรียนกี่หน่วยกิต", catalog_key="it-2565"
        )
        self.assertEqual(current.status, "complete")
        self.assertEqual(current.value, 129)


if __name__ == "__main__":
    unittest.main()
