"""Unit tests for bounded previous-answer follow-up grammar and state."""

import unittest

from rag.prev_answer import (
    classify_prev_followup,
    parse_last_answer,
)
from rag.query_spec import parse_query_spec


def _spec(question):
    return parse_query_spec(question)


class PrevFollowupGrammarTests(unittest.TestCase):
    def test_explain_phrases(self):
        for question in ("ขยายความหน่อย", "อธิบายเพิ่มหน่อย", "หมายความว่ายังไง"):
            self.assertEqual(
                classify_prev_followup(question, _spec(question)), "explain", question
            )

    def test_source_phrases(self):
        for question in ("อยู่ในข้อไหน", "อ้างอิงจากข้อไหน", "มาจากกฎข้อไหน"):
            self.assertEqual(
                classify_prev_followup(question, _spec(question)), "source", question
            )

    def test_rationale_phrase(self):
        question = "ทำไมถึงกำหนด 30 วัน"
        self.assertEqual(classify_prev_followup(question, _spec(question)), "rationale")

    def test_substance_questions_are_not_followups(self):
        for question in (
            "06016414 กี่หน่วย",
            "gpa เท่าไหร่ถึงติดโปร",
            "อธิบายหน่อยว่าวิชานี้เรียนเกี่ยวกับอะไร",
        ):
            self.assertIsNone(
                classify_prev_followup(question, _spec(question)), question
            )

    def test_new_explain_aliases(self):
        for question in (
            "ขยายอีกหน่อย",
            "ช่วยขยายอีกหน่อย",
            "ขอขยายอีกหน่อย",
            "อธิบายอีกที",
            "ช่วยอธิบายอีกที",
            "ขออธิบายอีกที",
            "หมายถึงอะไร",
            "หมายถึงยังไง",
            "ตรงนี้หมายถึงอะไร",
            "อันนี้หมายถึงอะไร",
            "ที่บอกมาหมายถึงอะไร",
            "สรุปว่าหมายถึงอะไร",
            "ขอรายละเอียดเพิ่มหน่อย",
            "ขอรายละเอียดอีกหน่อย",
            "ช่วยลงรายละเอียดหน่อย",
            "ลงรายละเอียดอีกนิด",
        ):
            self.assertEqual(
                classify_prev_followup(question, _spec(question)),
                "explain",
                question,
            )

    def test_new_source_aliases(self):
        for question in (
            "มาจากไหน",
            "เอามาจากไหน",
            "ข้อมูลนี้มาจากไหน",
            "มีที่มาไหม",
            "มีที่มามั้ย",
            "ขอดูที่มา",
        ):
            self.assertEqual(
                classify_prev_followup(question, _spec(question)),
                "source",
                question,
            )

    def test_new_rationale_aliases(self):
        for question in (
            "เพราะอะไร",
            "เพราะอะไรถึงกำหนดแบบนี้",
            "กำหนดแบบนี้เพราะอะไร",
        ):
            self.assertEqual(
                classify_prev_followup(question, _spec(question)),
                "rationale",
                question,
            )

    def test_ambiguous_phrasing_stays_unsupported(self):
        for question in (
            "ยังไงนะ",
            "คือไงอะ",
            "ขออีกนิด",
            "เพิ่มอีกหน่อย",
            "เล่าเพิ่มหน่อย",
            "บอกเพิ่มหน่อย",
            "พูดง่ายๆหน่อย",
            "ง่ายกว่านี้ได้มั้ย",
            "ขอแบบชัดๆหน่อย",
            "ขอให้เห็นภาพหน่อย",
            "แปลว่าอะไร",
            "แปลว่าไง",
            "คืออะไรอะ",
            "คือยังไง",
            "ข้ออะไร",
        ):
            self.assertIsNone(
                classify_prev_followup(question, _spec(question)), question
            )

    def test_substantive_new_queries_are_not_captured(self):
        for question in (
            "อธิบายวิชา 06016414 ให้หน่อย",
            "อธิบาย prerequisite ของ 06016414",
            "หมายความว่า IT มีทั้งหมดกี่หน่วย",
            "ขอรายละเอียดวิชา cybersecurity",
            "ขอรายละเอียดปี 2 เทอม 1",
            "อธิบายว่ามีวิชาอะไรในเทอม 2",
            "อ้างอิงรายวิชา 06016438",
            "ทำไมวิชา 06016414 มี 3 หน่วยกิต",
            "ขอรายละเอียดวิชา 06016414",
            "ข้อมูลนี้มาจากวิชาอะไร",
            "หมายถึง 06016414 ใช่ไหม",
            "เพราะอะไรวิชา 06016414 ถึงมี 3 หน่วยกิต",
        ):
            self.assertIsNone(
                classify_prev_followup(question, _spec(question)), question
            )

    def test_followup_anchor_does_not_capture_independent_subject(self):
        for question in (
            "เกียรตินิยมมาจากกฎข้อไหน",
            "ขยายความเกียรตินิยมหน่อย",
            "เกียรตินิยมอยู่ข้อไหน",
            "ภาคทัณฑ์มาจากไหน",
            "การลาเรียนมาจากข้อไหน",
            "พ้นสภาพมาจากไหน",
            "06016414 มาจากไหน",
            "ขยายความ 06016414 หน่อย",
            "วิชา database มาจากไหน",
            "IT ปี 2565 มาจากไหน",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    classify_prev_followup(question, _spec(question))
                )


class LastAnswerValidationTests(unittest.TestCase):
    def test_valid_policy_reference(self):
        parsed = parse_last_answer(
            {
                "route": "policy",
                "policy_kind": "sanction_appeal_deadline",
                "evidence_ids": ["rule:43"],
            },
            program=None,
            catalog_key=None,
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["route"], "policy")

    def test_valid_course_reference(self):
        parsed = parse_last_answer(
            {
                "route": "course",
                "course_code": "06016414",
                "operations": ["sum_credits"],
                "program": "IT",
                "catalog_key": "it-2565",
            },
            program="IT",
            catalog_key="it-2565",
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["course_code"], "06016414")

    def test_malformed_reference_rejected(self):
        for value in (
            {"route": "policy", "policy_kind": "no_such_kind"},
            {"route": "policy", "policy_kind": "probation_entry",
             "evidence_ids": "rule:26"},
            {"route": "policy", "policy_kind": "probation_entry",
             "evidence_ids": ["rule:1"] * 51},
            {"route": "policy", "policy_kind": "probation_entry",
             "evidence_ids": ["not-a-rule"]},
            {"route": "course", "course_code": "06016414",
             "operations": ["sum_credits"], "injected_fact": "x"},
            {"route": "course", "course_code": "abc",
             "operations": ["sum_credits"]},
            {"route": "unknown"},
            ["rule:43"],
        ):
            with self.assertRaises((TypeError, ValueError), msg=repr(value)):
                parse_last_answer(value, program="IT", catalog_key="it-2565")

    def test_stale_scope_returns_none(self):
        self.assertIsNone(
            parse_last_answer(
                {"route": "course", "course_code": "06016414",
                 "operations": ["sum_credits"], "program": "IT",
                 "catalog_key": "it-2560"},
                program="IT",
                catalog_key="it-2565",
            )
        )


if __name__ == "__main__":
    unittest.main()
