"""P0.2/P0.3 render-level controls: forged synthesis must fall back.

Companion to ``test_semantic_answer_fact_integrity`` (which asserts
validator rejection). This module proves the production boundary:
``render_semantic_answer`` returns ``deterministic`` grounded output for
every proven-forged shape and keeps ``grounded_synthesis`` for correct
shapes. No provider or database is used.
"""

import unittest

from tests.rag import test_semantic_answer_fact_integrity as red
from rag.semantic.answerer import (
    render_semantic_answer,
    render_verified_comparison,
    render_verified_fallback,
)


class ForgedSynthesisFallbackTests(unittest.TestCase):
    def assert_fallback(self, question, verified, forged, must_contain=(),
                        must_not_contain=()):
        expected = render_verified_fallback(question, verified)
        answer, mode = render_semantic_answer(
            question, verified, lambda _prompt: forged
        )
        self.assertEqual(mode, "deterministic")
        self.assertEqual(answer, expected)
        for token in must_contain:
            self.assertIn(token, answer)
        for token in must_not_contain:
            self.assertNotIn(token, answer)

    def test_wrong_credit_falls_back(self):
        verified = red._single_course_verified()
        self.assert_fallback(
            "q", verified, "06016405 CALCULUS 1 มี 99 หน่วยกิต",
            must_contain=("06016405", "3"), must_not_contain=("99",),
        )

    def test_invented_requirement_falls_back(self):
        verified = red._single_course_verified()
        self.assert_fallback(
            "q", verified,
            "06016405 CALCULUS 1 มี 3 หน่วยกิต และต้องสอบภาษาอังกฤษเพิ่ม",
            must_contain=("06016405",),
        )

    def test_wrong_placement_falls_back(self):
        verified = red._single_course_verified(
            extra_fact="ชั้นปีที่ 1 ภาคการศึกษาที่ 1"
        )
        self.assert_fallback(
            "q", verified,
            "06016405 CALCULUS 1 อยู่ชั้นปีที่ 3 ภาคการศึกษาที่ 2",
            must_contain=("06016405",),
        )

    def test_negated_credit_falls_back(self):
        verified = red._single_course_verified()
        self.assert_fallback(
            "q", verified, "06016405 CALCULUS 1 ไม่ใช่ 3 หน่วยกิต",
            must_contain=("06016405",),
        )

    def test_list_value_swap_falls_back(self):
        rows = [
            {"course_code": "10000001", "name_th": "ชื่อหนึ่ง", "credits": 3},
            {"course_code": "10000002", "name_th": "ชื่อสอง", "credits": 1},
        ]
        verified = red._list_verified(rows)
        self.assert_fallback(
            "list", verified,
            "- 10000001 — ชื่อหนึ่ง 1 หน่วยกิต\n"
            "- 10000002 — ชื่อสอง 3 หน่วยกิต",
            must_contain=("10000001", "10000002"),
        )

    def test_comparison_side_swap_falls_back(self):
        comparison, verified = red._program_comparison()
        forged = (
            "เปรียบเทียบหน่วยกิต\n"
            "- IT: 132 หน่วยกิต\n"
            "- DSBA: 129 หน่วยกิต\n"
            "\nสรุป: มีผลต่าง 3 หน่วยกิต"
        )
        expected = render_verified_comparison(verified)
        answer, mode = render_semantic_answer(
            "compare", verified, lambda _prompt: forged,
            numeric_comparison=comparison,
        )
        self.assertEqual(mode, "deterministic")
        self.assertEqual(answer, expected)
        self.assertIn("129", answer)
        self.assertIn("132", answer)

    def test_wrong_winner_falls_back(self):
        comparison, verified = red._program_comparison()
        forged = (
            "เปรียบเทียบหน่วยกิต\n"
            "- IT: 129 หน่วยกิต\n"
            "- DSBA: 132 หน่วยกิต\n"
            "\nสรุป: IT มีหน่วยกิตมากกว่า DSBA มีผลต่าง 3 หน่วยกิต"
        )
        expected = render_verified_comparison(verified)
        answer, mode = render_semantic_answer(
            "compare", verified, lambda _prompt: forged,
            numeric_comparison=comparison,
        )
        self.assertEqual(mode, "deterministic")
        self.assertEqual(answer, expected)

    def test_wrong_prerequisite_falls_back(self):
        verified = red._single_course_verified(
            extra_fact="06016405 มีวิชาบังคับก่อน: 06016400"
        )
        self.assert_fallback(
            "q", verified,
            "06016405 CALCULUS 1 มีวิชาบังคับก่อน: 06016999",
            must_contain=("06016405",),
            must_not_contain=("06016999",),
        )


class CorrectSynthesisKeptTests(unittest.TestCase):
    def test_correct_credit_stays_grounded(self):
        verified = red._single_course_verified()
        text = "06016405 CALCULUS 1 มี 3 หน่วยกิต"
        answer, mode = render_semantic_answer("q", verified, lambda _p: text)
        self.assertEqual(mode, "grounded_synthesis")
        self.assertEqual(answer, text)

    def test_correct_program_comparison_stays_grounded(self):
        comparison, verified = red._program_comparison()
        text = render_verified_comparison(verified)
        answer, mode = render_semantic_answer(
            "compare", verified, lambda _p: text,
            numeric_comparison=comparison,
        )
        self.assertEqual(mode, "grounded_synthesis")
        self.assertEqual(answer, text)

    def test_correct_course_comparison_stays_grounded(self):
        comparison, verified = red._course_comparison()
        text = render_verified_comparison(verified)
        answer, mode = render_semantic_answer(
            "compare", verified, lambda _p: text,
            numeric_comparison=comparison,
        )
        self.assertEqual(mode, "grounded_synthesis")
        self.assertEqual(answer, text)

    def test_correct_list_stays_grounded(self):
        rows = [
            {"course_code": "10000001", "name_th": "ชื่อหนึ่ง"},
            {"course_code": "10000002", "name_th": "ชื่อสอง"},
        ]
        verified = red._list_verified(rows)
        text = render_verified_fallback("list", verified)
        answer, mode = render_semantic_answer(
            "list", verified, lambda _p: text
        )
        self.assertEqual(mode, "grounded_synthesis")
        self.assertEqual(answer, text)


if __name__ == "__main__":
    unittest.main()
