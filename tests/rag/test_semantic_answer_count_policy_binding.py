"""P0 follow-up bindings: bare counts and decimal measurements.

Red-team residuals R1/R4: a forged bare count ("11 วิชา" for verified
10) and a forged policy decimal ("GPA 3.00" for verified 3.50) were
accepted as grounded synthesis. Both must reject to deterministic
fallback while exact matches stay accepted.
"""

import unittest
from types import SimpleNamespace

from rag.semantic.answerer import (
    render_semantic_answer,
    render_verified_fallback,
    validate_answer_text,
)
from rag.semantic.schema import VerifiedResult


def _count_verified(value=10):
    return VerifiedResult(
        status="answer",
        summary_facts=("จำนวน: 10",),
        claims=(SimpleNamespace(operation="count", status="complete",
                                value=value),),
        provenance=({"source": "count-control"},),
    )


def _policy_verified(fact="เกียรตินิยมอันดับหนึ่งต้องมี GPA ไม่ต่ำกว่า 3.50"):
    return VerifiedResult(
        status="answer",
        summary_facts=(fact,),
        claims=(),
        provenance=({"source": "policy-control"},),
    )


class CountBindingTests(unittest.TestCase):
    def test_correct_count_accepted(self):
        verified = _count_verified()
        self.assertTrue(validate_answer_text("มี 10 วิชา", verified))
        self.assertTrue(validate_answer_text("- จำนวน: 10", verified))

    def test_forged_count_rejected(self):
        verified = _count_verified()
        self.assertFalse(validate_answer_text("มี 11 วิชา", verified))
        self.assertFalse(validate_answer_text("มี 10 วิชา รวม 11 รายวิชา",
                                              verified))

    def test_negated_count_rejected(self):
        verified = _count_verified()
        self.assertFalse(validate_answer_text("ไม่ใช่ 10 วิชา", verified))

    def test_forged_count_falls_back(self):
        verified = _count_verified()
        expected = render_verified_fallback("q", verified)
        answer, mode = render_semantic_answer(
            "q", verified, lambda _prompt: "มี 11 วิชา")
        self.assertEqual(mode, "deterministic")
        self.assertEqual(answer, expected)
        self.assertNotIn("11", answer)

    def test_unrelated_numbers_are_not_count_bound(self):
        verified = _count_verified()
        self.assertTrue(validate_answer_text("มีทั้งหมด 10 วิชา", verified))
        # Placement-scope restatement without placement evidence fails
        # closed by design (same conservative rule as list answers).
        self.assertFalse(validate_answer_text(
            "ชั้นปีที่ 2 มี 10 วิชา", verified))


class DecimalBindingTests(unittest.TestCase):
    def test_correct_threshold_accepted(self):
        verified = _policy_verified()
        self.assertTrue(validate_answer_text(
            "เกียรตินิยมอันดับหนึ่งต้องมี GPA ไม่ต่ำกว่า 3.50", verified))
        self.assertTrue(validate_answer_text(
            "เกียรตินิยมอันดับหนึ่งต้องมี GPA ไม่ต่ำกว่า 3.5", verified))

    def test_forged_threshold_rejected(self):
        verified = _policy_verified()
        self.assertFalse(validate_answer_text(
            "เกียรตินิยมอันดับหนึ่งต้องมี GPA ไม่ต่ำกว่า 3.00", verified))

    def test_forged_threshold_falls_back(self):
        verified = _policy_verified()
        expected = render_verified_fallback("q", verified)
        answer, mode = render_semantic_answer(
            "q", verified,
            lambda _prompt: "เกียรตินิยมอันดับหนึ่งต้องมี GPA ไม่ต่ำกว่า 3.00")
        self.assertEqual(mode, "deterministic")
        self.assertEqual(answer, expected)
        self.assertNotIn("3.00", answer)


if __name__ == "__main__":
    unittest.main()
