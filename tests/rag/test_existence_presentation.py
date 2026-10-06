import unittest

from rag.answer import render_grounded_claim
from rag.grounded_answer import GroundedClaim
from rag.evidence_planner import StructuralScope


class ExistencePresentationTests(unittest.TestCase):
    def test_positive_existence_renders_naturally(self):
        claim = GroundedClaim(
            claim_id="exists",
            operation="existence",
            effective_scope=StructuralScope(program="IT"),
            status="complete",
            value=True,
        )
        answer = render_grounded_claim(claim)
        self.assertNotIn("existence:", answer)
        self.assertIn("มีครับ", answer)

    def test_negative_grounded_existence_renders_naturally(self):
        claim = GroundedClaim(
            claim_id="exists",
            operation="existence",
            effective_scope=StructuralScope(program="IT"),
            status="complete",
            value=False,
        )
        answer = render_grounded_claim(claim)
        self.assertNotIn("existence:", answer)
        self.assertIn("ไม่พบรายวิชา", answer)


if __name__ == "__main__":
    unittest.main()
