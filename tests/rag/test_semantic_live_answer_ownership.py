"""Offline canonical-evidence reproductions for relationship ownership/presentation."""

import json
import unittest
from types import SimpleNamespace

from rag.semantic.answerer import render_semantic_answer
from rag.semantic.pipeline import semantic_answer
from tests.rag.test_semantic_mixed_scope_composition import DB, QUESTION, payload, resolve
from rag.semantic import executor
from rag.semantic.executor import execute_mixed_scope


class LiveAnswerOwnershipTests(unittest.TestCase):
    def prerequisite_claim(self, values):
        return SimpleNamespace(operation="prerequisite", status="complete", value=values,
            effective_scope=SimpleNamespace(course_targets=({"course_code": "12345678", "name_en": "FOUNDATION"},)))

    def test_projection_uses_canonical_owner_not_recursive_nested_code(self):
        claim = self.prerequisite_claim(({"prerequisite_code": "87654321", "raw_text": "87654321"},))
        line = executor._claim_line_with_evidence(claim)
        self.assertEqual(line, "12345678 FOUNDATION มีวิชาบังคับก่อน: 87654321")

    def test_prerequisite_objects_deduplicate_without_losing_owner(self):
        claim = self.prerequisite_claim(tuple({"prerequisite_code": code, "raw_text": code}
                                             for code in ("87654321", "11223344", "87654321")))
        line = executor._claim_line_with_evidence(claim)
        self.assertEqual(line, "12345678 FOUNDATION มีวิชาบังคับก่อน: 87654321, 11223344")

    def test_retained_relationship_subject_is_not_dependency(self):
        claim = self.prerequisite_claim(({"prerequisite_code": "87654321"},))
        entries, program = executor._retained_from_claims((claim,), "EXAMPLE", "example-edition")
        self.assertEqual([entry["course_code"] for entry in entries], ["12345678"])

    def test_explicit_no_prerequisite_keeps_subject(self):
        claim = self.prerequisite_claim(({"prerequisite_state": "explicit_none"},))
        self.assertEqual(executor._claim_line_with_evidence(claim), "12345678 FOUNDATION: ไม่มีวิชาบังคับก่อน")

    def test_prerequisite_answer_retains_queried_owner(self):
        data = payload(("prerequisites",), "prerequisite")
        data.update(task="lookup", aggregation=None)
        data["scope"].update(year=None, semester=None)
        outcome = semantic_answer(DB, "IT no_coop: prerequisites of 06016420?",
            {"program": "IT", "catalog_key": "it-2565"},
            interpret_callable=lambda prompt: json.dumps(data),
            answer_callable=lambda prompt: "06016413 มีวิชาบังคับก่อน: 06016413")
        self.assertEqual(outcome.result.status, "answer")
        self.assertIn("06016420", outcome.result.final_answer)
        self.assertNotIn("06016413 มีวิชาบังคับก่อน: 06016413", outcome.result.final_answer)
        self.assertEqual([entry["course_code"] for entry in outcome.next_context["result_courses"]], ["06016420"])

    def test_planning_retains_explicit_target_and_prerequisite_placements(self):
        data = payload(("prerequisites", "prerequisite_placement", "placement"))
        verified = execute_mixed_scope(DB, resolve(data), QUESTION)
        self.assertEqual(verified.status, "answer", verified.missing_information)
        omitted = "06016420 มีวิชาบังคับก่อน: 06016413\n06016413 ชั้นปีที่ 2 ภาคการศึกษาที่ 1\nหน่วยกิตรวมทั้งเทอม: 30"
        answer, mode = render_semantic_answer(QUESTION, verified, lambda prompt: omitted)
        target_lines = [line for line in answer.splitlines() if "06016420" in line]
        self.assertTrue(any("ชั้นปีที่ 2" in line and "ภาคการศึกษาที่ 2" in line for line in target_lines), answer)
        self.assertIn("ภาคการศึกษาที่ 1", answer)
        self.assertIn("หน่วยกิตรวมทั้งเทอม: 30", answer)


if __name__ == "__main__":
    unittest.main()
