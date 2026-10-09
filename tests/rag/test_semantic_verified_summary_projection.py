"""Regressions for verified placement and description summary projection."""

import json
from collections.abc import Mapping
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.hybrid_demo import DEFAULT_CURRICULUM_DB_PATH
from rag.semantic.answerer import validate_answer_text
from rag.semantic.executor import execute_deterministic as _execute_deterministic
from rag.semantic.pipeline import semantic_answer


DB_PATH = Path(DEFAULT_CURRICULUM_DB_PATH)
COURSE_CODE = "06026239"
COURSE_TITLE = "DATA PIPELINE ARCHITECTURE"
SCOPE = {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "no_coop"}


def _intent_payload(*, relation, requested_field):
    return json.dumps(
        {
            "task": "lookup",
            "subject": "course",
            "relation": relation,
            "target": {
                "kind": "literal",
                "raw_text": "data pipeline",
                "normalized_hint": None,
                "ordinal": None,
            },
            "scope": {
                "program": None,
                "catalog": None,
                "plan": None,
                "plan_hint": None,
                "year": None,
                "semester": None,
            },
            "filters": [],
            "aggregation": None,
            "ranking": None,
            "comparison": None,
            "requested_fields": [requested_field],
            "clarification": None,
            "policy_topic": None,
            "observed_value": None,
        },
        ensure_ascii=False,
    )


def _run_lookup(question, *, relation, requested_field):
    captured = []

    def capture_execution(db_path, spec, resolution_context, execution_question):
        verified = _execute_deterministic(
            db_path, spec, resolution_context, execution_question
        )
        captured.append((spec, verified))
        return verified

    with patch(
        "rag.semantic.pipeline.execute_deterministic",
        side_effect=capture_execution,
    ):
        outcome = semantic_answer(
            DB_PATH,
            question,
            conversation_context=dict(SCOPE),
            home_program="DSBA",
            interpret_callable=lambda _prompt: _intent_payload(
                relation=relation, requested_field=requested_field
            ),
            answer_callable=None,
        )
    if not captured:
        raise AssertionError("semantic pipeline did not execute deterministic evidence")
    return captured[0][0], captured[0][1], outcome


class SemanticVerifiedSummaryProjectionTests(unittest.TestCase):
    def test_no_coop_flexible_choices_survive_summary_and_answer(self):
        spec, verified, outcome = _run_lookup(
            "วิชา data pipeline ต้องเรียนปีไหน",
            relation="placement",
            requested_field="placement",
        )
        expected = ((3, 1), (3, 2), (4, 1))
        placement_claims = [claim for claim in verified.claims if claim.operation == "placement"]
        evidence_rows = [
            row
            for claim in placement_claims
            for row in claim.evidence
            if isinstance(row, Mapping)
        ]

        self.assertEqual(spec.course_codes, (COURSE_CODE,))
        self.assertEqual(spec.plans, ("no_coop",))
        self.assertEqual(verified.status, "answer")
        self.assertTrue(placement_claims)
        self.assertIn(expected, [tuple(tuple(choice) for choice in row["year_semester_choices"])
                                 for row in evidence_rows])
        self.assertTrue(verified.provenance)
        self.assertEqual(outcome.result.status, "answer")

        summary = "\n".join(verified.summary_facts)
        self.assertIn(COURSE_CODE, summary)
        self.assertIn(COURSE_TITLE, summary)
        for year, semester in expected:
            self.assertIn(f"ชั้นปีที่ {year} ภาคการศึกษาที่ {semester}", summary)
            self.assertIn(f"ชั้นปีที่ {year} ภาคการศึกษาที่ {semester}", outcome.result.final_answer)
        self.assertNotIn("ไม่ได้ระบุ", outcome.result.final_answer)
        self.assertNotIn("ไม่มีข้อมูล", outcome.result.final_answer)

    def test_verified_flexible_choices_reject_absence_claim(self):
        _, verified, _ = _run_lookup(
            "วิชา data pipeline ต้องเรียนปีไหน",
            relation="placement",
            requested_field="placement",
        )
        unsupported_absence = (
            f"{COURSE_CODE} {COURSE_TITLE} แผนไม่สหกิจ ไม่มีข้อมูลระบุปีที่ต้องเรียน"
        )

        self.assertFalse(validate_answer_text(unsupported_absence, verified))

    def test_description_summary_and_answer_preserve_verified_description_text(self):
        spec, verified, outcome = _run_lookup(
            "data pipeline เรียนเกี่ยวกับอะไร",
            relation="description",
            requested_field="description",
        )
        description_claims = [claim for claim in verified.claims if claim.operation == "describe"]
        evidence_rows = [
            row
            for claim in description_claims
            for row in claim.evidence
            if isinstance(row, Mapping)
        ]
        descriptions = [
            row["text"]
            for row in evidence_rows
            if row.get("chunk_type") == "description"
            and isinstance(row.get("text"), str)
            and row["text"].strip()
        ]

        self.assertEqual(spec.course_codes, (COURSE_CODE,))
        self.assertEqual(spec.operations, ("describe",))
        self.assertEqual(verified.status, "answer")
        self.assertTrue(descriptions)
        self.assertTrue(any("การออกแบบ" in text for text in descriptions))

        summary = "\n".join(verified.summary_facts)
        self.assertIn(COURSE_CODE, summary)
        self.assertIn(COURSE_TITLE, summary)
        self.assertIn("การออกแบบ", summary)
        self.assertIn("การออกแบบ", outcome.result.final_answer)
        self.assertNotEqual(outcome.result.final_answer.strip(), COURSE_CODE)


if __name__ == "__main__":
    unittest.main()
