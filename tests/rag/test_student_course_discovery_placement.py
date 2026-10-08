"""SD3 baseline: semantic student wording through verified placement evidence."""

import json
from collections.abc import Mapping
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.hybrid_demo import DEFAULT_CURRICULUM_DB_PATH
from rag.semantic.executor import execute_deterministic as _execute_deterministic
from rag.semantic.pipeline import semantic_answer


DB_PATH = Path(DEFAULT_CURRICULUM_DB_PATH)
QUESTION = "วิชา data pipeline ต้องเรียนในปีไหน"


def _placement_intent_payload():
    return json.dumps(
        {
            "task": "lookup",
            "subject": "course",
            "relation": "placement",
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
            "requested_fields": ["placement"],
            "clarification": None,
            "policy_topic": None,
            "observed_value": None,
        },
        ensure_ascii=False,
    )


class StudentCourseDiscoveryPlacementTests(unittest.TestCase):
    def test_student_data_pipeline_request_preserves_flexible_placement_evidence(self):
        executions = []

        def capture_execution(db_path, spec, resolution_context, question):
            verified = _execute_deterministic(
                db_path, spec, resolution_context, question
            )
            executions.append((spec, verified))
            return verified

        with patch(
            "rag.semantic.pipeline.execute_deterministic",
            side_effect=capture_execution,
        ):
            outcome = semantic_answer(
                DB_PATH,
                QUESTION,
                conversation_context={
                    "program": "DSBA",
                    "catalog_key": "dsba-2565",
                },
                interpret_callable=lambda _prompt: _placement_intent_payload(),
            )

        self.assertEqual(outcome.result.status, "answer")
        self.assertTrue(outcome.trace.validation["valid"])
        self.assertEqual(len(executions), 1)
        spec, verified = executions[0]

        # The compiled target is the resolver-verified canonical identity.
        self.assertEqual(spec.course_codes, ("06026239",))
        self.assertEqual(spec.operations, ("placement",))
        self.assertEqual(verified.status, "answer")

        placement_claims = [
            claim for claim in verified.claims if claim.operation == "placement"
        ]
        self.assertTrue(placement_claims)
        evidence_rows = [
            row
            for claim in placement_claims
            for row in claim.evidence
            if isinstance(row, Mapping)
        ]
        self.assertTrue(evidence_rows)
        self.assertTrue(
            all(row.get("course_code") == "06026239" for row in evidence_rows)
        )
        self.assertTrue(
            any(row.get("name_en") == "DATA PIPELINE ARCHITECTURE" for row in evidence_rows)
        )
        self.assertIn(
            ((3, 1), (3, 2), (4, 1)),
            [row.get("year_semester_choices") for row in evidence_rows],
        )
        self.assertTrue(verified.provenance)

        # Also require the structured evidence to survive into the final claim.
        final_placement_rows = [
            row
            for claim in outcome.result.claims
            if claim.operation == "placement"
            for row in claim.evidence
            if isinstance(row, Mapping)
        ]
        self.assertIn(
            ((3, 1), (3, 2), (4, 1)),
            [row.get("year_semester_choices") for row in final_placement_rows],
        )


if __name__ == "__main__":
    unittest.main()
