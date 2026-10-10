"""Regression coverage for free-elective category wording across editions."""

import json
from pathlib import Path
import unittest

from rag.semantic.interpreter import parse_semantic_intent_payload
from rag.semantic.modes import semantic_ask_response
from rag.semantic.validation import validate_semantic_intent
from tests.rag.test_semantic_clarification_transport import payload


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"
QUESTION = "วิชาเสรีเลือกอะไรได้บ้าง"
CONTEXT = {"program": "DSBA", "catalog_key": "dsba-2565"}


def _free_elective_payload(category="หมวดวิชาเสรี"):
    data = payload(None)
    data.update(task="list", subject="course", relation=None, comparison=None)
    data["target"] = {
        "kind": "none", "raw_text": None, "normalized_hint": None, "ordinal": None
    }
    data["scope"] = {
        "program": None, "catalog": None, "plan": None, "plan_hint": None,
        "year": None, "semester": None,
    }
    data["filters"] = [
        {"field": "category", "operator": "eq", "value": category}
    ]
    data["requested_fields"] = ["code", "name"]
    data["clarification"] = None
    return data


class FreeElectiveCategoryRegressionTests(unittest.TestCase):
    def test_current_question_grounds_both_canonical_free_elective_labels(self):
        for category in ("หมวดวิชาเสรี", "หมวดวิชาเลือกเสรี"):
            with self.subTest(category=category):
                intent = parse_semantic_intent_payload(
                    json.dumps(_free_elective_payload(category), ensure_ascii=False)
                )
                result = validate_semantic_intent(intent, QUESTION)
                self.assertTrue(result.valid, result.reason)

    def test_dsba_2565_normalizes_legacy_free_elective_label_to_scoped_label(self):
        raw = json.dumps(_free_elective_payload("หมวดวิชาเสรี"), ensure_ascii=False)
        response = semantic_ask_response(
            DB,
            QUESTION,
            CONTEXT,
            home_program="DSBA",
            interpret_callable=lambda prompt: raw,
        )
        self.assertEqual(response["status"], "answer")
        answer = response["answer"]
        self.assertIn("แผนสหกิจ", answer)
        self.assertIn("แผนไม่สหกิจ", answer)
        self.assertIn("xxxxxxxx — วิชาเลือกเสรี 1", answer)
        self.assertIn("xxxxxxxx — วิชาเลือกเสรี 2", answer)


if __name__ == "__main__":
    unittest.main()
