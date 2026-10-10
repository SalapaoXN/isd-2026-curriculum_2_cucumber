"""Provider-free regression for ordinary IT term course lists."""

import json
from pathlib import Path
import re
import unittest

from rag.semantic.modes import semantic_ask_response
from tests.rag.test_semantic_clarification_transport import payload


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"
QUESTION = "ปีที่ 1 ภาคการศึกษาที่ 2 เรียนวิชาอะไรบ้าง"
EXPECTED = {
    "06016408",
    "06066001",
    "06066101",
    "06066301",
    "90641002",
    "90644008",
}


def list_payload(*, target_kind="none", raw_text=None, ordinal=None):
    data = payload(None)
    data.update(task="list", subject="course", relation=None, comparison=None)
    data["target"] = {
        "kind": target_kind, "raw_text": raw_text, "normalized_hint": None,
        "ordinal": ordinal, "members": [],
    }
    data["scope"].update(program=None, catalog=None, plan=None, plan_hint=None,
                         year=1, semester=2)
    data["filters"] = []
    data["aggregation"] = None
    data["ranking"] = None
    data["requested_fields"] = ["code", "name"]
    return json.dumps(data, ensure_ascii=False)


class ITTermListRegressionTests(unittest.TestCase):
    def test_year_one_semester_two_returns_complete_course_set_for_each_plan(self):
        for plan in ("coop", "no_coop"):
            with self.subTest(plan=plan):
                context = {"program": "IT", "catalog_key": "it-2565", "plan": plan}
                response = semantic_ask_response(
                    DB, QUESTION, context, home_program="IT",
                    interpret_callable=lambda prompt: list_payload(),
                )
                self.assertEqual(response["status"], "answer", response)
                codes = set(re.findall(r"(?m)^- ([0-9]{8})", response["answer"]))
                self.assertEqual(codes, EXPECTED)
                self.assertEqual(response["next_context"]["years"], [1])
                self.assertEqual(response["next_context"]["semesters"], [2])
                self.assertEqual(response["next_context"]["plan"], plan)

    def test_explicit_result_ordinal_still_resolves_retained_course(self):
        data = payload(None)
        data.update(task="lookup", subject="course", relation="credits", comparison=None)
        data["target"] = {
            "kind": "result_ordinal", "raw_text": None, "normalized_hint": None,
            "ordinal": 1, "members": [],
        }
        data["scope"].update(program=None, catalog=None, plan=None, plan_hint=None,
                             year=None, semester=None)
        data["filters"] = []
        data["aggregation"] = None
        data["ranking"] = None
        data["requested_fields"] = ["credits"]
        context = {
            "program": "IT", "catalog_key": "it-2565", "plan": "coop",
            "result_courses": [
                {"course_code": "90641002", "program": "IT", "catalog_key": "it-2565"}
            ],
            "result_scope_program": "IT",
        }
        response = semantic_ask_response(
            DB, "ตัวแรกกี่หน่วยกิต", context, home_program="IT",
            interpret_callable=lambda prompt: json.dumps(data, ensure_ascii=False),
        )
        self.assertEqual(response["status"], "answer", response)
        self.assertIn("90641002", response["answer"])
        self.assertIn("3", response["answer"])

    def test_year_number_cannot_be_reused_as_previous_result_ordinal(self):
        context = {
            "program": "IT", "catalog_key": "it-2565", "plan": "coop",
            "result_courses": [
                {"course_code": "90641002", "program": "IT", "catalog_key": "it-2565"}
            ],
            "result_scope_program": "IT",
        }
        response = semantic_ask_response(
            DB, QUESTION, context, home_program="IT",
            interpret_callable=lambda prompt: list_payload(target_kind="result_ordinal", ordinal=1),
        )
        self.assertEqual(response["status"], "insufficient_evidence", response)
        self.assertEqual(response["provenance"], [])
        self.assertNotIn("90641002", response["answer"])


if __name__ == "__main__":
    unittest.main()
