"""Regression coverage for distinct masked elective slots in semantic lists."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from rag.semantic.answerer import render_semantic_answer, validate_answer_text
from rag.semantic.executor import _claim_line, _collection_course_identities
from rag.semantic.modes import semantic_ask_response
from rag.semantic.schema import VerifiedResult
from tests.rag.test_semantic_clarification_transport import payload


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


def _verified(rows):
    claim = SimpleNamespace(operation="list", status="complete", value=tuple(rows))
    return VerifiedResult(
        status="answer",
        summary_facts=(_claim_line("list", rows),),
        claims=(claim,),
        provenance=({"source_page": 1},),
    )


def _list_payload(year, semester):
    data = payload(None)
    data.update(task="list", subject="course", comparison=None)
    data["scope"].update(year=year, semester=semester)
    return json.dumps(data, ensure_ascii=False)


class MaskedSlotProjectionTests(unittest.TestCase):
    def test_all_eight_character_mask_shapes_are_displayable(self):
        rows = [
            {"course_code": "9064xxxx", "placement_id": 201, "name_th": "วิชาเลือกศึกษาทั่วไป"},
            {"course_code": "xxxxxxxx", "placement_id": 202, "name_th": "วิชาเลือกเสรี 1"},
            {"course_code": "xxxxxxxx", "placement_id": 203, "name_th": "วิชาเลือกเสรี 2"},
            {"course_code": "060164xx", "placement_id": 204, "name_th": "วิชาเลือกอีกแบบ"},
        ]
        expected = [
            ("9064xxxx", "วิชาเลือกศึกษาทั่วไป"),
            ("xxxxxxxx", "วิชาเลือกเสรี 1"),
            ("xxxxxxxx", "วิชาเลือกเสรี 2"),
            ("060164xx", "วิชาเลือกอีกแบบ"),
        ]
        self.assertEqual(_collection_course_identities(rows), expected)
        result = _verified(rows)
        answer, mode = render_semantic_answer("list", result, None)
        self.assertEqual(mode, "deterministic")
        for code, title in expected:
            self.assertIn(f"- {code} — {title}", answer)
        self.assertTrue(validate_answer_text(answer, result))


class MaskedSlotProductionTests(unittest.TestCase):
    def test_dsba_y4_s1_no_coop_keeps_both_specific_elective_slots(self):
        context = {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "no_coop"}
        response = semantic_ask_response(
            DB,
            "ปี 4 เทอม 1 มีวิชาอะไรบ้าง",
            context,
            home_program="DSBA",
            interpret_callable=lambda prompt: _list_payload(4, 1),
        )
        self.assertEqual(response["status"], "answer")
        answer = response["answer"]
        self.assertIn("06026215 — โครงงานวิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ 2", answer)
        self.assertIn("06026xxx — วิชาเลือกในกลุ่มวิชาเลือกหรือกลุ่มวิชาชีพเฉพาะด้าน 1", answer)
        self.assertIn("06026xxx — วิชาเลือกในกลุ่มวิชาเลือกหรือกลุ่มวิชาชีพเฉพาะด้าน 2", answer)
        self.assertEqual(answer.count("06026xxx"), 2)
        self.assertEqual(len([line for line in answer.splitlines() if line.startswith("- ")]), 3)

    def test_dsba_y4_s1_coop_keeps_general_and_two_free_elective_slots(self):
        context = {"program": "DSBA", "catalog_key": "dsba-2565", "plan": "coop"}
        response = semantic_ask_response(
            DB,
            "ปี 4 เทอม 1 มีวิชาอะไรบ้าง",
            context,
            home_program="DSBA",
            interpret_callable=lambda prompt: _list_payload(4, 1),
        )
        self.assertEqual(response["status"], "answer")
        answer = response["answer"]
        self.assertIn("06026215 — โครงงานวิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ 2", answer)
        self.assertIn("9064xxxx — วิชาเลือกหมวดวิชาศึกษาทั่วไป", answer)
        self.assertIn("xxxxxxxx — วิชาเลือกเสรี 1", answer)
        self.assertIn("xxxxxxxx — วิชาเลือกเสรี 2", answer)
        self.assertEqual(answer.count("xxxxxxxx"), 2)
        self.assertEqual(len([line for line in answer.splitlines() if line.startswith("- ")]), 4)

    def test_dsba_y4_s1_without_plan_labels_each_plan_result(self):
        context = {"program": "DSBA", "catalog_key": "dsba-2565"}
        response = semantic_ask_response(
            DB,
            "ปี 4 เทอม 1 มีวิชาอะไรบ้าง",
            context,
            home_program="DSBA",
            interpret_callable=lambda prompt: _list_payload(4, 1),
        )
        self.assertEqual(response["status"], "answer")
        answer = response["answer"]
        self.assertIn("แผนสหกิจ\nรายวิชาที่พบ:", answer)
        self.assertIn("แผนไม่สหกิจ\nรายวิชาที่พบ:", answer)
        self.assertIn("9064xxxx — วิชาเลือกหมวดวิชาศึกษาทั่วไป", answer)
        self.assertIn("06026xxx — วิชาเลือกในกลุ่มวิชาเลือกหรือกลุ่มวิชาชีพเฉพาะด้าน 2", answer)


if __name__ == "__main__":
    unittest.main()
