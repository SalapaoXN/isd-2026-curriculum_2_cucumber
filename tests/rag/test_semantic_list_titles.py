"""Grounded list pairs, bounded display, synthesis safety and temporal flow."""

import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
import unittest

from rag.semantic.answerer import MAX_ANSWER_LEN, render_semantic_answer, validate_answer_text
from rag.semantic.executor import _claim_line, _collection_course_identities, _retained_from_claims
from rag.semantic.modes import semantic_ask_response
from rag.semantic.schema import VerifiedResult
from tests.rag.test_semantic_clarification_transport import payload


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"
CONTEXT = {"program": "IT", "catalog_key": "it-2565", "plan": "coop"}


def verified(rows):
    claim = SimpleNamespace(operation="list", status="complete", value=tuple(rows))
    return VerifiedResult(status="answer", summary_facts=(_claim_line("list", rows),),
                          claims=(claim,), provenance=({"source_page": 1},))


def canonical_rows():
    with sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(
            """SELECT DISTINCT co.course_code, co.name_th, co.name_en
            FROM plan_placements pp JOIN curriculum_plans cp ON cp.plan_id=pp.plan_id
            JOIN programs p ON p.program_id=cp.program_id
            JOIN catalogs ca ON ca.catalog_id=cp.catalog_id
            JOIN courses co ON co.course_id=pp.course_id
            WHERE p.program_code='IT' AND ca.catalog_key='it-2565'
              AND cp.plan_key='coop' AND pp.year_number=2 AND pp.semester_number=2"""
        )]


def list_payload(year, semester):
    data = payload(None)
    data.update(task="list", subject="course", comparison=None)
    data["scope"].update(year=year, semester=semester)
    return json.dumps(data, ensure_ascii=False)


class ListTitleProjectionTests(unittest.TestCase):
    def test_paired_source_order_dedupe_and_placeholders(self):
        rows = [
            {"course_code": "10000002", "name_th": "ชื่อสอง", "name_en": "Second"},
            {"course_code": "10000001", "name_th": "ชื่อหนึ่ง"},
            {"course_code": "10000002", "name_th": "ชื่อซ้ำ"},
            {"course_code": "90644xxx", "name_th": "ช่องเลือก"},
        ]
        self.assertEqual(_collection_course_identities(rows),
                         [("10000002", "ชื่อสอง"), ("10000001", "ชื่อหนึ่ง"),
                          ("90644xxx", "ช่องเลือก")])
        line = _claim_line("list", rows)
        self.assertIn("10000002 — ชื่อสอง", line)
        self.assertIn("10000001 — ชื่อหนึ่ง", line)
        self.assertNotIn("ชื่อซ้ำ", line)
        self.assertIn("90644xxx — ช่องเลือก", line)
        self.assertEqual(_collection_course_identities({"reference": "10000003", "courses": rows}),
                         _collection_course_identities(rows))

    def test_title_priority_english_fallback_and_untitled_identity(self):
        rows = [
            {"course_code": "10000001", "name_th": "ชื่อไทย", "name_en": "English"},
            {"course_code": "10000002", "name_th": "  ", "name_en": "English fallback"},
            {"course_code": "10000003"},
        ]
        self.assertEqual(_collection_course_identities(rows), [
            ("10000001", "ชื่อไทย"), ("10000002", "English fallback"), ("10000003", None),
        ])
        answer, mode = render_semantic_answer("list", verified(rows), None)
        self.assertEqual(mode, "deterministic")
        self.assertIn("- 10000003\n", answer + "\n")
        self.assertNotIn("10000003 —", answer)
        self.assertTrue(validate_answer_text(answer, verified(rows)))

    def test_twenty_display_and_retention_with_late_identities(self):
        rows = [{"course_code": f"{10000000+i:08d}", "name_th": f"ชื่อ {i}"} for i in range(21)]
        result = verified(rows)
        answer, _ = render_semantic_answer("list", result, None)
        self.assertEqual(len(re.findall(r"(?m)^- \d{8}", answer)), 20)
        self.assertIn("10000008 — ชื่อ 8", answer)
        self.assertIn("10000009 — ชื่อ 9", answer)
        self.assertIn("10000010 — ชื่อ 10", answer)
        self.assertNotIn("10000020", answer)
        self.assertIn("แสดง 20 จาก 21 รายวิชา", answer)
        retained, _ = _retained_from_claims(result.claims, "IT", "it-2565")
        self.assertEqual([row["course_code"] for row in retained], [row["course_code"] for row in rows[:20]])
        self.assertTrue(all(set(row) == {"course_code", "program", "catalog_key"} for row in retained))

    def test_fallback_preserves_pairs_and_rejects_code_only_synthesis(self):
        rows = canonical_rows()
        result = verified(rows)
        fallback, _ = render_semantic_answer("list", result, None)
        code_only = ", ".join(row["course_code"] for row in rows)
        self.assertFalse(validate_answer_text(code_only, result))
        answer, mode = render_semantic_answer("list", result, lambda prompt: code_only)
        self.assertEqual((answer, mode), (fallback, "deterministic"))
        answer, mode = render_semantic_answer("list", result, lambda prompt: fallback)
        self.assertEqual(mode, "grounded_synthesis")
        self.assertEqual(answer, fallback)
        self.assertEqual(result.provenance, ({"source_page": 1},))

    def test_swapped_titles_are_rejected_and_single_item_list_uses_only_thai(self):
        rows = [{"course_code": "10000001", "name_th": "ชื่อหนึ่ง", "name_en": "One"},
                {"course_code": "10000002", "name_th": "ชื่อสอง", "name_en": "Two"}]
        self.assertFalse(validate_answer_text("- 10000001 — ชื่อสอง\n- 10000002 — ชื่อหนึ่ง", verified(rows)))
        one = verified(rows[:1])
        answer, _ = render_semantic_answer("list", one, None)
        self.assertTrue(validate_answer_text(answer, one))
        self.assertNotIn("One", answer)

    def test_conservative_canonical_ten_title_length_budget(self):
        with sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            raw = connection.execute("SELECT course_code,name_th,name_en FROM courses").fetchall()
        longest = {}
        for code, thai, english in raw:
            if isinstance(code, str) and len(code) == 8 and code.isascii() and code.isdigit():
                title = (thai or "").strip() or (english or "").strip()
                if len(title) > len(longest.get(code, "")):
                    longest[code] = title
        rows = [{"course_code": code, "name_th": title}
                for code, title in sorted(longest.items(), key=lambda item: len(item[1]), reverse=True)[:10]]
        # Include subset/footer overhead as a conservative budget allowance.
        text = _claim_line("list", rows) + "\n(แสดง 10 จาก 999 รายวิชา)"
        self.assertLessEqual(len(text), MAX_ANSWER_LEN)


class ListTitleProductionTests(unittest.TestCase):
    def assert_canonical_pairs(self, response):
        rows = canonical_rows()
        self.assertEqual(len(rows), 10)
        self.assertEqual(response["status"], "answer")
        self.assertTrue(response["provenance"])
        for row in rows:
            self.assertTrue(row["name_th"])
            self.assertIn(f"{row['course_code']} — {row['name_th']}", response["answer"])
        self.assertEqual(len(re.findall(r"(?m)^- \d{8}", response["answer"])), 10)
        self.assertLessEqual(len(response["answer"]), MAX_ANSWER_LEN)
        retained = response["next_context"]["result_courses"]
        self.assertEqual({row["course_code"] for row in retained}, {row["course_code"] for row in rows})

    def test_exact_y2_s2_production_path_once(self):
        response = semantic_ask_response(DB, "ปี 2 เทอม 2 มีวิชาอะไรบ้าง", CONTEXT,
            home_program="IT", interpret_callable=lambda prompt: list_payload(2, 2))
        self.assert_canonical_pairs(response)

    def test_ordinary_temporal_followup_once(self):
        first = semantic_ask_response(DB, "ปี 2 เทอม 1 มีวิชาอะไรบ้าง", CONTEXT,
            home_program="IT", interpret_callable=lambda prompt: list_payload(2, 1))
        self.assertEqual(first["status"], "answer")
        followup = semantic_ask_response(DB, "แล้วเทอม 2 ล่ะ", first["next_context"],
            home_program="IT", interpret_callable=lambda prompt: list_payload(None, 2))
        self.assert_canonical_pairs(followup)
        self.assertEqual(followup["next_context"]["years"], [2])
        self.assertEqual(followup["next_context"]["semesters"], [2])
        self.assertEqual(followup["next_context"]["plan"], "coop")
