"""COUNT-2: required-load semester-credit semantics.

Distinct elective slots that share a masked placeholder code are separate
required placements and must each count toward the semester total.
True duplicate rows (same placement) still collapse.
"""

import unittest
from pathlib import Path

from backend.hard_qa import _format_h4
from rag.aggregation import aggregate_sum_credits
from rag.qa import ask
from rag.resolution import QueryContext

DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


def _component(program, code, credits, placement_id=None):
    component = {
        "program": program,
        "course_code": code,
        "counted_credit_units": credits,
        "partition": {"program": program},
    }
    if placement_id is not None:
        component["placement_id"] = placement_id
    return component


def _ask_sum(question, plan):
    return ask(
        DB_PATH,
        question,
        conversation_context=QueryContext(
            program="DSBA", catalog_key="dsba-2565", plan=plan
        ),
    )


def _sum_value(result):
    for claim in result.claims:
        if claim.operation == "sum_credits" and claim.status == "complete":
            return claim.value
    return None


class RequiredLoadAggregationTests(unittest.TestCase):
    def test_distinct_slots_sharing_masked_code_all_count(self):
        result = aggregate_sum_credits(
            [
                _component("DSBA", "06026215", 3, placement_id=589),
                _component("DSBA", "06026xxx", 3, placement_id=590),
                _component("DSBA", "06026xxx", 3, placement_id=591),
            ]
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.value, 9)

    def test_true_duplicate_placement_still_collapses(self):
        result = aggregate_sum_credits(
            [
                _component("DSBA", "06026215", 3, placement_id=589),
                _component("DSBA", "06026215", 3, placement_id=589),
            ]
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.value, 3)

    def test_concrete_code_repeated_across_slots_still_collapses(self):
        result = aggregate_sum_credits(
            [
                _component("IT", "06016418", 3, placement_id=1289),
                _component("IT", "06016418", 3, placement_id=1295),
            ]
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.value, 3)

    def test_components_without_placement_keep_identity_dedup(self):
        result = aggregate_sum_credits(
            [
                _component("DSBA", "06026xxx", 3),
                _component("DSBA", "06026xxx", 3),
            ]
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.value, 3)

    def test_conflicting_credits_on_same_placement_fail_closed(self):
        result = aggregate_sum_credits(
            [
                _component("DSBA", "06026215", 3, placement_id=589),
                _component("DSBA", "06026215", 5, placement_id=589),
            ]
        )
        self.assertEqual(result.status, "insufficient_evidence")

    def test_conflicting_credits_across_distinct_slots_fail_closed(self):
        result = aggregate_sum_credits(
            [
                _component("DSBA", "06026xxx", 3, placement_id=590),
                _component("DSBA", "06026xxx", 5, placement_id=591),
            ]
        )
        self.assertEqual(result.status, "insufficient_evidence")


class RequiredLoadPublicSeamTests(unittest.TestCase):
    def test_no_coop_y4t1_counts_both_elective_slots(self):
        result = _ask_sum("DSBA ปี 4 เทอม 1 รวมทั้งหมดกี่หน่วยกิต", "no_coop")[
            "result"
        ]
        self.assertEqual(result.status, "answer")
        self.assertEqual(_sum_value(result), 9)

    def test_coop_y4t1_counts_both_free_electives(self):
        result = _ask_sum("DSBA ปี 4 เทอม 1 รวมทั้งหมดกี่หน่วยกิต", "coop")[
            "result"
        ]
        self.assertEqual(result.status, "answer")
        self.assertEqual(_sum_value(result), 12)

    def test_fully_resolved_term_unchanged(self):
        for plan in ("coop", "no_coop"):
            with self.subTest(plan=plan):
                result = _ask_sum(
                    "DSBA ปี 1 เทอม 1 รวมทั้งหมดกี่หน่วยกิต", plan
                )["result"]
                self.assertEqual(result.status, "answer")
                self.assertEqual(_sum_value(result), 18)

    def test_unresolved_slots_noted_in_sum_answer(self):
        result = _ask_sum("DSBA ปี 4 เทอม 1 รวมทั้งหมดกี่หน่วยกิต", "no_coop")[
            "result"
        ]
        self.assertIn("ยังไม่ได้ระบุรายวิชา", result.final_answer)
        self.assertIn("9 หน่วยกิต", result.final_answer)

    def test_resolved_term_answer_has_no_unresolved_notice(self):
        result = _ask_sum("DSBA ปี 1 เทอม 1 รวมทั้งหมดกี่หน่วยกิต", "coop")[
            "result"
        ]
        self.assertNotIn("ยังไม่ได้ระบุรายวิชา", result.final_answer)


class H4FixedCreditWordingTests(unittest.TestCase):
    def _terms(self):
        return [
            {
                "term_index": index,
                "year": (index + 1) // 2,
                "semester": 1 if index % 2 else 2,
                "courses": [],
                "choice_slots": [],
                "required_selection_slots": [],
                "total_known_credits": 0,
            }
            for index in range(1, 8)
        ]

    def test_uniform_choice_slot_gives_exact_total_with_notice(self):
        terms = self._terms()
        terms[0]["choice_slots"] = [
            {
                "minimum_choices": 1,
                "candidates": [
                    {"course_code": "06026259", "credit_units": 6},
                    {"course_code": "06026260", "credit_units": 6},
                ],
            }
        ]
        _, answer, _ = _format_h4(
            {
                "status": "incomplete_evidence",
                "terms": terms,
                "required_selection_slots": [],
                "limitations": [],
            }
        )
        self.assertIn("รวม 6 หน่วยกิต", answer)
        self.assertIn("ยังไม่ได้ระบุรายวิชา", answer)
        self.assertNotIn("รวมหน่วยกิตที่ยืนยันได้", answer)

    def test_non_uniform_choice_slot_stays_partial(self):
        terms = self._terms()
        terms[0]["courses"] = [
            {"course_code": "06026200", "name_th": "แคลคูลัส 1", "credit_units": 3},
        ]
        terms[0]["choice_slots"] = [
            {
                "minimum_choices": 1,
                "candidates": [
                    {"course_code": "06026259", "credit_units": 6},
                    {"course_code": "06026260", "credit_units": 3},
                ],
            }
        ]
        _, answer, _ = _format_h4(
            {
                "status": "incomplete_evidence",
                "terms": terms,
                "required_selection_slots": [],
                "limitations": [],
            }
        )
        self.assertIn("รวมหน่วยกิตที่ยืนยันได้ 3 หน่วยกิต", answer)
        self.assertNotIn("รวม 6 หน่วยกิต", answer)
        self.assertNotIn("รวม 9 หน่วยกิต", answer)


if __name__ == "__main__":
    unittest.main()
