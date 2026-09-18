import unittest
from pathlib import Path
import tempfile
from decimal import Decimal
from unittest.mock import patch

from rag.aggregation import ComponentAggregation
from rag.answer import EMPTY_ANSWER
from rag.evidence_planner import StructuralScope
from rag.grounded_answer import GroundedAnswerResult, GroundedClaim
from scripts.evaluate_gold_questions import (
    _evaluate_question,
    _merge_gold_with_runtime_results,
    _json_safe,
    _load_gold_questions,
    _project_unseen_item,
    _project_typed_result,
    _provenance_correct_for_result,
    _provenance_matches,
    _semantic_provenance_items,
    _grade_answer,
    _semantic_checks,
    _structured_provenance_items,
    _structured_checks,
    _write_output_atomically,
    grade_results,
)


def _result(
    question_id,
    question_type,
    final_answer,
    *,
    structured_result=None,
    semantic_results=None,
    route=None,
    runtime_status=None,
):
    return {
        "id": question_id,
        "type": question_type,
        "question": "คำถามทดสอบ",
        "expected": {},
        "actual_route": route or question_type,
        "route_match": None if question_type == "unknown" else True,
        "structured_result": structured_result,
        "semantic_results": semantic_results,
        "final_answer": final_answer,
        "execution_success": True,
        "runtime_status": runtime_status,
        "error": None,
        "top_k": 10,
        "model_retry_count": 0,
    }


class GoldEvaluationTest(unittest.TestCase):
    def test_unseen_root_object_unwraps_items(self):
        path = Path(__file__).resolve().parents[1] / "ground_truth" / "rag" / "unseen_factual_v1.json"

        projected = _load_gold_questions(path)

        self.assertEqual(len(projected), 30)
        self.assertEqual([item["id"] for item in projected], [f"U{index:02d}" for index in range(1, 31)])

    def test_unseen_scalar_projection_uses_native_predicates(self):
        projected = _project_unseen_item(
            {
                "id": "scalar",
                "difficulty": "easy",
                "question": "วิชา 06046405 ใน AIT มีกี่หน่วยกิต?",
                "ground_truth_status": "complete",
                "canonical_answer": "3 หน่วยกิต",
                "atomic_expected_facts": [
                    "Program: AIT",
                    "Course code: 06046405",
                    "Credits: 3",
                    "Credit structure: 3(3-0-6)",
                ],
                "accepted_answer_variants": ["3 หน่วยกิต"],
                "provenance": [],
            }
        )

        self.assertEqual(projected["type"], "structured")
        self.assertEqual(
            projected["expected"],
            {
                "program": "AIT",
                "course_code": "06046405",
                "total_credits": 3,
                "credits_raw": "3(3-0-6)",
            },
        )

    def test_unseen_course_list_projection_is_unordered(self):
        projected = _project_unseen_item(
            {
                "id": "list",
                "difficulty": "medium",
                "question": "DSBA มีวิชาอะไรบ้าง?",
                "ground_truth_status": "complete",
                "canonical_answer": ["06026206", "06066300", "90644xxx"],
                "atomic_expected_facts": ["These three course entries are scheduled in DSBA."],
                "accepted_answer_variants": [],
                "provenance": [],
            }
        )

        self.assertEqual(projected["expected"]["course_codes"], ["06026206", "06066300", "90644xxx"])

    def test_unseen_multi_fact_projection_composes_native_fields(self):
        projected = _project_unseen_item(
            {
                "id": "hybrid",
                "difficulty": "hard",
                "question": "วิชา 06036127 ของ BIT ต้องผ่านวิชาอะไรก่อนและเรียนช่วงไหน?",
                "ground_truth_status": "complete",
                "canonical_answer": "ผ่าน 06036112 และเรียนปี 4 เทอม 2",
                "atomic_expected_facts": [
                    "Program: BIT",
                    "Target course: 06036127",
                    "Direct prerequisite: 06036112",
                    "06036127 is flexibly placed in year 4, semester 2.",
                    "Description covers database systems.",
                ],
                "accepted_answer_variants": [],
                "provenance": [],
            }
        )

        self.assertEqual(projected["type"], "hybrid")
        self.assertEqual(projected["expected"]["course_code"], "06036127")
        self.assertEqual(projected["expected"]["prerequisites"], [{"course_code": "06036112"}])
        self.assertEqual(projected["expected"]["placements"][0]["flexible_year_semester_raw"], "4/2")

    def test_unseen_alternative_projection_preserves_members_and_count(self):
        projected = _project_unseen_item(
            {
                "id": "alternative",
                "difficulty": "easy",
                "question": "AIT เลือกวิชาอะไรได้บ้างและต้องเลือกกี่วิชา?",
                "ground_truth_status": "complete",
                "canonical_answer": "เลือก 1 จาก 2 วิชา",
                "atomic_expected_facts": [
                    "Program: AIT",
                    "Alternative member: 06046443",
                    "Alternative member: 06046444",
                    "Required selection count: 1",
                ],
                "accepted_answer_variants": [],
                "provenance": [],
            }
        )

        self.assertEqual(projected["expected"]["course_codes"], ["06046443", "06046444"])
        self.assertEqual(projected["expected"]["alternative"], {"minimum_choices": 1, "maximum_choices": 1})

    def test_unseen_provenance_page_is_mapped(self):
        projected = _project_unseen_item(
            {
                "id": "provenance-page",
                "difficulty": "easy",
                "question": "วิชา 06016402 คืออะไร?",
                "ground_truth_status": "complete",
                "canonical_answer": "course",
                "atomic_expected_facts": [],
                "accepted_answer_variants": [],
                "provenance": [{"source": "it_page_328.png", "page": "328; document page 324"}],
            }
        )

        self.assertEqual(projected["expected"]["provenance"], [{"source_document_key": "it_page_328.png", "source_page": 328}])

    def test_unseen_provenance_null_page_is_omitted(self):
        projected = _project_unseen_item(
            {
                "id": "provenance-null-page",
                "difficulty": "easy",
                "question": "AIT คืออะไร?",
                "ground_truth_status": "complete",
                "canonical_answer": "course",
                "atomic_expected_facts": [],
                "accepted_answer_variants": [],
                "provenance": [{"source": "AIT_academic_plan.json", "page": None}],
            }
        )

        self.assertEqual(projected["expected"]["provenance"], [{"source_document_key": "AIT_academic_plan.json"}])
        self.assertNotIn("source_page", projected["expected"]["provenance"][0])

    def test_unseen_valid_empty_uses_existing_typed_empty_path(self):
        projected = _project_unseen_item(
            {
                "id": "valid-empty",
                "difficulty": "easy",
                "question": "BIT แบบไม่สหกิจ ปี 5 เทอม 1 มีข้อมูลไหม",
                "ground_truth_status": "valid_empty",
                "canonical_answer": "ไม่มีรายการ",
                "atomic_expected_facts": [],
                "accepted_answer_variants": [],
                "provenance": [],
            }
        )
        runtime = {
            "runtime_status": "valid_empty",
            "structured_result": {
                "rows": [
                    {
                        "status": "valid_empty",
                        "effective_scope": {
                            "program": "BIT",
                            "plans": ["no_coop"],
                            "years": [5],
                            "semesters": [1],
                        },
                    }
                ]
            },
        }

        status, _, details = _grade_answer(projected, runtime)

        self.assertEqual(projected["type"], "unknown")
        self.assertEqual(status, "PASS")
        self.assertTrue(details["typed_valid_empty"])

    def test_unseen_comparison_items_project_to_native_shapes(self):
        path = Path(__file__).resolve().parents[1] / "ground_truth" / "rag" / "unseen_factual_v1.json"
        projected = {item["id"]: item for item in _load_gold_questions(path)}

        self.assertEqual(projected["U26"]["type"], "structured")
        self.assertEqual(projected["U26"]["expected"]["comparison"]["kind"], "earliest_comparison")
        self.assertEqual(projected["U26"]["expected"]["comparison"]["winner_plan"], "no_coop")
        self.assertEqual(projected["U27"]["type"], "structured")
        self.assertEqual(projected["U27"]["expected"]["comparison"]["kind"], "maximum_with_ties")
        self.assertEqual(projected["U27"]["expected"]["comparison"]["maximum"], 19)
        self.assertEqual(projected["U30"]["type"], "structured")
        self.assertEqual(projected["U30"]["expected"]["comparison"]["kind"], "earliest_comparison")

    def _earliest_gold(self, *, tie=False):
        operands = [
            {"plan": "coop", "year": 3 if tie else 4, "semester": 2},
            {"plan": "no_coop", "year": 3, "semester": 2},
        ]
        expected = {
            "comparison": {
                "kind": "earliest_comparison",
                "operands": operands,
            }
        }
        if tie:
            expected["comparison"]["tie"] = True
        else:
            expected["comparison"]["winner_plan"] = "no_coop"
        return {
            "id": "earliest",
            "type": "structured",
            "question": "แผนไหนเรียนวิชาได้เร็วกว่า",
            "expected": expected,
        }

    def _earliest_result(self, left, right, relation):
        return _result(
            "earliest",
            "structured",
            "no_coop เรียนก่อน",
            structured_result={
                "rows": [
                    {
                        "operation": "compare",
                        "status": "complete",
                        "value": {
                            "status": "complete",
                            "relation": relation,
                            "left": {"partitions": [{"partition": left, "value": [left["year"], left["semester"]]}]},
                            "right": {"partitions": [{"partition": right, "value": [right["year"], right["semester"]]}]},
                        },
                    }
                ]
            },
        )

    def test_earliest_comparison_correct_operands_and_winner_pass(self):
        gold = self._earliest_gold()
        result = self._earliest_result(
            {"plan": "coop", "year": 4, "semester": 2},
            {"plan": "no_coop", "year": 3, "semester": 2},
            "greater",
        )
        status, _, details = _grade_answer(gold, result)
        self.assertEqual(status, "PASS")
        self.assertTrue(details["structured_checks"][0]["evidence"])

    def test_earliest_comparison_wrong_operand_placement_fails(self):
        gold = self._earliest_gold()
        result = self._earliest_result(
            {"plan": "coop", "year": 3, "semester": 1},
            {"plan": "no_coop", "year": 3, "semester": 2},
            "less",
        )
        status, _, _ = _grade_answer(gold, result)
        self.assertEqual(status, "FAIL")

    def test_earliest_comparison_wrong_winner_fails(self):
        gold = self._earliest_gold()
        result = self._earliest_result(
            {"plan": "coop", "year": 3, "semester": 2},
            {"plan": "no_coop", "year": 4, "semester": 2},
            "less",
        )
        status, _, _ = _grade_answer(gold, result)
        self.assertEqual(status, "FAIL")

    def test_earliest_comparison_reversed_inputs_compute_relation(self):
        gold = self._earliest_gold()
        result = self._earliest_result(
            {"plan": "no_coop", "year": 3, "semester": 2},
            {"plan": "coop", "year": 4, "semester": 2},
            "less",
        )
        status, _, _ = _grade_answer(gold, result)
        self.assertEqual(status, "PASS")

    def test_earliest_comparison_tie_is_supported(self):
        gold = self._earliest_gold(tie=True)
        result = self._earliest_result(
            {"plan": "no_coop", "year": 3, "semester": 2},
            {"plan": "coop", "year": 3, "semester": 2},
            "equal",
        )
        status, _, _ = _grade_answer(gold, result)
        self.assertEqual(status, "PASS")

    def _maximum_gold(self):
        return {
            "id": "maximum",
            "type": "structured",
            "question": "AIT เทอมไหนหน่วยกิตเยอะสุด",
            "expected": {
                "comparison": {
                    "kind": "maximum_with_ties",
                    "maximum": 19,
                    "winners": [
                        {"year": 1, "semester": 2},
                        {"year": 2, "semester": 2},
                    ],
                }
            },
        }

    def _maximum_result(self, rows):
        return _result(
            "maximum",
            "structured",
            "สูงสุด 19 หน่วยกิต ในปี 1 เทอม 2 และปี 2 เทอม 2",
            structured_result={"rows": rows},
        )

    def _credit_row(self, year, semester, credits):
        return {
            "operation": "sum_credits",
            "status": "complete",
            "effective_scope": {"years": [year], "semesters": [semester]},
            "value": {"value": credits},
        }

    def test_maximum_with_ties_correct_maximum_and_winners_pass(self):
        rows = [self._credit_row(1, 2, 19), self._credit_row(2, 2, 19)]
        status, _, _ = _grade_answer(self._maximum_gold(), self._maximum_result(rows))
        self.assertEqual(status, "PASS")

    def test_maximum_with_ties_only_one_winner_fails(self):
        rows = [self._credit_row(1, 2, 19)]
        status, _, _ = _grade_answer(self._maximum_gold(), self._maximum_result(rows))
        self.assertEqual(status, "FAIL")

    def test_maximum_with_ties_reversed_order_passes(self):
        rows = [self._credit_row(2, 2, 19), self._credit_row(1, 2, 19)]
        status, _, _ = _grade_answer(self._maximum_gold(), self._maximum_result(rows))
        self.assertEqual(status, "PASS")

    def test_maximum_with_ties_extra_winner_fails(self):
        rows = [
            self._credit_row(1, 2, 19),
            self._credit_row(2, 2, 19),
            self._credit_row(3, 1, 19),
        ]
        status, _, _ = _grade_answer(self._maximum_gold(), self._maximum_result(rows))
        self.assertEqual(status, "FAIL")

    def test_maximum_with_ties_wrong_maximum_fails(self):
        rows = [self._credit_row(1, 2, 18), self._credit_row(2, 2, 18)]
        status, _, _ = _grade_answer(self._maximum_gold(), self._maximum_result(rows))
        self.assertEqual(status, "FAIL")

    def _typed_result(self, operation, value, evidence, *, kind="deterministic_fact"):
        provenance = ({"program": "IT", "source_page": 7},)
        claim = GroundedClaim(
            "typed-claim",
            operation,
            effective_scope=StructuralScope(
                program="IT",
                plans=("no_coop",),
                years=(2,),
                semesters=(2,),
            ),
            kind=kind,
            value=value,
            evidence=evidence,
            provenance=provenance,
        )
        return GroundedAnswerResult(
            "answer",
            "deterministic" if kind == "deterministic_fact" else "grounded_synthesis",
            "typed rendered answer",
            (claim,),
            provenance,
        )

    def test_typed_success_uses_rendered_answer_and_claim_projection(self):
        evidence = ComponentAggregation(
            "sum_credits",
            "complete",
            30,
            ({"course_code": "06016420", "credits_raw": "3(3-0-6)"},),
        )
        typed = self._typed_result("sum_credits", 30, evidence)
        gold = {
            "id": "typed-structured",
            "type": "structured",
            "question": "IT แบบไม่สหกิจ ปี 2 เทอม 2 รวม 30 หน่วยกิต",
            "expected": {
                "total_credits": 30,
                "provenance": [{"program": "IT", "source_page": 7}],
            },
        }

        with patch("scripts.evaluate_gold_questions.route_question", return_value="structured"), patch(
            "scripts.evaluate_gold_questions.ask",
            return_value={"route": None, "result": typed},
        ), patch(
            "scripts.evaluate_gold_questions.answer_question",
            side_effect=AssertionError("typed results must not use legacy answer generation"),
        ):
            result = _evaluate_question(gold, type("NoCallGemini", (), {"retry_count": 0, "begin_question": lambda self: None})(), Path("unused.db"))

        self.assertTrue(result["execution_success"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["runtime_status"], "answer")
        self.assertEqual(result["final_answer"], "typed rendered answer")
        self.assertEqual(result["actual_route"], "structured")
        self.assertTrue(result["route_match"])
        self.assertEqual(result["structured_result"]["rows"][0]["total_credits"], 30)
        self.assertEqual(result["structured_result"]["rows"][0]["plan"], "no_coop")
        self.assertTrue(result["provenance_correct"])

    def test_structured_provenance_inherits_enclosing_plan(self):
        result = {
            "rows": [
                {
                    "effective_scope": {"plans": ["no_coop"]},
                    "provenance": [
                        {"program": "IT", "source_page": 7, "source_filename": "it_page_007.png"}
                    ],
                }
            ]
        }
        actual = _structured_provenance_items(result)
        self.assertEqual(actual[0]["plan"], "no_coop")
        self.assertEqual(actual[0]["source_page"], 7)
        self.assertEqual(actual[0]["source_filename"], "it_page_007.png")

    def test_provenance_explicit_plan_wins_over_enclosing_plan(self):
        result = {
            "rows": [
                {
                    "plan": "no_coop",
                    "provenance": [{"plan": "coop", "source_page": 7}],
                }
            ]
        }
        self.assertEqual(_structured_provenance_items(result)[0]["plan"], "coop")

    def test_provenance_without_any_plan_remains_without_plan(self):
        result = {"rows": [{"provenance": [{"source_page": 7}]}]}
        actual = _structured_provenance_items(result)
        self.assertNotIn("plan", actual[0])

    def test_semantic_provenance_inherits_effective_scope_plan(self):
        chunks = [
            {
                "effective_scope": {"plans": ["coop"]},
                "provenance": [{"source_page": 12, "source_filename": "it_page_012.png"}],
            }
        ]
        actual = _semantic_provenance_items(chunks)
        self.assertEqual(actual[0]["plan"], "coop")
        self.assertEqual(actual[0]["source_page"], 12)
        self.assertEqual(actual[0]["source_filename"], "it_page_012.png")

    def test_plan_aware_provenance_match_uses_structured_scope(self):
        gold = {
            "expected": {
                "provenance": [
                    {"program": "IT", "plan": "no_coop", "source_page": 7}
                ]
            }
        }
        structured = {
            "rows": [
                {
                    "plan": "no_coop",
                    "provenance": [{"program": "IT", "source_page": 7}],
                }
            ]
        }
        self.assertTrue(_provenance_correct_for_result(gold, structured, []))

    def test_validated_ocr_and_image_source_keys_match_by_page_identity(self):
        expected = {
            "program": "DSBA",
            "source_document_key": "dsba_page_317_ocr.txt",
            "source_page": 317,
        }
        actual = {
            "program": "DSBA",
            "source_filename": "dsba_page_317.png",
            "source_page": 317,
            "document_page": 311,
        }
        self.assertTrue(_provenance_matches(expected, actual))
        self.assertEqual(actual["source_page"], 317)
        self.assertEqual(actual["document_page"], 311)

    def test_validated_ocr_json_and_image_source_keys_match_by_page_identity(self):
        expected = {"source_document_key": "it_page_328_ocr.json", "source_page": 328}
        actual = {"source_filename": "it_page_328.png", "source_page": 328}
        self.assertTrue(_provenance_matches(expected, actual))

    def test_validated_source_keys_with_different_pages_do_not_match(self):
        expected = {"source_document_key": "dsba_page_317_ocr.txt", "source_page": 317}
        actual = {"source_filename": "dsba_page_318.png", "source_page": 318}
        self.assertFalse(_provenance_matches(expected, actual))

    def test_validated_source_keys_with_different_programs_do_not_match(self):
        expected = {"source_document_key": "dsba_page_317_ocr.txt", "source_page": 317}
        actual = {"source_filename": "it_page_317.png", "source_page": 317}
        self.assertFalse(_provenance_matches(expected, actual))

    def test_arbitrary_extensions_are_not_canonicalized(self):
        expected = {"source_document_key": "foo.txt"}
        actual = {"source_filename": "foo.png"}
        self.assertFalse(_provenance_matches(expected, actual))

    def test_exact_source_key_match_remains_supported(self):
        provenance = {"source_document_key": "it_page_328.png", "source_page": 328}
        self.assertTrue(_provenance_matches(provenance, dict(provenance)))

    def test_validated_page_range_matches_complete_individual_pages(self):
        expected = {
            "program": "AIT",
            "document_category": "description",
            "source_document_key": "ait_page_292.png–ait_page_297.png",
        }
        actual_pages = [
            {
                "program": "AIT",
                "document_category": "description",
                "source_filename": f"ait_page_{page}.png",
                "source_page": page,
                "document_page": page - 4,
            }
            for page in range(292, 298)
        ]
        gold = {"expected": {"provenance": [expected]}}
        structured = {"rows": [{"provenance": [reference]} for reference in actual_pages]}
        self.assertTrue(_provenance_correct_for_result(gold, structured, []))

    def test_page_range_requires_complete_coverage(self):
        expected = {
            "program": "AIT",
            "source_document_key": "ait_page_292.png–ait_page_297.png",
        }
        actual_pages = [
            {"program": "AIT", "source_filename": f"ait_page_{page}.png", "source_page": page}
            for page in range(292, 297)
        ]
        gold = {"expected": {"provenance": [expected]}}
        structured = {"rows": [{"provenance": [reference]} for reference in actual_pages]}
        self.assertFalse(_provenance_correct_for_result(gold, structured, []))

    def test_page_range_rejects_wrong_physical_page(self):
        expected = {"source_document_key": "ait_page_292.png–ait_page_297.png"}
        actual_pages = [
            {"source_filename": f"ait_page_{page}.png", "source_page": page}
            for page in (292, 293, 294, 295, 296, 298)
        ]
        gold = {"expected": {"provenance": [expected]}}
        structured = {"rows": [{"provenance": [reference]} for reference in actual_pages]}
        self.assertFalse(_provenance_correct_for_result(gold, structured, []))

    def test_page_range_rejects_different_program_identity(self):
        expected = {"source_document_key": "ait_page_292.png–ait_page_297.png"}
        actual_pages = [
            {"source_filename": f"it_page_{page}.png", "source_page": page}
            for page in range(292, 298)
        ]
        gold = {"expected": {"provenance": [expected]}}
        structured = {"rows": [{"provenance": [reference]} for reference in actual_pages]}
        self.assertFalse(_provenance_correct_for_result(gold, structured, []))

    def test_single_page_provenance_behavior_is_unchanged(self):
        expected = {"source_document_key": "ait_page_292.png", "source_page": 292}
        actual = {"source_filename": "ait_page_292.png", "source_page": 292}
        self.assertTrue(_provenance_matches(expected, actual))

    def test_exact_page_range_representation_remains_matching(self):
        expected = {"source_document_key": "ait_page_292.png–ait_page_297.png"}
        actual = {"source_document_key": "ait_page_292.png–ait_page_297.png"}
        self.assertTrue(_provenance_matches(expected, actual))

    def test_page_range_uses_source_page_not_document_page(self):
        expected = {"source_document_key": "ait_page_292.png–ait_page_297.png"}
        actual_pages = [
            {
                "source_filename": f"ait_page_{page}.png",
                "source_page": page,
                "document_page": 1,
            }
            for page in range(292, 298)
        ]
        gold = {"expected": {"provenance": [expected]}}
        structured = {"rows": [{"provenance": [reference]} for reference in actual_pages]}
        self.assertTrue(_provenance_correct_for_result(gold, structured, []))

    def test_typed_semantic_and_hybrid_views_use_the_same_claim_evidence(self):
        cases = (
            ("semantic", "grounded_summary", "semantic-typed"),
            ("hybrid", "deterministic_fact", "hybrid-typed"),
        )
        for route, kind, question_id in cases:
            with self.subTest(route=route):
                if route == "hybrid":
                    evidence = ComponentAggregation(
                        "sum_credits",
                        "complete",
                        7,
                        ({"course_code": "CLOUD", "name_en": "CLOUD TOPIC"},),
                    )
                    typed = self._typed_result("sum_credits", 7, evidence)
                else:
                    typed = self._typed_result(
                        "describe",
                        "PROJECT 1 description",
                        {"text": "PROJECT 1 description"},
                        kind=kind,
                    )
                gold = {
                    "id": question_id,
                    "type": route,
                    "question": "typed evidence question",
                    "expected": {},
                }
                with patch("scripts.evaluate_gold_questions.route_question", return_value=route), patch(
                    "scripts.evaluate_gold_questions.ask",
                    return_value={"route": None, "result": typed},
                ), patch(
                    "scripts.evaluate_gold_questions.answer_question",
                    side_effect=AssertionError("typed results must not use legacy answer generation"),
                ):
                    result = _evaluate_question(
                        gold,
                        type("NoCallGemini", (), {"retry_count": 0, "begin_question": lambda self: None})(),
                        Path("unused.db"),
                    )

                self.assertTrue(result["execution_success"])
                self.assertEqual(result["actual_route"], route)
                self.assertEqual(result["final_answer"], "typed rendered answer")
                self.assertTrue(result["semantic_results"])
                if route == "hybrid":
                    self.assertTrue(result["structured_result"]["rows"])
                    self.assertIn("CLOUD", result["semantic_results"][0]["text"])

    def test_typed_description_claim_is_one_logical_evidence_unit(self):
        typed = self._typed_result(
            "describe",
            "DATA MANAGEMENT DATABASE TECHNOLOGY",
            {"text": "DATA MANAGEMENT DATABASE TECHNOLOGY"},
            kind="grounded_summary",
        )
        _, semantic_results = _project_typed_result(typed)
        details = _semantic_checks(
            "วิชา 06016402 เรียนเกี่ยวกับอะไรบ้าง",
            {
                "description_evidence": ["DATA MANAGEMENT", "DATABASE TECHNOLOGY"],
            },
            semantic_results,
            "DATA MANAGEMENT DATABASE TECHNOLOGY",
        )

        description_checks = [
            check for check in details["checks"] if check["label"].startswith("description_evidence")
        ]
        self.assertEqual([check["label"] for check in description_checks], ["description_evidence"])
        self.assertEqual(details["evidence_count"], 1)
        self.assertEqual(details["evidence_total"], 1)
        self.assertEqual(details["description_found"], ["DATA MANAGEMENT", "DATABASE TECHNOLOGY"])

    def test_semantic_course_list_and_description_checks_do_not_crash(self):
        details = _semantic_checks(
            "AIT มีวิชาอะไรบ้างที่เรียนเกี่ยวกับ computer vision",
            {
                "course_codes": ["06046407", "06046409"],
                "description_evidence": ["COMPUTER VISION"],
            },
            [{"text": "06046407 06046409 COMPUTER VISION"}],
            "06046407 และ 06046409 COMPUTER VISION",
        )

        labels = [check["label"] for check in details["checks"]]
        self.assertIn("course_codes", labels)
        self.assertIn("description_evidence[0]", labels)
        self.assertEqual(details["evidence_count"], 2)
        self.assertEqual(details["evidence_total"], 2)

    def test_typed_placement_fields_are_verified_without_rendered_text(self):
        gold = [
            {
                "id": "typed-placement-fields",
                "type": "structured",
                "question": "วิชาอยู่ปีไหน เทอมไหน",
                "expected": {
                    "placements": [{"plan": "coop", "year": 4, "semester": 2}],
                },
            }
        ]
        result = _result(
            "typed-placement-fields",
            "structured",
            "พบข้อมูลการจัดวางรายวิชา",
            structured_result={
                "rows": [
                    {
                        "operation": "placement",
                        "status": "complete",
                        "effective_scope": {"plans": ["coop"]},
                        "value": [{"plan_key": "coop", "year_number": 4, "semester_number": 2}],
                    }
                ]
            },
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "REVIEW")
        self.assertTrue(graded[0]["evidence_correct"])
        self.assertTrue(graded[0]["structured_checks"][0]["evidence"])

    def test_typed_flexible_placement_uses_year_semester_choices(self):
        gold = [
            {
                "id": "typed-flexible-placement",
                "type": "structured",
                "question": "วิชา 06016481 ของ IT ในแต่ละแผนเรียนช่วงไหนบ้าง",
                "expected": {
                    "placements": [
                        {
                            "plan": "no_coop",
                            "year": None,
                            "semester": None,
                            "flexible_year_semester_raw": "3/1, 3/2, 4/1",
                        }
                    ]
                },
            }
        ]
        result = _result(
            "typed-flexible-placement",
            "structured",
            "ไม่สหกิจเปิด 3/1, 3/2, 4/1",
            structured_result={
                "rows": [
                    {
                        "operation": "placement",
                        "status": "complete",
                        "value": [
                            {
                                "plan_key": "no_coop",
                                "year_number": None,
                                "semester_number": None,
                                "year_semester_choices": [[3, 1], [3, 2], [4, 1]],
                            }
                        ],
                    }
                ]
            },
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertTrue(graded[0]["evidence_correct"])

    def test_missing_typed_placement_values_fail_closed(self):
        details = _structured_checks(
            "วิชาอยู่ปีไหน เทอมไหน",
            {"placements": [{"plan": "coop", "year": 3, "semester": 1}]},
            {"rows": [{"value": [{"plan_key": "coop"}]}]},
            "ปี 3 เทอม 1",
        )

        self.assertFalse(details[0]["evidence"])

    def test_excluded_unknown_fallback_behavior_remains_unchanged(self):
        gold = [
            {
                "id": "excluded-year5",
                "type": "unknown",
                "question": "BIT ปี 5 เทอม 1 มีข้อมูลไหม",
                "expected": EMPTY_ANSWER,
            }
        ]
        result = _result(
            "excluded-year5",
            "unknown",
            "existence: false\nsum_credits: 0",
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "FAIL")

    def test_excluded_structured_review_remains_non_pass(self):
        gold = [
            {
                "id": "excluded-review",
                "type": "structured",
                "question": "วิชา 06036107 อยู่ปีไหน เทอมไหน",
                "expected": {"placement": {"year": 3, "semester": 1}},
            }
        ]
        result = _result(
            "excluded-review",
            "structured",
            "ข้อมูลรายวิชา 06036107",
            structured_result={
                "rows": [
                    {
                        "value": [{"year_number": 3, "semester_number": 1}],
                    }
                ]
            },
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "REVIEW")

    def test_blocked_statuses_normalize_without_legacy_result_keys(self):
        for status in ("clarify_program", "insufficient_evidence"):
            with self.subTest(status=status):
                gold = {
                    "id": f"blocked-{status}",
                    "type": "structured",
                    "question": "blocked question",
                    "expected": {},
                }
                blocked = {"status": status, "action": status}
                with patch("scripts.evaluate_gold_questions.route_question", return_value="structured"), patch(
                    "scripts.evaluate_gold_questions.ask",
                    return_value={"route": None, "result": blocked},
                ):
                    result = _evaluate_question(
                        gold,
                        type("NoCallGemini", (), {"retry_count": 0, "begin_question": lambda self: None})(),
                        Path("unused.db"),
                    )

                self.assertTrue(result["execution_success"])
                self.assertEqual(result["runtime_status"], status)
                self.assertEqual(result["final_answer"], "")
                self.assertIsNone(result["error"])

    def test_no_data_blocked_result_preserves_exact_not_found_scoring(self):
        gold = {
            "id": "blocked-no-data",
            "type": "unknown",
            "question": "missing course",
            "expected": {},
        }
        with patch("scripts.evaluate_gold_questions.route_question", return_value="structured"), patch(
            "scripts.evaluate_gold_questions.ask",
            return_value={"route": None, "result": {"status": "no_data", "action": "no_data"}},
        ):
            result = _evaluate_question(
                gold,
                type("NoCallGemini", (), {"retry_count": 0, "begin_question": lambda self: None})(),
                Path("unused.db"),
            )

        self.assertTrue(result["execution_success"])
        self.assertEqual(result["runtime_status"], "no_data")
        self.assertEqual(result["final_answer"], EMPTY_ANSWER)
        self.assertEqual(result["answer_correctness"], "PASS")

    def test_grade_existing_uses_current_gold_and_preserves_runtime_observations(self):
        gold = [
            {
                "id": "q1",
                "type": "structured",
                "difficulty": "hard",
                "question": "คำถามปัจจุบัน",
                "expected": {"total_credits": 30},
                "required_evidence": ["semester total"],
                "required_provenance": [{"source_page": 12}],
            }
        ]
        raw = [
            {
                "id": "q1",
                "type": "semantic",
                "difficulty": "easy",
                "question": "คำถามเก่า",
                "expected": {"total_credits": 21},
                "required_evidence": ["stale"],
                "required_provenance": [{"source_page": 99}],
                "actual_route": "structured",
                "route_match": False,
                "structured_result": {
                    "columns": ["total_credits"],
                    "rows": [[30]],
                    "provenance": [{"source_page": 12}],
                },
                "semantic_results": None,
                "final_answer": "รวม 30 หน่วยกิต",
                "execution_success": True,
                "error": None,
                "latency_sec": 0.4,
                "model_retry_count": 2,
            }
        ]

        merged = _merge_gold_with_runtime_results(gold, raw)
        graded, _ = grade_results(gold, merged)

        self.assertEqual(merged[0]["expected"], {"total_credits": 30})
        self.assertEqual(merged[0]["type"], "structured")
        self.assertEqual(merged[0]["difficulty"], "hard")
        self.assertEqual(merged[0]["question"], "คำถามปัจจุบัน")
        self.assertEqual(merged[0]["required_evidence"], ["semester total"])
        self.assertEqual(merged[0]["actual_route"], "structured")
        self.assertFalse(merged[0]["route_match"])
        self.assertEqual(merged[0]["structured_result"]["rows"], [[30]])
        self.assertEqual(merged[0]["latency_sec"], 0.4)
        self.assertEqual(merged[0]["model_retry_count"], 2)
        self.assertEqual(graded[0]["answer_correctness"], "PASS")

    def test_grade_existing_rejects_missing_unexpected_and_duplicate_ids(self):
        gold = [
            {
                "id": "q1",
                "type": "unknown",
                "question": "ไม่มีข้อมูลอะไร",
                "expected": EMPTY_ANSWER,
            }
        ]

        with self.assertRaisesRegex(ValueError, "missing Gold ids: q1"):
            _merge_gold_with_runtime_results(gold, [])

        with self.assertRaisesRegex(ValueError, "unexpected runtime ids: extra"):
            _merge_gold_with_runtime_results(gold, [{"id": "extra"}])

        with self.assertRaisesRegex(ValueError, "duplicate id: q1"):
            _merge_gold_with_runtime_results(
                gold,
                [{"id": "q1"}, {"id": "q1"}],
            )

    def test_grade_existing_rejects_duplicate_gold_ids(self):
        gold = [
            {"id": "q1", "type": "unknown", "question": "ก", "expected": EMPTY_ANSWER},
            {"id": "q1", "type": "unknown", "question": "ข", "expected": EMPTY_ANSWER},
        ]

        with self.assertRaisesRegex(ValueError, "Gold Questions contain duplicate id: q1"):
            _merge_gold_with_runtime_results(gold, [{"id": "q1"}])

    def test_output_write_creates_parent_and_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "eval_result.json"

            _write_output_atomically(output_path, {"version": 1})
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{\n  "version": 1\n}\n',
            )

            _write_output_atomically(output_path, {"version": 2})
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{\n  "version": 2\n}\n',
            )
            self.assertEqual(list(output_path.parent.glob("*.tmp")), [])

    def test_json_safe_converts_integral_and_fractional_decimals(self):
        self.assertEqual(_json_safe(Decimal("3")), 3)
        self.assertIs(type(_json_safe(Decimal("3"))), int)
        self.assertEqual(_json_safe(Decimal("3.5")), 3.5)
        self.assertIs(type(_json_safe(Decimal("3.5"))), float)

    def test_json_safe_recurses_through_result_containers_with_decimals(self):
        value = {
            "structured_result": {
                "columns": ["credit_units"],
                "rows": [(Decimal("3"),)],
                "components": [{"credit_units": Decimal("3.5")}],
            }
        }

        self.assertEqual(
            _json_safe(value),
            {
                "structured_result": {
                    "columns": ["credit_units"],
                    "rows": [[3]],
                    "components": [{"credit_units": 3.5}],
                }
            },
        )

    def test_atomic_writer_succeeds_with_decimal_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "eval_result.json"
            payload = {"structured_result": {"credit_units": Decimal("3.5")}}

            _write_output_atomically(output_path, payload)

            self.assertIn('"credit_units": 3.5', output_path.read_text(encoding="utf-8"))

    def test_unknown_requires_exact_fallback_and_non_unknown_fallback_fails(self):
        gold = [
            {
                "id": "unknown",
                "type": "unknown",
                "question": "ข้อมูลอะไร",
                "expected": EMPTY_ANSWER,
            },
            {
                "id": "structured",
                "type": "structured",
                "question": "มีกี่หน่วยกิต",
                "expected": {"total_credits": 3},
            },
        ]
        results = [
            _result("unknown", "unknown", EMPTY_ANSWER, route="semantic"),
            _result(
                "structured",
                "structured",
                EMPTY_ANSWER,
                structured_result={"columns": ["total_credits"], "rows": [[3]]},
            ),
        ]

        graded, summary = grade_results(gold, results)

        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertEqual(graded[1]["answer_correctness"], "FAIL")
        self.assertEqual(summary["unknown_exact_fallback"], {"count": 1, "total": 1})

    def test_typed_valid_empty_supported_year5_scope_does_not_require_fallback(self):
        gold = [
            {
                "id": "typed-year5-empty",
                "type": "unknown",
                "question": "BIT แบบไม่สหกิจ ปี 5 เทอม 1 มีข้อมูลไหม",
                "expected": EMPTY_ANSWER,
            }
        ]
        result = _result(
            "typed-year5-empty",
            "unknown",
            "existence: false\nsum_credits: 0",
            runtime_status="valid_empty",
            structured_result={
                "rows": [
                    {
                        "operation": "existence",
                        "status": "valid_empty",
                        "effective_scope": {
                            "program": "BIT",
                            "plans": ["no_coop"],
                            "years": [5],
                            "semesters": [1],
                        },
                        "value": False,
                        "evidence": [],
                    },
                    {
                        "operation": "sum_credits",
                        "status": "valid_empty",
                        "effective_scope": {
                            "program": "BIT",
                            "plans": ["no_coop"],
                            "years": [5],
                            "semesters": [1],
                        },
                        "value": 0,
                        "evidence": [],
                    },
                ]
            },
        )

        graded, summary = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertTrue(graded[0]["typed_valid_empty"])
        self.assertIsNone(graded[0]["not_found_correct"])
        self.assertEqual(summary["unknown_exact_fallback"], {"count": 0, "total": 1})

    def test_typed_valid_empty_rejects_unsupported_or_incomplete_scopes(self):
        cases = (
            ("unsupported-year", 6, 1, "valid_empty"),
            ("missing-semester", 5, None, "valid_empty"),
            ("insufficient", 5, 1, "insufficient_evidence"),
        )
        for case_id, year, semester, status in cases:
            with self.subTest(case_id=case_id):
                gold = [
                    {
                        "id": case_id,
                        "type": "unknown",
                        "question": "scoped empty question",
                        "expected": EMPTY_ANSWER,
                    }
                ]
                result = _result(
                    case_id,
                    "unknown",
                    "structured result without fallback",
                    runtime_status=status,
                    structured_result={
                        "rows": [
                            {
                                "status": status,
                                "effective_scope": {
                                    "program": "BIT",
                                    "plans": ["no_coop"],
                                    "years": [year],
                                    "semesters": [] if semester is None else [semester],
                                },
                            }
                        ]
                    },
                )

                graded, _ = grade_results(gold, [result])

                self.assertEqual(graded[0]["answer_correctness"], "FAIL")
                self.assertFalse(graded[0].get("typed_valid_empty", False))

    def test_true_unknown_still_requires_exact_not_found_fallback(self):
        gold = [
            {
                "id": "unknown-course",
                "type": "unknown",
                "question": "unknown course",
                "expected": EMPTY_ANSWER,
            }
        ]
        result = _result(
            "unknown-course",
            "unknown",
            EMPTY_ANSWER,
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertTrue(graded[0]["not_found_correct"])

    def test_fallback_like_denial_fails_for_non_unknown(self):
        gold = [
            {
                "id": "structured",
                "type": "structured",
                "question": "มีกี่หน่วยกิต",
                "expected": {"total_credits": 3},
            }
        ]
        results = [
            _result(
                "structured",
                "structured",
                "ไม่พบข้อมูลในหลักสูตร",
                structured_result={"columns": ["total_credits"], "rows": [[3]]},
            )
        ]

        graded, _ = grade_results(gold, results)

        self.assertEqual(graded[0]["answer_correctness"], "FAIL")

    def test_structured_facts_require_evidence_and_final_answer(self):
        gold = [
            {
                "id": "structured",
                "type": "structured",
                "question": "IT ไม่สหกิจ ปี 2 เทอม 2 มีกี่หน่วยกิต",
                "expected": {
                    "program": "IT",
                    "plan": "no_coop",
                    "year": 2,
                    "semester": 2,
                    "total_credits": 30,
                },
            }
        ]
        result = _result(
            "structured",
            "structured",
            "IT แบบไม่สหกิจ ปี 2 เทอม 2 รวม 30 หน่วยกิต",
            structured_result={
                "sql": "WHERE program = 'IT' AND plan_key = 'no_coop' AND year = 2 AND semester = 2",
                "columns": ["total_credits"],
                "rows": [[30]],
            },
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertEqual(graded[0]["structured_fact_correctness"], "PASS")

    def test_semantic_partial_when_synthesis_omits_supported_topic(self):
        gold = [
            {
                "id": "semantic",
                "type": "semantic",
                "question": "วิชานี้สอนอะไร",
                "expected": {
                    "program": "IT",
                    "plan": "no_coop",
                    "course_code": "06016402",
                    "name_en": "INFORMATION TECHNOLOGY FUNDAMENTALS",
                    "description_evidence": ["DATA MANAGEMENT", "DATABASE TECHNOLOGY"],
                    "provenance": [
                        {
                            "document_category": "description",
                            "source_document_key": "it_page_328.png",
                            "source_page": 328,
                        }
                    ],
                },
            }
        ]
        result = _result(
            "semantic",
            "semantic",
            "06016402 INFORMATION TECHNOLOGY FUNDAMENTALS: DATA MANAGEMENT",
            semantic_results=[
                {
                    "text": "06016402 INFORMATION TECHNOLOGY FUNDAMENTALS IT no_coop DATA MANAGEMENT DATABASE TECHNOLOGY",
                    "provenance": [
                        {
                            "document_category": "description",
                            "source_document_key": "it_page_328.png",
                            "source_page": 328,
                        }
                    ],
                }
            ],
        )

        graded, summary = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "PARTIAL")
        self.assertEqual(graded[0]["semantic_evidence_coverage"]["matched"], 2)
        self.assertEqual(graded[0]["semantic_evidence_coverage"]["expected"], 2)
        self.assertEqual(summary["semantic_evidence_coverage"]["ratio"], 1.0)

    def test_hybrid_requires_structured_and_semantic_sides(self):
        gold = [
            {
                "id": "hybrid",
                "type": "hybrid",
                "question": "เรียนช่วงไหนและเนื้อหาอะไร",
                "expected": {
                    "program": "IT",
                    "plan": "coop",
                    "course_code": "06016406",
                    "placement": {"plan": "coop", "year": 2, "semester": 1},
                    "description_evidence": ["PROJECT WORK"],
                    "provenance": [
                        {
                            "document_category": "description",
                            "source_document_key": "it_page_330.png",
                            "source_page": 330,
                        }
                    ],
                },
            }
        ]
        result = _result(
            "hybrid",
            "hybrid",
            "IT แบบสหกิจ วิชา 06016406 เรียนปี 2 เทอม 1 และมี PROJECT WORK",
            structured_result={
                "columns": ["program", "plan_key", "course_code", "year", "semester"],
                "rows": [["IT", "coop", "06016406", 2, 1]],
            },
            semantic_results=[
                {
                    "text": "06016406 IT coop PROJECT WORK",
                    "provenance": [
                        {
                            "document_category": "description",
                            "source_document_key": "it_page_330.png",
                            "source_page": 330,
                        }
                    ],
                }
            ],
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "PASS")

    def test_unasked_expected_fields_are_diagnostic_only(self):
        gold = [
            {
                "id": "timing",
                "type": "structured",
                "question": "วิชา 06016465 ของ IT ในแต่ละแผนเรียนช่วงไหนบ้าง?",
                "expected": {
                    "program": "IT",
                    "course_code": "06016465",
                    "placements": [
                        {
                            "plan": "coop",
                            "year": None,
                            "semester": None,
                            "flexible_year_semester_raw": "4/1",
                            "credits_raw": "3(3-0-6)",
                        }
                    ],
                },
            },
            {
                "id": "credits",
                "type": "structured",
                "question": "IT แบบไม่สหกิจ ปี 4 เทอม 2 มีหน่วยกิตรวมเท่าไร?",
                "expected": {
                    "program": "IT",
                    "plan": "no_coop",
                    "year": 4,
                    "semester": 2,
                    "total_credits": 9,
                    "course_codes": ["06016407", "9064xxxx", "xxxxxxxx"],
                },
            },
            {
                "id": "prerequisite",
                "type": "structured",
                "question": "ก่อนลงวิชา 06016420 ใน IT แบบไม่สหกิจ ต้องผ่านวิชาใด?",
                "expected": {
                    "program": "IT",
                    "plan": "no_coop",
                    "course_code": "06016420",
                    "prerequisites": [
                        {
                            "course_code": "06016413",
                            "requirement_type": "required",
                            "raw_text": "06016413",
                        }
                    ],
                },
            },
        ]
        results = [
            _result(
                "timing",
                "structured",
                "coop เปิด 4/1",
                structured_result={
                    "columns": ["plan_key", "year", "semester", "flexible_year_semester_raw"],
                    "rows": [["coop", None, None, "4/1"]],
                },
            ),
            _result(
                "credits",
                "structured",
                "IT แบบไม่สหกิจ ปี 4 เทอม 2 รวม 9 หน่วยกิต",
                structured_result={
                    "sql": "WHERE program = 'IT' AND plan_key = 'no_coop' AND year = 4 AND semester = 2",
                    "columns": ["total_credits"],
                    "rows": [[9]],
                },
            ),
            _result(
                "prerequisite",
                "structured",
                "ต้องผ่านวิชา 06016413 ก่อน",
                structured_result={
                    "sql": "WHERE program = 'IT' AND plan_key = 'no_coop' AND course_code = '06016420'",
                    "columns": ["course_code"],
                    "rows": [["06016413"]],
                },
            ),
        ]

        graded, _ = grade_results(gold, results)

        self.assertEqual([item["answer_correctness"] for item in graded], ["PASS"] * 3)

    def test_comparative_placement_aliases_match_each_plan_exactly(self):
        gold = [
            {
                "id": "comparative-placement",
                "type": "structured",
                "question": "วิชา 06016481 แบบสหกิจและไม่สหกิจอยู่ปีไหน เทอมไหน และเปิดช่วงใดบ้าง",
                "expected": {
                    "program": "IT",
                    "course_code": "06016481",
                    "placements": [
                        {"plan": "coop", "year": 3, "semester": 2},
                        {
                            "plan": "no_coop",
                            "year": None,
                            "semester": None,
                            "flexible_year_semester_raw": "3/1, 3/2, 4/1",
                        },
                    ],
                },
            }
        ]
        results = [
            _result(
                "comparative-placement",
                "structured",
                "สหกิจปี 3 เทอม 2 และไม่สหกิจเปิด 3/1, 3/2, 4/1",
                structured_result={
                    "columns": [
                        "coop_year",
                        "coop_semester",
                        "no_coop_placement_raw",
                    ],
                    "rows": [[3, 2, "3/1, 3/2, 4/1"]],
                },
            )
        ]

        graded, _ = grade_results(gold, results)

        self.assertEqual(graded[0]["answer_correctness"], "PASS")

    def test_prerequisite_code_alias_matches_expected_prerequisite(self):
        gold = [
            {
                "id": "prerequisite-alias",
                "type": "structured",
                "question": "ก่อนลงวิชา 06016420 ต้องผ่านวิชาใด",
                "expected": {
                    "course_code": "06016420",
                    "prerequisites": [{"course_code": "06016413"}],
                },
            }
        ]
        results = [
            _result(
                "prerequisite-alias",
                "structured",
                "ต้องผ่านวิชา 06016413 ก่อน",
                structured_result={
                    "columns": ["course_code", "prerequisite_code"],
                    "rows": [["06016420", "06016413"]],
                },
            )
        ]

        graded, _ = grade_results(gold, results)

        self.assertEqual(graded[0]["answer_correctness"], "PASS")

    def test_credits_column_satisfies_expected_credits_raw(self):
        gold = [
            {
                "id": "credits-raw-alias",
                "type": "structured",
                "question": "วิชานี้มีกี่หน่วยกิต",
                "expected": {"credits_raw": "3(3-0-6)"},
            }
        ]
        results = [
            _result(
                "credits-raw-alias",
                "structured",
                "วิชานี้ 3(3-0-6) หน่วยกิต",
                structured_result={
                    "columns": ["credits"],
                    "rows": [["3(3-0-6)"]],
                },
            )
        ]

        graded, _ = grade_results(gold, results)

        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertTrue(graded[0]["evidence_correct"])

    def test_semantic_evidence_can_satisfy_prerequisite_for_structured_gold(self):
        gold = [
            {
                "id": "semantic-prerequisite",
                "type": "structured",
                "question": "ก่อนลงวิชา 06016420 ต้องผ่านวิชาใด",
                "expected": {
                    "prerequisites": [{"course_code": "06016413"}],
                },
                "required_provenance": [
                    {
                        "document_category": "plan",
                        "source_document_key": "it_page_035.png",
                        "source_page": 35,
                    }
                ],
            }
        ]
        results = [
            _result(
                "semantic-prerequisite",
                "structured",
                "ต้องผ่านวิชา 06016413 ก่อน",
                route="semantic",
                semantic_results=[
                    {
                        "text": "วิชาที่ต้องเรียนก่อน 06016413",
                        "provenance": [
                            {
                                "document_category": "plan",
                                "source_document_key": "it_page_035.png",
                                "source_page": 35,
                            }
                        ],
                    }
                ],
            )
        ]

        graded, _ = grade_results(gold, results)

        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertTrue(graded[0]["evidence_correct"])
        self.assertTrue(graded[0]["provenance_correct"])

    def test_nested_multifield_structured_facts_are_shape_independent(self):
        gold = [
            {
                "id": "nested-planning-facts",
                "type": "structured",
                "question": "ปี 2 เทอม 2 มีกี่หน่วยกิต และก่อนลงวิชานี้ต้องผ่านอะไร",
                "expected": {
                    "total_credits": 30,
                    "prerequisites": [{"course_code": "06016413"}],
                },
            }
        ]
        results = [
            _result(
                "nested-planning-facts",
                "structured",
                "ปี 2 เทอม 2 รวม 30 หน่วยกิต และต้องผ่าน 06016413 ก่อน",
                structured_result={
                    "rows": [
                        {
                            "facts": {
                                "total_credits": 30,
                                "prerequisite": {"course_code": "06016413"},
                            }
                        }
                    ]
                },
            )
        ]

        graded, _ = grade_results(gold, results)

        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertTrue(graded[0]["evidence_correct"])

    def test_credit_only_question_does_not_require_scope_year_in_answer(self):
        gold = [
            {
                "id": "credit-only",
                "type": "structured",
                "question": "IT ไม่สหกิจ ปี 2 เทอม 2 ลงทะเบียนรวมกี่หน่วยกิต",
                "expected": {"year": 2, "semester": 2, "total_credits": 30},
            }
        ]
        results = [
            _result(
                "credit-only",
                "structured",
                "รวม 30 หน่วยกิต",
                structured_result={"columns": ["year", "semester", "total_credits"], "rows": [[2, 2, 30]]},
            )
        ]

        graded, _ = grade_results(gold, results)

        self.assertEqual(graded[0]["answer_correctness"], "PASS")

    def test_semantic_thai_paraphrase_is_pass_when_evidence_is_complete(self):
        gold = [
            {
                "id": "thai-paraphrase",
                "type": "semantic",
                "question": "วิชานี้สอนอะไร",
                "expected": {"description_evidence": ["DATA MANAGEMENT", "DATABASE TECHNOLOGY"]},
            }
        ]
        results = [
            _result(
                "thai-paraphrase",
                "semantic",
                "เนื้อหาเกี่ยวกับการจัดการข้อมูลและเทคโนโลยีฐานข้อมูล",
                semantic_results=[
                    {
                        "text": (
                            "IT no_coop 06016402 INFORMATION TECHNOLOGY FUNDAMENTALS "
                            "DATA MANAGEMENT DATABASE TECHNOLOGY เนื้อหาเกี่ยวกับการจัดการข้อมูลและเทคโนโลยีฐานข้อมูล"
                        ),
                        "provenance": [],
                    }
                ],
            )
        ]

        graded, _ = grade_results(gold, results)

        self.assertEqual(graded[0]["answer_correctness"], "PASS")

    def test_unasked_credit_representation_does_not_fail_answer_correctness(self):
        gold = [
            {
                "id": "unasked-credit-representation",
                "type": "structured",
                "question": "วิชา 06016420 ชื่ออะไรและมีกี่หน่วยกิต",
                "expected": {
                    "course_code": "06016420",
                    "name_en": "INFRASTRUCTURE SYSTEMS AND SERVICES",
                    "credits_raw": "3(2-2-5)",
                },
            }
        ]
        result = _result(
            "unasked-credit-representation",
            "structured",
            "INFRASTRUCTURE SYSTEMS AND SERVICES มี 3 หน่วยกิต",
            structured_result={
                "columns": ["course_code", "name_en", "credit_units"],
                "rows": [["06016420", "INFRASTRUCTURE SYSTEMS AND SERVICES", 3]],
            },
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "PASS")

    def test_provenance_remains_independently_strict(self):
        gold = [
            {
                "id": "missing-provenance",
                "type": "structured",
                "question": "วิชา 06016420 ชื่ออะไรและมีกี่หน่วยกิต",
                "expected": {
                    "course_code": "06016420",
                    "name_en": "INFRASTRUCTURE SYSTEMS AND SERVICES",
                    "credits_raw": "3(2-2-5)",
                },
                "required_provenance": [
                    {
                        "document_category": "description",
                        "source_document_key": "it_page_338.png",
                        "source_page": 338,
                    }
                ],
            }
        ]
        result = _result(
            "missing-provenance",
            "structured",
            "INFRASTRUCTURE SYSTEMS AND SERVICES มี 3 หน่วยกิต",
            structured_result={
                "columns": ["course_code", "name_en", "credit_units"],
                "rows":[["06016420", "INFRASTRUCTURE SYSTEMS AND SERVICES", 3]],
            },
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertTrue(graded[0]["evidence_correct"])
        self.assertFalse(graded[0]["provenance_correct"])

    def test_incomplete_semantic_summary_remains_non_pass(self):
        gold = [
            {
                "id": "incomplete-semantic-summary",
                "type": "semantic",
                "question": "วิชานี้เรียนเกี่ยวกับอะไรบ้าง",
                "expected": {
                    "course_code": "06016421",
                    "description_evidence": [
                        "RESOURCE VIRTUALIZATION",
                        "SOFTWARE DEFINED INFRASTRUCTURE",
                        "INFRASTRUCTURE SECURITY",
                    ],
                },
            }
        ]
        result = _result(
            "incomplete-semantic-summary",
            "semantic",
            "เนื้อหาเกี่ยวกับความมั่นคงปลอดภัยในโครงสร้างพื้นฐาน",
            semantic_results=[
                {
                    "text": (
                        "06016421 RESOURCE VIRTUALIZATION SOFTWARE DEFINED INFRASTRUCTURE "
                        "INFRASTRUCTURE SECURITY การจัดการความมั่นคงปลอดภัยในโครงสร้างพื้นฐาน"
                    ),
                    "provenance": [],
                }
            ],
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "REVIEW")

    def test_missing_semantic_evidence_remains_failure(self):
        gold = [
            {
                "id": "missing-semantic-evidence",
                "type": "semantic",
                "question": "วิชานี้เรียนเกี่ยวกับอะไรบ้าง",
                "expected": {
                    "description_evidence": ["TOPIC A", "TOPIC B"],
                },
            }
        ]
        result = _result(
            "missing-semantic-evidence",
            "semantic",
            "มีเนื้อหาเกี่ยวกับ TOPIC A",
            semantic_results=[{"text": "TOPIC A", "provenance": []}],
        )

        graded, _ = grade_results(gold, [result])

        self.assertEqual(graded[0]["answer_correctness"], "FAIL")

    def test_result_contract_tracks_evidence_provenance_and_strictness(self):
        gold = [
            {
                "id": "contract",
                "type": "structured",
                "difficulty": "easy",
                "question": "มีกี่หน่วยกิต",
                "expected": {"total_credits": 3},
                "required_provenance": [
                    {
                        "document_category": "plan",
                        "source_document_key": "it_page_035.png",
                        "source_page": 35,
                    }
                ],
            }
        ]
        result = _result(
            "contract",
            "structured",
            "รวม 3 หน่วยกิต",
            structured_result={
                "columns": ["total_credits", "provenance"],
                "rows": [
                    [
                        3,
                        [
                            {
                                "document_category": "plan",
                                "source_document_key": "it_page_035.png",
                                "source_page": 35,
                            }
                        ],
                    ]
                ],
            },
        )
        result["latency_sec"] = 0.25

        graded, summary = grade_results(gold, [result])

        self.assertEqual(graded[0]["difficulty"], "easy")
        self.assertEqual(graded[0]["answer_correctness"], "PASS")
        self.assertTrue(graded[0]["evidence_correct"])
        self.assertTrue(graded[0]["provenance_correct"])
        self.assertIsNone(graded[0]["not_found_correct"])
        self.assertIsNone(graded[0]["earliest_failure_stage"])
        self.assertEqual(summary["strict_correct"], {"count": 1, "rate": 1.0})
        self.assertEqual(summary["by_difficulty"]["easy"]["total"], 1)

    def test_summary_reports_rates_latency_and_failure_stage(self):
        gold = [
            {
                "id": "good",
                "type": "structured",
                "difficulty": "medium",
                "question": "มีกี่หน่วยกิต",
                "expected": {"total_credits": 3},
            },
            {
                "id": "bad",
                "type": "structured",
                "difficulty": "hard",
                "question": "มีกี่หน่วยกิต",
                "expected": {"total_credits": 3},
            },
        ]
        good = _result(
            "good",
            "structured",
            "รวม 3 หน่วยกิต",
            structured_result={"columns": ["total_credits"], "rows": [[3]]},
        )
        good["latency_sec"] = 0.1
        bad = _result("bad", "structured", "เกิดข้อผิดพลาด")
        bad["execution_success"] = False
        bad["error"] = "RuntimeError: failed"
        bad["latency_sec"] = 0.3

        _, summary = grade_results(gold, [good, bad])

        self.assertEqual(summary["execution_success"], {"count": 1, "total": 2, "rate": 0.5})
        self.assertEqual(summary["evidence_correct"]["rate"], 1.0)
        self.assertEqual(summary["latency_avg_sec"], 0.2)
        self.assertEqual(summary["latency_p95_sec"], 0.3)
        self.assertIn("nearest-rank", summary["latency_p95_method"])
        self.assertEqual(summary["failure_ids_by_earliest_stage"], {"execution": ["bad"]})
        self.assertEqual(summary["error_count"], 1)

    def test_synthesis_fallback_retry_is_counted_in_model_retry_count(self):
        class FakeGemini:
            retry_count = 0

            def __init__(self):
                self.calls = []

            def begin_question(self):
                self.calls.clear()

            def __call__(self, prompt):
                self.calls.append(prompt)
                return EMPTY_ANSWER if len(self.calls) == 1 else "PROJECT 1 เนื้อหาตามหลักฐาน"

        gold = {
            "id": "semantic-retry",
            "type": "semantic",
            "question": "วิชา PROJECT 1 เรียนเกี่ยวกับอะไร",
            "expected": {"description_evidence": ["PROJECT 1"]},
        }
        fake = FakeGemini()
        with patch(
            "scripts.evaluate_gold_questions.ask",
            return_value={"route": "semantic", "result": [{"text": "PROJECT 1"}]},
        ):
            result = _evaluate_question(gold, fake, Path("unused.db"))

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(result["model_retry_count"], 1)


if __name__ == "__main__":
    unittest.main()
