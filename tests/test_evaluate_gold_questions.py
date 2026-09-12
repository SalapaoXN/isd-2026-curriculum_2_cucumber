import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from rag.answer import EMPTY_ANSWER
from scripts.evaluate_gold_questions import (
    _evaluate_question,
    _merge_gold_with_runtime_results,
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
        "error": None,
        "top_k": 10,
        "model_retry_count": 0,
    }


class GoldEvaluationTest(unittest.TestCase):
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
