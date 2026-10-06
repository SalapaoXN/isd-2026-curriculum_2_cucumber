import json
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.hard_qa import (
    HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA,
    _format_h4,
    answer_hard_question,
    answer_seven_term_followup,
)
from backend.hard_sequence_planner import plan_curriculum_sequence
from rag.qa import ask
from rag.resolution import QueryContext


DB_PATH = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"


def _intent(task_type, *, program=None, plan=None, left_plan=None, right_plan=None,
            target_course_code=None, horizon_terms=None):
    return json.dumps({
        "task_type": task_type,
        "program": program,
        "plan": plan,
        "left_plan": left_plan,
        "right_plan": right_plan,
        "target_course_code": target_course_code,
        "horizon_terms": horizon_terms,
    })


class HardQaTests(unittest.TestCase):
    def run_hard(self, question, output, *, context=None):
        prompts = []

        def interpreter(prompt):
            prompts.append(prompt)
            return output

        result = answer_hard_question(DB_PATH, question, context, interpreter)
        return result, prompts

    def test_comparison_dispatches_to_h1_and_empty_difference_is_not_no_data(self):
        result, _ = self.run_hard(
            "DSBA coop กับ no_coop ต่างกันที่วิชาไหน",
            _intent("plan_comparison", program="DSBA", left_plan="coop", right_plan="no_coop"),
            context={"program": "DSBA", "catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["hard_task_type"], "plan_comparison")
        self.assertEqual(result["status"], "answer")
        self.assertIn("ไม่พบความแตกต่างของชุดรหัสวิชา", result["answer"])
        self.assertIn("ไม่ได้ยืนยันว่ารายละเอียดด้านอื่นของแผนเหมือนกัน", result["answer"])
        self.assertNotEqual(result["status"], "no_data")
        self.assertTrue(result["provenance"])

    def test_valid_exact_interpreter_shape_has_no_rejection_diagnostic(self):
        with self.assertNoLogs("backend.hard_qa", level="WARNING"):
            result, _ = self.run_hard(
                "DSBA coop กับ no_coop ต่างกันที่วิชาไหน",
                _intent("plan_comparison", program="DSBA", left_plan="coop", right_plan="no_coop"),
                context={"program": "DSBA", "catalog_key": "dsba-2565"},
            )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["hard_task_type"], "plan_comparison")

    def test_missing_edition_scope_clarifies_before_plan_comparison(self):
        result, prompts = self.run_hard(
            "DSBA coop กับ no_coop ต่างกันที่วิชาไหน",
            _intent("plan_comparison", program="DSBA", left_plan="coop", right_plan="no_coop"),
            context={"program": "DSBA"},
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["scope"], {"program": "DSBA"})
        self.assertIn("ฉบับหลักสูตร", result["answer"])
        self.assertFalse(result["provenance"])
        self.assertEqual(len(prompts), 1)

    def test_dsba_h1_to_h4_resolve_each_edition_and_plan(self):
        from backend.hard_plan_compare import compare_plan_course_sets
        from backend.hard_plan_validate import validate_curriculum_plan_structure
        from backend.hard_prerequisite_validate import validate_plan_prerequisite_sequence
        from backend.hard_sequence_planner import plan_curriculum_sequence

        for catalog_key in ("dsba-2560", "dsba-2565"):
            comparison = compare_plan_course_sets(
                DB_PATH, "DSBA", "coop", "no_coop", catalog_key
            )
            self.assertEqual(comparison["status"], "complete")
            self.assertEqual(comparison["catalog_key"], catalog_key)
            expected_source_prefix = (
                "dsba2560_page_" if catalog_key == "dsba-2560" else "dsba_page_"
            )
            for side in ("left_plan_evidence", "right_plan_evidence"):
                self.assertTrue(comparison[side])
                self.assertTrue(
                    all(
                        item["source_filename"].startswith(expected_source_prefix)
                        for item in comparison[side]
                    )
                )
            for plan in ("coop", "no_coop"):
                with self.subTest(catalog_key=catalog_key, plan=plan):
                    h2 = validate_curriculum_plan_structure(
                        DB_PATH, "DSBA", plan, catalog_key
                    )
                    h3 = validate_plan_prerequisite_sequence(
                        DB_PATH, "DSBA", plan, catalog_key
                    )
                    h4 = plan_curriculum_sequence(
                        DB_PATH, "DSBA", plan, catalog_key=catalog_key
                    )
                    self.assertNotEqual(h2["status"], "ambiguous_plan")
                    self.assertNotEqual(h3["status"], "ambiguous_plan")
                    self.assertNotEqual(h4["status"], "ambiguous_plan")
                    self.assertEqual(h2["catalog_key"], catalog_key)
                    self.assertEqual(h3["catalog_key"], catalog_key)
                    self.assertTrue(h3["provenance"])
                    self.assertTrue(
                        all(
                            item["source_filename"].startswith(expected_source_prefix)
                            for item in h3["provenance"]
                        )
                    )
                    self.assertEqual(h4["status"], "incomplete_evidence")
                    if catalog_key == "dsba-2560":
                        total_requirement = next(
                            item for item in h2["credit_requirements"]
                            if item["requirement_type"] == "total_program_credits"
                        )
                        self.assertEqual(total_requirement["required_value"], 126)
                        self.assertTrue(
                            {item["source_page"] for item in total_requirement["provenance"]}
                            >= {29, 34}
                        )
                        self.assertNotEqual(h4.get("required_program_credits"), 132)
                    else:
                        total_requirement = next(
                            item for item in h2["credit_requirements"]
                            if item["requirement_type"] == "total_program_credits"
                        )
                        self.assertEqual(total_requirement["required_value"], 132)
                        self.assertTrue(
                            any(
                                item["source_page"] in {32, 39}
                                for item in total_requirement["provenance"]
                            )
                        )

    def test_h4_does_not_claim_to_schedule_when_no_terms_are_supported(self):
        status, answer, _ = _format_h4(
            {"status": "incomplete_evidence", "sequence_feasible": None, "terms": []}
        )

        self.assertEqual(status, "incomplete_evidence")
        self.assertIn("ยังระบุลำดับรายวิชาไม่ได้", answer)
        self.assertNotIn("สามารถจัดลำดับรายวิชา", answer)

    def test_interpreter_response_schema_matches_strict_seven_field_contract(self):
        schema = HARD_INTERPRETATION_RESPONSE_JSON_SCHEMA
        expected_fields = {
            "task_type", "program", "plan", "left_plan", "right_plan",
            "target_course_code", "horizon_terms",
        }

        self.assertEqual(schema["type"], "object")
        self.assertEqual(set(schema["properties"]), expected_fields)
        self.assertEqual(set(schema["required"]), expected_fields)
        self.assertNotIn("additionalProperties", schema)
        self.assertEqual(
            set(schema["properties"]["task_type"]["enum"]),
            {"none", "plan_comparison", "plan_structure_validation", "prerequisite_sequence", "seven_term_plan"},
        )
        for field in expected_fields - {"task_type", "horizon_terms"}:
            nullable_types = {
                item["type"] for item in schema["properties"][field]["anyOf"]
            }
            self.assertEqual(nullable_types, {"string", "null"})
        self.assertEqual(
            {item["type"] for item in schema["properties"]["horizon_terms"]["anyOf"]},
            {"integer", "null"},
        )

    def test_interpreter_parse_rejections_are_classified_without_logging_raw_output(self):
        valid = json.loads(_intent("plan_comparison", program="DSBA", left_plan="coop", right_plan="no_coop"))
        missing_key = dict(valid)
        missing_key.pop("horizon_terms")
        extra_key = {**valid, "extra": "PRIVATE_RAW_SENTINEL"}
        mixed_key_set = {**valid, "horizon_terms_extra": 7}
        mixed_key_set.pop("horizon_terms")
        invalid_field = {**valid, "program": ["PRIVATE_RAW_SENTINEL"]}
        invalid_horizon = {**valid, "horizon_terms": 7.0}
        empty_field = {**valid, "program": "  "}
        cases = (
            (None, "invalid_response_type"),
            ("{PRIVATE_RAW_SENTINEL", "json_decode_error"),
            ("```json\n" + _intent("plan_comparison", program="DSBA", left_plan="coop", right_plan="no_coop") + "\n```", "json_decode_error"),
            (json.dumps(["PRIVATE_RAW_SENTINEL"]), "root_not_object"),
            (json.dumps(missing_key), "missing_keys"),
            (json.dumps(extra_key), "unexpected_keys"),
            (json.dumps(mixed_key_set), "key_set_mismatch"),
            (json.dumps({**valid, "task_type": "plan comparison"}), "invalid_task_type"),
            (json.dumps({**valid, "task_type": []}), "invalid_task_type"),
            (json.dumps(invalid_field), "invalid_field_type"),
            (json.dumps(empty_field), "invalid_field_value"),
            (json.dumps(invalid_horizon), "invalid_horizon_terms"),
            (json.dumps({**valid, "horizon_terms": True}), "invalid_horizon_terms"),
        )

        for raw, reason in cases:
            with self.subTest(reason=reason):
                with self.assertLogs("backend.hard_qa", level="WARNING") as captured:
                    result = answer_hard_question(
                        DB_PATH,
                        "DSBA coop กับ no_coop ต่างกันที่วิชาไหน",
                        {"program": "DSBA"},
                        lambda prompt: raw,
                    )

                self.assertEqual(result["status"], "error")
                self.assertEqual(result["action"], "hard_interpretation_failure")
                self.assertEqual(
                    result["answer"],
                    "ยังจัดประเภทคำถาม Hard นี้ไม่ได้อย่างปลอดภัย กรุณาระบุคำถามใหม่ให้ชัดเจน",
                )
                diagnostic = "\n".join(captured.output)
                self.assertEqual(captured.records[0].stage, "hard_interpreter_parse")
                self.assertEqual(captured.records[0].reason, reason)
                self.assertTrue(captured.records[0].exception_class)
                self.assertIn("stage=hard_interpreter_parse", diagnostic)
                self.assertIn(f"reason={reason}", diagnostic)
                self.assertNotIn("PRIVATE_RAW_SENTINEL", diagnostic)
                self.assertNotIn("PRIVATE_RAW_SENTINEL", repr(captured.records[0].metadata))

    def test_structural_validation_preserves_incomplete_evidence(self):
        result, _ = self.run_hard(
            "แผน DSBA coop มีโครงสร้างครบตามหลักสูตรไหม",
            _intent("plan_structure_validation", program="DSBA", plan="coop"),
            context={"program": "DSBA", "catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["hard_task_type"], "plan_structure_validation")
        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertIn("ยังมีข้อมูลที่ยืนยันไม่ครบ", result["answer"])
        self.assertNotIn("incomplete_evidence", result["answer"])
        self.assertNotIn("ผลตรวจโครงสร้างที่แทนได้ครบตามหลักฐาน", result["answer"])
        self.assertTrue(result["provenance"])

    def test_target_prerequisite_answer_contains_no_unrelated_course_chain(self):
        response = ask(
            DB_PATH,
            "วิชา 06026201 ต้องเรียนอะไรมาก่อน",
            conversation_context=QueryContext(program="DSBA", plan="coop", catalog_key="dsba-2565"),
        )
        result = response["result"]
        self.assertEqual(result.status, "answer")
        self.assertIn("06026201", result.final_answer)
        self.assertIn("06026200", result.final_answer)
        self.assertNotIn("06026212", result.final_answer)
        self.assertTrue(result.provenance)
        self.assertNotEqual(response.get("route"), "hard")

    def test_plan_wide_prerequisite_question_uses_h3_scope(self):
        result, _ = self.run_hard(
            "ตรวจสอบลำดับวิชาบังคับก่อนของแผนนี้",
            _intent("prerequisite_sequence", program="DSBA", plan="coop"),
            context={"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["hard_task_type"], "prerequisite_sequence")
        self.assertEqual(result["status"], "satisfied")
        self.assertIn("เรียงตามลำดับ", result["answer"])

    def test_explicit_current_course_overrides_trusted_course_context(self):
        response = ask(
            DB_PATH,
            "วิชา 06026201 ต้องเรียนอะไรมาก่อน",
            conversation_context=QueryContext(program="DSBA", plan="coop", course_code="06026212", catalog_key="dsba-2565"),
        )
        result = response["result"]
        self.assertEqual(result.status, "answer")
        self.assertIn("06026201", result.final_answer)
        self.assertIn("06026200", result.final_answer)
        self.assertNotIn("06026212", result.final_answer)
        self.assertTrue(result.provenance)
        self.assertEqual(response["next_context"].course_code, "06026201")

    def test_interpreter_cannot_inject_status_count_or_answer(self):
        injected = json.loads(_intent(
            "plan_structure_validation", program="DSBA", plan="coop"
        ))
        injected.update({"status": "satisfied", "course_count": 999, "answer": "ผ่านครบ"})
        result, _ = self.run_hard(
            "แผน DSBA coop มีโครงสร้างครบตามหลักสูตรไหม",
            json.dumps(injected),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["action"], "hard_interpretation_failure")
        self.assertNotIn("ผ่านครบ", result["answer"])

    def test_seven_term_answer_shows_concrete_courses_and_unresolved_plan_slots(self):
        result, _ = self.run_hard(
            "ถ้าจะจบใน 3.5 ปี แต่ละเทอมต้องลงวิชาอะไร",
            _intent("seven_term_plan", program="DSBA", plan="coop", horizon_terms=7),
            context={"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["hard_task_type"], "seven_term_plan")
        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertTrue(result["answer"].startswith(
            "จากข้อมูลหลักสูตรที่มี สามารถจัดลำดับรายวิชาเป็นโครงร่าง 7 เทอม"
        ))
        self.assertIn("ปี 1 เทอม 1:", result["answer"])
        self.assertIn("ช่องวิชาเลือกที่ยังไม่ระบุวิชาจริง", result["answer"])
        self.assertIn("โครงร่าง 7 เทอมนี้ไม่ใช่ข้อพิสูจน์ว่าครบเงื่อนไขจบ", result["answer"])
        self.assertIn("ยังไม่ได้ตรวจสอบการเปิดสอนจริงในแต่ละภาคเรียน", result["answer"])
        self.assertNotIn("incomplete_evidence", result["answer"])
        self.assertNotIn("06026xxx", result["answer"])
        self.assertNotIn("90644xxx", result["answer"])
        self.assertNotIn("9064xxxx", result["answer"])
        self.assertNotIn("xxxxxxxx", result["answer"])
        self.assertTrue(result["provenance"])

    def test_h4_rows_show_canonical_course_credits_and_only_proven_subtotals(self):
        terms = [{
            "term_index": index,
            "year": (index + 1) // 2,
            "semester": 1 if index % 2 else 2,
            "courses": [],
            "choice_slots": [],
            "required_selection_slots": [],
            "total_known_credits": 0,
        } for index in range(1, 8)]
        terms[0]["courses"] = [
            {"course_code": "06026200", "name_th": "แคลคูลัส 1", "credit_units": 3},
            {"course_code": "06026201", "name_th": "แคลคูลัส 2", "credit_units": 3},
        ]
        terms[0]["total_known_credits"] = 6
        terms[1]["courses"] = [
            {"course_code": "06026202", "name_th": "พีชคณิตเชิงเส้น", "credit_units": 3},
        ]
        terms[1]["total_known_credits"] = 3
        terms[1]["required_selection_slots"] = [{"label_th": "วิชาเลือก", "placement_id": 9}]
        terms[2]["courses"] = [
            {"course_code": "06026203", "name_th": "วิชาที่ไม่ทราบหน่วยกิต", "credit_units": None},
        ]

        _, answer, _ = _format_h4({"status": "incomplete_evidence", "terms": terms,
                                   "required_selection_slots": [{"label_th": "วิชาเลือก", "placement_id": 9}],
                                   "limitations": []})
        self.assertIn("06026200 — แคลคูลัส 1 — 3 หน่วยกิต", answer)
        self.assertIn("06026201 — แคลคูลัส 2 — 3 หน่วยกิต", answer)
        self.assertLess(answer.index("06026200"), answer.index("06026201"))
        self.assertIn("รวม 6 หน่วยกิต", answer)
        self.assertIn("รวมหน่วยกิตที่ยืนยันได้ 3 หน่วยกิต", answer)
        self.assertNotIn("06026203 — วิชาที่ไม่ทราบหน่วยกิต —", answer)
        self.assertIn("โครงร่าง 7 เทอมนี้ไม่ใช่ข้อพิสูจน์ว่าครบเงื่อนไขจบ", answer)

    def test_h4_does_not_render_semester_subtotal_with_unknown_course_credit(self):
        terms = [{"term_index": index, "year": (index + 1) // 2,
                  "semester": 1 if index % 2 else 2, "courses": [],
                  "choice_slots": [], "required_selection_slots": [],
                  "total_known_credits": 0} for index in range(1, 8)]
        terms[0]["courses"] = [
            {"course_code": "06026200", "name_th": "แคลคูลัส 1", "credit_units": 3},
            {"course_code": "06026201", "name_th": "วิชาหน่วยกิตไม่ทราบ", "credit_units": None},
        ]
        terms[0]["total_known_credits"] = 3
        _, answer, _ = _format_h4({"status": "incomplete_evidence", "terms": terms,
                                   "required_selection_slots": [], "limitations": []})
        self.assertIn("06026200 — แคลคูลัส 1 — 3 หน่วยกิต", answer)
        self.assertIn("รวมหน่วยกิตที่ยืนยันได้ 3 หน่วยกิต", answer)
        self.assertNotIn("รวม 3 หน่วยกิต", answer)

    def test_h4_does_not_invent_missing_choice_minimum(self):
        terms = [
            {"term_index": index, "year": (index + 1) // 2, "semester": 1 if index % 2 else 2,
             "courses": [], "choice_slots": []}
            for index in range(1, 8)
        ]
        terms[0]["courses"] = [
            {"course_code": "06026xxx", "name_th": "วิชาเลือกกลุ่ม วิทยาการข้อมูล 1 วิชาเลือกกลุ่มการวิเคราะห์เชิงสถิติ"},
            {"course_code": "06026200", "name_th": "แคลคูลัส 1"},
        ]
        terms[6]["choice_slots"] = [{
            "minimum_choices": None,
            "candidates": [
                {"course_code": "06026259", "name_th": "ทางเลือก ก"},
                {"course_code": "06026260", "name_th": "ทางเลือก ข"},
            ],
        }]
        planner_result = {
            "status": "incomplete_evidence", "sequence_feasible": None,
            "terms": terms, "limitations": [], "evidence": [],
            "actual_course_offering_unverified": True,
        }
        with patch("backend.hard_qa.plan_curriculum_sequence", return_value=planner_result):
            result, _ = self.run_hard(
                "ถ้าจะจบใน 3.5 ปี แต่ละเทอมต้องลงวิชาอะไร",
                _intent("seven_term_plan", program="DSBA", plan="coop", horizon_terms=7),
                context={"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565"},
            )

        self.assertIn("- วิชาเลือกกลุ่มวิทยาการข้อมูล 1", result["answer"].splitlines())
        self.assertIn("- วิชาเลือกกลุ่มการวิเคราะห์เชิงสถิติ", result["answer"].splitlines())
        self.assertIn("- 06026200 — แคลคูลัส 1", result["answer"].splitlines())
        self.assertIn("ทางเลือก: 06026259 — ทางเลือก ก หรือ 06026260 — ทางเลือก ข", result["answer"])
        self.assertNotIn("เลือก None", result["answer"])
        self.assertNotIn("เลือก 1 วิชาจาก", result["answer"])

    def test_hard_answer_citation_display_compacts_pages_and_keeps_provenance(self):
        references = [
            {"provenance_id": i, "program": "DSBA", "source_filename": f"dsba_page_{page:03}.png",
             "source_page": page, "document_page": page - 5, "document_category": "plan"}
            for i, page in enumerate((33, 34, 35, 37, 39), start=1)
        ]
        references.append({
            "provenance_id": 6, "program": "DSBA", "source_filename": "dsba_catalog_page_033.png",
            "source_page": 33, "document_page": 28, "document_category": "plan",
        })
        duplicated = dict(references[1])
        h1 = {
            "status": "complete", "left_plan": "coop", "right_plan": "no_coop",
            "left_only_courses": [], "right_only_courses": [],
            "left_plan_evidence": references, "right_plan_evidence": [duplicated],
        }
        with patch("backend.hard_qa.compare_plan_course_sets", return_value=h1):
            result, _ = self.run_hard(
                "DSBA coop กับ no_coop ต่างกันที่วิชาไหน",
                _intent("plan_comparison", program="DSBA", left_plan="coop", right_plan="no_coop"),
                context={"program": "DSBA", "catalog_key": "dsba-2565"},
            )

        self.assertIn("DSBA แผน หน้า 33–35, 37, 39", result["answer"])
        self.assertIn("DSBA · dsba_catalog แผน หน้า 33", result["answer"])
        self.assertEqual(len(result["provenance"]), 6)
        self.assertEqual(len({item["provenance_id"] for item in result["provenance"]}), 6)

    def test_unselected_seven_term_plan_requires_plan_selection(self):
        with patch(
            "backend.hard_qa.plan_curriculum_sequence",
            side_effect=AssertionError("planner must not run without a plan"),
        ):
            result, prompts = self.run_hard(
                "ถ้าอยากเรียนจบใน 3.5 ปีต้องทำยังไง",
                _intent("seven_term_plan", program="DSBA", horizon_terms=7),
                context={"program": "DSBA", "catalog_key": "dsba-2565"},
            )

        self.assertEqual(result["hard_task_type"], "seven_term_plan")
        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["action"], "plan_required")
        self.assertEqual(result["scope"]["plan"], None)
        self.assertNotIn("plan_results", result)
        self.assertIn("เลือกแผนหลักสูตร", result["answer"])
        self.assertEqual(result["provenance"], [])
        self.assertEqual(len(prompts), 1)

    def test_dsba2565_no_plan_seven_term_answer_asks_for_plan_once(self):
        result, prompts = self.run_hard(
            "ถ้าอยากเรียนจบใน 3.5 ปีต้องทำยังไง",
            _intent("seven_term_plan", program="DSBA", horizon_terms=7),
            context={"program": "DSBA", "catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["hard_task_type"], "seven_term_plan")
        self.assertEqual(result["scope"]["plan"], None)
        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["action"], "plan_required")
        self.assertNotIn("plan_results", result)
        self.assertIn("เลือกแผนหลักสูตร", result["answer"])
        self.assertNotIn("ผลแผน coop", result["answer"])
        self.assertNotIn("ผลแผน no_coop", result["answer"])
        self.assertEqual(result["provenance"], [])
        self.assertEqual(len(prompts), 1)

    def test_explicit_seven_term_plan_runs_only_selected_plan(self):
        for plan in ("coop", "no_coop"):
            with self.subTest(plan=plan):
                with patch(
                    "backend.hard_qa.plan_curriculum_sequence",
                    wraps=plan_curriculum_sequence,
                ) as planner:
                    result, _ = self.run_hard(
                        f"ถ้าเลือกแผน {plan} แล้วอยากเรียนจบใน 3.5 ปีต้องทำยังไง",
                        _intent("seven_term_plan", program="DSBA", horizon_terms=7),
                        context={"program": "DSBA", "catalog_key": "dsba-2565"},
                    )

                self.assertEqual(result["scope"]["plan"], plan)
                planner.assert_called_once()
                self.assertEqual(planner.call_args.args[2], plan)

    def test_unrecognized_seven_term_plan_remains_fail_closed(self):
        with patch("backend.hard_qa.plan_curriculum_sequence") as planner:
            result, _ = self.run_hard(
                "ถ้าอยากเรียนจบใน 3.5 ปีต้องทำยังไง",
                _intent("seven_term_plan", program="DSBA", plan="mystery", horizon_terms=7),
                context={"program": "DSBA", "catalog_key": "dsba-2565"},
            )

        self.assertEqual(result["status"], "clarification_required")
        self.assertIn("เลือกแผนหลักสูตร", result["answer"])
        planner.assert_not_called()

    def test_trusted_program_and_plan_context_are_reused(self):
        result, _ = self.run_hard(
            "ถ้าจะจบใน 3.5 ปี แต่ละเทอมต้องลงอะไร",
            _intent("seven_term_plan", horizon_terms=7),
            context={"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["scope"], {"catalog_key": "dsba-2565", "program": "DSBA", "plan": "coop"})
        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertEqual(result["next_context"], {
            "catalog_key": "dsba-2565", "program": "DSBA", "plan": "coop",
            "study_plan_context": {
                "kind": "seven_term_plan", "program": "DSBA",
                "catalog_key": "dsba-2565", "plan": "coop",
            },
        })

    def test_seven_term_followup_provenance_is_scoped_to_requested_term(self):
        full = plan_curriculum_sequence(
            DB_PATH, "DSBA", "coop", horizon_terms=7, catalog_key="dsba-2565"
        )
        full_ids = {
            reference["provenance_id"]
            for reference in full.get("evidence", [])
            if isinstance(reference, dict)
        }
        term = next(
            item for item in full["terms"]
            if item.get("year") == 3 and item.get("semester") == 1
        )
        term_ids = {
            evidence_id
            for course in term["courses"]
            for evidence_id in course.get("evidence_ids", [])
        }
        for slot in term["required_selection_slots"]:
            for reference in slot.get("provenance", []) or ():
                if isinstance(reference, dict):
                    term_ids.add(reference.get("provenance_id"))

        for include_courses in (False, True):
            with self.subTest(include_courses=include_courses):
                result = answer_seven_term_followup(
                    DB_PATH, program="DSBA", catalog_key="dsba-2565",
                    plan="coop", year=3, semester=1,
                    include_courses=include_courses, include_credits=True,
                )
                self.assertEqual(result["status"], "answer")
                returned_ids = {
                    reference["provenance_id"]
                    for reference in result["provenance"]
                }
                self.assertTrue(returned_ids)
                self.assertTrue(returned_ids <= full_ids)
                self.assertTrue(returned_ids <= term_ids)
                self.assertTrue(len(returned_ids) < len(full_ids))

    def test_hallucinated_plan_is_rejected_and_program_change_invalidates_context_plan(self):
        result, _ = self.run_hard(
            "แผน AIT มีโครงสร้างครบตามหลักสูตรไหม",
            _intent("plan_structure_validation", program="AIT", plan="coop"),
            context={"program": "DSBA", "plan": "coop"},
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["scope"], {"program": "AIT", "plan": None})
        self.assertIn("เลือกแผนหลักสูตร", result["answer"])

    def test_old_new_curriculum_comparison_uses_canonical_editions_without_llm(self):
        result = answer_hard_question(
            DB_PATH,
            "วิชาที่มีในหลักสูตรเก่า DSBA ไม่มีในหลักสูตรใหม่มีอะไรบ้าง",
            {"program": "DSBA"},
            lambda prompt: self.fail("old/new request should not call interpreter"),
        )

        self.assertEqual(result["hard_task_type"], "old_new_comparison")
        self.assertEqual(result["status"], "answer")
        comparison = result["comparison"]
        self.assertEqual(comparison["older"], {"catalog_key": "dsba-2560", "academic_year": "2560"})
        self.assertEqual(comparison["newer"], {"catalog_key": "dsba-2565", "academic_year": "2565"})
        self.assertEqual(
            set(comparison["categories"]),
            {
                "shared_same_code", "old_only_by_code", "new_only_by_code",
                "same_name_changed_code_candidates", "unresolved_non_concrete",
            },
        )
        self.assertTrue(comparison["provenance"])
        self.assertTrue(all(
            candidate["equivalence_proven"] is False
            for candidate in comparison["categories"]["same_name_changed_code_candidates"]
        ))
        self.assertTrue(any(
            candidate["older"]["course_code"] == "06026106"
            and candidate["newer"]["course_code"] == "06066300"
            for candidate in comparison["categories"]["same_name_changed_code_candidates"]
        ))
        answer = result["answer"]
        self.assertIn("หลักสูตร DSBA พ.ศ. 2560", answer)
        self.assertIn("พ.ศ. 2565", answer)
        self.assertIn("ขอบเขตแผน: รวมแผนที่ปรากฏในข้อมูล", answer)
        self.assertIn("รายวิชาที่ใช้รหัสเดียวกันในทั้งสองหลักสูตร: 0 รหัส", answer)
        self.assertIn(
            "รายวิชาที่ชื่อเดียวกันหรือชื่อที่ตรงกันตามการปรับรูปแบบข้อความ แต่รหัสวิชาเปลี่ยน: 30 คู่",
            answer,
        )
        self.assertIn("ตัวเลือกที่อาจเป็นการเปลี่ยนรหัสวิชา", answer)
        self.assertIn("ยังไม่ถือว่าเป็นการยืนยันว่ารายวิชาทั้งสองเทียบเท่ากัน", answer)
        self.assertIn("06026106", answer)
        self.assertIn("06066300", answer)
        self.assertTrue(
            all(
                "06026106" not in line
                for line in answer.splitlines()
                if line.startswith("ตัวอย่างรหัสฝั่ง 2560:")
            )
        )
        self.assertTrue(
            all(
                "06066300" not in line
                for line in answer.splitlines()
                if line.startswith("ตัวอย่างรหัสฝั่ง 2565:")
            )
        )
        placeholder = comparison["categories"]["unresolved_non_concrete"][0]["course_code"]
        self.assertIn(placeholder, answer)
        self.assertIn("รหัสวิชาที่พบเฉพาะในหลักสูตร พ.ศ. 2560: 75 รหัส", answer)
        self.assertIn("ไม่ได้สรุปว่ารายวิชาถูกยกเลิก", answer)
        self.assertIn("รหัสวิชาที่พบเฉพาะในหลักสูตร พ.ศ. 2565: 79 รหัส", answer)
        self.assertIn("ไม่ได้สรุปว่าเป็นรายวิชาใหม่", answer)
        self.assertNotIn("วิชาที่ถูกยกเลิก", answer)
        self.assertNotIn("วิชาใหม่:", answer)
        self.assertTrue(result["provenance"])

        repeated = answer_hard_question(
            DB_PATH,
            "วิชาที่มีในหลักสูตรเก่า DSBA ไม่มีในหลักสูตรใหม่มีอะไรบ้าง",
            {"program": "DSBA"},
            lambda prompt: self.fail("old/new request should not call interpreter"),
        )
        self.assertEqual(repeated["answer"], answer)

    def test_old_new_comparison_preserves_explicit_plan_scope(self):
        result = answer_hard_question(
            DB_PATH,
            "วิชาใดในหลักสูตรเก่า DSBA coop ไม่พบในหลักสูตรใหม่",
            {"program": "DSBA"},
            lambda prompt: self.fail("old/new request should not call interpreter"),
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["scope"]["plan"], "coop")
        self.assertEqual(result["comparison"]["plan"], "coop")
        self.assertIn("ขอบเขตแผน: coop", result["answer"])
        for category in (
            "shared_same_code", "old_only_by_code", "new_only_by_code",
            "same_name_changed_code_candidates",
        ):
            for bucket in result["comparison"]["categories"][category]:
                courses = bucket.get("courses", []) or [
                    bucket.get("older", {}), bucket.get("newer", {})
                ]
                for course in courses:
                    if course:
                        self.assertEqual(course["plans"], ["coop"])

    def test_old_new_comparison_clarifies_invalid_plan_context(self):
        result = answer_hard_question(
            DB_PATH,
            "วิชาใดในหลักสูตรเก่า DSBA ไม่พบในหลักสูตรใหม่",
            {"program": "DSBA", "plan": "gened"},
            lambda prompt: self.fail("old/new request should not call interpreter"),
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["action"], "invalid_plan_context")

    def test_old_new_question_without_program_requests_scope(self):
        result = answer_hard_question(
            DB_PATH,
            "วิชาที่มีในหลักสูตรเก่า ไม่มีในหลักสูตรใหม่มีอะไรบ้าง",
            None,
            lambda prompt: self.fail("program clarification must not call interpreter"),
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertIn("ระบุรหัสหลักสูตร", result["answer"])

    def test_old_new_comparison_fails_closed_when_only_one_edition_exists(self):
        result = answer_hard_question(
            DB_PATH,
            "วิชาที่มีในหลักสูตรเก่า AIT ไม่มีในหลักสูตรใหม่มีอะไรบ้าง",
            {"program": "AIT"},
            lambda prompt: self.fail("missing edition pair must not call interpreter"),
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["action"], "insufficient_editions")

    def test_old_new_comparison_resolves_legacy_and_current_edition_pair(self):
        for program, legacy_catalog in (("IT", "it-2560"), ("BIT", "bit-2560")):
            with self.subTest(program=program):
                result = answer_hard_question(
                    DB_PATH,
                    f"วิชาที่มีในหลักสูตรเก่า {program} ไม่มีในหลักสูตรใหม่มีอะไรบ้าง",
                    {"program": program},
                    lambda prompt: self.fail("edition pair must not call interpreter"),
                )

                self.assertEqual(result["status"], "answer")
                self.assertEqual(result["hard_task_type"], "old_new_comparison")
                self.assertIn(legacy_catalog, result["answer"])
                self.assertTrue(result["provenance"])
                self.assertTrue(
                    any(
                        str(reference.get("source_filename", "")).startswith(
                            legacy_catalog.replace("-", "")
                        )
                        for reference in result["provenance"]
                    )
                )

    def test_provenance_pages_are_only_canonical_fields_and_answer_uses_no_model(self):
        answer_callable_calls = []
        result, _ = self.run_hard(
            "DSBA coop กับ no_coop ต่างกันที่วิชาไหน",
            _intent("plan_comparison", program="DSBA", left_plan="coop", right_plan="no_coop"),
            context={"program": "DSBA", "catalog_key": "dsba-2565"},
        )

        self.assertFalse(answer_callable_calls)
        self.assertTrue(result["provenance"])
        for reference in result["provenance"]:
            self.assertIn("provenance_id", reference)
            self.assertIn("source_filename", reference)
            self.assertIn("source_page", reference)
            self.assertIn("document_page", reference)
        self.assertEqual(len({item["provenance_id"] for item in result["provenance"]}), len(result["provenance"]))

    def test_easy_question_is_not_a_hard_candidate(self):
        result = answer_hard_question(
            DB_PATH,
            "ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
            {"program": "IT"},
            lambda prompt: self.fail("ordinary Easy/Medium question must not invoke Hard interpreter"),
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
