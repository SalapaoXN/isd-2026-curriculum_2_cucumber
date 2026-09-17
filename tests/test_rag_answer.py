import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from rag.answer import EMPTY_ANSWER, answer_question, render_grounded_answer, render_grounded_claim
from rag.grounded_answer import GroundedAnswerResult, GroundedClaim
from rag.hybrid_demo import run_hybrid_demo
from rag.retrieval.retrieve import SimilarityEvidence, SimilarityPair


class RagAnswerTest(unittest.TestCase):
    def test_complete_similarity_renders_grounded_course_descriptions(self):
        value = SimilarityEvidence(
            status="complete",
            pairs=(
                SimilarityPair(
                    "complete",
                    {"program": "IT", "plan": "no_coop"},
                    {
                        "program": "IT",
                        "course_code": "06016402",
                        "partition": {"plan": "no_coop"},
                        "text": "IT grounded description",
                        "provenance": ({"source_page": 10},),
                    },
                    {
                        "program": "DSBA",
                        "course_code": "06026207",
                        "partition": {"plan": "no_coop"},
                        "text": "DSBA grounded description",
                        "provenance": ({"source_page": 20},),
                    },
                    cosine_distance=0.2193,
                    cosine_similarity=0.7807,
                ),
            ),
            mean_distance=0.2193,
            min_distance=0.2193,
            max_distance=0.2193,
        )
        claim = GroundedClaim(
            "claim_001",
            "similarity",
            value=value,
            evidence=value,
            provenance=(
                {"source_page": 10},
                {"source_page": 20},
            ),
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("IT 06016402 (no_coop)", rendered)
        self.assertIn("DSBA 06026207 (no_coop)", rendered)
        self.assertIn("IT grounded description", rendered)
        self.assertIn("DSBA grounded description", rendered)
        self.assertIn("cosine_similarity", rendered)
        self.assertNotIn('"pairs"', rendered)
        self.assertNotIn("เนื้อหาเหมือนกัน", rendered)

    def test_structured_placement_renders_thai_year_semester_and_credits(self):
        claim = GroundedClaim(
            "placement_001",
            "placement",
            value={
                "program": "BIT",
                "plan_key": "coop",
                "year_number": 3,
                "semester_number": 1,
                "credits": "3(3-0-6)",
            },
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("แผนสหกิจ", rendered)
        self.assertIn("เรียนในปี 3 ภาคเรียนที่ 1", rendered)
        self.assertIn("3(3-0-6) หน่วยกิต", rendered)
        self.assertNotIn("year_number", rendered)

    def test_same_timing_plan_comparison_is_rendered_explicitly(self):
        claim = GroundedClaim(
            "placement_compare_001",
            "compare",
            value=(
                {"plan_key": "coop", "year_semester_choices": ((2, 1),)},
                {"plan_key": "no_coop", "year_semester_choices": ((2, 1),)},
            ),
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("ทั้งแผนสหกิจและแผนไม่สหกิจ", rendered)
        self.assertIn("ปี 2 ภาคเรียนที่ 1", rendered)
        self.assertIn("ไม่ต่างกันด้านช่วงเรียน", rendered)

    def test_flexible_placement_lists_each_grounded_choice(self):
        claim = GroundedClaim(
            "placement_flexible_001",
            "placement",
            value={
                "program": "IT",
                "plan_key": "no_coop",
                "year_semester_choices": ((3, 1), (3, 2), (4, 1)),
            },
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("สามารถเรียนได้ในปี 3 ภาคเรียนที่ 1 หรือ ปี 3 ภาคเรียนที่ 2 หรือ ปี 4 ภาคเรียนที่ 1", rendered)
        self.assertNotIn("year_semester_choices", rendered)

    def test_hybrid_rendering_keeps_placement_and_description_grounded(self):
        result = GroundedAnswerResult(
            status="answer",
            answer_mode="deterministic",
            claims=(
                GroundedClaim(
                    "placement_002",
                    "placement",
                    value={"plan_key": "coop", "year_number": 3, "semester_number": 1},
                ),
                GroundedClaim(
                    "description_001",
                    "describe",
                    value={"text": "DATABASE TECHNOLOGY"},
                    evidence={"text": "DATABASE TECHNOLOGY"},
                ),
            ),
        )

        rendered = render_grounded_answer(result).final_answer

        self.assertIn("ปี 3 ภาคเรียนที่ 1", rendered)
        self.assertIn("DATABASE TECHNOLOGY", rendered)

    def test_structured_prompt_is_grounded_in_sql_rows(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            return "วิชา C100 มี 3 หน่วยกิต"

        answer = answer_question(
            "วิชา C100 มีกี่หน่วยกิต",
            "structured",
            structured_result={
                "columns": ["course_code", "credit_units", "source_page"],
                "rows": [("C100", 3, 12)],
            },
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "วิชา C100 มี 3 หน่วยกิต")
        self.assertEqual(len(prompts), 1)
        self.assertIn("ผลลัพธ์จาก SQL", prompts[0])
        self.assertIn("ห้ามนับ รวม คำนวณ", prompts[0])
        self.assertIn("C100", prompts[0])
        self.assertIn("source_page", prompts[0])
        self.assertIn("12", prompts[0])

    def test_structured_derived_facts_reach_final_prompt_with_rows(self):
        prompts = []

        structured_result = {
            "operation": "course_placement",
            "status": "ok",
            "columns": ["plan_key", "year_semester_choices"],
            "rows": [["coop", [(4, 1)]], ["no_coop", [(3, 1)]]],
            "earliest_plan": "no_coop",
            "provenance": [{"source_page": 360}],
        }

        answer = answer_question(
            "วิชา 06016465 ใน IT ควรเลือกแผนไหนให้เร็วที่สุด?",
            "structured",
            structured_result=structured_result,
            answer_model_callable=lambda prompt: prompts.append(prompt) or "คำตอบ",
        )

        self.assertEqual(answer, "คำตอบ")
        self.assertEqual(len(prompts), 1)
        self.assertIn('"derived_facts":{"earliest_plan":"no_coop"}', prompts[0])
        self.assertIn('"sql_rows":[{"plan_key":"coop"', prompts[0])
        self.assertIn('"plan_key":"no_coop"', prompts[0])
        self.assertNotIn("earliest_plan_keys", prompts[0])

    def test_complete_structured_comparison_answer_needs_no_retry(self):
        prompts = []
        structured_result = {
            "operation": "course_placement_comparison",
            "status": "ok",
            "requested_plan_keys": ["coop", "no_coop"],
            "columns": ["plan_key", "year", "semester", "flexible_year_semester_raw"],
            "rows": [
                ["coop", 3, 1, None],
                ["coop", None, None, "4/1"],
                ["no_coop", 3, 1, None],
                ["no_coop", None, None, "3/1, 3/2, 4/1"],
            ],
        }

        answer = answer_question(
            "วางแผนสองวิชาใน IT ควรเลือกแผนไหนและแต่ละแผนเรียนช่วงใด",
            "structured",
            structured_result=structured_result,
            answer_model_callable=lambda prompt: prompts.append(prompt)
            or "แผน coop: วิชาแรก 3/1 และวิชาที่สอง 4/1; "
            "แผน no_coop: วิชาแรก 3/1 และวิชาที่สอง 3/1, 3/2, 4/1",
        )

        self.assertIn("แผน no_coop", answer)
        self.assertEqual(len(prompts), 1)

    def test_incomplete_structured_comparison_retries_once_with_all_evidence(self):
        prompts = []
        answers = iter(
            [
                "แผน no_coop: วิชาแรก 3/1 และวิชาที่สอง 3/1, 3/2, 4/1",
                "แผน coop: วิชาแรก 3/1 และวิชาที่สอง 4/1; "
                "แผน no_coop: วิชาแรก 3/1 และวิชาที่สอง 3/1, 3/2, 4/1",
            ]
        )
        structured_result = {
            "operation": "course_placement_comparison",
            "status": "ok",
            "requested_plan_keys": ["coop", "no_coop"],
            "derived_facts": {
                "earliest_plan": "no_coop",
                "plan_completion_earliest": [
                    {"plan_key": "coop", "completion_year_semester": [4, 1]},
                    {"plan_key": "no_coop", "completion_year_semester": [3, 1]},
                ],
            },
            "columns": ["plan_key", "year", "semester", "flexible_year_semester_raw"],
            "rows": [
                ["coop", 3, 1, None],
                ["coop", None, None, "4/1"],
                ["no_coop", 3, 1, None],
                ["no_coop", None, None, "3/1, 3/2, 4/1"],
            ],
        }

        answer = answer_question(
            "วางแผนสองวิชาใน IT ควรเลือกแผนไหนและแต่ละแผนเรียนช่วงใด",
            "structured",
            structured_result=structured_result,
            answer_model_callable=lambda prompt: prompts.append(prompt)
            or next(answers),
        )

        self.assertIn("แผน coop", answer)
        self.assertEqual(len(prompts), 2)
        self.assertIn('"earliest_plan":"no_coop"', prompts[1])
        self.assertIn('"plan_key":"coop"', prompts[1])
        self.assertIn('"plan_key":"no_coop"', prompts[1])

    def test_incomplete_structured_comparison_uses_appendix_after_one_retry(self):
        prompts = []
        structured_result = {
            "operation": "course_placement_comparison",
            "status": "ok",
            "requested_plan_keys": ["coop", "no_coop"],
            "columns": ["plan_key", "year", "semester", "flexible_year_semester_raw"],
            "rows": [
                ["coop", 3, 1, None],
                ["coop", None, None, "4/1"],
                ["no_coop", 3, 1, None],
                ["no_coop", None, None, "3/1, 3/2, 4/1"],
            ],
        }

        answer = answer_question(
            "วางแผนสองวิชาใน IT ควรเลือกแผนไหนและแต่ละแผนเรียนช่วงใด",
            "structured",
            structured_result=structured_result,
            answer_model_callable=lambda prompt: prompts.append(prompt)
            or "แผน no_coop เรียนวิชาแรก 3/1",
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("ข้อมูลการจัดวางที่ยืนยันได้เพิ่มเติม", answer)
        self.assertIn('"plan":"coop"', answer)
        self.assertIn('"plan":"no_coop"', answer)

    def test_semantic_prompt_is_grounded_in_retrieved_chunks(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            return "หลักฐานระบุเนื้อหาของวิชา C200"

        answer = answer_question(
            "วิชา C200 เรียนเกี่ยวกับอะไร",
            "semantic",
            semantic_chunks=[
                {
                    "chunk_id": "chunk-1",
                    "text": "C200 ครอบคลุมระบบฐานข้อมูล",
                    "source_page": [21],
                }
            ],
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "หลักฐานระบุเนื้อหาของวิชา C200")
        self.assertEqual(len(prompts), 1)
        self.assertIn("ชิ้นส่วนหลักฐานที่ค้นคืนได้", prompts[0])
        self.assertIn("C200 ครอบคลุมระบบฐานข้อมูล", prompts[0])
        self.assertIn('"source_page":[21]', prompts[0])

    def test_complete_paraphrased_topic_answer_needs_no_retry(self):
        prompts = []

        answer = answer_question(
            "วิชานี้เรียนเกี่ยวกับอะไร",
            "semantic",
            semantic_chunks=[
                {
                    "text": (
                        "คำอธิบายรายวิชาภาษาอังกฤษ: RESOURCE VIRTUALIZATION, "
                        "SOFTWARE DEFINED INFRASTRUCTURE, INFRASTRUCTURE SECURITY"
                    )
                }
            ],
            answer_model_callable=lambda prompt: prompts.append(prompt)
            or "เนื้อหาครอบคลุมการทำทรัพยากรเสมือน โครงสร้างพื้นฐานที่กำหนดด้วยซอฟต์แวร์ "
            "และความปลอดภัยของโครงสร้างพื้นฐาน",
        )

        self.assertIn("ความปลอดภัย", answer)
        self.assertEqual(len(prompts), 1)

    def test_incomplete_topic_answer_retries_once_and_can_recover(self):
        prompts = []
        answers = iter(
            [
                "เนื้อหาเกี่ยวกับความปลอดภัยของโครงสร้างพื้นฐาน",
                "เนื้อหาครอบคลุมการทำทรัพยากรเสมือน โครงสร้างพื้นฐานที่กำหนดด้วยซอฟต์แวร์ "
                "และความปลอดภัยของโครงสร้างพื้นฐาน",
            ]
        )

        answer = answer_question(
            "วิชานี้เรียนเกี่ยวกับอะไร",
            "semantic",
            semantic_chunks=[
                {
                    "text": (
                        "คำอธิบายรายวิชาภาษาอังกฤษ: RESOURCE VIRTUALIZATION, "
                        "SOFTWARE DEFINED INFRASTRUCTURE, INFRASTRUCTURE SECURITY"
                    )
                }
            ],
            answer_model_callable=lambda prompt: prompts.append(prompt)
            or next(answers),
        )

        self.assertIn("ทรัพยากรเสมือน", answer)
        self.assertEqual(len(prompts), 2)
        self.assertIn("สรุปหัวข้อที่รองรับคำถามอย่างครบถ้วน", prompts[1])

    def test_incomplete_topic_answer_is_never_retried_more_than_once(self):
        prompts = []

        answer = answer_question(
            "วิชานี้เรียนเกี่ยวกับอะไร",
            "semantic",
            semantic_chunks=[
                {
                    "text": (
                        "คำอธิบายรายวิชาภาษาอังกฤษ: RESOURCE VIRTUALIZATION, "
                        "SOFTWARE DEFINED INFRASTRUCTURE"
                    )
                }
            ],
            answer_model_callable=lambda prompt: prompts.append(prompt)
            or "กล่าวถึงทรัพยากรเสมือน",
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("ทรัพยากรเสมือน", answer)

    def test_nonempty_evidence_retries_an_unjustified_fallback_once(self):
        prompts = []
        answers = iter([EMPTY_ANSWER, "คำตอบจากหลักฐาน"])

        def answer_model(prompt):
            prompts.append(prompt)
            return next(answers)

        answer = answer_question(
            "วิชา C200 เรียนเกี่ยวกับอะไร",
            "semantic",
            semantic_chunks=[
                {
                    "chunk_id": "chunk-1",
                    "text": "C200 ครอบคลุมระบบฐานข้อมูล",
                    "source_page": [21],
                }
            ],
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "คำตอบจากหลักฐาน")
        self.assertEqual(len(prompts), 2)
        self.assertIn("ห้ามเดาหรือสร้างคำตอบ", prompts[0])
        self.assertIn("ถ้าหลักฐานรองรับคำถาม", prompts[1])

    def test_fallback_like_denial_is_retried_once(self):
        prompts = []
        answers = iter(["ไม่สามารถตอบคำถามนี้ได้จากหลักฐานที่มี", "มี 30 หน่วยกิต"])

        def answer_model(prompt):
            prompts.append(prompt)
            return next(answers)

        answer = answer_question(
            "IT ปี 2 เทอม 2 มีกี่หน่วยกิต",
            "structured",
            structured_result={"columns": ["total_credits"], "rows": [[30]]},
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "มี 30 หน่วยกิต")
        self.assertEqual(len(prompts), 2)

    def test_placeholder_course_name_does_not_override_description_evidence(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            if "ไม่ใช่เหตุผลให้ปฏิเสธ" in prompt and "description" in prompt:
                return "เนื้อหาครอบคลุมระบบฐานข้อมูล"
            return EMPTY_ANSWER

        answer = answer_question(
            "วิชา C300 เรียนเกี่ยวกับอะไร",
            "semantic",
            semantic_chunks=[
                {
                    "chunk_id": "chunk-description",
                    "text": "course_code: C300 name_en: ไม่ระบุ description: ระบบฐานข้อมูล",
                    "source_page": [44],
                }
            ],
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "เนื้อหาครอบคลุมระบบฐานข้อมูล")
        self.assertEqual(len(prompts), 1)

    def test_structured_rows_are_safe_fallback_after_two_denials(self):
        prompts = []
        answers = iter(["ไม่มีข้อมูลที่ตอบคำถามได้", "ไม่พบข้อมูลในเล่มหลักสูตร"])

        def answer_model(prompt):
            prompts.append(prompt)
            return next(answers)

        answer = answer_question(
            "IT ไม่สหกิจ ปี 2 เทอม 2 และวิชา 06016420 ต้องผ่านวิชาใด",
            "structured",
            structured_result={
                "columns": ["total_credits", "prerequisite_course_code"],
                "rows": [[30, "06016413"]],
            },
            answer_model_callable=answer_model,
        )

        self.assertEqual(len(prompts), 2)
        self.assertNotEqual(answer, EMPTY_ANSWER)
        self.assertIn('"total_credits":30', answer)
        self.assertIn('"prerequisite_course_code":"06016413"', answer)

    def test_two_fallback_like_denials_without_structured_rows_are_canonical(self):
        answers = iter(["ไม่พบข้อมูลในหลักสูตร", "ไม่มีข้อมูลที่ยืนยันได้"])

        answer = answer_question(
            "วิชา C200 เรียนเกี่ยวกับอะไร",
            "semantic",
            semantic_chunks=[{"text": "C200 เป็นวิชาอื่น", "source_page": [21]}],
            answer_model_callable=lambda prompt: next(answers),
        )

        self.assertEqual(answer, EMPTY_ANSWER)

    def test_related_but_insufficient_evidence_keeps_fallback_after_one_retry(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            return EMPTY_ANSWER

        answer = answer_question(
            "วิชา C200 ต้องผ่านวิชาอะไรมาก่อน",
            "semantic",
            semantic_chunks=[
                {
                    "chunk_id": "chunk-related",
                    "text": "C200 เป็นวิชาฐานข้อมูลเบื้องต้น",
                    "source_page": [21],
                }
            ],
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, EMPTY_ANSWER)
        self.assertEqual(len(prompts), 2)
        self.assertIn("ถ้าหลักฐานไม่พอ", prompts[1])

    def test_empty_evidence_returns_exact_fallback_without_calling_model(self):
        calls = []

        def answer_model(prompt):
            calls.append(prompt)
            return "ไม่ควรถูกเรียก"

        answer = answer_question(
            "ข้อมูลที่ไม่มี",
            "structured",
            structured_result={"columns": ["course_code"], "rows": []},
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, EMPTY_ANSWER)
        self.assertEqual(calls, [])

    def test_source_page_from_chunk_provenance_is_included(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            return "คำตอบ"

        answer_question(
            "คำถาม",
            "semantic",
            semantic_chunks=[
                {
                    "chunk_id": "chunk-2",
                    "text": "ข้อมูลอ้างอิง",
                    "provenance": [{"source_page": 33}],
                }
            ],
            answer_model_callable=answer_model,
        )

        self.assertEqual(len(prompts), 1)
        self.assertIn('"source_page":[33]', prompts[0])

    def test_combined_evidence_is_grounded_in_sql_and_chunks(self):
        prompts = []

        def answer_model(prompt):
            prompts.append(prompt)
            return "คำตอบจากหลักฐานร่วม"

        answer = answer_question(
            "วิชาฐานข้อมูลปี 1 เรียนเรื่องอะไร",
            "hybrid",
            structured_result={
                "columns": ["course_code", "source_page"],
                "rows": [("C100", 12)],
            },
            semantic_chunks=[
                {
                    "chunk_id": "chunk-1",
                    "text": "C100 ครอบคลุมฐานข้อมูล",
                    "source_page": [13],
                }
            ],
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "คำตอบจากหลักฐานร่วม")
        self.assertEqual(len(prompts), 1)
        self.assertIn("ผลลัพธ์จาก SQL และชิ้นส่วนหลักฐาน", prompts[0])
        self.assertIn('"sql_rows":[{"course_code":"C100","source_page":12}]', prompts[0])
        self.assertIn("C100 ครอบคลุมฐานข้อมูล", prompts[0])

    def test_cost_request_does_not_use_credit_total_as_cost_evidence(self):
        calls = []

        answer = answer_question(
            "ค่าเทอมของหลักสูตรนี้เท่าไร",
            "structured",
            structured_result={"columns": ["total_credits"], "rows": [[30]]},
            answer_model_callable=lambda prompt: calls.append(prompt) or "30 บาท",
        )

        self.assertEqual(answer, EMPTY_ANSWER)
        self.assertEqual(calls, [])

    def test_hybrid_retries_when_a_requested_plan_is_omitted(self):
        prompts = []
        answers = iter(
            [
                "แผนสหกิจอยู่ปี 3 เทอม 2 และมีเนื้อหาตามหลักฐาน",
                "แผนสหกิจอยู่ปี 3 เทอม 2 ส่วนแผนไม่สหกิจเปิด 3/1, 3/2, 4/1",
            ]
        )

        def answer_model(prompt):
            prompts.append(prompt)
            return next(answers)

        answer = answer_question(
            "วิชานี้ของ IT แบบสหกิจและไม่สหกิจจัดช่วงเรียนอย่างไร",
            "hybrid",
            structured_result={
                "columns": ["plan_key", "year", "semester", "flexible_year_semester_raw"],
                "rows": [
                    ["coop", 3, 2, None],
                    ["no_coop", None, None, "3/1, 3/2, 4/1"],
                ],
            },
            semantic_chunks=[{"text": "เนื้อหาตามหลักฐาน"}],
            answer_model_callable=answer_model,
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("3/1, 3/2, 4/1", answer)
        self.assertIn("ไม่สหกิจ", prompts[1])

    def test_injected_callable_is_used(self):
        calls = []

        def answer_model(prompt):
            calls.append(prompt)
            return "คำตอบจาก callable"

        answer = answer_question(
            "คำถาม",
            "semantic",
            semantic_chunks=[{"text": "หลักฐาน"}],
            answer_model_callable=answer_model,
        )

        self.assertEqual(answer, "คำตอบจาก callable")
        self.assertEqual(len(calls), 1)

    @patch("rag.hybrid_demo.ask")
    def test_hybrid_demo_prints_grounded_final_answer(self, ask_mock):
        claim = GroundedClaim(
            "claim_001",
            "describe",
            kind="grounded_summary",
            value=({"text": "C100 เป็นวิชาพื้นฐาน"},),
            evidence=({"text": "C100 เป็นวิชาพื้นฐาน"},),
            provenance=({"source_page": 8},),
        )
        ask_mock.return_value = {
            "route": None,
            "result": GroundedAnswerResult(
                "answer",
                "grounded_synthesis",
                "คำตอบจากหลักฐาน",
                (claim,),
                claim.provenance,
            ),
        }

        output = io.StringIO()
        with redirect_stdout(output):
            response = run_hybrid_demo(
                "curriculum.db",
                "C100 คืออะไร",
                answer_model_callable=lambda prompt: "คำตอบจากหลักฐาน",
            )

        self.assertEqual(response["result"].final_answer, "คำตอบจากหลักฐาน")
        self.assertIn("Question: C100 คืออะไร", output.getvalue())
        self.assertIn("Final Answer: คำตอบจากหลักฐาน", output.getvalue())


if __name__ == "__main__":
    unittest.main()
