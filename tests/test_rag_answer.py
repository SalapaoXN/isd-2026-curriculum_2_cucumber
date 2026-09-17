import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from rag.answer import (
    EMPTY_ANSWER,
    _critical_facts,
    answer_question,
    render_grounded_answer,
    render_grounded_claim,
)
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
        self.assertNotIn("cosine_similarity", rendered)
        self.assertNotIn("cosine_distance", rendered)
        self.assertNotIn("similarity: {", rendered)
        self.assertNotIn('"pairs"', rendered)
        self.assertNotIn("เนื้อหาเหมือนกัน", rendered)
        self.assertEqual(value.pairs[0].cosine_distance, 0.2193)
        self.assertEqual(value.pairs[0].cosine_similarity, 0.7807)
        self.assertEqual(
            claim.provenance,
            ({"source_page": 10}, {"source_page": 20}),
        )

    def test_prerequisite_renders_required_course_without_internal_fields(self):
        claim = GroundedClaim(
            "prerequisite_001",
            "prerequisite",
            effective_scope={"plans": ("no_coop",)},
            evidence=(
                {
                    "prerequisite_id": 41,
                    "course_id": 739,
                    "prerequisite_course_id": 728,
                    "prerequisite_code": "06016413",
                    "prerequisite_name_en": "INTRODUCTION TO NETWORK SYSTEMS",
                    "requirement_type": "required",
                    "provenance": ({"provenance_id": 200, "source_page": 338},),
                },
            ),
            provenance=({"provenance_id": 200, "source_page": 338},),
        )

        rendered = render_grounded_claim(claim, scope_dimensions=("plan",))

        self.assertIn("plan=no_coop", rendered)
        self.assertIn("ต้องเรียนวิชา 06016413 INTRODUCTION TO NETWORK SYSTEMS มาก่อน", rendered)
        for field in (
            "prerequisite_id",
            "course_id",
            "prerequisite_course_id",
            "provenance_id",
            "source_page",
        ):
            self.assertNotIn(field, rendered)

    def test_prerequisite_plan_claims_remain_distinguishable(self):
        claims = tuple(
            GroundedClaim(
                f"prerequisite_{plan}",
                "prerequisite",
                effective_scope={"plans": (plan,)},
                evidence=({"prerequisite_code": code, "prerequisite_name_en": name},),
            )
            for plan, code, name in (
                ("coop", "06016413", "INTRODUCTION TO NETWORK SYSTEMS"),
                ("no_coop", "06016414", "SYSTEM ANALYSIS"),
            )
        )

        rendered = render_grounded_answer(
            GroundedAnswerResult(
                status="answer",
                answer_mode="deterministic",
                claims=claims,
            )
        ).final_answer

        self.assertIn("plan=coop", rendered)
        self.assertIn("plan=no_coop", rendered)
        self.assertIn("06016413", rendered)
        self.assertIn("06016414", rendered)

    def test_prerequisite_alternative_group_renders_or_semantics(self):
        claim = GroundedClaim(
            "prerequisite_alternative_001",
            "prerequisite",
            evidence={
                "alternative_courses": (
                    {"course_code": "06016413", "course_name": "NETWORK SYSTEMS"},
                    {"course_code": "06016414", "course_name": "SYSTEM ANALYSIS"},
                )
            },
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("อย่างน้อยหนึ่งวิชาจาก", rendered)
        self.assertIn("06016413 NETWORK SYSTEMS", rendered)
        self.assertIn("06016414 SYSTEM ANALYSIS", rendered)
        self.assertIn(" หรือ ", rendered)
        self.assertNotIn("ต้องเรียนวิชา 06016413", rendered)

    def test_prerequisite_multiple_required_entries_are_preserved(self):
        claim = GroundedClaim(
            "prerequisite_multiple_001",
            "prerequisite",
            evidence=(
                {"prerequisite_code": "06016413", "prerequisite_name_en": "NETWORK SYSTEMS"},
                {"prerequisite_code": "06016414", "prerequisite_name_en": "SYSTEM ANALYSIS"},
            ),
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("ต้องเรียนวิชา 06016413 NETWORK SYSTEMS มาก่อน", rendered)
        self.assertIn("ต้องเรียนวิชา 06016414 SYSTEM ANALYSIS มาก่อน", rendered)

    def test_empty_prerequisite_evidence_is_fail_closed(self):
        claim = GroundedClaim(
            "prerequisite_incomplete_001",
            "prerequisite",
            status="complete",
            evidence=(),
        )
        self.assertEqual(render_grounded_claim(claim), "หลักฐานไม่เพียงพอ")

    def test_valid_empty_prerequisite_explicitly_reports_no_requirement(self):
        claim = GroundedClaim(
            "prerequisite_empty_001",
            "prerequisite",
            status="valid_empty",
            evidence=(),
        )
        self.assertEqual(render_grounded_claim(claim), "ไม่มีวิชาบังคับก่อน")

    def test_prerequisite_rendering_keeps_result_provenance_unchanged(self):
        provenance = ({"source_page": 338, "program": "IT"},)
        claim = GroundedClaim(
            "prerequisite_provenance_001",
            "prerequisite",
            evidence=({"prerequisite_code": "06016413", "prerequisite_name_en": "NETWORK SYSTEMS"},),
            provenance=provenance,
        )
        result = GroundedAnswerResult(
            status="answer",
            answer_mode="deterministic",
            claims=(claim,),
            provenance=provenance,
        )

        rendered = render_grounded_answer(result)

        self.assertIn("06016413", rendered.final_answer)
        self.assertEqual(rendered.provenance, result.provenance)
        self.assertEqual(rendered.claims, result.claims)

    def test_identity_renders_english_name_without_internal_fields(self):
        claim = GroundedClaim(
            "identity_001",
            "identity",
            value=(
                {
                    "catalog_id": 7,
                    "course_id": 609,
                    "course_code": "06016401",
                    "name_en": "MATHEMATICS FOR INFORMATION TECHNOLOGY",
                    "name_en_variants": ["MATHEMATICS FOR INFORMATION TECHNOLOGY"],
                    "program": "IT",
                    "provenance": ({"provenance_id": 182, "source_page": 39},),
                },
            ),
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("06016401", rendered)
        self.assertIn("MATHEMATICS FOR INFORMATION TECHNOLOGY", rendered)
        self.assertNotIn("identity: [{", rendered)
        for field in (
            "course_id",
            "catalog_id",
            "provenance_id",
            "name_en_variants",
            "name_th_variants",
        ):
            self.assertNotIn(field, rendered)

    def test_identity_renders_thai_name(self):
        claim = GroundedClaim(
            "identity_th_001",
            "identity",
            value=({"course_code": "06016401", "name_th": "คณิตศาสตร์สำหรับเทคโนโลยีสารสนเทศ"},),
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("06016401", rendered)
        self.assertIn("คณิตศาสตร์สำหรับเทคโนโลยีสารสนเทศ", rendered)

    def test_identity_renders_both_names(self):
        claim = GroundedClaim(
            "identity_both_001",
            "identity",
            value=(
                {
                    "course_code": "06016401",
                    "name_th": "คณิตศาสตร์สำหรับเทคโนโลยีสารสนเทศ",
                    "name_en": "MATHEMATICS FOR INFORMATION TECHNOLOGY",
                },
            ),
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("06016401", rendered)
        self.assertIn("ชื่อภาษาไทย: คณิตศาสตร์สำหรับเทคโนโลยีสารสนเทศ", rendered)
        self.assertIn("ชื่อภาษาอังกฤษ: MATHEMATICS FOR INFORMATION TECHNOLOGY", rendered)

    def test_identity_with_multiple_distinct_codes_preserves_all(self):
        claim = GroundedClaim(
            "identity_multi_001",
            "identity",
            value=(
                {"course_code": "06016401", "name_en": "MATHEMATICS FOR INFORMATION TECHNOLOGY"},
                {"course_code": "06016402", "name_en": "INFORMATION TECHNOLOGY FUNDAMENTALS"},
            ),
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("06016401", rendered)
        self.assertIn("06016402", rendered)

    def test_incomplete_identity_is_fail_closed(self):
        empty_list = GroundedClaim("identity_empty_001", "identity", value=())
        self.assertEqual(render_grounded_claim(empty_list), "หลักฐานไม่เพียงพอ")

        missing_names = GroundedClaim(
            "identity_noname_001",
            "identity",
            value=({"course_id": 609, "program": "IT"},),
        )
        self.assertEqual(render_grounded_claim(missing_names), "หลักฐานไม่เพียงพอ")

    def test_identity_rendering_keeps_result_provenance_unchanged(self):
        provenance = ({"source_page": 39, "program": "IT"},)
        claim = GroundedClaim(
            "identity_provenance_001",
            "identity",
            value=({"course_code": "06016401", "name_en": "MATHEMATICS FOR INFORMATION TECHNOLOGY"},),
            provenance=provenance,
        )
        result = GroundedAnswerResult(
            status="answer",
            answer_mode="deterministic",
            claims=(claim,),
            provenance=provenance,
        )

        rendered = render_grounded_answer(result)

        self.assertIn("06016401", rendered.final_answer)
        self.assertEqual(rendered.provenance, result.provenance)
        self.assertEqual(rendered.claims, result.claims)

    def test_combined_identity_and_credits_preserves_both_claims(self):
        result = GroundedAnswerResult(
            status="answer",
            answer_mode="deterministic",
            claims=(
                GroundedClaim(
                    "identity_combined_001",
                    "identity",
                    value=({"course_code": "06016401", "name_en": "MATHEMATICS FOR INFORMATION TECHNOLOGY"},),
                ),
                GroundedClaim(
                    "credits_combined_001",
                    "sum_credits",
                    effective_scope={"plans": ("no_coop",)},
                    value=3,
                ),
            ),
        )

        rendered = render_grounded_answer(result).final_answer

        self.assertIn("06016401", rendered)
        self.assertIn("MATHEMATICS FOR INFORMATION TECHNOLOGY", rendered)
        self.assertIn("3", rendered)
        self.assertNotIn("identity: [{", rendered)

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

    def test_describe_claim_renders_grounded_text_without_typed_dump(self):
        claim = GroundedClaim(
            "describe_001",
            "describe",
            value=({"text": "DATABASE TECHNOLOGY"},),
            evidence=({"text": "DATABASE TECHNOLOGY"},),
        )

        rendered = render_grounded_claim(claim)

        self.assertEqual(rendered, "DATABASE TECHNOLOGY")
        self.assertNotIn('"text"', rendered)

    def test_describe_claim_preserves_all_grounded_text_items(self):
        claim = GroundedClaim(
            "describe_002",
            "describe",
            value=({"text": "DATA MANAGEMENT"}, {"text": "DATABASE TECHNOLOGY"}),
            evidence=({"text": "DATA MANAGEMENT"}, {"text": "DATABASE TECHNOLOGY"}),
        )

        rendered = render_grounded_claim(claim)

        self.assertIn("DATA MANAGEMENT", rendered)
        self.assertIn("DATABASE TECHNOLOGY", rendered)
        self.assertNotIn('"text"', rendered)

    def test_insufficient_describe_claim_remains_fail_closed(self):
        claim = GroundedClaim(
            "describe_003",
            "describe",
            status="insufficient_evidence",
            value=({"text": "UNSUPPORTED"},),
            evidence=(),
        )

        self.assertEqual(render_grounded_claim(claim), "หลักฐานไม่เพียงพอ")

    def _polish_result(self):
        claim = GroundedClaim(
            "polish_001",
            "placement",
            value={
                "program": "IT",
                "plan_key": "no_coop",
                "course_code": "06016420",
                "year_number": 2,
                "semester_number": 1,
                "credits": "3(2-2-5)",
            },
            provenance=({"program": "IT", "source_page": 35},),
        )
        return GroundedAnswerResult(
            "answer",
            "deterministic",
            claims=(claim,),
            provenance=claim.provenance,
        )

    def test_optional_polish_uses_deterministic_answer_without_model(self):
        result = render_grounded_answer(self._polish_result(), question="ช่วงเรียน")
        self.assertIn("06016420", result.final_answer)
        self.assertIn("ปี 2 ภาคเรียนที่ 1", result.final_answer)

    def test_optional_polish_falls_back_on_exception_or_empty_output(self):
        deterministic = render_grounded_answer(self._polish_result()).final_answer
        for model in (
            lambda prompt: (_ for _ in ()).throw(RuntimeError("offline")),
            lambda prompt: "   ",
        ):
            with self.subTest(model=model):
                result = render_grounded_answer(
                    self._polish_result(), model, question="ช่วงเรียน"
                )
                self.assertEqual(result.final_answer, deterministic)

    def test_valid_polish_is_accepted_and_prompt_contains_grounded_content(self):
        prompts = []

        def model(prompt):
            prompts.append(prompt)
            return "วิชา 06016420 แผน no_coop เรียนในปี 2 ภาคเรียนที่ 1 มี 3(2-2-5) หน่วยกิต"

        result = render_grounded_answer(
            self._polish_result(), model, question="วิชา 06016420 อยู่ช่วงไหน"
        )
        self.assertIn("วิชา 06016420", result.final_answer)
        self.assertIn("GROUNDED_CONTENT", prompts[0])
        self.assertIn("USER_QUESTION", prompts[0])

    def test_polish_that_drops_or_changes_critical_facts_is_rejected(self):
        outputs = (
            "วิชานี้เรียนในปี 2 ภาคเรียนที่ 1 มี 3(2-2-5) หน่วยกิต",
            "วิชา 06016420 แผน no_coop เรียนในปี 3 ภาคเรียนที่ 1 มี 3(2-2-5) หน่วยกิต",
            "วิชา 06016420 แผน no_coop เรียนในปี 2 ภาคเรียนที่ 1 มี 4(2-2-5) หน่วยกิต",
        )
        deterministic = render_grounded_answer(self._polish_result()).final_answer
        for output in outputs:
            with self.subTest(output=output):
                result = render_grounded_answer(
                    self._polish_result(), lambda prompt, output=output: output,
                    question="วิชา 06016420 อยู่ช่วงไหน",
                )
                self.assertEqual(result.final_answer, deterministic)

    def test_safe_year_and_semester_wording_is_accepted(self):
        result = render_grounded_answer(
            self._polish_result(),
            lambda prompt: "วิชา 06016420 หลักสูตร IT ไม่สหกิจ เรียนในปีที่ 2 เทอม 1 มี 3(2-2-5) หน่วยกิต",
            question="ช่วงเรียน",
        )
        self.assertIn("ปีที่ 2 เทอม 1", result.final_answer)

    def test_wrong_year_semester_or_plan_is_rejected(self):
        deterministic = render_grounded_answer(self._polish_result()).final_answer
        outputs = (
            "วิชา 06016420 หลักสูตร IT ไม่สหกิจ เรียนในปีที่ 3 เทอม 2 มี 3(2-2-5) หน่วยกิต",
            "วิชา 06016420 หลักสูตร IT เรียนในปีที่ 2 เทอม 1 มี 3(2-2-5) หน่วยกิต",
            "วิชา 06016420 หลักสูตร IT สหกิจ เรียนในปีที่ 2 เทอม 1 มี 3(2-2-5) หน่วยกิต",
        )
        for output in outputs:
            with self.subTest(output=output):
                result = render_grounded_answer(
                    self._polish_result(), lambda prompt, output=output: output,
                    question="ช่วงเรียน",
                )
                self.assertEqual(result.final_answer, deterministic)

    def test_plan_and_description_paraphrase_are_guarded_separately(self):
        result = render_grounded_answer(
            self._polish_result(),
            lambda prompt: "วิชา 06016420 ในหลักสูตร IT แผนไม่สหกิจ เรียนปีที่ 2 เทอม 1 มี 3(2-2-5) หน่วยกิต และกล่าวถึงโครงสร้างพื้นฐาน",
            question="ชื่อ ช่วงเรียน และเนื้อหา",
        )
        self.assertIn("แผนไม่สหกิจ", result.final_answer)
        self.assertIn("โครงสร้างพื้นฐาน", result.final_answer)

    def test_lowercase_english_it_does_not_satisfy_program_guard(self):
        deterministic = render_grounded_answer(self._polish_result()).final_answer
        result = render_grounded_answer(
            self._polish_result(),
            lambda prompt: "วิชา 06016420 it แผนไม่สหกิจ เรียนปีที่ 2 เทอม 1 มี 3(2-2-5) หน่วยกิต",
            question="ช่วงเรียน",
        )
        self.assertEqual(result.final_answer, deterministic)

    def test_closed_semester_credit_and_spaced_plan_variants_are_accepted(self):
        result = render_grounded_answer(
            self._polish_result(),
            lambda prompt: "วิชา 06016420 หลักสูตร IT ไม่ สหกิจ เรียนปีที่ 2 เทอมที่ 1 มี 3 (2-2-5) หน่วยกิต",
            question="ช่วงเรียน",
        )
        self.assertIn("ไม่ สหกิจ", result.final_answer)

    def test_wrong_semester_and_credit_structure_are_rejected(self):
        deterministic = render_grounded_answer(self._polish_result()).final_answer
        outputs = (
            "วิชา 06016420 หลักสูตร IT ไม่ สหกิจ เรียนปีที่ 2 เทอมที่ 2 มี 3 (2-2-5) หน่วยกิต",
            "วิชา 06016420 หลักสูตร IT ไม่ สหกิจ เรียนปีที่ 2 เทอมที่ 1 มี 3 (3-0-6) หน่วยกิต",
        )
        for output in outputs:
            with self.subTest(output=output):
                result = render_grounded_answer(
                    self._polish_result(), lambda prompt, output=output: output,
                    question="ช่วงเรียน",
                )
                self.assertEqual(result.final_answer, deterministic)

    def test_plan_canonicalization_distinguishes_spaced_no_coop_from_coop(self):
        self.assertIn("plan:no_coop", _critical_facts("แผนไม่สหกิจ"))
        self.assertIn("plan:no_coop", _critical_facts("แผนไม่ สหกิจ"))
        self.assertNotIn("plan:coop", _critical_facts("แผนไม่ สหกิจ"))
        self.assertEqual(_critical_facts("แผนสหกิจ"), ("plan:coop",))

    def test_polish_preserves_provenance_and_works_for_hybrid_content(self):
        result = self._polish_result()
        result = GroundedAnswerResult(
            result.status,
            result.answer_mode,
            claims=(
                result.claims[0],
                GroundedClaim(
                    "polish_desc",
                    "describe",
                    value=({"text": "DATABASE TECHNOLOGY"},),
                    evidence=({"text": "DATABASE TECHNOLOGY"},),
                ),
            ),
            provenance=result.provenance,
        )
        polished = "วิชา 06016420 ในหลักสูตร IT แผน no_coop เรียนในปี 2 ภาคเรียนที่ 1 มี 3(2-2-5) หน่วยกิต และเรียน DATABASE TECHNOLOGY"
        rendered = render_grounded_answer(
            result, lambda prompt: polished, question="ชื่อและช่วงเรียน"
        )
        self.assertEqual(rendered.final_answer, polished)
        self.assertEqual(rendered.provenance, result.provenance)

    def test_blocked_result_is_not_polished(self):
        result = GroundedAnswerResult("no_data", "deterministic", "ไม่พบข้อมูลนี้ในเล่มหลักสูตร")
        calls = []
        rendered = render_grounded_answer(
            result, lambda prompt: calls.append(prompt) or "เปลี่ยนข้อความ", question="ถาม"
        )
        self.assertEqual(rendered.final_answer, result.final_answer)
        self.assertEqual(calls, [])

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
