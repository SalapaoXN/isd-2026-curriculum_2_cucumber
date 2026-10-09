"""G2 regressions for preserving verified detail through presentation."""

import unittest
from types import SimpleNamespace

from rag.semantic.answerer import (
    MAX_ANSWER_LEN,
    render_semantic_answer,
    render_verified_fallback,
    validate_answer_text,
)
from rag.semantic.answerer import _displayed_list_pairs
from rag.semantic.executor import (
    _claim_line,
    _collection_course_codes,
    _collection_course_identities,
)
from rag.semantic.schema import VerifiedResult
from rag.hybrid_demo import DEFAULT_CURRICULUM_DB_PATH
from rag.query_spec import QuerySpec
from rag.resolution import QueryContext
from rag.semantic.executor import execute_deterministic


def _verified(claim, fact):
    return VerifiedResult(
        status="answer",
        claims=(claim,),
        summary_facts=(fact,),
        provenance=({"source_filename": "canonical-source"},),
    )


class SemanticPresentationCompletenessTests(unittest.TestCase):
    def test_authoritative_course_credit_components_reach_verified_summary(self):
        for code, raw in (
            ("06016401", "3(3-0-6)"),
            ("06016420", "3(2-2-5)"),
            ("06016406", "3(0-9-0)"),
        ):
            spec = QuerySpec(
                original_question="", normalized_question="", program="IT",
                plans=("no_coop",), years=(), semesters=(), course_codes=(code,),
                course_name=None, category=None, topic=None,
                operations=("identity", "sum_credits"), group_by=(),
                judgement="none", credit_units=None,
                references_previous_result_set=False, result_ordinal=None,
            )
            verified = execute_deterministic(
                DEFAULT_CURRICULUM_DB_PATH, spec,
                QueryContext(program="IT", catalog_key="it-2565", plan="no_coop"), "",
            )
            credit_claim = next(c for c in verified.claims if c.operation == "sum_credits")
            self.assertEqual(credit_claim.value, 3)
            self.assertEqual(credit_claim.evidence.components[0]["credits_raw"], raw)
            self.assertIn(raw, "\n".join(verified.summary_facts))

    def test_real_multiplan_placement_sets_survive_rendering(self):
        spec = QuerySpec(
            original_question="", normalized_question="", program="IT",
            plans=("coop", "no_coop"), years=(), semesters=(),
            course_codes=("06016465",), course_name=None, category=None, topic=None,
            operations=("placement",), group_by=(), judgement="none",
            credit_units=None, references_previous_result_set=False, result_ordinal=None,
        )
        verified = execute_deterministic(
            DEFAULT_CURRICULUM_DB_PATH, spec,
            QueryContext(program="IT", catalog_key="it-2565"), "",
        )
        placement_claims = [c for c in verified.claims if c.operation == "placement"]
        self.assertEqual({row["plan_key"] for c in placement_claims for row in c.value}, {"coop", "no_coop"})
        answer, mode = render_semantic_answer("earliest plan", verified, lambda _p: "06016465")
        self.assertEqual(mode, "deterministic")
        self.assertIn("แผนสหกิจ", answer)
        self.assertIn("แผนไม่สหกิจ", answer)
        self.assertIn("ชั้นปีที่ 4 ภาคการศึกษาที่ 1", answer)
        self.assertIn("ชั้นปีที่ 3 ภาคการศึกษาที่ 1", answer)


    def test_credit_claim_keeps_authoritative_raw_structure_and_numeric_value(self):
        claim = SimpleNamespace(
            operation="sum_credits", status="complete", value=3,
            evidence=SimpleNamespace(components=({"credits_raw": "3(2-2-5)"},)),
        )
        fact = _claim_line("sum_credits", claim.value)
        # Simulate component-backed canonical notation at the projection seam.
        from rag.semantic.executor import _claim_line_with_evidence
        fact = _claim_line_with_evidence(claim)
        self.assertIn("3(2-2-5)", fact)
        self.assertIn("3 หน่วยกิต", fact)
        self.assertEqual(claim.value, 3)
        verified = _verified(claim, "06016420 INFRASTRUCTURE SYSTEMS AND SERVICES: " + fact)
        self.assertTrue(validate_answer_text("06016420 INFRASTRUCTURE SYSTEMS AND SERVICES 3(2-2-5)", verified))
        self.assertFalse(validate_answer_text("06016420 INFRASTRUCTURE SYSTEMS AND SERVICES 3 หน่วยกิต", verified))
        self.assertFalse(validate_answer_text("06016420 INFRASTRUCTURE SYSTEMS AND SERVICES 3(3-0-6)", verified))

    def test_absent_canonical_credit_raw_is_not_reconstructed(self):
        claim = SimpleNamespace(
            operation="sum_credits", status="complete", value=3,
            evidence=SimpleNamespace(components=({"credit_units": 3},)),
        )
        from rag.semantic.executor import _claim_line_with_evidence
        self.assertNotIn("3(0-0-0)", _claim_line_with_evidence(claim))

    def test_canonical_display_placeholder_is_distinct_from_course_identity(self):
        row = {"course_code": "90644xxx", "name_th": "วิชาเลือก"}
        claim = SimpleNamespace(operation="list", status="complete", value=(row,))
        line = _claim_line("list", (row,))
        self.assertIn("90644xxx", line)
        self.assertEqual(_collection_course_identities((row,)), [("90644xxx", "วิชาเลือก")])
        # Display references never become exact course identities.
        from rag.semantic.executor import _collection_course_codes
        self.assertEqual(_collection_course_codes((row,)), [])
        verified = _verified(claim, line)
        self.assertTrue(validate_answer_text(line, verified))

    def test_eleven_item_collection_displays_every_reference_including_last_placeholder(self):
        rows = [
            {"course_code": f"{10000000 + index:08d}", "name_th": f"วิชา {index}"}
            for index in range(10)
        ] + [{"course_code": "90644xxx", "name_th": "ช่องเลือก"}]
        claim = SimpleNamespace(operation="list", status="complete", value=tuple(rows))
        fact = _claim_line("list", rows)
        verified = _verified(claim, fact)

        self.assertEqual(len(_displayed_list_pairs(verified)), 11)
        answer, _ = render_semantic_answer("list", verified, None)
        self.assertTrue(all(row["course_code"] in answer for row in rows))
        self.assertIn("90644xxx", answer)
        self.assertNotIn("แสดง 10 จาก 11", answer)
        self.assertTrue(validate_answer_text(answer, verified))

    def test_list_validation_rejects_omission_from_authoritative_collection(self):
        rows = [
            {"course_code": f"{10000000 + index:08d}", "name_th": f"วิชา {index}"}
            for index in range(10)
        ] + [{"course_code": "90644xxx", "name_th": "ช่องเลือก"}]
        claim = SimpleNamespace(operation="list", status="complete", value=tuple(rows))
        verified = _verified(claim, _claim_line("list", rows))
        omitted = "\n".join(
            f"- {row['course_code']} — {row['name_th']}" for row in rows[:-1]
        )
        self.assertFalse(validate_answer_text(omitted, verified))

    def test_small_collection_output_is_unchanged(self):
        rows = [
            {"course_code": "10000001", "name_th": "วิชาหนึ่ง"},
            {"course_code": "10000002", "name_th": "วิชาสอง"},
        ]
        self.assertEqual(
            _claim_line("list", rows),
            "รายวิชาที่พบ:\n- 10000001 — วิชาหนึ่ง\n- 10000002 — วิชาสอง",
        )

    def test_display_bound_and_answer_length_remain_bounded(self):
        rows = [
            {"course_code": f"{10000000 + index:08d}", "name_th": f"วิชา {index}"}
            for index in range(25)
        ]
        claim = SimpleNamespace(operation="list", status="complete", value=tuple(rows))
        fact = _claim_line("list", rows)
        verified = _verified(claim, fact)
        self.assertIn("แสดง 20 จาก 25", fact)
        answer = render_verified_fallback("list", verified)
        self.assertLessEqual(len(answer), MAX_ANSWER_LEN)

    def test_placeholder_remains_rejected_by_exact_code_collection(self):
        self.assertEqual(_collection_course_codes({"course_code": "90644xxx"}), [])

    def test_all_verified_multiplan_placement_choices_survive_fallback(self):
        value = (
            {"course_code": "06016465", "plan_key": "coop", "year_semester_choices": ((3, 2),)},
            {"course_code": "06016465", "plan_key": "no_coop", "year_semester_choices": ((3, 1),)},
        )
        claim = SimpleNamespace(operation="placement", status="complete", value=value)
        fact = _claim_line("placement", value)
        verified = _verified(claim, fact)
        answer = render_verified_fallback("", verified)
        self.assertIn("ชั้นปีที่ 3 ภาคการศึกษาที่ 2", answer)
        self.assertIn("ชั้นปีที่ 3 ภาคการศึกษาที่ 1", answer)

    def test_description_omission_rejected_and_fallback_keeps_verified_detail(self):
        text = "Canonical course description includes network security architecture and threat analysis."
        claim = SimpleNamespace(
            operation="describe", status="complete",
            value={"course_code": "06016421", "name_en": "INFORMATION TECHNOLOGY INFRASTRUCTURE SECURITY"},
            evidence=({"chunk_type": "description", "entity_type": "course", "text": text},),
        )
        fact = "06016421 INFORMATION TECHNOLOGY INFRASTRUCTURE SECURITY: " + text
        verified = _verified(claim, fact)
        answer, mode = render_semantic_answer("description", verified, lambda _p: "06016421 INFORMATION TECHNOLOGY INFRASTRUCTURE SECURITY")
        self.assertEqual(mode, "deterministic")
        self.assertIn(text, answer)
        self.assertFalse(validate_answer_text("06016421 INFORMATION TECHNOLOGY INFRASTRUCTURE SECURITY", verified))

    def test_factual_hallucination_guard_stays_fail_closed(self):
        claim = SimpleNamespace(operation="sum_credits", status="complete", value=3, evidence=())
        verified = _verified(claim, "06016420 INFRASTRUCTURE SYSTEMS AND SERVICES: 3 หน่วยกิต")
        self.assertFalse(validate_answer_text("06016420 INFRASTRUCTURE SYSTEMS AND SERVICES มี 99 หน่วยกิต", verified))


if __name__ == "__main__":
    unittest.main()
