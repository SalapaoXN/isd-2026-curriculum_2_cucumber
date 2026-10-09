"""G5-C provider-free sequence, canonical ownership and atomicity gates."""

from dataclasses import replace
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.semantic import executor
from rag.semantic.answerer import render_semantic_answer
from rag.semantic.compiler import compile_resolved_intent_to_query_spec
from rag.semantic.context import MergedContext
from rag.semantic.interpreter import SemanticSchemaError, semantic_intent_json_schema
from rag.semantic.pipeline import semantic_answer
from rag.semantic.planner import plan_semantic_query
from rag.semantic.prompts import build_semantic_interpreter_prompt
from rag.semantic.resolver import resolve_semantic_intent, _resolve_ordinal
from rag.semantic.validation import validate_semantic_intent
from tests.rag.test_semantic_course_set_contract import payload, parse
from tests.rag.test_semantic_course_set_resolution import DB

CODES = ("06016413", "06016420", "06016421")
FIELDS = ("placement_sequence", "placement", "prerequisites")
CONTEXT = {"program": "IT", "catalog_key": "it-2565", "plan": "no_coop"}
QUESTION = ("ถ้าจะวางแผนเรียนสาย infrastructure ใน IT แบบไม่สหกิจ "
            "ควรเรียง INTRODUCTION TO NETWORK SYSTEMS (06016413), "
            "INFRASTRUCTURE SYSTEMS AND SERVICES (06016420) "
            "และ INFORMATION TECHNOLOGY INFRASTRUCTURE SECURITY (06016421) "
            "ตามปี/เทอมอย่างไร และแต่ละวิชาต้องผ่านวิชาอะไรมาก่อน?")


def resolve(codes=CODES, fields=FIELDS, context=None):
    return resolve_semantic_intent(DB, parse(payload(codes, fields)),
                                  MergedContext(**(context or CONTEXT)))


def pipeline(codes=CODES, fields=FIELDS, question=None, context=None):
    question = question or "Chronological placement sequence by year/semester: " + " ".join(codes)
    public = semantic_answer(DB, question, context or CONTEXT,
                           interpret_callable=lambda _prompt: json.dumps(payload(codes, fields)),
                           answer_callable=lambda _prompt: (_ for _ in ()).throw(AssertionError("provider forbidden")))
    verified = executor.execute_explicit_course_set(DB, resolve(codes, fields, context), question)
    return SimpleNamespace(result=verified, answer=public.result.final_answer, public=public, trace=public.trace)


class PlacementSequenceTests(unittest.TestCase):
    def successful(self, **kwargs):
        answer = pipeline(**kwargs)
        self.assertEqual(answer.result.status, "answer", answer.trace.failure_reason)
        self.assertEqual(answer.public.result.status, "answer", answer.trace.failure_reason)
        return answer

    def fault(self, mutate):
        original = executor.execute_deterministic
        calls = []
        def evidence(db, spec, context, question):
            calls.append(spec.course_codes)
            result = original(db, spec, context, question)
            return mutate(result) if spec.course_codes == ("06016420",) else result
        with patch.object(executor, "execute_deterministic", side_effect=evidence):
            answer = pipeline()
        self.assertIn(("06016420",), calls, "must reach the evidence boundary")
        return answer

    def assert_atomic_failure(self, answer):
        self.assertNotEqual(answer.result.status, "answer")
        self.assertEqual(answer.result.claims, ())
        self.assertEqual(answer.result.scoped_results, ())
        self.assertEqual(answer.result.result_courses, ())
        self.assertEqual(answer.result.summary_facts, ())
        self.assertNotEqual(answer.public.result.status, "answer")

    def test_sequence_executable_and_compiled_without_language_parsing(self):
        resolved = resolve()
        self.assertEqual(plan_semantic_query(resolved).execution, "deterministic")
        spec = compile_resolved_intent_to_query_spec(resolved, QUESTION)
        self.assertEqual(spec.course_codes, CODES)
        self.assertEqual(set(spec.operations), {"placement", "prerequisite"})
        self.assertEqual(spec.normalized_question, "")

    def test_primary_acceptance_all_six_facts_visible(self):
        answer = self.successful(question=QUESTION)
        for code, term in zip(CODES, ((2, 1), (2, 2), (3, 1))):
            part = next(p for p in answer.result.scoped_results if p.course_code == code)
            text = " ".join(part.summary_facts)
            self.assertIn(f"ชั้นปีที่ {term[0]} ภาคการศึกษาที่ {term[1]}", text)
            self.assertIn("ไม่มีวิชาบังคับก่อน" if code == CODES[0] else "มีวิชาบังคับก่อน: 06016413", text)
            for fact in part.summary_facts:
                self.assertIn(fact, answer.answer)

    def test_three_identities_and_independent_resolution(self):
        resolved = resolve()
        self.assertFalse(resolved.needs_clarification)
        self.assertEqual(tuple(m.course_code for m in resolved.target.members), CODES)
        self.assertTrue(all(m.course_name and (m.program, m.catalog_key) == ("IT", "it-2565")
                            for m in resolved.target.members))
        self.assertEqual(tuple(c["course_code"] for c in self.successful().result.result_courses), CODES)

    def test_same_program_catalog_plan_for_every_member(self):
        for part in self.successful().result.scoped_results:
            self.assertEqual((part.scope.program, part.scope.catalog_key, part.scope.plan),
                             ("IT", "it-2565", "no_coop"))

    def test_order_uses_canonical_terms_not_first_mention(self):
        answer = self.successful(codes=(CODES[2], CODES[0], CODES[1]))
        self.assertEqual(tuple(p.course_code for p in answer.result.scoped_results), CODES)
        self.assertEqual(tuple(c["course_code"] for c in answer.result.result_courses), CODES)
        self.assertEqual(_resolve_ordinal(2, answer.result.result_courses)["course_code"], CODES[1])
        self.assertLess(answer.answer.index(CODES[0]), answer.answer.index(CODES[1]))
        self.assertLess(answer.answer.index(CODES[1]), answer.answer.index(CODES[2]))

    def test_prerequisites_owned_by_targets_without_fake_dependency(self):
        parts = self.successful().result.scoped_results
        for part in parts[1:]:
            claim = next(c for c in part.claims if c.operation == "prerequisite")
            self.assertEqual(tuple(r["prerequisite_code"] for r in claim.value), (CODES[0],))
            self.assertEqual(claim.effective_scope.course_targets[0]["course_code"], part.course_code)

    def test_explicit_no_prerequisite_preserved(self):
        part = self.successful().result.scoped_results[0]
        claim = next(c for c in part.claims if c.operation == "prerequisite")
        self.assertEqual(claim.value[0]["prerequisite_state"], "explicit_none")
        self.assertIn("ไม่มีวิชาบังคับก่อน", " ".join(part.summary_facts))

    def test_provenance_for_every_member_and_fact(self):
        result = self.successful().result
        for part in result.scoped_results:
            self.assertTrue(part.provenance)
            for claim in part.claims:
                self.assertTrue(claim.provenance)
                self.assertTrue(all(row.get("provenance") for row in claim.value))
                self.assertTrue(all(ref in result.provenance for ref in claim.provenance))

    def test_missing_placement_blocks_whole_sequence(self):
        self.assert_atomic_failure(self.fault(lambda r: replace(r, claims=tuple(c for c in r.claims if c.operation != "placement"))))

    def test_missing_prerequisite_blocks_whole_sequence(self):
        self.assert_atomic_failure(self.fault(lambda r: replace(r, claims=tuple(c for c in r.claims if c.operation != "prerequisite"))))

    def test_unresolved_member_blocks_all_execution(self):
        with patch.object(executor, "execute_deterministic") as evidence:
            answer = pipeline(codes=(CODES[0], CODES[1], "06019999"))
        self.assert_atomic_failure(answer)
        evidence.assert_not_called()

    def test_ambiguous_multi_placement_blocks_sequence(self):
        self.assert_atomic_failure(pipeline(codes=("06016481", "06016482")))

    def mutate_rows(self, operation, changes):
        def mutate(result):
            return replace(result, claims=tuple(replace(c, value=tuple(dict(row, **changes) for row in c.value))
                           if c.operation == operation else c for c in result.claims))
        return mutate

    def test_plan_row_leakage_rejected(self):
        self.assert_atomic_failure(self.fault(self.mutate_rows("placement", {"plan_key": "coop"})))

    def test_catalog_row_leakage_rejected(self):
        self.assert_atomic_failure(self.fault(self.mutate_rows("placement", {"catalog_id": -1})))

    def test_program_row_leakage_rejected(self):
        self.assert_atomic_failure(self.fault(self.mutate_rows("placement", {"program": "DSBA"})))

    def test_claim_scope_leakage_rejected(self):
        def mutate(result):
            return replace(result, claims=tuple(replace(c, effective_scope=replace(c.effective_scope, catalog_key="it-2560"))
                                                for c in result.claims))
        self.assert_atomic_failure(self.fault(mutate))

    def test_wrong_placement_owner_rejected(self):
        self.assert_atomic_failure(self.fault(self.mutate_rows("placement", {"course_code": CODES[2]})))

    def test_wrong_prerequisite_owner_rejected(self):
        self.assert_atomic_failure(self.fault(self.mutate_rows("prerequisite", {"course_id": -1})))

    def test_missing_fact_sources_rejected(self):
        self.assert_atomic_failure(self.fault(self.mutate_rows("prerequisite", {"provenance": ()})))

    def test_empty_prerequisite_is_not_explicit_absence(self):
        def mutate(result):
            return replace(result, claims=tuple(replace(c, value=()) if c.operation == "prerequisite" else c for c in result.claims))
        self.assert_atomic_failure(self.fault(mutate))

    def test_same_term_tie_does_not_invent_order(self):
        self.assert_atomic_failure(self.fault(self.mutate_rows("placement", {
            "year_number": 2, "semester_number": 1, "year_semester_choices": ((2, 1),)})))

    def test_multi_placement_overlap_with_next_member_fails(self):
        answer = self.fault(self.mutate_rows("placement", {
            "year_number": None, "semester_number": None, "year_semester_choices": ((2, 2), (3, 1))}))
        # This overlaps the last member; no earliest choice may rescue the order.
        self.assert_atomic_failure(answer)

    def test_multi_placement_overlap_with_previous_member_fails(self):
        answer = self.fault(self.mutate_rows("placement", {
            "year_number": None, "semester_number": None, "year_semester_choices": ((2, 2), (2, 1))}))
        self.assert_atomic_failure(answer)  # Overlap with the first member is also ambiguous.

    def test_disjoint_multi_placement_retains_all_choices(self):
        original = executor.execute_deterministic
        def evidence(db, spec, context, question):
            result = original(db, spec, context, question)
            if spec.course_codes == (CODES[0],):
                return self.mutate_rows("placement", {"year_number": None, "semester_number": None,
                    "year_semester_choices": ((1, 2), (2, 1))})(result)
            return result
        with patch.object(executor, "execute_deterministic", side_effect=evidence):
            answer = self.successful()
        self.assertEqual(tuple(p.course_code for p in answer.result.scoped_results), CODES)
        text = " ".join(answer.result.scoped_results[0].summary_facts)
        self.assertIn("ชั้นปีที่ 1 ภาคการศึกษาที่ 2", text)
        self.assertIn("ชั้นปีที่ 2 ภาคการศึกษาที่ 1", text)

    def test_explicit_course_omission_rejected(self):
        self.assertFalse(validate_semantic_intent(parse(payload(CODES[:2], FIELDS)), QUESTION).valid)

    def test_literal_set_alias_dedupe_remains_atomic(self):
        answer = self.successful(codes=(CODES[0], "INTRODUCTION TO NETWORK SYSTEMS", CODES[1], CODES[2]))
        self.assertEqual(tuple(p.course_code for p in answer.result.scoped_results), CODES)

    def test_duplicate_aliases_cannot_make_two_course_sequence(self):
        self.assert_atomic_failure(pipeline(codes=(CODES[0], "INTRODUCTION TO NETWORK SYSTEMS")))

    def test_sequence_requires_grounded_placement_order_request(self):
        intent = parse(payload(CODES, FIELDS))
        for question in ("credits of " + " ".join(CODES), "placement of " + " ".join(CODES)):
            with self.subTest(question=question):
                self.assertFalse(validate_semantic_intent(intent, question).valid)

    def test_thai_sequence_language_grounded(self):
        for phrase in ("ควรเรียง ตามปี/เทอมอย่างไร", "เรียนอะไรก่อนหลัง", "ลำดับการเรียน", "วางแผนเรียน ตามปี/เทอม"):
            with self.subTest(phrase=phrase):
                self.assertTrue(validate_semantic_intent(parse(payload(CODES, FIELDS)), phrase + " " + " ".join(CODES)).valid)

    def test_sequence_only_still_requires_placement_evidence(self):
        answer = self.successful(fields=("placement_sequence",))
        self.assertEqual(len(answer.result.scoped_results), 3)
        self.assertTrue(all(c.operation == "placement" for c in answer.result.claims))

    def test_invalid_canonical_plan_fails_closed(self):
        self.assert_atomic_failure(pipeline(context=dict(CONTEXT, plan="unknown_plan")))

    def test_missing_scope_fails_closed(self):
        resolved = resolve()
        for changes in ({"program": None}, {"catalog_key": None}, {"plan": None}):
            with self.subTest(changes=changes):
                bad = replace(resolved, scope=replace(resolved.scope, **changes))
                self.assertNotEqual(executor.execute_explicit_course_set(DB, bad, "").status, "answer")

    def test_strict_unknown_fields_and_existing_request_schema(self):
        schema = semantic_intent_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("placement_sequence", schema["properties"]["requested_fields"]["items"]["enum"])
        for location in ("root", "target", "member"):
            data = payload(CODES, FIELDS)
            target = data if location == "root" else data["target"] if location == "target" else data["target"]["members"][0]
            target["sequence_order"] = []
            with self.subTest(location=location), self.assertRaises(SemanticSchemaError):
                parse(data)

    def test_prompt_teaches_generic_sequence_without_teacher_fixture(self):
        prompt = build_semantic_interpreter_prompt("generic question")
        self.assertIn("CHRONOLOGICAL PLACEMENT SEQUENCE", prompt)
        self.assertNotIn("unsupported until sequence execution", prompt)
        self.assertNotIn("06016413", prompt)

    def test_presentation_bounded_and_provider_free(self):
        result = self.successful().result
        answer, mode = render_semantic_answer(QUESTION, result,
            lambda _prompt: self.fail("sequence answer must be deterministic"))
        self.assertEqual(mode, "deterministic")
        self.assertTrue(answer.startswith("1. 06016413"))
        self.assertIn("2. 06016420", answer)
        self.assertIn("3. 06016421", answer)


if __name__ == "__main__":
    unittest.main()
