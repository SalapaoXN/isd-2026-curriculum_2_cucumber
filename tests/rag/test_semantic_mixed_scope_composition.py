"""G5-A provider-free mixed course/term contract and atomic evidence gate."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.semantic import compiler, executor
from rag.semantic.answerer import render_semantic_answer
from rag.semantic.context import MergedContext
from rag.semantic.interpreter import (
    interpret_semantic_intent, parse_semantic_intent_payload,
    semantic_intent_json_schema, SemanticSchemaError,
)
from rag.semantic.pipeline import semantic_answer
from rag.semantic.modes import semantic_ask_response
from rag.semantic.planner import plan_semantic_query
from rag.semantic.resolver import resolve_semantic_intent
from rag.semantic.schema import VerifiedResult
from rag.semantic.validation import validate_semantic_intent


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"
QUESTION = "IT no_coop year 2 semester 2: prerequisites and placement of 06016420, and term total credits?"


def payload(fields=("prerequisites", "placement"), relation="prerequisite"):
    return {
        "task": "compose", "subject": "course", "relation": relation,
        "target": {"kind": "literal", "raw_text": "06016420", "normalized_hint": None, "ordinal": None},
        "scope": {"program": "IT", "catalog": None, "plan": "no_coop", "plan_hint": "no_coop", "year": 2, "semester": 2},
        "filters": [], "aggregation": {"function": "sum", "measure": "credits", "group_by": []},
        "ranking": None, "comparison": None, "requested_fields": list(fields),
        "clarification": None, "policy_topic": None, "observed_value": None,
    }


def parse(data):
    return parse_semantic_intent_payload(json.dumps(data))


def resolve(data=None):
    return resolve_semantic_intent(DB, parse(data or payload()),
                                   MergedContext(program="IT", catalog_key="it-2565", plan="no_coop", years=(2,), semesters=(2,)))


class MixedScopeContractTests(unittest.TestCase):
    def test_prerequisite_plus_term_total_survives_single_interpreter_call(self):
        calls = []
        def stub(prompt):
            calls.append(prompt)
            return json.dumps(payload())
        intent, _ = interpret_semantic_intent(QUESTION, stub)
        self.assertEqual(len(calls), 1)
        self.assertEqual(intent.target.raw_text, "06016420")
        self.assertEqual(intent.requested_fields, ("prerequisites", "placement"))
        self.assertEqual((intent.aggregation.function, intent.aggregation.measure), ("sum", "credits"))
        self.assertTrue(validate_semantic_intent(intent, QUESTION).valid)

    def test_placement_plus_term_total(self):
        intent = parse(payload(("placement",), "placement"))
        self.assertTrue(validate_semantic_intent(intent, QUESTION).valid)

    def test_course_credits_and_term_total_have_distinct_consumers(self):
        resolved = resolve(payload(("credits",), "credits"))
        requests = compiler.compile_mixed_scope_requests(resolved, QUESTION)
        self.assertEqual([r.scope_kind for r in requests], ["course", "term"])
        self.assertEqual(requests[0].spec.course_codes, ("06016420",))
        self.assertEqual(requests[0].scope.years, ())
        self.assertEqual(requests[1].spec.course_codes, ())
        self.assertEqual(requests[1].scope.years, (2,))
        self.assertEqual(requests[1].scope.semesters, (2,))
        self.assertIn("sum_credits", requests[0].spec.operations)
        self.assertEqual(requests[1].spec.operations, ("sum_credits",))

    def test_compiler_preserves_g1_union_and_dedupe(self):
        requests = compiler.compile_mixed_scope_requests(resolve(payload(("placement", "credits", "prerequisites", "credits"))), QUESTION)
        self.assertEqual(requests[0].spec.operations, ("prerequisite", "placement", "sum_credits"))
        self.assertEqual(requests[1].spec.operations, ("sum_credits",))

    def test_old_single_spec_compiler_cannot_discard_aggregate(self):
        with self.assertRaisesRegex(ValueError, "mixed.scope"):
            compiler.compile_resolved_intent_to_query_spec(resolve(), QUESTION)

    def test_omitted_either_scope_rejects(self):
        for field, value in (("aggregation", None), ("target", {"kind": "none", "raw_text": None, "normalized_hint": None, "ordinal": None}),
                             ("requested_fields", []), ("relation", None)):
            data = payload()
            data[field] = value
            with self.subTest(field=field):
                self.assertFalse(validate_semantic_intent(parse(data), QUESTION).valid)

    def test_unsupported_aggregate_shapes_reject(self):
        for key, value in (("function", "average"), ("measure", "course_count"), ("group_by", ["semester"])):
            data = payload()
            data["aggregation"][key] = value
            with self.subTest(key=key):
                self.assertFalse(validate_semantic_intent(parse(data), QUESTION).valid)

    def test_no_filters_comparison_or_ranking_can_be_dropped(self):
        data = payload()
        data["filters"] = [{"field": "topic", "operator": "related_to", "value": "infrastructure"}]
        self.assertFalse(validate_semantic_intent(parse(data), QUESTION + " infrastructure").valid)

    def test_lookup_cannot_silently_ignore_an_aggregate(self):
        data = payload()
        data["task"] = "lookup"
        self.assertFalse(validate_semantic_intent(parse(data), QUESTION).valid)
        with self.assertRaisesRegex(ValueError, "mixed.scope"):
            compiler.compile_resolved_intent_to_query_spec(resolve(data), QUESTION)

    def test_literal_set_cannot_be_mixed_implicitly(self):
        from tests.rag.test_semantic_course_set_contract import payload as set_payload
        data = set_payload()
        data.update(task="compose", aggregation={"function": "sum", "measure": "credits", "group_by": []})
        self.assertFalse(validate_semantic_intent(parse(data), "06016481 and 06016482").valid)

    def test_missing_resolved_term_dimension_blocks_compiler(self):
        resolved = resolve()
        for dimension, value in (("plan", None), ("years", ()), ("semesters", ())):
            with self.subTest(dimension=dimension):
                missing = replace(resolved, scope=replace(resolved.scope, **{dimension: value}))
                self.assertEqual(plan_semantic_query(missing).execution, "unsupported")
                with self.assertRaises(ValueError):
                    compiler.compile_mixed_scope_requests(missing, QUESTION)

    def test_strict_schema_stays_closed(self):
        schema = semantic_intent_json_schema()
        self.assertIn("compose", schema["properties"]["task"]["enum"])
        self.assertFalse(schema["additionalProperties"])
        data = payload()
        data["requests"] = "free form extra state"
        with self.assertRaises(SemanticSchemaError):
            parse(data)

    def test_legacy_g1_identity_credits_unchanged(self):
        data = payload(("name", "credits"), "identity")
        data.update(task="lookup", aggregation=None)
        spec = compiler.compile_resolved_intent_to_query_spec(resolve(data), QUESTION)
        self.assertEqual(spec.operations, ("identity", "sum_credits"))
        self.assertEqual(spec.course_codes, ("06016420",))

    def test_legacy_placement_credits_unchanged(self):
        data = payload(("placement", "credits"), "placement")
        data.update(task="lookup", aggregation=None)
        self.assertEqual(compiler.compile_resolved_intent_to_query_spec(resolve(data), QUESTION).operations, ("placement", "sum_credits"))


class MixedScopeEvidenceTests(unittest.TestCase):
    def execute(self, data=None):
        return executor.execute_mixed_scope(DB, resolve(data), QUESTION)

    def test_both_scopes_have_typed_evidence_and_provenance(self):
        result = self.execute()
        self.assertEqual(result.status, "answer", result.missing_information)
        self.assertEqual([part.scope_kind for part in result.scoped_results], ["course", "term"])
        course, term = result.scoped_results
        self.assertEqual(course.course_code, "06016420")
        self.assertIsNone(term.course_code)
        self.assertEqual(term.total_credits, 30)
        self.assertTrue(course.provenance)
        self.assertTrue(term.provenance)
        self.assertEqual((term.scope.plan, term.scope.years, term.scope.semesters), ("no_coop", (2,), (2,)))

    def test_course_credits_are_three_term_total_is_thirty(self):
        result = self.execute(payload(("credits", "placement"), "credits"))
        self.assertEqual(result.status, "answer", result.missing_information)
        course, term = result.scoped_results
        self.assertEqual(term.total_credits, 30)
        self.assertTrue(any("3(2-2-5)" in line for line in course.summary_facts))
        self.assertTrue(all("30" not in line for line in course.summary_facts))

    def test_aggregate_failure_discards_course_success(self):
        original = executor.execute_deterministic
        def evidence(db, spec, context, question):
            return original(db, spec, context, question) if spec.course_codes else VerifiedResult(status="missing_data")
        with patch("rag.semantic.executor.execute_deterministic", side_effect=evidence):
            result = self.execute()
        self.assertNotEqual(result.status, "answer")
        self.assertEqual(result.claims, ())
        self.assertEqual(result.scoped_results, ())

    def test_course_failure_discards_aggregate(self):
        with patch("rag.semantic.executor.execute_deterministic", return_value=VerifiedResult(status="missing_data")):
            result = self.execute()
        self.assertNotEqual(result.status, "answer")
        self.assertEqual(result.claims, ())

    def test_missing_accepted_operation_blocks_success(self):
        original = executor.execute_deterministic
        def evidence(*args):
            result = original(*args)
            return replace(result, claims=tuple(c for c in result.claims if c.operation != "prerequisite"))
        with patch("rag.semantic.executor.execute_deterministic", side_effect=evidence):
            self.assertNotEqual(self.execute().status, "answer")

    def test_term_evidence_cannot_be_relabelled_from_another_scope(self):
        original = executor.execute_deterministic
        def evidence(*args):
            result = original(*args)
            if not args[1].course_codes:
                result = replace(result, claims=tuple(replace(c, effective_scope=replace(c.effective_scope, semesters=(1,))) for c in result.claims))
            return result
        with patch("rag.semantic.executor.execute_deterministic", side_effect=evidence):
            self.assertNotEqual(self.execute().status, "answer")

    def test_missing_prerequisite_placement_blocks_entire_composition(self):
        original = executor.execute_deterministic
        def evidence(*args):
            if args[1].course_codes == ("06016413",):
                return VerifiedResult(status="missing_data")
            return original(*args)
        with patch("rag.semantic.executor.execute_deterministic", side_effect=evidence):
            result = self.execute(payload(("prerequisites", "prerequisite_placement", "placement")))
        self.assertNotEqual(result.status, "answer")
        self.assertEqual(result.scoped_results, ())

    def test_nonfinite_aggregate_evidence_fails_closed(self):
        original = executor.execute_deterministic
        def evidence(*args):
            result = original(*args)
            if not args[1].course_codes:
                result = replace(result, claims=tuple(replace(c, value=float("nan")) for c in result.claims))
            return result
        with patch("rag.semantic.executor.execute_deterministic", side_effect=evidence):
            self.assertNotEqual(self.execute().status, "answer")

    def test_invented_planning_equivalent_covers_direct_prerequisite_placement(self):
        data = payload(("prerequisites", "prerequisite_placement", "placement"))
        result = self.execute(data)
        self.assertEqual(result.status, "answer", result.missing_information)
        predecessor = [part for part in result.scoped_results if part.scope_kind == "prerequisite_course"]
        self.assertEqual([part.course_code for part in predecessor], ["06016413"])
        self.assertEqual(predecessor[0].parent_course_code, "06016420")
        self.assertTrue(any("ชั้นปีที่ 2" in fact and "ภาคการศึกษาที่ 1" in fact for fact in predecessor[0].summary_facts))
        self.assertEqual(result.scoped_results[-1].total_credits, 30)

    def test_presentation_keeps_subject_and_total_distinct_without_provider(self):
        result = self.execute(payload(("credits", "prerequisites", "placement"), "credits"))
        def forbidden(_prompt):
            self.fail("mixed-scope presentation must not call a real or stub answer model")
        text, mode = render_semantic_answer(QUESTION, result, forbidden)
        self.assertEqual(mode, "deterministic")
        self.assertIn("06016420", text)
        self.assertIn("3(2-2-5)", text)
        self.assertIn("หน่วยกิตรวมทั้งเทอม: 30", text)

    def test_pipeline_context_does_not_retain_whole_term_course_list(self):
        result = semantic_answer(DB, QUESTION, {"program": "IT", "catalog_key": "it-2565"},
                                 interpret_callable=lambda _prompt: json.dumps(payload()))
        self.assertEqual(result.result.status, "answer", result.trace.failure_reason)
        self.assertEqual(result.next_context["focus_course"]["course_code"], "06016420")
        self.assertEqual([row["course_code"] for row in result.next_context["result_courses"]], ["06016420"])
        self.assertEqual(result.trace.verified_summary["scope_kinds"], ["course", "term"])

    def test_missing_term_dimensions_clarify_without_evidence(self):
        for dimension in ("plan", "year", "semester"):
            data = payload()
            data["scope"][dimension] = None
            if dimension == "plan":
                data["scope"]["plan_hint"] = None
            with self.subTest(dimension=dimension), patch("rag.semantic.pipeline.execute_mixed_scope") as evidence:
                result = semantic_answer(DB, QUESTION, {"program": "IT", "catalog_key": "it-2565"},
                                         interpret_callable=lambda _prompt, data=data: json.dumps(data))
                self.assertEqual(result.result.status, "clarify_program")
                self.assertEqual(result.trace.verified_summary["scope_dimension"], dimension)
                evidence.assert_not_called()

    def test_public_adapter_reports_missing_year_without_live_api(self):
        data = payload()
        data["scope"]["year"] = None
        response = semantic_ask_response(DB, QUESTION, {"program": "IT", "catalog_key": "it-2565"},
                                         interpret_callable=lambda _prompt: json.dumps(data))
        self.assertEqual((response["status"], response["action"]), ("clarification_required", "year_required"))


if __name__ == "__main__":
    unittest.main()
