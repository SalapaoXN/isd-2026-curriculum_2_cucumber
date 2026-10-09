"""G5-B offline contract, complete matrix, and deterministic conclusion gates."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.semantic import compiler, executor
from rag.semantic.answerer import render_semantic_answer
from rag.semantic.context import MergedContext
from rag.semantic.interpreter import parse_semantic_intent_payload, semantic_intent_json_schema, SemanticSchemaError
from rag.semantic.pipeline import semantic_answer
from rag.semantic.planner import plan_semantic_query
from rag.semantic.resolver import resolve_semantic_intent, resolve_comparison_operand
from rag.semantic.validation import validate_semantic_intent

DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


def payload(codes=("06016418", "06016465"), earliest=True, alternative=False):
    target = {"kind": "literal", "raw_text": codes[0], "normalized_hint": None, "ordinal": None}
    if len(codes) > 1:
        target.update(kind="literal_set", raw_text=None,
                      members=[{"raw_text": code, "normalized_hint": None} for code in codes])
    return {
        "task": "compare", "subject": "course", "relation": "placement", "target": target,
        "scope": {"program": "IT", "catalog": None, "plan": None, "year": None, "semester": None},
        "filters": [], "aggregation": None, "ranking": None,
        "comparison": {"left": {"plan": "coop", "plan_hint": "coop"},
                       "right": {"plan": "no_coop", "plan_hint": "no_coop"},
                       "measure": "placement", "operation": "earliest_placement" if earliest else None},
        "requested_fields": ["placement"] + (["alternative_selection"] if alternative else []),
        "clarification": None, "policy_topic": None, "observed_value": None,
    }


def question(data):
    target = data["target"]
    codes = [m["raw_text"] for m in target.get("members", [])] or [target["raw_text"]]
    return "IT: placement of " + " and ".join(codes) + " across coop and no_coop" + (
        "; which plan allows earlier placement?" if data["comparison"]["operation"] else "?")


def parse(data):
    return parse_semantic_intent_payload(json.dumps(data))


def resolve(data=None):
    intent = parse(data or payload())
    resolved = resolve_semantic_intent(DB, intent, MergedContext(program="IT", catalog_key="it-2565"))
    sides = tuple(resolve_comparison_operand(DB, side, "IT", "it-2565")
                  for side in (intent.comparison.left, intent.comparison.right))
    return replace(resolved, comparison_sides=sides)


class PlanPlacementContractTests(unittest.TestCase):
    def test_single_course_non_credit_comparison(self):
        data = payload(("06016465",))
        self.assertTrue(validate_semantic_intent(parse(data), question(data)).valid)
        self.assertEqual(plan_semantic_query(resolve(data)).execution, "deterministic")

    def test_set_remains_subject_plans_remain_sides(self):
        resolved = resolve()
        self.assertEqual([m.course_code for m in resolved.target.members], ["06016418", "06016465"])
        self.assertEqual([s.scope.plan for s in resolved.comparison_sides], ["coop", "no_coop"])
        self.assertTrue(all(s.target.course_code is None for s in resolved.comparison_sides))
        self.assertTrue(validate_semantic_intent(resolved.intent, question(payload())).valid)

    def test_placement_without_derived_operation_is_valid(self):
        data = payload(earliest=False)
        self.assertTrue(validate_semantic_intent(parse(data), question(data)).valid)

    def test_earliest_must_be_requested(self):
        data = payload()
        self.assertFalse(validate_semantic_intent(parse(data), question(payload(earliest=False))).valid)

    def test_requested_earliest_cannot_be_dropped(self):
        data = payload(earliest=False)
        self.assertFalse(validate_semantic_intent(parse(data), question(payload())).valid)

    def test_placement_must_be_grounded(self):
        self.assertFalse(validate_semantic_intent(parse(payload(earliest=False)),
                         "IT 06016418 06016465 coop no_coop credits?").valid)

    def test_missing_course_mention_rejects(self):
        self.assertFalse(validate_semantic_intent(parse(payload(("06016418",))), question(payload())).valid)

    def test_duplicate_plan_cannot_hide_missing_plan(self):
        data = payload()
        data["comparison"]["right"] = data["comparison"]["left"].copy()
        self.assertFalse(validate_semantic_intent(parse(data), question(payload())).valid)

    def test_course_operand_cannot_change_subject(self):
        data = payload()
        data["comparison"]["left"]["course"] = "06016418"
        self.assertFalse(validate_semantic_intent(parse(data), question(data)).valid)

    def test_no_unconsumed_requested_fields(self):
        for field in ("credits", "prerequisites", "placement_sequence", "description"):
            data = payload()
            data["requested_fields"].append(field)
            with self.subTest(field=field):
                self.assertFalse(validate_semantic_intent(parse(data), question(data)).valid)

    def test_plan_hint_cannot_override_raw_negation(self):
        data = payload()
        data["comparison"]["right"] = {"plan": "ไม่สหกิจ", "plan_hint": "coop"}
        self.assertFalse(validate_semantic_intent(parse(data), question(data) + " ไม่สหกิจ").valid)
        self.assertTrue(resolve_comparison_operand(DB, tuple(data["comparison"]["right"].items()),
                                                  "IT", "it-2565").unresolved)

    def test_unknown_fields_still_rejected(self):
        for location in ("root", "comparison", "side"):
            data = payload()
            obj = data if location == "root" else data["comparison"] if location == "comparison" else data["comparison"]["left"]
            obj["winner"] = "no_coop"
            with self.subTest(location=location), self.assertRaises(SemanticSchemaError):
                parse(data)

    def test_transport_schema_alignment_and_no_aggregate_expansion(self):
        schema = semantic_intent_json_schema()["properties"]
        self.assertIn("placement", schema["comparison"]["anyOf"][0]["properties"]["measure"]["enum"])
        self.assertNotIn("placement", schema["aggregation"]["anyOf"][0]["properties"]["measure"]["enum"])

    def test_matrix_compiler_complete_and_isolated(self):
        requests = compiler.compile_plan_placement_requests(resolve(), "")
        self.assertEqual([(r.course_code, r.scope.plan) for r in requests],
                         [(c, p) for c in ("06016418", "06016465") for p in ("coop", "no_coop")])
        for request in requests:
            self.assertEqual(request.spec.operations, ("placement",))
            self.assertEqual(request.spec.course_codes, (request.course_code,))
            self.assertEqual(request.spec.plans, (request.scope.plan,))

    def test_incompatible_or_missing_operand_scope_blocks(self):
        base = resolve()
        for values in ({"program": "DSBA"}, {"catalog_key": "dsba-2565"}, {"plan": None}):
            bad = replace(base.comparison_sides[1], scope=replace(base.comparison_sides[1].scope, **values))
            with self.subTest(values=values):
                self.assertEqual(plan_semantic_query(replace(base, comparison_sides=(base.comparison_sides[0], bad))).execution,
                                 "unsupported")


class PlanPlacementEvidenceTests(unittest.TestCase):
    def execute(self, data=None):
        return executor.execute_comparison(DB, resolve(data))

    def test_complete_matrix_flexible_choices_and_member_isolation(self):
        result = self.execute()
        self.assertEqual(result.status, "answer", result.missing_information)
        cells = result.placement_comparison.cells
        self.assertEqual(len(cells), 4)
        actual = {(c.course_code, c.scope.plan): c.placements for c in cells}
        self.assertEqual(actual, {
            ("06016418", "coop"): ((3, 1),), ("06016418", "no_coop"): ((3, 1),),
            ("06016465", "coop"): ((4, 1),), ("06016465", "no_coop"): ((3, 1), (3, 2), (4, 1)),
        })
        self.assertTrue(all(c.provenance for c in cells))

    def test_deterministic_earliest_and_explicit_tie(self):
        result = self.execute()
        self.assertEqual(result.status, "answer", result.missing_information)
        conclusions = {c.course_code: c for c in result.placement_comparison.conclusions}
        self.assertEqual(conclusions["06016465"].earliest, ((4, 1), (3, 1)))
        self.assertEqual(conclusions["06016465"].earlier_plan, "no_coop")
        self.assertIsNone(conclusions["06016418"].earlier_plan)
        self.assertTrue(conclusions["06016418"].tie)

    def test_no_unrequested_earliest(self):
        result = self.execute(payload(earliest=False))
        self.assertEqual(result.status, "answer", result.missing_information)
        self.assertEqual(result.placement_comparison.conclusions, ())

    def test_alternative_set_shape_and_exactly_one(self):
        result = self.execute(payload(("06016481", "06016482"), earliest=False, alternative=True))
        self.assertEqual(result.status, "answer", result.missing_information)
        self.assertEqual(len(result.placement_comparison.cells), 4)
        for cell in result.placement_comparison.cells:
            self.assertEqual(cell.placements, ((3, 2),) if cell.scope.plan == "coop" else ((3, 1), (3, 2), (4, 1)))
        self.assertEqual([s.plan for s in result.alternative_selections], ["coop", "no_coop"])
        self.assertTrue(all((s.minimum_choices, s.maximum_choices) == (1, 1) for s in result.alternative_selections))

    def mutate_query(self, change):
        original = executor.execute_deterministic
        def plain(value):
            from collections.abc import Mapping
            if isinstance(value, Mapping):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value
        def corrupt(db, spec, context, question):
            result = original(db, spec, context, question)
            if spec.plans == ("no_coop",) and spec.course_codes == ("06016465",):
                claim = result.claims[0]
                data = {"courses": plain(claim.value)}
                change(data)
                return replace(result, claims=(replace(claim, value=tuple(data["courses"])),))
            return result
        with patch.object(executor, "execute_deterministic", side_effect=corrupt):
            result = self.execute()
        self.assertNotEqual(result.status, "answer")
        self.assertIsNone(result.placement_comparison)
        self.assertEqual(result.summary_facts, ())

    def test_missing_one_cell_atomic_failure(self):
        self.mutate_query(lambda data: data.update(courses=[]))

    def test_cross_plan_evidence_rejected(self):
        self.mutate_query(lambda data: data["courses"][0].update(plan_key="coop"))

    def test_cross_member_evidence_rejected(self):
        self.mutate_query(lambda data: data["courses"][0].update(course_code="06016418"))

    def test_missing_or_malformed_choices_never_hidden(self):
        for choices in ([], [(None, 1)], [(3, "1")], [(3, True)], [(3, 1), (0, 2)], [(3, 1), (4,)], [(3, 3)]):
            with self.subTest(choices=choices):
                self.mutate_query(lambda data: data["courses"][0].update(year_semester_choices=choices))

    def test_missing_provenance_rejects(self):
        self.mutate_query(lambda data: data["courses"][0].update(provenance=[]))

    def test_presentation_complete_without_provider(self):
        result = self.execute()
        def forbidden(*args):
            self.fail("presentation provider called")
        answer, mode = render_semantic_answer("", result, forbidden)
        self.assertEqual(mode, "deterministic")
        for code in ("06016418", "06016465"):
            self.assertIn(code, answer)
        self.assertIn("ภาคการศึกษาที่ 2", answer)
        self.assertIn("เร็วกว่า", answer)
        self.assertIn("เท่ากัน", answer)

    def test_pipeline_with_only_stub_interpretation(self):
        data = payload()
        prior = {"program": "IT", "catalog_key": "it-2565", "plan": "coop"}
        outcome = semantic_answer(DB, question(data), prior, interpret_callable=lambda prompt: json.dumps(data))
        self.assertEqual(outcome.result.status, "answer", outcome.trace.failure_reason)
        self.assertEqual(outcome.next_context, prior)
        self.assertEqual(outcome.trace.llm_request_count, 1)  # local stub only
        self.assertEqual(outcome.trace.semantic_intent["comparison"]["measure"], "placement")
