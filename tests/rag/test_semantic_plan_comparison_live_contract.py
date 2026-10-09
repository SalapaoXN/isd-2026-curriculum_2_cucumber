"""Provider-free RED replays for descriptive contrast and available-plan choice."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from rag.semantic import resolver, executor, compiler
from rag.semantic.context import MergedContext
from rag.semantic.interpreter import semantic_intent_json_schema, SemanticSchemaError
from rag.semantic.modes import semantic_ask_response
from rag.semantic.validation import validate_semantic_intent
from tests.rag.test_semantic_plan_placement_comparison import DB, payload, parse, resolve

CONTEXT = {"program": "IT", "catalog_key": "it-2565"}
Q28 = "ถ้าต้องเลือกระหว่างกลุ่มวิชา 06016481 กับ 06016482 ใน IT แผนสหกิจกับไม่สหกิจต่างกันอย่างไร ทั้งจำนวนวิชาที่เลือกและช่วงเรียน?"
Q29 = "ถ้าต้องวางแผนเรียน SERVER SIDE WEB DEVELOPMENT (06016418) และ DATA CENTER DESIGN (06016465) ให้เร็วที่สุดใน IT ควรเลือกแผนไหน และแต่ละวิชาเรียนได้ช่วงใด?"


def contrast():
    data = payload(("06016481", "06016482"), earliest=False, alternative=True)
    data["comparison"].update(operation="difference")
    data["comparison"]["left"] = {"plan": "แผนสหกิจ", "plan_hint": "coop"}
    data["comparison"]["right"] = {"plan": "ไม่สหกิจ", "plan_hint": "no_coop"}
    return data


def available():
    data = payload()
    data["target"]["members"] = [{"raw_text": text, "normalized_hint": None} for text in (
        "SERVER SIDE WEB DEVELOPMENT (06016418)", "DATA CENTER DESIGN (06016465)")]
    data["comparison"].update(left={}, right={}, plan_selector="available_plans")
    return data


def available_resolved(data=None, context=None):
    initial = resolver.resolve_semantic_intent(DB, parse(data or available()), context or MergedContext(**CONTEXT))
    return resolver.resolve_available_plan_operands(DB, initial)


class DescriptivePlacementContractTests(unittest.TestCase):
    def test_live_descriptive_difference_and_alternative_selection_validate(self):
        result = validate_semantic_intent(parse(contrast()), Q28)
        self.assertTrue(result.valid, result.reason)

    def test_bounded_thai_placement_vocabulary(self):
        data = contrast()
        for cue in ("อยู่ช่วงไหน", "ช่วงเรียน", "เรียนช่วงไหน", "เปิดให้ลงช่วงไหน",
                    "ปีไหนเทอมไหน", "ช่วงที่เรียน", "เรียนได้ช่วงใด"):
            with self.subTest(cue=cue):
                question = "IT 06016481 06016482 แผนสหกิจกับไม่สหกิจ " + cue + " ต่างกันอย่างไร?"
                result = validate_semantic_intent(parse(data), question)
                self.assertTrue(result.valid, result.reason)

    def test_generic_difference_alone_does_not_ground_placement(self):
        result = validate_semantic_intent(parse(contrast()), "IT 06016481 06016482 แผนสหกิจกับไม่สหกิจ ต่างกันอย่างไร?")
        self.assertFalse(result.valid)

    def test_explicit_raw_hint_authority_is_preserved(self):
        data = contrast()
        data["comparison"]["right"]["plan_hint"] = "coop"
        self.assertFalse(validate_semantic_intent(parse(data), Q28).valid)
        data = contrast()
        data["comparison"]["left"]["plan"] = "coop"
        self.assertFalse(validate_semantic_intent(parse(data), Q28).valid)

    def test_descriptive_difference_reuses_complete_matrix_without_arithmetic(self):
        result = executor.execute_comparison(DB, resolve(contrast()))
        self.assertEqual(result.status, "answer", result.missing_information)
        self.assertIsNone(result.numeric_comparison)
        self.assertEqual(result.placement_comparison.conclusions, ())
        self.assertEqual(len(result.placement_comparison.cells), 4)
        for cell in result.placement_comparison.cells:
            self.assertEqual(cell.placements, ((3, 2),) if cell.scope.plan == "coop" else ((3, 1), (3, 2), (4, 1)))
        self.assertEqual([selection.plan for selection in result.alternative_selections], ["coop", "no_coop"])
        self.assertTrue(all((s.minimum_choices, s.maximum_choices) == (1, 1) for s in result.alternative_selections))

    def test_live_shape_public_replay_has_selection_and_complete_choices(self):
        response = semantic_ask_response(DB, Q28, CONTEXT, interpret_callable=lambda prompt: json.dumps(contrast()))
        self.assertEqual(response["status"], "answer")
        self.assertIn("เลือก 1 วิชาจาก 2 วิชา", response["answer"])
        self.assertIn("ชั้นปีที่ 4 ภาคการศึกษาที่ 1", response["answer"])


class AvailablePlansContractTests(unittest.TestCase):
    def test_plan_choice_selector_contains_no_invented_raw_operands(self):
        intent = parse(available())
        self.assertEqual(intent.comparison.plan_selector, "available_plans")
        self.assertEqual((intent.comparison.left, intent.comparison.right), ((), ()))
        result = validate_semantic_intent(intent, Q29)
        self.assertTrue(result.valid, result.reason)

    def test_selector_requires_grounded_plan_choice(self):
        self.assertFalse(validate_semantic_intent(parse(available()),
            "IT SERVER SIDE WEB DEVELOPMENT (06016418) DATA CENTER DESIGN (06016465) เรียนได้ช่วงใด เร็วที่สุด?").valid)

    def test_selector_cannot_smuggle_explicit_plans_or_course_operands(self):
        for side in ({"plan": "coop"}, {"plan_hint": "no_coop"}, {"course": "06016418"}):
            data = available()
            data["comparison"]["left"] = side
            with self.subTest(side=side):
                self.assertFalse(validate_semantic_intent(parse(data), Q29).valid)

    def test_explicit_mentions_require_explicit_mode(self):
        self.assertFalse(validate_semantic_intent(parse(available()), Q29 + " coop no_coop").valid)

    def test_available_selector_is_not_a_numeric_comparison_shortcut(self):
        data = available()
        data["comparison"].update(measure="credits", operation="difference")
        self.assertFalse(validate_semantic_intent(parse(data), Q29).valid)

    def test_closed_parser_and_transport_schema(self):
        schema = semantic_intent_json_schema()["properties"]["comparison"]["anyOf"][0]
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("available_plans", schema["properties"]["plan_selector"]["anyOf"][0]["enum"])
        for key, value in (("winner", "coop"), ("plan_selector", "newest_plan")):
            data = available()
            data["comparison"][key] = value
            with self.subTest(key=key), self.assertRaises(SemanticSchemaError):
                parse(data)

    def test_canonical_program_catalog_supply_exactly_two_independent_plans(self):
        result = available_resolved()
        self.assertFalse(result.needs_clarification, result.clarification_reason)
        self.assertEqual([s.scope.plan for s in result.comparison_sides], ["coop", "no_coop"])
        self.assertTrue(all((s.scope.program, s.scope.catalog_key, s.target.kind) ==
                            ("IT", "it-2565", "none") for s in result.comparison_sides))
        requests = compiler.compile_plan_placement_requests(result, Q29)
        self.assertEqual([(r.course_code, r.scope.plan) for r in requests],
                         [(c, p) for c in ("06016418", "06016465") for p in ("coop", "no_coop")])

    def test_missing_or_ambiguous_catalog_fails_closed(self):
        for scope in (MergedContext(program="IT"), MergedContext(program="IT", catalog_key="2565-unknown")):
            with self.subTest(scope=scope):
                self.assertTrue(available_resolved(context=scope).needs_clarification)

    def test_one_plan_authority_cannot_become_two_plan_comparison(self):
        base = available_resolved()
        initial = replace(base, scope=replace(base.scope, program="AIT", catalog_key="ait-2566"), comparison_sides=())
        result = resolver.resolve_available_plan_operands(DB, initial)
        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.comparison_sides, ())

    def test_plan_inventory_absence_overflow_and_conflict_fail_closed(self):
        initial = resolver.resolve_semantic_intent(DB, parse(available()), MergedContext(**CONTEXT))
        for rows in ([], [{"plan_key": "a"}], [{"plan_key": "a"}, {"plan_key": "b"}, {"plan_key": "c"}],
                     [{"plan_key": "coop"}, {"plan_key": "coop"}], [{"plan_key": None}, {"plan_key": "coop"}]):
            from unittest.mock import MagicMock
            connection = MagicMock()
            connection.execute.return_value.fetchall.return_value = rows
            with (self.subTest(rows=rows), patch.object(resolver, "_connect_ro", return_value=connection),
                  patch.object(resolver, "canonical_program", return_value="IT"),
                  patch.object(resolver, "canonical_catalog_key", return_value="it-2565")):
                result = resolver.resolve_available_plan_operands(DB, initial)
                self.assertTrue(result.needs_clarification)
                self.assertEqual(result.comparison_sides, ())

    def test_existing_inherited_term_or_plan_does_not_narrow_selector(self):
        result = available_resolved(context=MergedContext(**CONTEXT, plan="coop", years=(4,), semesters=(1,)))
        self.assertFalse(result.needs_clarification, result.clarification_reason)
        self.assertTrue(all(not s.scope.years and not s.scope.semesters for s in result.comparison_sides))
        self.assertEqual([s.scope.plan for s in result.comparison_sides], ["coop", "no_coop"])

    def test_earliest_uses_existing_matrix_and_keeps_tie_and_full_choices(self):
        result = executor.execute_comparison(DB, available_resolved())
        self.assertEqual(result.status, "answer", result.missing_information)
        self.assertEqual(len(result.placement_comparison.cells), 4)
        conclusions = {c.course_code: c for c in result.placement_comparison.conclusions}
        self.assertTrue(conclusions["06016418"].tie)
        self.assertEqual(conclusions["06016465"].earlier_plan, "no_coop")
        flexible = [cell for cell in result.placement_comparison.cells if cell.course_code == "06016465" and cell.scope.plan == "no_coop"]
        self.assertEqual(flexible[0].placements, ((3, 1), (3, 2), (4, 1)))

    def test_available_matrix_still_fails_atomically_on_missing_cell(self):
        from rag.semantic.schema import VerifiedResult
        original = executor.execute_deterministic
        def missing(db, spec, context, question):
            if spec.course_codes == ("06016465",) and spec.plans == ("no_coop",):
                return VerifiedResult(status="missing_data")
            return original(db, spec, context, question)
        with patch.object(executor, "execute_deterministic", side_effect=missing):
            result = executor.execute_comparison(DB, available_resolved())
        self.assertNotEqual(result.status, "answer")
        self.assertIsNone(result.placement_comparison)

    def test_public_choice_replay_and_trace_keep_selector(self):
        from rag.semantic.pipeline import semantic_answer
        outcome = semantic_answer(DB, Q29, CONTEXT, interpret_callable=lambda prompt: json.dumps(available()))
        self.assertEqual(outcome.result.status, "answer", outcome.trace.failure_reason)
        self.assertEqual(outcome.trace.semantic_intent["comparison"]["plan_selector"], "available_plans")
        self.assertIn("06016418", outcome.result.final_answer)
        self.assertIn("06016465", outcome.result.final_answer)
        self.assertIn("เร็วที่สุดเท่ากัน", outcome.result.final_answer)
        self.assertIn("แผนไม่สหกิจ เรียนได้เร็วกว่า", outcome.result.final_answer)
