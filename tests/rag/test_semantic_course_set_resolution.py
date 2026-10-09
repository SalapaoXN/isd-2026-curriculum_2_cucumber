"""Atomic G4 resolution and complete compilation, without providers."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.semantic.compiler import compile_resolved_intent_to_query_spec
from rag.semantic.context import MergedContext
from rag.semantic.planner import plan_semantic_query
from rag.semantic.resolver import resolve_semantic_intent
from rag.semantic.schema import ComparisonSpec, SemanticTarget
from tests.rag.test_semantic_course_set_contract import parse, payload


DB = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"


def resolve(refs=("06016481", "06016482"), fields=("placement",), plan="coop"):
    return resolve_semantic_intent(
        DB, parse(payload(refs, fields)),
        MergedContext(program="IT", catalog_key="it-2565", plan=plan),
    )


class CourseSetResolutionTests(unittest.TestCase):
    def test_every_member_resolves_and_compiles(self):
        resolved = resolve()
        self.assertFalse(resolved.needs_clarification)
        self.assertIsNone(resolved.target.course_code)
        self.assertEqual([m.course_code for m in resolved.target.members], ["06016481", "06016482"])
        self.assertTrue(all((m.program, m.catalog_key) == ("IT", "it-2565") for m in resolved.target.members))
        spec = compile_resolved_intent_to_query_spec(resolved, "question")
        self.assertEqual(spec.course_codes, ("06016481", "06016482"))
        self.assertIsNone(spec.course_name)
        self.assertIsNone(spec.topic)

    def test_nonexistent_member_is_atomic_failure(self):
        resolved = resolve(("06016413", "06016420", "06019999"))
        self.assertTrue(resolved.needs_clarification)
        self.assertEqual(resolved.target.members, ())
        self.assertIn("member 3", resolved.clarification_reason)

    def test_ambiguous_member_is_atomic_failure(self):
        resolved = resolve(("06016413", "programming"))
        self.assertTrue(resolved.needs_clarification)
        self.assertEqual(resolved.target.members, ())
        self.assertIn("member 2", resolved.clarification_reason)

    def test_conflicting_compound_reference_blocks_whole_set(self):
        resolved = resolve(("06016413", "NOSQL DATABASE SYSTEMS (06016406)"))
        self.assertTrue(resolved.needs_clarification)
        self.assertEqual(resolved.target.members, ())

    def test_malformed_code_does_not_use_title_fallback(self):
        for reference in ("0601640", "106016406", "PROJECT 1 (0601640)"):
            with self.subTest(reference=reference):
                self.assertTrue(resolve(("06016413", reference)).needs_clarification)

    def test_all_members_checked_before_alias_dedupe(self):
        from rag.semantic.resolver import _lookup_literal
        with patch("rag.semantic.resolver._lookup_literal", wraps=_lookup_literal) as lookup:
            resolved = resolve(("06016413", "INTRODUCTION TO NETWORK SYSTEMS", "06016420"))
        self.assertFalse(resolved.needs_clarification)
        self.assertEqual(lookup.call_count, 3)
        self.assertEqual([m.course_code for m in resolved.target.members], ["06016413", "06016420"])

    def test_wrong_program_and_catalog_block_members(self):
        for program, catalog in (("DSBA", "dsba-2565"), ("IT", "it-2560")):
            with self.subTest(program=program, catalog=catalog):
                resolved = resolve_semantic_intent(DB, parse(payload()), MergedContext(program=program, catalog_key=catalog))
                self.assertTrue(resolved.needs_clarification)
                self.assertEqual(resolved.target.members, ())

    def test_partial_title_member_uses_existing_policy(self):
        resolved = resolve(("SERVER SIDE WEB", "DATA CENTER DESIGN"), plan="no_coop")
        self.assertFalse(resolved.needs_clarification)
        self.assertEqual([m.course_code for m in resolved.target.members], ["06016418", "06016465"])

    def test_hints_never_authorize_members(self):
        data = payload(("invented course", "06016413"))
        data["target"]["members"][0]["normalized_hint"] = "06016420"
        resolved = resolve_semantic_intent(DB, parse(data), MergedContext(program="IT", catalog_key="it-2565"), allow_hint_candidates=True)
        self.assertTrue(resolved.needs_clarification)
        self.assertEqual(resolved.target.members, ())

    def test_g1_union_for_sets_and_selection(self):
        resolved = resolve(fields=("placement", "prerequisites", "credits", "name", "credits", "alternative_selection"))
        spec = compile_resolved_intent_to_query_spec(resolved, "question")
        self.assertEqual(spec.operations, ("placement", "prerequisite", "sum_credits", "identity"))
        self.assertEqual(plan_semantic_query(resolved).execution, "deterministic")

    def test_sequence_preserved_and_plan_deterministically_executable(self):
        resolved = resolve(("06016413", "06016420", "06016421"),
                           ("placement", "prerequisites", "placement_sequence"), plan="no_coop")
        self.assertFalse(resolved.needs_clarification)
        self.assertIn("placement_sequence", resolved.intent.requested_fields)
        plan = plan_semantic_query(resolved)
        self.assertEqual(plan.execution, "deterministic")
        self.assertIn("placement_sequence", plan.reason)
        self.assertEqual(compile_resolved_intent_to_query_spec(resolved, "question").operations,
                         ("placement", "prerequisite"))

    def test_selection_requires_plan(self):
        self.assertEqual(plan_semantic_query(resolve(fields=("alternative_selection",), plan=None)).execution, "unsupported")

    def test_unknown_requested_field_blocks_planning(self):
        resolved = resolve()
        resolved = replace(resolved, intent=replace(resolved.intent, requested_fields=("invented",)))
        self.assertEqual(plan_semantic_query(resolved).execution, "unsupported")

    def test_comparison_shape_remains_independent(self):
        resolved = resolve()
        comparison = ComparisonSpec(left=(("course", "06016413"),), right=(("course", "06016420"),))
        intent = replace(resolved.intent, task="compare", target=SemanticTarget(), comparison=comparison)
        self.assertEqual(plan_semantic_query(replace(resolved, intent=intent, target=replace(resolved.target, kind="none", members=()))).execution, "deterministic")
        self.assertEqual(intent.comparison, comparison)


if __name__ == "__main__":
    unittest.main()
