"""Focused direct merge controls for explicit program switches."""

import unittest

from rag.semantic.context import merge_semantic_context
from rag.semantic.schema import ScopeMention, SemanticIntent


class ProgramSwitchContextTests(unittest.TestCase):
    def setUp(self):
        self.prior = {
            "program": "IT", "catalog_key": "it-2565", "plan": "coop",
            "years": [2], "semesters": [1],
            "focus_course": {"course_code": "06016454", "program": "IT"},
            "result_courses": [{"course_code": "06016454", "program": "IT"}],
            "result_scope_program": "IT", "result_set_empty": True,
        }

    def merge(self, **fields):
        return merge_semantic_context(
            SemanticIntent(scope=ScopeMention(**fields)), self.prior)

    def test_same_program_followup(self):
        merged = self.merge(semester=2)
        self.assertEqual((merged.program, merged.catalog_key, merged.plan,
                          merged.years, merged.semesters),
                         ("IT", "it-2565", "coop", (2,), (2,)))
        self.assertEqual(merged.invalidated, ())

    def test_switch_clears_catalog(self):
        merged = self.merge(program="DSBA")
        self.assertEqual(merged.program, "DSBA")
        self.assertIsNone(merged.catalog_key)

    def test_switch_clears_plan(self):
        self.assertIsNone(self.merge(program="DSBA").plan)

    def test_switch_clears_temporal_scope(self):
        merged = self.merge(program="DSBA")
        self.assertEqual((merged.years, merged.semesters), ((), ()))

    def test_switch_clears_referents(self):
        merged = self.merge(program="DSBA")
        self.assertIsNone(merged.focus_course)
        self.assertEqual(merged.result_courses, ())
        self.assertIsNone(merged.result_scope_program)
        self.assertFalse(merged.result_set_empty)

    def test_switch_preserves_explicit_replacements(self):
        merged = self.merge(program="DSBA", catalog="dsba-2565", year=1)
        self.assertEqual((merged.program, merged.catalog_key, merged.years),
                         ("DSBA", "dsba-2565", (1,)))
        self.assertIsNone(merged.plan)
        self.assertEqual(merged.semesters, ())
        self.assertIsNone(merged.focus_course)
        self.assertEqual(merged.result_courses, ())
        self.assertIsNone(merged.result_scope_program)
        self.assertFalse(merged.result_set_empty)
