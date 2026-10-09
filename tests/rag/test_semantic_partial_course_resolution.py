"""SD1 baseline contracts for deterministic partial course resolution."""

from pathlib import Path
import unittest
from unittest.mock import patch

from rag.hybrid_demo import DEFAULT_CURRICULUM_DB_PATH
from rag.semantic.context import MergedContext
from rag.semantic.resolver import resolve_comparison_operand, resolve_semantic_intent
from rag.semantic.schema import SemanticIntent, SemanticTarget, ScopeMention
from rag.structured.queries import exact_course_candidates


DB_PATH = Path(DEFAULT_CURRICULUM_DB_PATH)
DSBA_CATALOG = "dsba-2565"


class SemanticPartialCourseResolutionTests(unittest.TestCase):
    def resolve_literal(self, reference, *, program="DSBA", catalog=DSBA_CATALOG):
        intent = SemanticIntent(
            task="lookup",
            subject="course",
            relation="placement",
            target=SemanticTarget(kind="literal", raw_text=reference),
            scope=ScopeMention(program=program, catalog=catalog),
        )
        merged = MergedContext(program=program, catalog_key=catalog)
        return resolve_semantic_intent(DB_PATH, intent, merged)

    def resolve_operand(self, **side):
        return resolve_comparison_operand(
            DB_PATH,
            tuple(side.items()),
            default_program=None,
            default_catalog_key=None,
        )

    def test_unique_partial_data_pipeline_resolves_to_canonical_course(self):
        resolved = self.resolve_literal("data pipeline")

        self.assertFalse(resolved.needs_clarification)
        self.assertEqual(resolved.target.course_code, "06026239")

    def test_exact_full_title_still_resolves(self):
        resolved = self.resolve_literal("DATA PIPELINE ARCHITECTURE")

        self.assertFalse(resolved.needs_clarification)
        self.assertEqual(resolved.target.course_code, "06026239")

    def test_exact_eight_digit_code_still_resolves(self):
        resolved = self.resolve_literal("06026239")

        self.assertFalse(resolved.needs_clarification)
        self.assertEqual(resolved.target.course_code, "06026239")

    def test_real_ambiguous_partial_does_not_select_first_identity(self):
        candidates = exact_course_candidates(
            DB_PATH,
            course_name="programming",
            program="DSBA",
            catalog_key=DSBA_CATALOG,
        )
        self.assertEqual(
            {candidate["course_code"] for candidate in candidates},
            {"06026203", "06026206", "06066302", "06066303"},
        )

        with patch(
            "rag.semantic.resolver.exact_course_candidates",
            wraps=exact_course_candidates,
        ) as candidate_lookup:
            resolved = self.resolve_literal("programming")

        self.assertTrue(
            any(call.kwargs.get("exact_title") is False for call in candidate_lookup.call_args_list),
            "ambiguous control must exercise the partial-title lookup",
        )
        self.assertTrue(resolved.needs_clarification)
        self.assertIsNone(resolved.target.course_code)

    def test_nonexistent_literal_remains_unresolved_not_topic_discovery(self):
        resolved = self.resolve_literal("nonexistent literal course reference")

        self.assertTrue(resolved.needs_clarification)
        self.assertIsNone(resolved.target.course_code)
        self.assertEqual(resolved.intent.task, "lookup")
        self.assertEqual(resolved.intent.target.kind, "literal")

    def test_weak_references_do_not_resolve_to_course_identity(self):
        for reference in ("1", "!!!", "x"):
            with self.subTest(reference=reference):
                resolved = self.resolve_literal(reference)
                self.assertTrue(resolved.needs_clarification)
                self.assertIsNone(resolved.target.course_code)

    def test_partial_candidates_do_not_cross_program_or_catalog(self):
        in_scope = exact_course_candidates(
            DB_PATH,
            course_name="data pipeline",
            program="DSBA",
            catalog_key=DSBA_CATALOG,
        )
        other_program = exact_course_candidates(
            DB_PATH, course_name="data pipeline", program="IT", catalog_key=DSBA_CATALOG
        )
        other_catalog = exact_course_candidates(
            DB_PATH, course_name="data pipeline", program="DSBA", catalog_key="dsba-2560"
        )
        wrong_program = self.resolve_literal("data pipeline", program="IT")
        wrong_catalog = self.resolve_literal("data pipeline", catalog="dsba-2560")

        self.assertEqual([candidate["course_code"] for candidate in in_scope], ["06026239"])
        self.assertEqual(other_program, [])
        self.assertEqual(other_catalog, [])
        self.assertTrue(wrong_program.needs_clarification)
        self.assertIsNone(wrong_program.target.course_code)
        self.assertTrue(wrong_catalog.needs_clarification)
        self.assertIsNone(wrong_catalog.target.course_code)

    def test_comparison_operands_keep_independent_scope_and_identity(self):
        left = self.resolve_operand(
            course="06026239", program="DSBA", catalog=DSBA_CATALOG
        )
        right = self.resolve_operand(
            course="06016420", program="IT", catalog="it-2565"
        )

        self.assertFalse(left.unresolved)
        self.assertFalse(right.unresolved)
        self.assertEqual((left.scope.program, left.scope.catalog_key), ("DSBA", DSBA_CATALOG))
        self.assertEqual((right.scope.program, right.scope.catalog_key), ("IT", "it-2565"))
        self.assertEqual(left.target.course_code, "06026239")
        self.assertEqual(right.target.course_code, "06016420")

    def test_comparison_operand_resolves_unique_partial_independently(self):
        operand = self.resolve_operand(
            course="data pipeline", program="DSBA", catalog=DSBA_CATALOG
        )

        self.assertFalse(operand.unresolved)
        self.assertEqual(operand.scope.program, "DSBA")
        self.assertEqual(operand.scope.catalog_key, DSBA_CATALOG)
        self.assertEqual(operand.target.course_code, "06026239")

    def test_ambiguous_partial_on_either_comparison_side_fails_closed(self):
        exact_right = {"course": "06026239", "program": "DSBA", "catalog": DSBA_CATALOG}
        for side in ("left", "right"):
            with self.subTest(side=side):
                ambiguous = self.resolve_operand(
                    course="programming", program="DSBA", catalog=DSBA_CATALOG
                )
                other = self.resolve_operand(**exact_right)
                operands = (ambiguous, other) if side == "left" else (other, ambiguous)
                self.assertTrue(operands[0 if side == "left" else 1].unresolved)
                self.assertIsNone(operands[0 if side == "left" else 1].target.course_code)


if __name__ == "__main__":
    unittest.main()
