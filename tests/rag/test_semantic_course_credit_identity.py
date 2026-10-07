"""Focused producer-to-consumer checks for semantic course-credit lookups."""

import unittest
from pathlib import Path

from rag.resolution import QueryContext
from rag.semantic.answerer import (
    render_semantic_answer,
    render_verified_comparison,
    validate_answer_text,
)
from rag.semantic.compiler import compile_resolved_intent_to_query_spec
from rag.semantic.executor import (
    execute_comparison,
    execute_deterministic,
    execute_policy,
)
from rag.semantic.schema import (
    AggregationSpec,
    ComparisonSpec,
    ResolvedIntent,
    ResolvedOperand,
    ResolvedScope,
    ResolvedTarget,
    SemanticIntent,
    SemanticTarget,
)


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


def _course_credit_intent(
    program: str, catalog_key: str, course_code: str, course_name: str
) -> ResolvedIntent:
    return ResolvedIntent(
        intent=SemanticIntent(
            task="lookup",
            subject="course",
            relation="credits",
            target=SemanticTarget(kind="literal", raw_text=course_name),
        ),
        scope=ResolvedScope(program=program, catalog_key=catalog_key),
        target=ResolvedTarget(
            kind="literal",
            course_code=course_code,
            course_name=course_name,
            program=program,
            catalog_key=catalog_key,
        ),
    )


class SemanticCourseCreditIdentityTests(unittest.TestCase):
    def _run_credit_lookup(self, program, catalog_key, course_code, course_name):
        resolved = _course_credit_intent(
            program, catalog_key, course_code, course_name
        )
        spec = compile_resolved_intent_to_query_spec(resolved, course_name)
        self.assertEqual(spec.operations, ("identity", "sum_credits"))
        result = execute_deterministic(
            DB_PATH,
            spec,
            QueryContext(program=program, catalog_key=catalog_key),
            course_name,
        )
        answer, _ = render_semantic_answer(course_name, result, None)
        self.assertEqual(result.status, "answer")
        self.assertIn(course_code, answer)
        self.assertIn(course_name, answer)
        self.assertIn("3", answer)
        self.assertTrue(result.provenance)
        self.assertTrue(validate_answer_text(answer, result))
        normal_answer, mode = render_semantic_answer(
            course_name,
            result,
            lambda _prompt: f"{course_name} ({course_code}): 3 credits",
        )
        self.assertEqual(mode, "grounded_synthesis")
        self.assertIn(course_code, normal_answer)
        self.assertIn(course_name, normal_answer)
        return result

    def test_e01_e07_and_generic_course_credit_keep_identity(self):
        cases = (
            ("AIT", "ait-2566", "06046401", "CALCULUS 2"),
            ("DSBA", "dsba-2565", "06026200", "CALCULUS 1"),
            # Existing canonical course control, distinct from E01 and E07.
            ("AIT", "ait-2566", "06046400", "CALCULUS 1"),
        )
        for case in cases:
            with self.subTest(course_code=case[2]):
                self._run_credit_lookup(*case)

    def test_program_total_stays_on_program_requirement_path(self):
        resolved = ResolvedIntent(
            intent=SemanticIntent(
                task="lookup", subject="program", relation="credits"
            ),
            scope=ResolvedScope(program="AIT", catalog_key="ait-2566"),
        )
        result = execute_policy(DB_PATH, resolved)
        self.assertEqual(result.status, "answer")
        self.assertIn("120", result.summary_facts[0])
        self.assertFalse(result.claims)

    def test_year_credit_total_stays_aggregate(self):
        resolved = ResolvedIntent(
            intent=SemanticIntent(
                task="aggregate",
                subject="course",
                aggregation=AggregationSpec(function="sum", measure="credits"),
            ),
            scope=ResolvedScope(
                program="AIT", catalog_key="ait-2566", years=(1,)
            ),
        )
        spec = compile_resolved_intent_to_query_spec(resolved, "year aggregate")
        self.assertEqual(spec.operations, ("sum_credits",))
        result = execute_deterministic(
            DB_PATH,
            spec,
            QueryContext(program="AIT", catalog_key="ait-2566", years=(1,)),
            "year aggregate",
        )
        self.assertEqual(result.status, "answer")
        self.assertIn("52", result.summary_facts[0])
        self.assertTrue(result.provenance)

    def test_other_course_lookup_relations_keep_their_operations(self):
        expected = {
            "identity": ("identity",),
            "description": ("describe",),
            "prerequisite": ("prerequisite",),
            "placement": ("placement",),
        }
        for relation, operations in expected.items():
            with self.subTest(relation=relation):
                resolved = ResolvedIntent(
                    intent=SemanticIntent(
                        task="lookup", subject="course", relation=relation
                    ),
                    scope=ResolvedScope(program="AIT", catalog_key="ait-2566"),
                    target=ResolvedTarget(
                        kind="literal",
                        course_code="06046401",
                        course_name="CALCULUS 2",
                        program="AIT",
                        catalog_key="ait-2566",
                    ),
                )
                spec = compile_resolved_intent_to_query_spec(resolved, relation)
                self.assertEqual(spec.operations, operations)

    def test_h07_numeric_comparison_path_is_unchanged(self):
        scope = ResolvedScope(program="AIT", catalog_key="ait-2566")
        sides = (
            ResolvedOperand(
                scope=scope,
                target=ResolvedTarget(
                    kind="literal", course_code="06046400", course_name="CALCULUS 1"
                ),
                unresolved=False,
            ),
            ResolvedOperand(
                scope=scope,
                target=ResolvedTarget(
                    kind="literal", course_code="06046401", course_name="CALCULUS 2"
                ),
                unresolved=False,
            ),
        )
        resolved = ResolvedIntent(
            intent=SemanticIntent(
                task="compare",
                comparison=ComparisonSpec(measure="credits", operation="greater"),
            ),
            comparison_sides=sides,
        )
        result = execute_comparison(DB_PATH, resolved)
        self.assertEqual(result.status, "answer")
        self.assertEqual(result.numeric_comparison.actual_relation, "equal")
        self.assertTrue(result.provenance)
        answer = render_verified_comparison(result)
        self.assertIn("ทั้งสองวิชามีจำนวนหน่วยกิตเท่ากัน", answer)


if __name__ == "__main__":
    unittest.main()
