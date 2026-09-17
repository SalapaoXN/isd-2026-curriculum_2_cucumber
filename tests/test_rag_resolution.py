import unittest
from pathlib import Path
from unittest.mock import patch

from rag.query_spec import parse_query_spec
from rag.resolution import QueryContext, resolve_query_spec


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


class ResolutionTest(unittest.TestCase):
    def test_missing_program_is_filled_by_context(self):
        outcome = resolve_query_spec(
            parse_query_spec("ปี 2 เรียนอะไรบ้าง"),
            DB_PATH,
            QueryContext(program="IT"),
        )

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(outcome.resolved_program, "IT")

    def test_matching_program_context_is_accepted(self):
        outcome = resolve_query_spec(
            parse_query_spec("IT ปี 2 เรียนอะไรบ้าง"),
            DB_PATH,
            QueryContext(program="IT"),
        )

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(outcome.resolved_program, "IT")

    def test_calculus_name_context_scopes_to_ait_or_dsba(self):
        for program, code in (("AIT", "06046400"), ("DSBA", "06026200")):
            with self.subTest(program=program):
                outcome = resolve_query_spec(
                    parse_query_spec("วิชา Calculus 1 รหัสวิชาอะไร"),
                    DB_PATH,
                    QueryContext(program=program),
                )

                self.assertEqual(outcome.action, "answer")
                self.assertEqual(outcome.resolved_program, program)
                self.assertEqual(
                    outcome.course_references[0].candidates[0]["course_code"],
                    code,
                )

    def test_unscoped_calculus_name_requires_program_clarification(self):
        outcome = resolve_query_spec(
            parse_query_spec("วิชา Calculus 1 รหัสวิชาอะไร"),
            DB_PATH,
        )

        self.assertEqual(outcome.action, "clarify_program")
        self.assertEqual(outcome.blocking_ambiguity, ("program",))

    def test_calculus_roman_numeral_remains_no_data(self):
        outcome = resolve_query_spec(
            parse_query_spec("วิชา Calculus I รหัสวิชาอะไร"),
            DB_PATH,
        )

        self.assertEqual(outcome.action, "no_data")
        self.assertEqual(outcome.course_references[0].candidates, ())

    def test_conflicting_program_context_is_explicit(self):
        outcome = resolve_query_spec(
            parse_query_spec("AIT ปี 2 เรียนอะไรบ้าง"),
            DB_PATH,
            QueryContext(program="IT"),
        )

        self.assertEqual(outcome.action, "context_conflict")
        self.assertEqual(outcome.context_conflicts, ("program",))
        self.assertEqual(outcome.blocking_ambiguity, ())

    def test_context_plan_fills_missing_plan(self):
        outcome = resolve_query_spec(
            parse_query_spec("IT ปี 2 เรียนอะไรบ้าง"),
            DB_PATH,
            QueryContext(plan="coop"),
        )

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(outcome.resolved_plans, ("coop",))

    def test_matching_plan_context_is_accepted(self):
        outcome = resolve_query_spec(
            parse_query_spec("IT coop ปี 2 เรียนอะไรบ้าง"),
            DB_PATH,
            QueryContext(plan="coop"),
        )

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(outcome.resolved_plans, ("coop",))

    def test_conflicting_plan_context_is_explicit(self):
        outcome = resolve_query_spec(
            parse_query_spec("IT no_coop ปี 2 เรียนอะไรบ้าง"),
            DB_PATH,
            QueryContext(plan="coop"),
        )

        self.assertEqual(outcome.action, "context_conflict")
        self.assertEqual(outcome.context_conflicts, ("plan",))
        self.assertEqual(outcome.blocking_ambiguity, ())

    def test_multiple_plans_conflict_with_restricted_context(self):
        outcome = resolve_query_spec(
            parse_query_spec("IT coop no_coop เปรียบเทียบแผน"),
            DB_PATH,
            QueryContext(plan="coop"),
        )

        self.assertEqual(outcome.action, "context_conflict")
        self.assertEqual(outcome.context_conflicts, ("plan",))

    def test_program_context_does_not_restrict_plan_comparison(self):
        outcome = resolve_query_spec(
            parse_query_spec("IT coop no_coop เปรียบเทียบแผน"),
            DB_PATH,
            QueryContext(program="IT"),
        )

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(outcome.resolved_program, "IT")
        self.assertEqual(outcome.resolved_plans, ("coop", "no_coop"))

    def test_context_does_not_change_query_spec(self):
        spec = parse_query_spec("ปี 2 เรียนอะไรบ้าง")
        resolve_query_spec(spec, DB_PATH, QueryContext(program="IT", plan="coop"))

        self.assertIsNone(spec.program)
        self.assertEqual(spec.plans, ())

    def test_unsupported_stops_before_db_lookup(self):
        spec = parse_query_spec("วิชาไหนยากที่สุด")

        with patch("rag.resolution.exact_course_candidates") as lookup:
            outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "unsupported")
        self.assertEqual(outcome.blocking_ambiguity, ())
        lookup.assert_not_called()

    def test_unknown_exact_code_is_no_data_before_missing_program(self):
        spec = parse_query_spec("06019999 เรียนอะไร")

        outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "no_data")
        self.assertEqual(outcome.blocking_ambiguity, ())
        self.assertEqual(len(outcome.course_references), 1)
        self.assertEqual(outcome.course_references[0].candidates, ())

    def test_ambiguous_name_is_clarify_program(self):
        spec = parse_query_spec("วิชา NOSQL เรียนเรื่องอะไรบ้าง")

        outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "clarify_program")
        self.assertEqual(outcome.blocking_ambiguity, ("program",))
        self.assertEqual(
            {
                (candidate["program"], candidate["course_code"])
                for candidate in outcome.course_references[0].candidates
            },
            {("IT", "06016414"), ("DSBA", "06026207")},
        )

    def test_unique_code_supplies_program_scope(self):
        for question in (
            "06016414 เรียนอะไร",
            "06016414 เรียนเกี่ยวกับอะไร",
        ):
            with self.subTest(question=question):
                outcome = resolve_query_spec(parse_query_spec(question), DB_PATH)
                self.assertEqual(outcome.action, "answer")
                self.assertEqual(outcome.resolved_program, "IT")

    def test_multi_code_same_program_answers(self):
        spec = parse_query_spec("06016414 และ 06016419 คล้ายกันไหม")

        outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(outcome.resolved_program, "IT")
        self.assertEqual(
            [reference.reference for reference in outcome.course_references],
            ["06016414", "06016419"],
        )

    def test_explicit_programs_resolve_each_cross_program_reference(self):
        spec = parse_query_spec(
            "IT วิชา 06016402 และ DSBA วิชา 06026207 คล้ายกันไหม"
        )
        outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(
            [
                (reference.candidates[0]["program"], reference.reference)
                for reference in outcome.course_references
            ],
            [("IT", "06016402"), ("DSBA", "06026207")],
        )

    def test_code_first_program_reference_resolves_explicit_identity(self):
        spec = parse_query_spec("วิชา 06026207 ของ DSBA")
        outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(
            [
                (candidate["program"], candidate["course_code"])
                for candidate in outcome.course_references[0].candidates
            ],
            [("DSBA", "06026207")],
        )

    def test_mixed_program_and_code_first_references_resolve_independently(self):
        spec = parse_query_spec(
            "IT วิชา 06016402 และวิชา 06026207 ของ DSBA เรียนเกี่ยวกับ database อย่างไร?"
        )
        outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(
            [
                (reference.candidates[0]["program"], reference.reference)
                for reference in outcome.course_references
            ],
            [("IT", "06016402"), ("DSBA", "06026207")],
        )

    def test_multi_code_cross_program_clarifies(self):
        spec = parse_query_spec("06016414 และ 06026207 คล้ายกันไหม")

        outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "clarify_program")
        self.assertEqual(outcome.blocking_ambiguity, ("program",))
        self.assertEqual(len(outcome.course_references), 2)

    def test_explicit_program_miss_is_no_data_without_cross_program_fallback(self):
        spec = parse_query_spec("DSBA 06016420 เรียนอะไร")

        outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "no_data")
        self.assertEqual(outcome.resolved_program, "DSBA")

    def test_missing_program_for_topic_clarifies(self):
        spec = parse_query_spec("มีวิชาเกี่ยวกับ database อะไรบ้าง")

        outcome = resolve_query_spec(spec, DB_PATH)

        self.assertEqual(outcome.action, "clarify_program")
        self.assertEqual(outcome.blocking_ambiguity, ("program",))

    def test_missing_plan_is_nonblocking(self):
        outcome = resolve_query_spec(parse_query_spec("IT"), DB_PATH)

        self.assertEqual(outcome.action, "answer")
        self.assertEqual(outcome.resolved_program, "IT")


if __name__ == "__main__":
    unittest.main()
