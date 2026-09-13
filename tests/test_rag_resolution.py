import unittest
from pathlib import Path
from unittest.mock import patch

from rag.query_spec import parse_query_spec
from rag.resolution import resolve_query_spec


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


class ResolutionTest(unittest.TestCase):
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
