import unittest

from rag.aggregation import (
    ComparisonAggregation,
    CourseSetAggregation,
    aggregate_course_set,
    aggregate_option_count,
    aggregate_required_load,
    aggregate_sum_credits,
    compare_aggregates,
)
from rag.judgement import (
    JUDGEMENT_EVIDENCE_STATES,
    evaluate_preference,
    evaluate_quantity,
    evaluate_workload,
)


class RagJudgementTest(unittest.TestCase):
    def _course(self, code="00000001", provenance=None):
        return {
            "program": "IT",
            "course_code": code,
            "partition": {"plan": "coop", "year": 2, "semester": 1},
            "provenance": provenance or [{"source_page": 10}],
        }

    def _group(self, complete=True):
        return {
            "program": "IT",
            "course_code": None,
            "alternative_group_id": 7,
            "minimum_choices": 1,
            "membership_complete": complete,
            "alternative_courses": [self._course("00000002")],
            "partition": {"plan": "coop", "year": 2, "semester": 1},
            "provenance": [{"group": 7}],
        }

    def test_standalone_quantity_is_descriptive_only_and_exposes_facts(self):
        count = aggregate_course_set([self._course()])
        result = evaluate_quantity({"count": count})

        self.assertEqual(result.status, "descriptive_only")
        self.assertEqual(result.fact_value("count"), 1)
        self.assertIs(result.facts["count"], count)

    def test_explicit_quantity_comparison_is_supported(self):
        count = aggregate_course_set([self._course()])
        comparison = compare_aggregates(count, 2)
        result = evaluate_quantity({"count": count}, comparison=comparison)

        self.assertEqual(result.status, "supported")
        self.assertEqual(result.comparisons["comparison"].relation, "less")

    def test_sum_credits_passed_as_count_fails_closed(self):
        credits = aggregate_sum_credits(
            [{**self._course(), "counted_credit_units": 3}]
        )

        result = evaluate_quantity({"count": credits})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.fact_value("count"))

    def test_count_passed_as_credits_fails_closed(self):
        count = aggregate_course_set([self._course()])

        result = evaluate_workload(
            {
                "course_count": count,
                "required_load": aggregate_required_load([self._course()]),
                "credits": count,
            }
        )

        self.assertEqual(result.status, "insufficient_evidence")

    def test_option_count_and_required_load_mismatch_fails_closed(self):
        options = aggregate_option_count([self._group()])
        load = aggregate_required_load([self._group()])

        result = evaluate_quantity({"required_load": options})
        workload = evaluate_quantity({"option_count": load})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(workload.status, "insufficient_evidence")

    def test_correct_operation_mappings_still_pass(self):
        result = evaluate_quantity(
            {
                "count": aggregate_course_set([self._course()]),
                "option_count": aggregate_option_count([self._group()]),
                "required_load": aggregate_required_load([self._group()]),
            }
        )

        self.assertEqual(result.status, "descriptive_only")
        self.assertEqual(result.fact_value("count"), 1)

    def test_correct_operation_valid_zero_still_passes(self):
        result = evaluate_quantity(
            {"option_count": aggregate_option_count([])}
        )

        self.assertEqual(result.status, "descriptive_only")
        self.assertEqual(result.fact_value("option_count"), 0)

    def test_valid_zero_count_remains_factual(self):
        result = evaluate_quantity({"count": aggregate_course_set([])})

        self.assertEqual(result.status, "descriptive_only")
        self.assertEqual(result.fact_value("count"), 0)

    def test_insufficient_quantity_facts_propagate(self):
        result = evaluate_quantity(
            {"required_load": aggregate_required_load([self._group(False)])}
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.fact_value("required_load"))

    def test_workload_exposes_count_load_and_credit_proxies(self):
        facts = {
            "course_count": aggregate_course_set([self._course()]),
            "required_load": aggregate_required_load([self._course()]),
            "credits": aggregate_sum_credits(
                [{**self._course(), "counted_credit_units": 3}]
            ),
        }
        result = evaluate_workload(facts)

        self.assertEqual(result.status, "supported")
        self.assertEqual(result.fact_value("course_count"), 1)
        self.assertEqual(result.fact_value("required_load"), 1)
        self.assertEqual(result.fact_value("credits"), 3)
        self.assertIn({"source_page": 10}, result.provenance)
        self.assertNotIn("difficulty", result.facts)

    def test_workload_preserves_conflicting_proxy_relations(self):
        original_facts = {
            "course_count": aggregate_course_set([self._course()]),
            "required_load": aggregate_required_load([self._course()]),
            "credits": aggregate_sum_credits(
                [{**self._course(), "counted_credit_units": 6}]
            ),
        }
        facts = dict(original_facts)
        comparisons = {
            "course_count": compare_aggregates(3, 2),
            "credits": compare_aggregates(2, 3),
        }
        result = evaluate_workload(facts, comparisons=comparisons)

        self.assertEqual(result.status, "supported")
        self.assertEqual(result.comparisons["course_count"].relation, "greater")
        self.assertEqual(result.comparisons["credits"].relation, "less")
        self.assertNotIn("difficulty", result.comparisons)
        self.assertEqual(facts, original_facts)

    def test_mismatched_comparison_source_fails_closed(self):
        count = aggregate_course_set([self._course()])
        credits = aggregate_sum_credits(
            [{**self._course(), "counted_credit_units": 3}]
        )
        facts = {
            "course_count": count,
            "required_load": aggregate_required_load([self._course()]),
            "credits": credits,
        }

        result = evaluate_workload(
            facts,
            comparisons={"course_count": compare_aggregates(credits, 2)},
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.comparisons["course_count"].relation)

    def test_unknown_workload_fact_key_hides_value(self):
        result = evaluate_workload(
            {
                "course_count": aggregate_course_set([self._course()]),
                "required_load": aggregate_required_load([self._course()]),
                "credits": aggregate_sum_credits(
                    [{**self._course(), "counted_credit_units": 3}]
                ),
                "difficulty": aggregate_course_set([self._course()]),
            }
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.facts, {})

    def test_unknown_comparison_key_hides_relation(self):
        facts = {
            "course_count": aggregate_course_set([self._course()]),
            "required_load": aggregate_required_load([self._course()]),
            "credits": aggregate_sum_credits(
                [{**self._course(), "counted_credit_units": 3}]
            ),
        }
        result = evaluate_workload(
            facts,
            comparisons={"difficulty": compare_aggregates(3, 2)},
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.comparisons, {})

    def test_malformed_non_numeric_quantity_comparison_hides_relation(self):
        result = evaluate_quantity(
            {"count": aggregate_course_set([self._course()])},
            comparison=ComparisonAggregation("complete", "less", "one", 2),
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.comparisons["comparison"].relation)

    def test_malformed_non_numeric_workload_comparison_hides_relation(self):
        result = evaluate_workload(
            {
                "course_count": aggregate_course_set([self._course()]),
                "required_load": aggregate_required_load([self._course()]),
                "credits": aggregate_sum_credits(
                    [{**self._course(), "counted_credit_units": 3}]
                ),
            },
            comparisons={
                "course_count": ComparisonAggregation("complete", "less", "one", 2)
            },
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.comparisons["course_count"].relation)

    def test_bool_and_non_finite_comparison_operands_fail_closed(self):
        facts = {"count": aggregate_course_set([self._course()])}
        for left in (True, float("nan"), float("inf")):
            with self.subTest(left=left):
                result = evaluate_quantity(
                    facts,
                    comparison=ComparisonAggregation("complete", "less", left, 2),
                )
                self.assertEqual(result.status, "insufficient_evidence")
                self.assertIsNone(result.comparisons["comparison"].relation)

    def test_comparison_source_value_mismatch_fails_closed(self):
        facts = {"count": aggregate_course_set([self._course()])}
        mismatched_source = aggregate_course_set(
            [self._course(), self._course("00000002")]
        )
        result = evaluate_quantity(
            facts,
            comparison=ComparisonAggregation(
                "complete", "greater", mismatched_source, 0
            ),
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.comparisons["comparison"].relation)

    def test_inconsistent_fabricated_relation_fails_closed(self):
        result = evaluate_quantity(
            {"count": aggregate_course_set([self._course()])},
            comparison=ComparisonAggregation("complete", "greater", 1, 2),
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.comparisons["comparison"].relation)

    def test_invalid_comparison_redacts_both_operands(self):
        result = evaluate_quantity(
            {"count": aggregate_course_set([self._course()])},
            comparison=ComparisonAggregation("complete", "less", "one", 2),
        )

        comparison = result.comparisons["comparison"]
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(comparison.status, "insufficient_evidence")
        self.assertIsNone(comparison.left)
        self.assertIsNone(comparison.right)
        self.assertIsNone(comparison.relation)

    def test_equal_numeric_comparison_exposes_equal(self):
        result = evaluate_quantity(
            {"count": aggregate_course_set([self._course()])},
            comparison=ComparisonAggregation("complete", "equal", 2, 2),
        )

        self.assertEqual(result.status, "supported")
        self.assertEqual(result.comparisons["comparison"].relation, "equal")

    def test_malformed_workload_facts_fail_closed_without_exception(self):
        result = evaluate_workload(None)

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.facts, {})

    def test_malformed_aggregate_provenance_fails_closed_without_exception(self):
        malformed = CourseSetAggregation(
            "complete",
            (
                {
                    "program": "IT",
                    "course_code": "00000001",
                    "partition": {"plan": "coop"},
                    "provenance": 123,
                },
            ),
            1,
            True,
        )

        result = evaluate_quantity({"count": malformed})

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.facts, {})

    def test_malformed_workload_aggregate_provenance_fails_closed(self):
        malformed = CourseSetAggregation(
            "complete",
            (
                {
                    "program": "IT",
                    "course_code": "00000001",
                    "partition": {"plan": "coop"},
                    "provenance": 123,
                },
            ),
            1,
            True,
        )
        course = self._course()
        result = evaluate_workload(
            {
                "course_count": malformed,
                "required_load": aggregate_required_load([course]),
                "credits": aggregate_sum_credits(
                    [{**course, "counted_credit_units": 3}]
                ),
            }
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.facts, {})

    def test_malformed_comparison_fails_closed_without_exception(self):
        facts = {
            "course_count": aggregate_course_set([self._course()]),
            "required_load": aggregate_required_load([self._course()]),
            "credits": aggregate_sum_credits(
                [{**self._course(), "counted_credit_units": 3}]
            ),
        }
        result = evaluate_workload(facts, comparisons=["not-a-comparison"])

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.comparisons, {})

    def test_malformed_preference_provenance_fails_closed(self):
        option = {
            **self._course(),
            "description_evidence": (
                {
                    "chunk_id": "a",
                    "text": "topic",
                    "provenance": ({"source_page": 1},),
                },
            ),
        }
        option["provenance"] = 123
        malformed_content = dict(option)
        malformed_content["provenance"] = [123]

        for invalid in (option, malformed_content):
            with self.subTest(provenance=invalid["provenance"]):
                result = evaluate_preference([invalid])
                self.assertEqual(result.status, "insufficient_evidence")
                self.assertEqual(result.options, ())

    def test_workload_missing_or_insufficient_proxy_is_safe(self):
        missing = evaluate_workload({"course_count": aggregate_course_set([])})
        insufficient = evaluate_workload(
            {
                "course_count": aggregate_course_set([self._course()]),
                "required_load": aggregate_required_load([self._group(False)]),
                "credits": aggregate_sum_credits(
                    [{**self._course(), "counted_credit_units": 3}]
                ),
            }
        )

        self.assertEqual(missing.status, "insufficient_evidence")
        self.assertEqual(insufficient.status, "insufficient_evidence")

    def test_insufficient_comparison_does_not_become_factual_relation(self):
        left = aggregate_required_load([self._group(False)])
        result = evaluate_workload(
            {
                "course_count": aggregate_course_set([self._course()]),
                "required_load": aggregate_required_load([self._course()]),
                "credits": aggregate_sum_credits(
                    [{**self._course(), "counted_credit_units": 3}]
                ),
            },
            comparisons={"required_load": compare_aggregates(left, 1)},
        )

        self.assertEqual(result.status, "insufficient_evidence")

    def test_preference_preserves_options_and_provenance_without_ranking(self):
        options = [
            {
                **self._course("00000002", [{"source_page": 2}]),
                "description_evidence": (
                    {"chunk_id": "b", "text": "topic", "distance": 0.2,
                     "provenance": ({"source_page": 2},)},
                ),
            },
            {
                **self._course("00000001", [{"source_page": 1}]),
                "description_evidence": (
                    {"chunk_id": "a", "text": "topic", "distance": 0.8,
                     "provenance": ({"source_page": 1},)},
                ),
            },
        ]
        result = evaluate_preference(options)

        self.assertEqual(result.status, "supported")
        self.assertEqual(
            [option["course_code"] for option in result.options],
            ["00000002", "00000001"],
        )
        self.assertEqual(result.options[0]["course_code"], "00000002")
        self.assertEqual(result.options[0]["description_evidence"][0]["distance"], 0.2)
        self.assertNotIn("rank", result.options[0])
        self.assertIn({"source_page": 1}, result.provenance)
        self.assertIn({"source_page": 2}, result.provenance)

    def test_preference_missing_identity_fails_closed(self):
        option = {
            **self._course(),
            "description_evidence": (
                {"chunk_id": "a", "text": "topic", "provenance": ({"source_page": 1},)},
            ),
        }
        for field in ("program", "course_code"):
            invalid = dict(option)
            invalid.pop(field)
            with self.subTest(field=field):
                self.assertEqual(
                    evaluate_preference([invalid]).status,
                    "insufficient_evidence",
                )

    def test_preference_missing_grounded_description_fails_closed(self):
        self.assertEqual(
            evaluate_preference([self._course()]).status,
            "insufficient_evidence",
        )

    def test_preference_missing_provenance_fails_closed(self):
        option = {
            **self._course(provenance=[]),
            "description_evidence": (
                {"chunk_id": "a", "text": "topic", "provenance": ()},
            ),
        }
        self.assertEqual(
            evaluate_preference([option]).status,
            "insufficient_evidence",
        )

    def test_preference_inconsistent_evidence_identity_fails_closed(self):
        option = {
            **self._course(),
            "description_evidence": (
                {
                    "chunk_id": "a",
                    "text": "topic",
                    "program": "DSBA",
                    "provenance": ({"source_page": 1},),
                },
            ),
        }
        self.assertEqual(
            evaluate_preference([option]).status,
            "insufficient_evidence",
        )

    def test_preference_lexical_rescue_without_distance_is_grounded(self):
        option = {
            **self._course(),
            "description_evidence": (
                {
                    "chunk_id": "a",
                    "text": "Artificial intelligence",
                    "distance": None,
                    "provenance": ({"source_page": 1},),
                },
            ),
        }
        result = evaluate_preference([option])
        self.assertEqual(result.status, "supported")
        self.assertIsNone(result.options[0]["description_evidence"][0]["distance"])

    def test_preference_empty_options_are_not_fabricated(self):
        result = evaluate_preference([])

        self.assertEqual(result.status, "supported")
        self.assertEqual(result.options, ())

    def test_preference_retrieval_missing_state_is_insufficient(self):
        source = type(
            "RetrievalResult",
            (),
            {"status": "description_missing", "scored_candidates": ()},
        )()

        result = evaluate_preference(source)

        self.assertEqual(result.status, "insufficient_evidence")

    def test_results_are_immutable_and_state_vocabulary_is_frozen(self):
        self.assertEqual(
            JUDGEMENT_EVIDENCE_STATES,
            ("supported", "descriptive_only", "insufficient_evidence"),
        )
        result = evaluate_preference(
            [
                {
                    **self._course(),
                    "description_evidence": (
                        {
                            "chunk_id": "immutable",
                            "text": "topic",
                            "provenance": ({"source_page": 10},),
                        },
                    ),
                }
            ]
        )
        with self.assertRaises(AttributeError):
            result.status = "supported"
        with self.assertRaises(TypeError):
            result.options[0]["course_code"] = "changed"


if __name__ == "__main__":
    unittest.main()
