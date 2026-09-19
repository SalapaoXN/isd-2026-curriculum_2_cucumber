import unittest

from rag.aggregation import (
    AGGREGATION_OPERATIONS,
    AGGREGATION_STATES,
    ComponentAggregation,
    CourseSetAggregation,
    aggregate_earliest,
    aggregate_option_count,
    aggregate_plan_comparison,
    aggregate_required_load,
    aggregate_sum_credits,
    aggregate_course_set,
    compare_aggregates,
    PlanComparisonInput,
)


class RagAggregationTest(unittest.TestCase):
    def _course(self, program, code, plan, provenance=()):
        return {
            "program": program,
            "course_code": code,
            "course_id": f"{program}-{code}",
            "partition": {"plan": plan, "year": 2, "semester": 1},
            "provenance": list(provenance),
        }

    def _group(
        self,
        program="IT",
        group_id=1,
        plan="coop",
        minimum_choices=1,
        members=("00000001", "00000002"),
        membership_complete=True,
        counted_credit_units=6,
    ):
        return {
            "program": program,
            "course_code": None,
            "alternative_group_id": group_id,
            "minimum_choices": minimum_choices,
            "membership_complete": membership_complete,
            "counted_credit_units": counted_credit_units,
            "alternative_courses": [
                {
                    "program": program,
                    "course_code": code,
                    "course_id": f"{program}-{code}",
                    "provenance": [{"course": code}],
                }
                for code in members
            ],
            "partition": {"plan": plan, "year": 2, "semester": 1},
            "provenance": [{"group": group_id}],
        }

    def _placement(self, placement_id, plan="coop", year=2, semester=1, **extra):
        return {
            "placement_id": placement_id,
            "program": "IT",
            "course_code": extra.pop("course_code", "00000001"),
            "partition": {"plan": plan},
            "year": year,
            "semester": semester,
            "provenance": [{"placement": placement_id}],
            **extra,
        }

    def _plan_input(self, plan, codes, placements, *, complete=True):
        courses = [self._course("IT", code, plan) for code in codes]
        return PlanComparisonInput(
            plan=plan,
            course_set=aggregate_course_set(courses),
            placements=placements,
            placements_complete=complete,
        )

    def test_duplicate_identity_within_partition_is_one_course_and_merges_provenance(self):
        components = [
            self._course("IT", "00000001", "coop", [{"source_page": 2}]),
            self._course("IT", "00000001", "coop", [{"source_page": 1}]),
        ]
        result = aggregate_course_set(components)

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.count, 1)
        self.assertTrue(result.exists)
        self.assertEqual(len(result.courses), 1)
        self.assertEqual(
            result.courses[0]["provenance"],
            ({"source_page": 1}, {"source_page": 2}),
        )

    def test_same_course_across_plan_partitions_remains_separate(self):
        result = aggregate_course_set(
            [
                self._course("IT", "00000001", "no_coop"),
                self._course("IT", "00000001", "coop"),
            ]
        )

        self.assertEqual(result.count, 2)
        self.assertEqual(
            {course["partition"]["plan"] for course in result.courses},
            {"coop", "no_coop"},
        )

    def test_output_order_is_deterministic_independent_of_input_order(self):
        components = [
            self._course("IT", "00000002", "no_coop"),
            self._course("BIT", "00000001", "coop"),
            self._course("IT", "00000001", "coop"),
        ]
        forward = aggregate_course_set(components)
        reverse = aggregate_course_set(reversed(components))

        self.assertEqual(forward.courses, reverse.courses)

    def test_list_count_and_existence_agree(self):
        result = aggregate_course_set(
            [self._course("IT", "00000001", "coop")]
        )

        self.assertEqual(len(result.courses), result.count)
        self.assertEqual(result.exists, result.count > 0)

    def test_complete_empty_evidence_is_valid_empty(self):
        result = aggregate_course_set([])

        self.assertEqual(result.status, "valid_empty")
        self.assertEqual(result.courses, ())
        self.assertEqual(result.count, 0)
        self.assertFalse(result.exists)

    def test_incomplete_evidence_does_not_fabricate_zero_or_false(self):
        result = aggregate_course_set([], evidence_complete=False)

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.count)
        self.assertIsNone(result.exists)

    def test_incomplete_evidence_preserves_known_components(self):
        component = self._course("IT", "00000001", "coop")
        result = aggregate_course_set([component], evidence_complete=False)

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(len(result.courses), 1)
        self.assertIsNone(result.count)
        self.assertIsNone(result.exists)

    def test_partition_metadata_and_provenance_are_preserved(self):
        component = self._course(
            "IT", "00000001", "coop", [{"source_page": 42}]
        )
        result = aggregate_course_set([component])

        self.assertEqual(
            result.courses[0]["partition"],
            {"plan": "coop", "year": 2, "semester": 1},
        )
        self.assertEqual(result.courses[0]["provenance"], ({"source_page": 42},))

    def test_alternative_parent_is_one_course_set_item(self):
        component = self._group(group_id=10)
        result = aggregate_course_set([component])

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.count, 1)
        self.assertTrue(result.exists)
        item = result.courses[0]
        self.assertIsNone(item["course_code"])
        self.assertEqual(item["alternative_group_id"], 10)
        self.assertEqual(item["minimum_choices"], 1)
        self.assertEqual(
            {member["course_code"] for member in item["alternative_courses"]},
            {"00000001", "00000002"},
        )
        self.assertEqual(item["provenance"], ({"group": 10},))

    def test_duplicate_alternative_parent_merges_parent_and_member_provenance(self):
        first = self._group(group_id=11)
        second = self._group(group_id=11)
        second["provenance"] = [{"group_source": 11}]
        second["alternative_courses"][0]["provenance"].append(
            {"member_source": "00000001"}
        )

        result = aggregate_course_set([first, second])

        self.assertEqual(result.count, 1)
        item = result.courses[0]
        self.assertIn({"group": 11}, item["provenance"])
        self.assertIn({"group_source": 11}, item["provenance"])
        member = next(
            member
            for member in item["alternative_courses"]
            if member["course_code"] == "00000001"
        )
        self.assertIn(
            {"member_source": "00000001"}, member["provenance"]
        )

    def test_distinct_alternative_groups_remain_distinct(self):
        result = aggregate_course_set(
            [self._group(group_id=12), self._group(group_id=13)]
        )

        self.assertEqual(result.count, 2)
        self.assertEqual(
            {item["alternative_group_id"] for item in result.courses},
            {12, 13},
        )

    def test_same_alternative_group_across_plans_remains_separate(self):
        result = aggregate_course_set(
            [
                self._group(group_id=14, plan="coop"),
                self._group(group_id=14, plan="no_coop"),
            ]
        )

        self.assertEqual(result.count, 2)
        self.assertEqual(
            {item["partition"]["plan"] for item in result.courses},
            {"coop", "no_coop"},
        )

    def test_course_set_rejects_missing_concrete_or_group_identity(self):
        with self.assertRaises(ValueError):
            aggregate_course_set(
                [{"program": "IT", "course_code": None, "partition": {}}]
            )
        with self.assertRaises(ValueError):
            aggregate_course_set(
                [
                    {
                        "program": "IT",
                        "course_code": None,
                        "is_alternative": True,
                        "partition": {},
                    }
                ]
            )

    def test_inputs_are_not_mutated(self):
        component = self._course("IT", "00000001", "coop", [{"source_page": 1}])
        original = {
            key: value.copy() if isinstance(value, dict) else list(value)
            if isinstance(value, list)
            else value
            for key, value in component.items()
        }
        aggregate_course_set([component])

        self.assertEqual(component, original)

    def test_result_and_state_vocabulary_are_immutable(self):
        self.assertEqual(
            AGGREGATION_STATES,
            ("complete", "valid_empty", "insufficient_evidence"),
        )
        result = CourseSetAggregation("valid_empty", count=0, exists=False)
        with self.assertRaises(AttributeError):
            result.status = "complete"

    def test_required_load_counts_normal_courses(self):
        result = aggregate_required_load(
            [
                self._course("IT", "00000001", "coop"),
                self._course("IT", "00000001", "coop"),
                self._course("IT", "00000002", "coop"),
            ]
        )

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.value, 2)

    def test_required_load_complete_one_of_two_group(self):
        result = aggregate_required_load([self._group()])

        self.assertEqual(result.value, 1)
        self.assertEqual(result.components[0]["alternative_group_id"], 1)
        self.assertEqual(result.components[0]["alternative_courses"][0]["course_code"], "00000001")

    def test_required_load_uses_minimum_choices_without_selecting_members(self):
        result = aggregate_required_load(
            [self._group(group_id=2, minimum_choices=2)]
        )

        self.assertEqual(result.value, 2)
        self.assertEqual(
            len(result.components[0]["alternative_courses"]),
            2,
        )

    def test_option_count_counts_group_members_not_synthetic_group_row(self):
        result = aggregate_option_count(
            [
                self._group(),
                self._course("IT", "00000003", "coop"),
            ]
        )

        self.assertEqual(result.value, 3)

    def test_partial_group_is_valid_option_count_but_insufficient_required_load(self):
        partial = self._group(
            group_id=3,
            members=("00000001",),
            membership_complete=False,
        )

        option_result = aggregate_option_count([partial])
        load_result = aggregate_required_load([partial])

        self.assertEqual(option_result.status, "complete")
        self.assertEqual(option_result.value, 1)
        self.assertEqual(load_result.status, "insufficient_evidence")
        self.assertIsNone(load_result.value)

    def test_duplicate_group_within_partition_counts_once(self):
        first = self._group(group_id=4)
        second = self._group(group_id=4)
        result = aggregate_required_load([first, second])

        self.assertEqual(result.value, 1)
        self.assertEqual(len(result.components), 1)

    def test_same_group_across_partitions_remains_separate(self):
        coop = self._group(group_id=5, plan="coop")
        no_coop = self._group(group_id=5, plan="no_coop")
        result = aggregate_required_load([coop, no_coop])

        self.assertEqual(result.value, 2)
        self.assertEqual(
            {component["partition"]["plan"] for component in result.components},
            {"coop", "no_coop"},
        )

    def test_sum_credits_consumes_counted_credit_units_and_keeps_lineage(self):
        normal = self._course("IT", "00000003", "coop")
        normal["counted_credit_units"] = 3
        group = self._group(group_id=6, counted_credit_units=6)
        result = aggregate_sum_credits([normal, group])

        self.assertEqual(result.value, 9)
        grouped = next(
            component
            for component in result.components
            if component["alternative_group_id"] == 6
        )
        self.assertEqual(grouped["alternative_group_id"], 6)
        self.assertEqual(
            grouped["alternative_courses"][0]["provenance"],
            ({"course": "00000001"},),
        )

    def test_missing_or_malformed_credit_fact_is_insufficient(self):
        missing = self._course("IT", "00000001", "coop")
        malformed = self._course("IT", "00000002", "coop")
        malformed["counted_credit_units"] = "six"

        for component in (missing, malformed):
            result = aggregate_sum_credits([component])
            self.assertEqual(result.status, "insufficient_evidence")
            self.assertIsNone(result.value)

    def test_scalar_results_and_inputs_are_immutable(self):
        group = self._group(group_id=7)
        original_members = [dict(member) for member in group["alternative_courses"]]
        result = aggregate_option_count([group])

        self.assertIsInstance(result, ComponentAggregation)
        self.assertEqual(
            result.components[0]["alternative_courses"][0]["course_code"],
            "00000001",
        )
        self.assertEqual(group["alternative_courses"], original_members)
        with self.assertRaises(TypeError):
            result.components[0]["partition"] = {}

    def test_scalar_operation_and_state_vocabulary(self):
        self.assertEqual(
            AGGREGATION_OPERATIONS,
            ("option_count", "required_load", "sum_credits"),
        )
        self.assertEqual(
            AGGREGATION_STATES,
            ("complete", "valid_empty", "insufficient_evidence"),
        )

    def test_earliest_uses_lexicographic_year_semester_order(self):
        result = aggregate_earliest(
            [
                self._placement(1, year=2, semester=1),
                self._placement(2, year=1, semester=2),
                self._placement(3, year=2, semester=1),
            ]
        )

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.value, (1, 2))
        self.assertEqual([p["placement_id"] for p in result.partitions[0].placements], [2])

    def test_earliest_preserves_all_ties_and_provenance(self):
        result = aggregate_earliest(
            [
                self._placement(2, year=1, semester=1),
                self._placement(1, year=1, semester=1),
                self._placement(3, year=2, semester=1),
            ]
        )
        partition = result.partitions[0]

        self.assertEqual(partition.value, (1, 1))
        self.assertEqual(
            [p["placement_id"] for p in partition.placements],
            [1, 2],
        )
        self.assertEqual(
            partition.provenance,
            ({"placement": 1}, {"placement": 2}),
        )

    def test_earliest_keeps_partitions_isolated(self):
        result = aggregate_earliest(
            [
                self._placement(1, plan="coop", year=2, semester=1),
                self._placement(2, plan="no_coop", year=1, semester=2),
            ]
        )

        self.assertEqual(len(result.partitions), 2)
        self.assertEqual(
            {
                partition.partition["plan"]: partition.value
                for partition in result.partitions
            },
            {"coop": (2, 1), "no_coop": (1, 2)},
        )

    def test_earliest_empty_and_incomplete_states_are_distinct(self):
        self.assertEqual(aggregate_earliest([]).status, "valid_empty")
        self.assertEqual(
            aggregate_earliest([self._placement(1, year="2")]).status,
            "insufficient_evidence",
        )
        self.assertEqual(
            aggregate_earliest(
                [self._placement(1, year=2, semester=1)],
                evidence_complete=False,
            ).status,
            "insufficient_evidence",
        )

    def test_earliest_accepts_existing_flexible_placement_choices(self):
        result = aggregate_earliest(
            [
                self._placement(
                    1,
                    year=None,
                    semester=None,
                    year_semester_choices=((3, 2), (2, 2)),
                )
            ]
        )

        self.assertEqual(result.value, (2, 2))

    def test_numeric_comparison_has_deterministic_relations(self):
        self.assertEqual(compare_aggregates(1, 2).relation, "less")
        self.assertEqual(compare_aggregates(2, 1).relation, "greater")
        self.assertEqual(compare_aggregates(2, 2).relation, "equal")

    def test_tuple_comparison_uses_year_then_semester(self):
        self.assertEqual(
            compare_aggregates((2, 1), (2, 2)).relation,
            "less",
        )
        self.assertEqual(
            compare_aggregates((3, 1), (2, 2)).relation,
            "greater",
        )
        self.assertEqual(
            compare_aggregates((2, 2), (2, 2)).relation,
            "equal",
        )

    def test_valid_zero_is_comparable(self):
        zero = aggregate_option_count([])
        result = compare_aggregates(zero, 1)

        self.assertEqual(zero.status, "valid_empty")
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.relation, "less")

    def test_insufficient_side_stops_comparison_and_preserves_sources(self):
        left = aggregate_required_load(
            [self._group(members=("00000001",), membership_complete=False)]
        )
        right = aggregate_option_count([self._group()])
        result = compare_aggregates(left, right)

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.relation)
        self.assertIs(result.left, left)
        self.assertIs(result.right, right)

    def test_earliest_does_not_mutate_input(self):
        placements = [self._placement(1, year=2, semester=1)]
        original = dict(placements[0])

        aggregate_earliest(placements)

        self.assertEqual(placements[0], original)

    def test_plan_comparison_identical_sets_and_placements_are_complete(self):
        left = self._plan_input(
            "coop",
            ("00000001", "00000002"),
            [
                self._placement(1, "coop", 1, 1, course_code="00000001"),
                self._placement(2, "coop", 2, 1, course_code="00000002"),
            ],
        )
        right = self._plan_input(
            "no_coop",
            ("00000001", "00000002"),
            [
                self._placement(3, "no_coop", 1, 1, course_code="00000001"),
                self._placement(4, "no_coop", 2, 1, course_code="00000002"),
            ],
        )

        result = aggregate_plan_comparison(left, right)

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.only_left, ())
        self.assertEqual(result.only_right, ())
        self.assertEqual(result.placement_differences, ())
        self.assertTrue(result.provenance)

    def test_plan_comparison_reports_left_and_right_only_courses(self):
        left = self._plan_input(
            "coop",
            ("00000001", "00000002"),
            [self._placement(1, "coop", course_code="00000001")],
        )
        right = self._plan_input(
            "no_coop",
            ("00000001", "00000003"),
            [self._placement(2, "no_coop", course_code="00000001")],
        )

        result = aggregate_plan_comparison(left, right)

        self.assertEqual(
            [course["course_code"] for course in result.only_left], ["00000002"]
        )
        self.assertEqual(
            [course["course_code"] for course in result.only_right], ["00000003"]
        )

    def test_plan_comparison_reports_shared_placement_difference(self):
        left = self._plan_input(
            "coop",
            ("00000001",),
            [self._placement(1, "coop", 2, 1, course_code="00000001")],
        )
        right = self._plan_input(
            "no_coop",
            ("00000001",),
            [self._placement(2, "no_coop", 3, 2, course_code="00000001")],
        )

        result = aggregate_plan_comparison(left, right)

        self.assertEqual(len(result.placement_differences), 1)
        difference = result.placement_differences[0]
        self.assertEqual(difference.course_key, ("course", ("IT", "00000001")))
        self.assertEqual(difference.left_periods, ((2, 1),))
        self.assertEqual(difference.right_periods, ((3, 2),))
        self.assertEqual(
            [reference["placement"] for reference in difference.provenance], [1, 2]
        )

    def test_plan_comparison_keeps_multiple_placement_differences_keyed(self):
        left = self._plan_input(
            "coop",
            ("00000001", "00000002"),
            [
                self._placement(1, "coop", 1, 1, course_code="00000001"),
                self._placement(2, "coop", 2, 1, course_code="00000002"),
            ],
        )
        right = self._plan_input(
            "no_coop",
            ("00000001", "00000002"),
            [
                self._placement(3, "no_coop", 2, 1, course_code="00000001"),
                self._placement(4, "no_coop", 3, 1, course_code="00000002"),
            ],
        )

        result = aggregate_plan_comparison(left, right)

        self.assertEqual(
            [difference.course_key for difference in result.placement_differences],
            [
                ("course", ("IT", "00000001")),
                ("course", ("IT", "00000002")),
            ],
        )

    def test_plan_comparison_incomplete_side_fails_closed(self):
        complete = self._plan_input(
            "coop",
            ("00000001",),
            [self._placement(1, "coop", course_code="00000001")],
        )
        incomplete = self._plan_input(
            "no_coop",
            ("00000001",),
            [self._placement(2, "no_coop", course_code="00000001")],
            complete=False,
        )

        result = aggregate_plan_comparison(complete, incomplete)

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.placement_differences, ())

    def test_plan_comparison_ambiguous_duplicate_identity_fails_closed(self):
        first = self._course("IT", "00000001", "coop")
        second = self._course("IT", "00000001", "coop")
        left = PlanComparisonInput(
            "coop",
            CourseSetAggregation("complete", (first, second), count=2, exists=True),
            (self._placement(1, "coop", course_code="00000001"),),
        )
        right = self._plan_input(
            "no_coop",
            ("00000001",),
            [self._placement(2, "no_coop", course_code="00000001")],
        )

        result = aggregate_plan_comparison(left, right)

        self.assertEqual(result.status, "insufficient_evidence")

    def test_plan_comparison_rejects_placement_not_in_course_set(self):
        left = self._plan_input(
            "coop",
            ("00000001",),
            [self._placement(1, "coop", course_code="00000002")],
        )
        right = self._plan_input(
            "no_coop",
            ("00000001",),
            [self._placement(2, "no_coop", course_code="00000001")],
        )

        result = aggregate_plan_comparison(left, right)

        self.assertEqual(result.status, "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
