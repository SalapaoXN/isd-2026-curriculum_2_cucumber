"""Fail-closed validation for the deliberately narrow semantic filter contract."""

import unittest

from rag.semantic.schema import AggregationSpec, SemanticFilter, SemanticIntent
from rag.semantic.validation import validate_semantic_intent


def _intent(
    filters, *, task="list", subject="course", aggregation=None, relation=None
):
    return SemanticIntent(
        task=task,
        subject=subject,
        relation=relation,
        filters=tuple(filters),
        aggregation=aggregation,
    )


class SemanticFilterValidationTests(unittest.TestCase):
    def test_proven_collection_filters_remain_valid(self):
        cases = (
            (
                SemanticFilter(field="topic", operator="related_to", value="networks"),
                "list courses related to networks",
            ),
            (
                SemanticFilter(field="category", operator="eq", value="required"),
                "list courses in required category",
            ),
        )
        for filter_item, question in cases:
            with self.subTest(field=filter_item.field):
                result = validate_semantic_intent(
                    _intent((filter_item,)), question
                )
                self.assertTrue(result.valid, result.reason)

    def test_proven_filters_can_be_composed_without_losing_either_field(self):
        result = validate_semantic_intent(
            _intent(
                (
                    SemanticFilter(
                        field="topic", operator="related_to", value="networks"
                    ),
                    SemanticFilter(
                        field="category", operator="eq", value="required"
                    ),
                )
            ),
            "list courses related to networks in required category",
        )
        self.assertTrue(result.valid, result.reason)

    def test_proven_filters_are_valid_for_supported_scalar_aggregates(self):
        aggregations = (
            AggregationSpec(function="sum", measure="credits"),
            AggregationSpec(function="count", measure="course_count"),
        )
        for aggregation in aggregations:
            with self.subTest(aggregation=aggregation):
                result = validate_semantic_intent(
                    _intent(
                        (SemanticFilter(field="category", operator="eq", value="required"),),
                        task="aggregate",
                        aggregation=aggregation,
                    ),
                    "aggregate required category",
                )
                self.assertTrue(result.valid, result.reason)

    def test_positive_prerequisite_filter_is_valid_only_for_course_list(self):
        filter_item = SemanticFilter(
            field="has_prerequisite", operator="eq", value=True
        )
        result = validate_semantic_intent(
            _intent((filter_item,), task="list", subject="course"),
            "list courses with prerequisites",
        )
        self.assertTrue(result.valid, result.reason)

        rejected_shapes = (
            _intent((filter_item,), task="search", subject="course"),
            _intent(
                (filter_item,),
                task="aggregate",
                aggregation=AggregationSpec(function="count", measure="course_count"),
            ),
            _intent(
                (
                    filter_item,
                    SemanticFilter(field="category", operator="eq", value="required"),
                ),
                task="list",
                subject="course",
            ),
        )
        for intent in rejected_shapes:
            with self.subTest(task=intent.task, subject=intent.subject, count=len(intent.filters)):
                rejected = validate_semantic_intent(intent, "list or search courses")
                self.assertFalse(rejected.valid)
                self.assertIn("filter", rejected.reason or "")

    def test_unsupported_filter_shapes_fail_closed(self):
        cases = (
            ("topic", "eq", "networks"),
            ("topic", "contains", "networks"),
            ("category", "related_to", "required"),
            ("has_prerequisite", "eq", False),
            ("credits", "eq", 3),
            ("credits", "ne", 3),
            ("credits", "lt", 3),
            ("credits", "lte", 3),
            ("credits", "gt", 3),
            ("credits", "gte", 3),
            ("credits", "between", (2, 4)),
            ("year", "eq", 2),
            ("semester", "eq", 1),
            ("plan", "eq", "coop"),
            ("topic", "ne", "networks"),
        )
        for field, operator, value in cases:
            with self.subTest(field=field, operator=operator):
                result = validate_semantic_intent(
                    _intent((SemanticFilter(field, operator, value),)),
                    "list courses",
                )
                self.assertFalse(result.valid)
                self.assertIn("filter", result.reason or "")

    def test_proven_filter_shapes_fail_on_unproven_tasks_or_subjects(self):
        filter_item = SemanticFilter(
            field="category", operator="eq", value="required"
        )
        cases = (
            _intent(
                (filter_item,),
                task="lookup",
                subject="program",
                relation="credits",
            ),
            _intent((filter_item,), task="list", subject="semester"),
            _intent(
                (filter_item,),
                task="aggregate",
                aggregation=AggregationSpec(function="average", measure="credits"),
            ),
            _intent(
                (filter_item,),
                task="aggregate",
                aggregation=AggregationSpec(
                    function="sum", measure="credits", group_by=("year",)
                ),
            ),
        )
        for intent in cases:
            with self.subTest(task=intent.task, subject=intent.subject):
                result = validate_semantic_intent(intent, "valid question")
                self.assertFalse(result.valid)
                self.assertIn("filter", result.reason or "")

    def test_duplicate_supported_field_is_rejected_before_compilation(self):
        result = validate_semantic_intent(
            _intent(
                (
                    SemanticFilter(field="category", operator="eq", value="required"),
                    SemanticFilter(field="category", operator="eq", value="elective"),
                )
            ),
            "list courses",
        )
        self.assertFalse(result.valid)
        self.assertIn("duplicate semantic filter field", result.reason or "")


if __name__ == "__main__":
    unittest.main()
