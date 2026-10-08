import unittest

from rag.semantic.compiler import _operations_for
from rag.semantic.schema import (
    ResolvedIntent,
    SemanticIntent,
)


class RequestedFieldCompilationTests(unittest.TestCase):
    def operations(self, relation, fields):
        return _operations_for(
            ResolvedIntent(
                intent=SemanticIntent(
                    task="lookup",
                    subject="course",
                    relation=relation,
                    requested_fields=tuple(fields),
                )
            )
        )

    def test_identity_and_credits(self):
        self.assertEqual(self.operations("identity", ["name", "credits"]), ("identity", "sum_credits"))

    def test_placement_and_credits(self):
        self.assertEqual(self.operations("placement", ["placement", "credits"]), ("placement", "sum_credits"))

    def test_placement_and_description(self):
        self.assertEqual(self.operations("placement", ["placement", "description"]), ("placement", "describe"))

    def test_prerequisite_and_another_field(self):
        self.assertEqual(self.operations("prerequisite", ["prerequisites", "credits"]), ("prerequisite", "sum_credits"))

    def test_duplicates_collapse_in_stable_order(self):
        self.assertEqual(self.operations("credits", ["credits", "name", "description", "name"]), ("identity", "sum_credits", "describe"))

    def test_relation_operation_survives_requested_fields(self):
        self.assertEqual(self.operations("description", ["credits"]), ("describe", "sum_credits"))

    def test_pure_single_field_lookup_unchanged(self):
        self.assertEqual(self.operations("credits", []), ("identity", "sum_credits"))

    def test_non_lookup_tasks_unchanged(self):
        intent = SemanticIntent(task="compare", subject="course", requested_fields=("credits",))
        self.assertEqual(_operations_for(ResolvedIntent(intent=intent)), ())


if __name__ == "__main__":
    unittest.main()
