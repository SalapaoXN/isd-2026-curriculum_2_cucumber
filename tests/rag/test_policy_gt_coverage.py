import json
import unittest
from pathlib import Path

from rag.policy.routing import (
    POLICY_GT_CATEGORY_COVERAGE,
    POLICY_ROUTE_ALLOWLIST,
)


ROOT = Path(__file__).parents[2]
GT_PATH = ROOT / "ground_truth" / "rules_ground_truth.json"
POLICY_PATH = ROOT / "data" / "output" / "final" / "institution_policy.json"


class PolicyGtCoverageTest(unittest.TestCase):
    def test_gt_category_presence_matches_canonical_policy_taxonomy(self):
        ground_truth = json.loads(GT_PATH.read_text(encoding="utf-8"))
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

        presence_by_category: dict[str, set[bool | None]] = {}
        for program_records in ground_truth["programs"].values():
            for record in program_records.values():
                presence_by_category.setdefault(record["category"], set()).add(
                    record.get("present")
                )

        self.assertTrue(
            all(len(states) == 1 for states in presence_by_category.values())
        )
        expected_presence = {
            category: next(iter(states))
            for category, states in presence_by_category.items()
        }
        canonical_presence = {
            category["category"]: category.get("present")
            for category in policy["categories"]
        }

        self.assertEqual(canonical_presence, expected_presence)

    def test_every_gt_present_category_has_declared_qa_coverage(self):
        ground_truth = json.loads(GT_PATH.read_text(encoding="utf-8"))
        presence_by_category: dict[str, set[bool | None]] = {}
        for program_records in ground_truth["programs"].values():
            for record in program_records.values():
                presence_by_category.setdefault(record["category"], set()).add(
                    record.get("present")
                )

        required_categories = {
            category
            for category, states in presence_by_category.items()
            if states == {True}
        }
        unknown_categories = {
            category
            for category, states in presence_by_category.items()
            if states == {None}
        }

        self.assertEqual(unknown_categories, {"ระเบียบอื่น ๆ"})
        self.assertEqual(set(POLICY_GT_CATEGORY_COVERAGE), required_categories)

        routed_or_special = set(POLICY_ROUTE_ALLOWLIST) | {
            "program_total_credits",
            "registration_compare",
        }
        for category, kinds in POLICY_GT_CATEGORY_COVERAGE.items():
            with self.subTest(category=category):
                self.assertTrue(kinds)
                self.assertTrue(set(kinds) <= routed_or_special)

    def test_every_gt_present_category_keeps_canonical_rule_evidence(self):
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        categories = {
            category["category"]: category for category in policy["categories"]
        }
        for category in POLICY_GT_CATEGORY_COVERAGE:
            with self.subTest(category=category):
                record = categories[category]
                self.assertIs(record.get("present"), True)
                evidence = record.get("evidence") or {}
                self.assertTrue(evidence.get("supporting_rule_text"))
                self.assertTrue(evidence.get("source_provenance"))


if __name__ == "__main__":
    unittest.main()
