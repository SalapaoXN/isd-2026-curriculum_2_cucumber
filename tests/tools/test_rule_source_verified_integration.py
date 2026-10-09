import json
import unittest
from pathlib import Path

from src.pipeline.tools.extraction.rules import apply_source_verified_rule_corrections
from src.pipeline.tools.merge.policy import (
    APPEAL_CATEGORY,
    GRADING_CATEGORY,
    GRADUATION_CATEGORY,
    HONORS_CATEGORY,
    RulesPolicyMapper,
)


ROOT = Path(__file__).resolve().parents[2]


def category(payload, name):
    return next(item for item in payload["categories"] if item["category"] == name)


class RuleSourceVerifiedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = json.loads(
            (ROOT / "data" / "output" / "rules_extracted.json").read_text(
                encoding="utf-8"
            )
        )
        cls.corrected = apply_source_verified_rule_corrections(raw)
        cls.policy = RulesPolicyMapper().map_data(cls.corrected)

    def test_all_reviewed_corrections_match_real_rule_provenance(self):
        self.assertEqual(self.corrected["source_verified_correction_count"], 13)
        verified = {
            rule["rule_id"]: rule
            for rule in self.corrected["rules"]
            if rule.get("source_verified") is True
        }
        self.assertEqual(len(verified), 13)
        self.assertIn("๓๐ ชั่วโมง", verified["rule:6.3.2"]["rule_text"])
        self.assertIn("ทุจริตในการสอบมากกว่า ๑ ครั้ง", verified["rule:33.8"]["rule_text"])
        self.assertIn("ภายใน ๓๐ วัน", verified["rule:43"]["rule_text"])

    def test_grade_table_is_complete_after_source_correction(self):
        grading = category(self.policy, GRADING_CATEGORY)
        grade_points = {
            item["grade"]: item["value"]
            for item in grading["values"]
            if item.get("condition") == "grade_point"
        }
        self.assertEqual(
            grade_points,
            {
                "A": "4.00",
                "B+": "3.50",
                "B": "3.00",
                "C+": "2.50",
                "C": "2.00",
                "D+": "1.50",
                "D": "1.00",
                "F": "0",
            },
        )

    def test_graduation_gpa_is_recovered_from_rules_25_1_and_25_2(self):
        graduation = category(self.policy, GRADUATION_CATEGORY)
        values = {
            (item.get("source_rule_id"), item.get("value"))
            for item in graduation["values"]
        }
        self.assertIn(("rule:25.1", "2.00"), values)
        self.assertIn(("rule:25.2", "2.00"), values)

    def test_honors_uses_clean_source_values_not_ocr_symbols(self):
        honors = category(self.policy, HONORS_CATEGORY)
        values = {item["value"] for item in honors["values"]}
        self.assertTrue({"3.75", "3.50", "3.25", "B or S", "2/3"}.issubset(values))
        evidence = {
            item["rule_id"]: item
            for item in honors["evidence"]["supporting_rule_text"]
        }
        self.assertEqual(evidence["rule:27.1.2"]["rule_text"], "ไม่มีรายวิชาใดได้ค่าระดับคะแนน F หรือ U")
        self.assertTrue(
            any(
                snippet.get("status") == "source_verified"
                for snippet in evidence["rule:27.1.2"]["snippets"]
            )
        )

    def test_appeal_deadlines_are_recovered_from_verified_text(self):
        appeal = category(self.policy, APPEAL_CATEGORY)
        deadlines = {
            (item.get("source_rule_id"), item.get("value"), item.get("unit"))
            for item in appeal["values"]
        }
        self.assertIn(("rule:43", "30", "วัน"), deadlines)
        self.assertIn(("rule:48", "15", "วันทำการ"), deadlines)
        self.assertIn(("rule:51.2", "30", "วัน"), deadlines)


if __name__ == "__main__":
    unittest.main()
