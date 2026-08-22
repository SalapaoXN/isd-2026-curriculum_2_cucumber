import unittest

from src.rules_policy_mapper import (
    PROBATION_CATEGORY,
    REENTRY_CATEGORY,
    RulesPolicyMapper,
)


def rule(section_number, text, page):
    return {
        "rule_id": f"rule:{section_number}",
        "section_number": section_number,
        "category": "institutional",
        "section_path": section_number.split("."),
        "parent_rule_id": None,
        "rule_text": text,
        "references": [],
        "source_provenance": [
            {
                "source_filename": f"rule_page_{page:03d}_ocr.json",
                "source_page": page,
                "document_category": "rule",
            }
        ],
    }


def category(result, name):
    return next(item for item in result["categories"] if item["category"] == name)


class RulesPolicyMapperTests(unittest.TestCase):
    def test_required_rules_present(self):
        result = RulesPolicyMapper().map_data(
            {
                "source": "Academic Rules",
                "total_rules": 3,
                "rules": [
                    rule("22", "ภาคทัณฑ์ตามเกณฑ์", 7),
                    rule("33.11", "ภาคทัณฑ์ตามข้อ ๒๑", 9),
                    rule("36", "ไม่เกิน ๑ ปีนับจากวันที่พ้นสภาพนักศึกษา", 10),
                ],
            }
        )

        self.assertTrue(category(result, PROBATION_CATEGORY)["present"])
        self.assertTrue(category(result, REENTRY_CATEGORY)["present"])

    def test_missing_required_evidence_sets_present_null(self):
        result = RulesPolicyMapper().map_data(
            {"rules": [rule("22", "ภาคทัณฑ์ตามเกณฑ์", 7)]}
        )

        probation = category(result, PROBATION_CATEGORY)
        self.assertIsNone(probation["present"])
        self.assertEqual(probation["evidence"]["missing_rule_ids"], ["rule:33.11"])

    def test_complete_numeric_contexts_are_captured_and_normalized(self):
        result = RulesPolicyMapper().map_data(
            {
                "rules": [
                    rule(
                        "22",
                        "ค่าระดับคะแนนเฉลี่ยสะสมต่ำกว่า ๒.๐๐ ต้องถูกภาคทัณฑ์ "
                        "และค่าระดับคะแนนเฉลี่ยสะสมไม่ต่ำกว่า ๒.๐๐",
                        7,
                    ),
                    rule(
                        "33.11",
                        "ภาคทัณฑ์และค่าระดับคะแนนเฉลี่ยประจำภาคการศึกษาถัดไปต่ำกว่า ๒.๐๐",
                        9,
                    ),
                    rule("36", "ต้องไม่เกิน ๑ ปีนับจากวันที่พ้นสภาพนักศึกษา", 10),
                ]
            }
        )

        probation_values = category(result, PROBATION_CATEGORY)["values"]
        self.assertEqual(
            {(item["condition"], item["value"]) for item in probation_values},
            {("below", "2.00"), ("at_least", "2.00")},
        )
        reentry_values = category(result, REENTRY_CATEGORY)["values"]
        self.assertEqual(reentry_values[0]["value"], "1")
        self.assertEqual(reentry_values[0]["unit"], "ปี")

    def test_bounded_ocr_zero_tokens_are_normalized(self):
        result = RulesPolicyMapper().map_data(
            {
                "rules": [
                    rule(
                        "22",
                        "ค่าระดับคะแนนเฉลี่ยสะสมต่ำกว่า ๒.OO ต้องถูกภาคทัณฑ์ "
                        "และจะพ้นภาคทัณฑ์เมื่อได้รับค่าระดับคะแนนเฉลี่ยสะสมไม่ต่ำกว่า ๒.OO",
                        7,
                    ),
                    rule(
                        "33.11",
                        "ภาคทัณฑ์และค่าระดับคะแนนเฉลี่ยประจำภาคการศึกษาถัดไปต่ำกว่า ๒.OO",
                        9,
                    ),
                    rule("36", "ต้องไม่เกิน O ปีนับจากวันที่พ้นสภาพนักศึกษา", 10),
                ]
            }
        )

        probation_values = category(result, PROBATION_CATEGORY)["values"]
        self.assertEqual(
            {(item["condition"], item["value"]) for item in probation_values},
            {("below", "2.00"), ("at_least", "2.00")},
        )
        self.assertEqual(probation_values[0]["raw_value"], "๒.OO")
        self.assertEqual(category(result, REENTRY_CATEGORY)["values"], [])

    def test_ambiguous_ocr_only_numeric_token_is_rejected(self):
        result = RulesPolicyMapper().map_data(
            {
                "rules": [
                    rule(
                        "22",
                        "ค่าระดับคะแนนเฉลี่ยสะสมต่ำกว่า OO ต้องถูกภาคทัณฑ์",
                        7,
                    ),
                    rule("33.11", "ภาคทัณฑ์ตามข้อ ๒๑", 9),
                ]
            }
        )

        self.assertEqual(category(result, PROBATION_CATEGORY)["values"], [])

    def test_provenance_and_supporting_text_are_preserved(self):
        text_22 = "ภาคทัณฑ์ตามข้อ ๒๒"
        text_3311 = "ภาคทัณฑ์ตามข้อ ๓๓.๑๑"
        result = RulesPolicyMapper().map_data(
            {
                "rules": [
                    rule("22", text_22, 7),
                    rule("33.11", text_3311, 9),
                    rule("36", "ไม่เกิน ๑ ปี", 10),
                ]
            }
        )

        probation = category(result, PROBATION_CATEGORY)
        self.assertEqual(
            [item["source_page"] for item in probation["evidence"]["source_provenance"]],
            [7, 9],
        )
        supporting = probation["evidence"]["supporting_rule_text"]
        self.assertEqual([item["rule_id"] for item in supporting], ["rule:22", "rule:33.11"])
        self.assertEqual(supporting[0]["rule_text"], text_22)

    def test_mapper_does_not_require_instructor_gt_shape(self):
        result = RulesPolicyMapper().map_data(
            {
                "source": "Academic Rules",
                "total_rules": 1,
                "rules": [rule("36", "ไม่เกิน ๑ ปี", 10)],
            }
        )

        self.assertEqual(result["source"], "Academic Rules")
        self.assertEqual(category(result, REENTRY_CATEGORY)["present"], True)


if __name__ == "__main__":
    unittest.main()
