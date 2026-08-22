import unittest

from src.rules_policy_mapper import (
    ACADEMIC_DISHONESTY_CATEGORY,
    APPEAL_CATEGORY,
    CATEGORY_RULES,
    CONDUCT_CATEGORY,
    DISCIPLINARY_PENALTY_CATEGORY,
    EXAM_CATEGORY,
    GRADING_CATEGORY,
    LEAVE_CATEGORY,
    PROBATION_CATEGORY,
    REENTRY_CATEGORY,
    RESIGNATION_CATEGORY,
    RulesPolicyMapper,
    STATUS_CATEGORY,
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


def mapped_category_rules(category_name, overrides=None):
    overrides = overrides or {}
    return [
        rule(section_number, overrides.get(section_number, f"ข้อความข้อ {section_number}"), 1)
        for section_number in CATEGORY_RULES[category_name]
    ]


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

    def test_mapper_emits_all_eleven_categories(self):
        result = RulesPolicyMapper().map_data({"rules": []})

        self.assertEqual(
            [item["category"] for item in result["categories"]],
            list(CATEGORY_RULES),
        )

    def test_status_category_extracts_complete_threshold(self):
        result = RulesPolicyMapper().map_data(
            {"rules": mapped_category_rules(STATUS_CATEGORY, {"33.12": "ค่าระดับคะแนนเฉลี่ยสะสมต่ำกว่า ๑.00"})}
        )

        status = category(result, STATUS_CATEGORY)
        self.assertTrue(status["present"])
        self.assertEqual(status["values"][0]["value"], "1.00")
        self.assertEqual(status["values"][0]["source_rule_id"], "rule:33.12")

    def test_grading_category_extracts_only_complete_grade_rows(self):
        result = RulesPolicyMapper().map_data(
            {
                "rules": mapped_category_rules(
                    GRADING_CATEGORY,
                    {
                        "19.3": "A\n๔.00\nB+\n๓.๕๑\nB\n๓.OD\nC+\n๒.๕0\nC\n๒.OO",
                    },
                )
            }
        )

        grading = category(result, GRADING_CATEGORY)
        self.assertTrue(grading["present"])
        self.assertEqual(
            {(item["grade"], item["value"]) for item in grading["values"]},
            {("A", "4.00"), ("C+", "2.50"), ("C", "2.00")},
        )
        snippets = category(result, GRADING_CATEGORY)["evidence"]["supporting_rule_text"][0]["snippets"]
        self.assertEqual(
            {item["grade"]: item["status"] for item in snippets if "status" in item},
            {"B+": "invalid_grade_point", "B": "invalid_numeric_token"},
        )
        self.assertEqual(
            {(item["grade"], item["raw_value"]) for item in snippets if "status" in item},
            {("B+", "๓.๕๑"), ("B", "๓.OD")},
        )

    def test_leave_category_preserves_evidence_without_guessing_values(self):
        result = RulesPolicyMapper().map_data(
            {"rules": mapped_category_rules(LEAVE_CATEGORY)}
        )

        leave = category(result, LEAVE_CATEGORY)
        self.assertTrue(leave["present"])
        self.assertEqual(leave["values"], [])
        self.assertEqual(len(leave["evidence"]["rule_ids"]), len(CATEGORY_RULES[LEAVE_CATEGORY]))

    def test_resignation_category_preserves_evidence_without_guessing_values(self):
        result = RulesPolicyMapper().map_data(
            {"rules": mapped_category_rules(RESIGNATION_CATEGORY)}
        )

        resignation = category(result, RESIGNATION_CATEGORY)
        self.assertTrue(resignation["present"])
        self.assertEqual(resignation["values"], [])

    def test_exam_category_extracts_suspension_period(self):
        result = RulesPolicyMapper().map_data(
            {
                    "rules": mapped_category_rules(
                        EXAM_CATEGORY,
                    {"20": "นักศึกษาทุจริตในการสอบและพักการเรียนในภาคการศึกษาปกติถัดไปอีก ๑ ภาคการศึกษา"},
                )
            }
        )

        exam = category(result, EXAM_CATEGORY)
        self.assertTrue(exam["present"])
        self.assertEqual(exam["values"][0]["value"], "1")
        self.assertEqual(exam["values"][0]["unit"], "ภาคการศึกษา")
        self.assertEqual(exam["values"][0]["label"], "ระยะพักการเรียนจากการทุจริตในการสอบ")

    def test_academic_dishonesty_category_extracts_evidence_bound_value(self):
        result = RulesPolicyMapper().map_data(
            {
                    "rules": mapped_category_rules(
                        ACADEMIC_DISHONESTY_CATEGORY,
                        {"20": "ทุจริตในการสอบและพักการเรียนในภาคการศึกษาปกติถัดไปอีก ๑ ภาคการศึกษา"},
                )
            }
        )

        dishonesty = category(result, ACADEMIC_DISHONESTY_CATEGORY)
        self.assertTrue(dishonesty["present"])
        self.assertEqual(dishonesty["values"][0]["source_rule_id"], "rule:20")
        self.assertEqual(dishonesty["values"][0]["condition"], "exam_dishonesty_suspension")
        self.assertEqual(dishonesty["values"][0]["label"], "ระยะพักการเรียนจากการทุจริตในการสอบ")

    def test_conduct_category_preserves_rule_evidence(self):
        result = RulesPolicyMapper().map_data(
            {"rules": mapped_category_rules(CONDUCT_CATEGORY)}
        )

        conduct = category(result, CONDUCT_CATEGORY)
        self.assertTrue(conduct["present"])
        self.assertEqual(conduct["values"], [])
        self.assertEqual(
            conduct["evidence"]["rule_ids"],
            [f"rule:{section}" for section in CATEGORY_RULES[CONDUCT_CATEGORY]],
        )

    def test_disciplinary_penalty_category_extracts_counts(self):
        result = RulesPolicyMapper().map_data(
            {
                "rules": mapped_category_rules(
                    DISCIPLINARY_PENALTY_CATEGORY,
                    {
                        "38": "โทษทางวินัยอย่างไมร้ายแรงมี ๓ สถาน คือ",
                        "39": "โทษทางวินัยอย่างร้ายแรงมี ๓ สถาน คือ",
                    },
                )
            }
        )

        penalties = category(result, DISCIPLINARY_PENALTY_CATEGORY)
        self.assertTrue(penalties["present"])
        self.assertEqual(
            {(item["severity"], item["value"]) for item in penalties["values"]},
            {("ไม่ร้ายแรง", "3"), ("ร้ายแรง", "3")},
        )
        minor = next(item for item in penalties["values"] if item["severity"] == "ไม่ร้ายแรง")
        self.assertEqual(minor["supporting_rule_ids"], ["rule:38.1", "rule:38.2", "rule:38.3"])

    def test_disciplinary_count_requires_complete_child_rules(self):
        result = RulesPolicyMapper().map_data(
            {"rules": [rule("38", "โทษทางวินัยอย่างไมร้ายแรงมี ๓ สถาน คือ", 11)]}
        )

        penalties = category(result, DISCIPLINARY_PENALTY_CATEGORY)
        self.assertEqual(penalties["values"], [])

    def test_appeal_category_normalizes_bounded_deadlines(self):
        result = RulesPolicyMapper().map_data(
            {
                "rules": mapped_category_rules(
                    APPEAL_CATEGORY,
                    {
                        "43": "ผู้ถูกลงโทษตามข้อ ๓๘ มีสิทธิอุทธรณ์ ให้อุทธรณ์ภายใน ๓O วัน",
                        "48": "ไม่เสนอชื่อให้ได้รับปริญญาและมีสิทธิอุทธรณ์ ภายใน ๑๕ วันทำการ",
                        "49": "ส่งคำชี้แจงเกี่ยวกับการอุทธรณ์มายังสถาบันภายใน ๗ วันทำการ",
                        "51.2": "การยื่นคำขอพิจารณาใหม่ต้องกระทำภายใน ๓O วัน",
                    },
                )
            }
        )

        appeal = category(result, APPEAL_CATEGORY)
        self.assertTrue(appeal["present"])
        self.assertEqual(
            {(item["source_rule_id"], item["value"], item["unit"]) for item in appeal["values"]},
            {
                ("rule:43", "30", "วัน"),
                ("rule:48", "15", "วันทำการ"),
                ("rule:49", "7", "วันทำการ"),
                ("rule:51.2", "30", "วัน"),
            },
        )
        self.assertEqual(
            {item["procedure"]: item["label"] for item in appeal["values"]},
            {
                "student_sanction_appeal": "ยื่นอุทธรณ์คำสั่งลงโทษ",
                "degree_decision_appeal": "ยื่นอุทธรณ์กรณีไม่เสนอชื่อรับปริญญา",
                "institution_response": "ส่งคำชี้แจงของส่วนงานวิชาการ",
                "reconsideration_request": "ยื่นคำขอพิจารณาอุทธรณ์ใหม่",
            },
        )


if __name__ == "__main__":
    unittest.main()
