import unittest

from src.pipeline.tools.merge.consolidator import (
    CurriculumConsolidator,
    _courses_from_persisted_description_pages,
    _preserve_authoritative_extracted_credit,
)


def _source_provenance(program="IT", filename="it_page_354.png", page=354):
    return [
        {
            "program": program,
            "source_filename": filename,
            "source_page": page,
            "document_category": "description",
        }
    ]


class MergeCreditProvenanceTests(unittest.TestCase):
    def test_persisted_page_edge_continuation_joins_previous_course(self):
        first_page = {
            "program": "AIT",
            "courses": [{
                "code": "06046410",
                "desc_th": "คำอธิบายไทยต้นหน้า",
                "source_provenance": _source_provenance("AIT", "ait_page_293.png", 293),
            }],
        }
        second_page = {
            "program": "AIT",
            "courses": [{"code": "06046411", "desc_th": "วิชาถัดไป"}],
            "leading_description_continuation": {
                "lines": ["คำอธิบายไทยต่อหน้า", "ENGLISH CONTINUATION"],
                "source_provenance": [
                    {
                        "program": "AIT",
                        "source_filename": "ait_page_294.png",
                        "source_page": 294,
                        "document_category": "description",
                    }
                ],
            },
        }

        courses = _courses_from_persisted_description_pages(
            [("coop", 294, second_page), ("coop", 293, first_page)]
        )

        self.assertIn("คำอธิบายไทยต่อหน้า", courses[0]["desc_th"])
        self.assertEqual(courses[0]["desc_en"], "ENGLISH CONTINUATION")
        self.assertEqual(
            [entry["source_page"] for entry in courses[0]["source_provenance"]],
            [293, 294],
        )

    def test_empty_description_fields_do_not_clobber_plan_content(self):
        plan = {
            "program": "IT",
            "courses": [
                {
                    "code": "06016454",
                    "desc_th": "คำอธิบายเดิม",
                    "desc_en": "Existing description",
                }
            ],
        }
        description = {
            "descriptions": [
                {
                    "code": "06016454",
                    "desc_th": None,
                    "desc_en": "",
                    "source_provenance": _source_provenance(),
                }
            ]
        }

        merged = CurriculumConsolidator(plan, description).consolidate()["courses"][0]

        self.assertEqual(merged["desc_th"], "คำอธิบายเดิม")
        self.assertEqual(merged["desc_en"], "Existing description")

    def test_reparsed_description_preserves_verified_credit_metadata(self):
        reparsed = {
            "code": "06016454",
            "credits": "3(3-0-6)",
            "source_provenance": _source_provenance(),
        }
        persisted = {
            **reparsed,
            "credit_source_verified": True,
            "credit_source_provenance": {
                "source_filename": "it_page_354.png",
                "source_page": 354,
                "document_category": "description",
            },
        }

        preserved = _preserve_authoritative_extracted_credit(
            reparsed,
            [persisted],
        )

        self.assertIs(preserved["credit_source_verified"], True)
        self.assertEqual(
            preserved["credit_source_provenance"],
            persisted["credit_source_provenance"],
        )

    def test_source_verified_credit_metadata_survives_merge(self):
        plan = {
            "program": "IT",
            "courses": [{"code": "06016454", "credits": "3(3-0-6)"}],
        }
        description = {
            "descriptions": [
                {
                    "code": "06016454",
                    "credits": "3(3-0-6)",
                    "credit_source_verified": True,
                    "credit_source_provenance": {
                        "source_filename": "it_page_354.png",
                        "source_page": 354,
                        "document_category": "description",
                    },
                    "source_provenance": _source_provenance(),
                }
            ]
        }

        merged = CurriculumConsolidator(plan, description).consolidate()["courses"][0]

        self.assertEqual(merged["credits"], "3(3-0-6)")
        self.assertIs(merged["credit_source_verified"], True)
        self.assertEqual(
            merged["credit_source_provenance"],
            {
                "source_filename": "it_page_354.png",
                "source_page": 354,
                "document_category": "description",
            },
        )
        self.assertEqual(merged["source_provenance"], _source_provenance())

    def test_unresolved_bit_credit_stays_unresolved(self):
        plan = {
            "program": "BIT",
            "courses": [{"code": "06036135", "credits": ""}],
        }
        description = {
            "descriptions": [
                {
                    "code": "06036135",
                    "credits": "",
                    "source_provenance": _source_provenance(
                        "BIT", "bit_page_252.png", 252
                    ),
                }
            ]
        }

        merged = CurriculumConsolidator(plan, description).consolidate()["courses"][0]

        self.assertEqual(merged["credits"], "")
        self.assertNotIn("credit_source_verified", merged)
        self.assertNotIn("credit_source_provenance", merged)

    def test_ordinary_course_does_not_gain_verification_metadata(self):
        plan = {"program": "IT", "courses": [{"code": "06016455", "credits": "3(3-0-6)"}]}
        description = {
            "descriptions": [
                {
                    "code": "06016455",
                    "credits": "3(3-0-6)",
                    "source_provenance": _source_provenance(),
                }
            ]
        }

        merged = CurriculumConsolidator(plan, description).consolidate()["courses"][0]

        self.assertEqual(merged["credits"], "3(3-0-6)")
        self.assertNotIn("credit_source_verified", merged)
        self.assertNotIn("credit_source_provenance", merged)


if __name__ == "__main__":
    unittest.main()
