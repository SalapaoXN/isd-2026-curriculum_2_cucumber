import unittest

from src.pipeline.tools.merge.consolidator import (
    CurriculumConsolidator,
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
