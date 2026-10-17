import unittest

from src.pipeline.tools.repair_cached_descriptions import _unique_sources, repair_document


class RepairCachedDescriptionsTests(unittest.TestCase):
    def test_combined_course_records_get_exact_per_code_descriptions(self):
        document = {
            "courses": [
                {
                    "code": "06000001 หรือ 06000002",
                    "desc_th": "Combined Thai descriptions",
                    "desc_en": "Combined English descriptions",
                }
            ]
        }
        sources = {
            code: {
                "code": code,
                "desc_th": f"Thai {code}",
                "desc_en": None if code.endswith("2") else f"English {code}",
                "source_provenance": [
                    {
                        "source_page": int(code[-1]),
                        "document_category": "description",
                    }
                ],
            }
            for code in ("06000001", "06000002")
        }

        repairs = repair_document(document, sources)

        course = document["courses"][0]
        self.assertEqual(course["desc_th"], "Combined Thai descriptions")
        self.assertEqual(course["desc_en"], "Combined English descriptions")
        self.assertEqual(
            course["descriptions_by_code"],
            {
                code: {
                    "desc_th": f"Thai {code}",
                    "desc_en": None if code.endswith("2") else f"English {code}",
                    "source_provenance": sources[code]["source_provenance"],
                }
                for code in sources
            },
        )
        self.assertEqual(repairs[0]["fields"], ["descriptions_by_code"])

    def test_fills_only_missing_language_and_merges_page_provenance(self):
        document = {
            "courses": [
                {
                    "code": "06000001",
                    "desc_th": "Thai text",
                    "name_en": "COURSE",
                    "source_provenance": [
                        {"program": "IT", "source_filename": "plan.png", "source_page": 1,
                         "document_category": "plan"}
                    ],
                }
            ]
        }
        sources = {
            "06000001": {
                "code": "06000001",
                "desc_th": "Other Thai text",
                "desc_en": "English text",
                "source_provenance": [
                    {"program": "IT", "source_filename": "desc.png", "source_page": 2,
                     "document_category": "description"}
                ],
            }
        }

        repairs = repair_document(document, sources)

        course = document["courses"][0]
        self.assertEqual(repairs[0]["fields"], ["desc_en"])
        self.assertEqual(course["desc_th"], "Thai text")
        self.assertEqual(course["desc_en"], "English text")
        self.assertEqual(course["name_en"], "COURSE")
        self.assertEqual(
            [source["source_page"] for source in course["source_provenance"]],
            [1, 2],
        )

    def test_duplicate_ocr_matches_are_not_used_as_a_winner(self):
        sources = _unique_sources([
            {"code": "06000001", "desc_en": "first"},
            {"code": "06000001", "desc_en": "second"},
        ])
        document = {"courses": [{"code": "06000001"}]}

        self.assertNotIn("06000001", sources)
        self.assertEqual(repair_document(document, sources), [])

    def test_missing_source_does_not_create_description(self):
        document = {"courses": [{"code": "06000001", "desc_en": ""}]}

        self.assertEqual(repair_document(document, {}), [])
        self.assertEqual(document["courses"][0]["desc_en"], "")


if __name__ == "__main__":
    unittest.main()
