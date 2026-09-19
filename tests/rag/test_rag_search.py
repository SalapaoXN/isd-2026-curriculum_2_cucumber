import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.retrieval.search import search
from rag.structured.loader import load_json_to_sqlite


class RagSearchTest(unittest.TestCase):
    def test_joins_hits_to_text_and_provenance_in_stable_order(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {
                    "code": "C101",
                    "name_th": "วิชาหนึ่ง",
                    "credits": "3(3-0-6)",
                    "year": 1,
                    "semester": 1,
                    "desc_th": "คำอธิบายหนึ่ง",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-10.png",
                            "source_page": 10,
                            "document_category": "plan",
                        }
                    ],
                },
                {
                    "code": "C102",
                    "name_th": "วิชาสอง",
                    "credits": "3(3-0-6)",
                    "year": 1,
                    "semester": 2,
                    "desc_th": "คำอธิบายสอง",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-11.png",
                            "source_page": 11,
                            "document_category": "plan",
                        }
                    ],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "curriculum.db"
            input_path = Path(directory) / "curriculum.json"
            input_path.write_text(
                __import__("json").dumps(document, ensure_ascii=False),
                encoding="utf-8",
            )
            load_json_to_sqlite(input_path, database_path)

            hits = [
                {"chunk_id": "course-2-placement-2-metadata", "distance": 0.25},
                {"chunk_id": "course-1-placement-1-metadata", "distance": 0.25},
            ]
            with patch("rag.retrieval.search.nearest_neighbor_search", return_value=hits) as search_mock:
                results = search(database_path, [0.0] * 384, k=2)

        search_mock.assert_called_once_with(database_path, [0.0] * 384, limit=2)
        self.assertEqual(
            [result["chunk_id"] for result in results],
            ["course-1-placement-1-metadata", "course-2-placement-2-metadata"],
        )
        self.assertIn("รหัสวิชา C101", results[0]["text"])
        self.assertEqual(results[0]["provenance"][0]["source_page"], 10)
        self.assertIn("รหัสวิชา C102", results[1]["text"])
        self.assertEqual(results[1]["provenance"][0]["source_page"], 11)


if __name__ == "__main__":
    unittest.main()
