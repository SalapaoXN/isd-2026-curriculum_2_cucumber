import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.retrieval.chunks import build_chunks
from rag.structured.loader import load_json_to_sqlite


class RagChunksTest(unittest.TestCase):
    def test_formats_flexible_year_semester_without_fixed_term(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {
                    "code": "C401",
                    "name_th": "วิชาที่ยืดหยุ่น",
                    "credits": "3(3-0-6)",
                    "year": 0,
                    "semester": 0,
                    "flexible_year_semester": "4/1",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "test_page_43.png",
                            "source_page": 43,
                            "document_category": "description",
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)
            metadata = next(
                chunk for chunk in build_chunks(database_path)
                if chunk["chunk_type"] == "metadata"
            )

        self.assertIn("ช่วงที่สามารถลงได้: ปี 4 ภาคเรียน 1", metadata["text"])
        self.assertNotIn("เปิดสอนปีที่ 0", metadata["text"])
        self.assertNotIn("เปิดสอนภาคเรียนที่ 0", metadata["text"])

    def test_builds_metadata_and_description_chunks_with_provenance(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {
                    "code": "C101",
                    "name_th": "วิชาทดสอบ",
                    "name_en": "TEST COURSE",
                    "credits": "3(3-0-6)",
                    "category": "core",
                    "type": "required",
                    "year": 1,
                    "semester": 2,
                    "prerequisite": "ไม่มี",
                    "desc_th": "คำอธิบายวิชาทดสอบ",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "test_page_42.png",
                            "source_page": 42,
                            "document_category": "plan",
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)
            chunks = build_chunks(database_path)

            self.assertEqual([chunk["chunk_type"] for chunk in chunks], ["metadata", "description"])
            self.assertEqual(
                [chunk["chunk_id"] for chunk in chunks],
                ["course-1-placement-1-metadata", "course-1-description"],
            )
            self.assertEqual(chunks[0]["course_id"], 1)
            self.assertEqual(chunks[0]["placement_id"], 1)
            self.assertIn("รหัสวิชา C101", chunks[0]["text"])
            self.assertIn("เปิดสอนปีที่ 1 ภาคเรียนที่ 2", chunks[0]["text"])
            self.assertIn("คำอธิบายวิชาทดสอบ", chunks[1]["text"])
            self.assertEqual(chunks[0]["provenance_ids"], [1])
            self.assertEqual(chunks[1]["provenance_ids"], [1])
            self.assertEqual(chunks[0]["provenance"][0]["source_page"], 42)
            self.assertEqual(build_chunks(database_path), chunks)

            with closing(sqlite3.connect(database_path)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM courses").fetchone(), (1,))


if __name__ == "__main__":
    unittest.main()
