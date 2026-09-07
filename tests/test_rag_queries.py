import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.structured.loader import load_json_to_sqlite
from rag.structured.queries import (
    courses_in_year_semester,
    courses_requiring_prerequisite,
    prerequisites_of_course,
    semester_total_credits,
)


class RagQueriesTest(unittest.TestCase):
    def test_queries_preserve_placements_prerequisites_and_alternatives(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {
                    "code": "C100",
                    "name_th": "First placement",
                    "credits": "3(3-0-6)",
                    "year": 1,
                    "semester": 1,
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
                    "code": "C100",
                    "name_th": "Repeated placement",
                    "credits": "3(3-0-6)",
                    "year": 1,
                    "semester": 1,
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-11.png",
                            "source_page": 11,
                            "document_category": "plan",
                        }
                    ],
                },
                {
                    "code": "C300",
                    "name_th": "Prerequisite",
                    "credits": "2(2-0-4)",
                    "year": 1,
                    "semester": 1,
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-12.png",
                            "source_page": 12,
                            "document_category": "plan",
                        }
                    ],
                },
                {
                    "code": "C400",
                    "name_th": "Dependent",
                    "credits": "3(3-0-6)",
                    "year": 1,
                    "semester": 1,
                    "prerequisite": "C300",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-13.png",
                            "source_page": 13,
                            "document_category": "description",
                        }
                    ],
                },
                {
                    "code": "C500 หรือ C600",
                    "name_th": "Choice A\nChoice B",
                    "credits": "4(4-0-8)",
                    "year": 1,
                    "semester": 1,
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-14.png",
                            "source_page": 14,
                            "document_category": "plan",
                        }
                    ],
                },
                {
                    "code": "C700",
                    "name_th": "Alternative dependent",
                    "credits": "3(3-0-6)",
                    "year": 2,
                    "semester": 1,
                    "prerequisite": "C500 หรือ C600",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-15.png",
                            "source_page": 15,
                            "document_category": "description",
                        }
                    ],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(
                json.dumps(document, ensure_ascii=False), encoding="utf-8"
            )
            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                plan_id = connection.execute(
                    "SELECT plan_id FROM curriculum_plans"
                ).fetchone()[0]
                c300_id = connection.execute(
                    "SELECT course_id FROM courses WHERE course_code = 'C300'"
                ).fetchone()[0]
                c500_id = connection.execute(
                    "SELECT course_id FROM courses WHERE course_code = 'C500'"
                ).fetchone()[0]
                c400_id = connection.execute(
                    "SELECT course_id FROM courses WHERE course_code = 'C400'"
                ).fetchone()[0]
                c700_id = connection.execute(
                    "SELECT course_id FROM courses WHERE course_code = 'C700'"
                ).fetchone()[0]

            self.assertEqual(semester_total_credits(database_path, plan_id, 1, 1), 15)

            placements = courses_in_year_semester(database_path, plan_id, 1, 1)
            self.assertEqual(len(placements), 5)
            self.assertEqual(
                [placement["course_code"] for placement in placements[:4]],
                ["C100", "C100", "C300", "C400"],
            )
            self.assertEqual(
                len({placement["placement_id"] for placement in placements[:2]}), 2
            )
            self.assertTrue(placements[4]["is_alternative"])
            self.assertEqual(
                [
                    member["course_code"]
                    for member in placements[4]["alternative_courses"]
                ],
                ["C500", "C600"],
            )
            self.assertIn(14, placements[4]["source_pages"])

            direct = prerequisites_of_course(database_path, c400_id)
            self.assertEqual(len(direct), 1)
            self.assertEqual(direct[0]["prerequisite_course_id"], c300_id)
            self.assertEqual(direct[0]["prerequisite_code"], "C300")
            self.assertIn(13, direct[0]["source_pages"])

            alternative = prerequisites_of_course(database_path, c700_id)
            self.assertEqual(len(alternative), 1)
            self.assertTrue(alternative[0]["is_alternative"])
            self.assertEqual(
                [
                    member["course_code"]
                    for member in alternative[0]["alternative_courses"]
                ],
                ["C500", "C600"],
            )

            requiring_direct = courses_requiring_prerequisite(database_path, c300_id)
            self.assertEqual([course["course_id"] for course in requiring_direct], [c400_id])
            requiring_alternative = courses_requiring_prerequisite(database_path, c500_id)
            self.assertEqual(
                [course["course_id"] for course in requiring_alternative], [c700_id]
            )


if __name__ == "__main__":
    unittest.main()
