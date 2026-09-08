import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.structured.loader import load_json_to_sqlite, load_jsons_to_sqlite


class RagLoaderTest(unittest.TestCase):
    def test_program_and_plan_identity_are_normalized_and_unique(self):
        document = {
            "program": "  TEST-PROGRAM  ",
            "plan": "  Regular  ",
            "courses": [{"code": "C100"}],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                identity = connection.execute(
                    """
                    SELECT programs.program_code,
                           programs.program_code_normalized,
                           curriculum_plans.program_id,
                           curriculum_plans.plan_code,
                           curriculum_plans.plan_key
                    FROM programs
                    JOIN curriculum_plans
                        ON curriculum_plans.program_id = programs.program_id
                    """
                ).fetchone()
                catalog_id, program_id = connection.execute(
                    "SELECT catalog_id, program_id FROM curriculum_plans"
                ).fetchone()

                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """
                        INSERT INTO programs (
                            catalog_id, program_code, program_code_normalized
                        ) VALUES (?, ?, ?)
                        """,
                        (catalog_id, "OTHER", "test-program"),
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """
                        INSERT INTO curriculum_plans (
                            catalog_id, program_id, program_code, plan_key
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (catalog_id, program_id, "TEST-PROGRAM", "regular"),
                    )

            self.assertEqual(
                identity,
                ("  TEST-PROGRAM  ", "test-program", program_id, "  Regular  ", "regular"),
            )

    def test_provenance_has_document_key_and_preserves_source_fields(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [{"code": "C100"}],
            "source_provenance": [
                {
                    "source_filename": "curriculum.pdf",
                    "source_page": 7,
                    "document_page": 3,
                    "document_category": "plan",
                    "source_uri": "file:///curriculum.pdf",
                    "source_locator": "page=7",
                    "excerpt": "C100",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                provenance = connection.execute(
                    """
                    SELECT source_document_key, source_filename, source_page,
                           document_page, document_category, source_uri,
                           source_locator, excerpt
                    FROM provenance
                    """
                ).fetchone()

            self.assertEqual(
                provenance,
                (
                    "curriculum.pdf",
                    "curriculum.pdf",
                    7,
                    3,
                    "plan",
                    "file:///curriculum.pdf",
                    "page=7",
                    "C100",
                ),
            )

    def test_production_provenance_without_source_identity_fails_clearly(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [{"code": "C100"}],
            "source_provenance": [
                {"source_page": 7, "document_category": "plan"}
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no usable source identity"):
                load_json_to_sqlite(input_path, database_path)

    def test_unknown_provenance_without_source_identity_is_retained(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [{"code": "C100"}],
            "source_provenance": [{"source_page": 7}],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                provenance = connection.execute(
                    "SELECT source_document_key, source_page FROM provenance"
                ).fetchone()

            self.assertEqual(provenance, ("unknown", 7))

    def test_repeated_course_code_uses_one_course_row_and_two_placements(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {"code": "C100", "year": 1, "semester": 1},
                {"code": " c100 ", "year": 2, "semester": 1},
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                course = connection.execute(
                    """
                    SELECT course_code, course_code_normalized
                    FROM courses
                    """
                ).fetchone()
                course_count = connection.execute(
                    "SELECT COUNT(*) FROM courses WHERE course_code = 'C100'"
                ).fetchone()
                placement_count = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM plan_placements
                    JOIN courses ON courses.course_id = plan_placements.course_id
                    WHERE courses.course_code = 'C100'
                    """
                ).fetchone()
                catalog_id = connection.execute(
                    "SELECT catalog_id FROM catalogs"
                ).fetchone()[0]
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """
                        INSERT INTO courses (catalog_id, course_code, course_code_normalized)
                        VALUES (?, ?, ?)
                        """,
                        (catalog_id, "C101", "c100"),
                    )

            self.assertEqual(course, ("C100", "c100"))
            self.assertEqual(course_count, (1,))
            self.assertEqual(placement_count, (2,))

    def test_prerequisite_resolves_to_repeated_course(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {"code": "C100", "year": 1, "semester": 1},
                {"code": "C100", "year": 2, "semester": 1},
                {"code": "C200", "prerequisite": "C100", "year": 2, "semester": 2},
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                prerequisite = connection.execute(
                    """
                    SELECT COUNT(*), target.course_code
                    FROM prerequisites AS prereq
                    JOIN courses AS course ON course.course_id = prereq.course_id
                    JOIN courses AS target
                        ON target.course_id = prereq.prerequisite_course_id
                    WHERE course.course_code = 'C200'
                    GROUP BY target.course_code
                    """
                ).fetchone()

            self.assertEqual(prerequisite, (1, "C100"))

    def test_alternative_course_members_remain_distinct(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [{"code": "C400 or C500", "type": "choose-one"}],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                alternatives = connection.execute(
                    """
                    SELECT courses.course_code
                    FROM alternative_course_group_members AS members
                    JOIN courses ON courses.course_id = members.course_id
                    ORDER BY members.member_order
                    """
                ).fetchall()

            self.assertEqual(alternatives, [("C400",), ("C500",)])

    def test_credit_units_preserve_raw_credit_values(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {"code": "C100", "credits": "3(2-2-5)"},
                {"code": "C200", "credits": "credit varies"},
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                credits = connection.execute(
                    """
                    SELECT course_code, credit_units, credits_raw
                    FROM courses
                    ORDER BY course_code
                    """
                ).fetchall()

            self.assertEqual(
                credits,
                [("C100", 3, "3(2-2-5)"), ("C200", None, "credit varies")],
            )

    def test_zero_year_and_semester_are_stored_as_null(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {
                    "code": "C000",
                    "name_th": "Description-only course",
                    "year": 0,
                    "semester": 0,
                    "flexible_year_semester": "4/1",
                    "notes": "Placement note",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                placement = connection.execute(
                    """
                    SELECT year_number, semester_number,
                           flexible_year_number, flexible_semester_number,
                           flexible_year_semester_raw, notes
                    FROM plan_placements
                    """
                ).fetchone()
            self.assertEqual(placement, (None, None, 4, 1, "4/1", "Placement note"))

    def test_loads_structured_records_without_mutating_source(self):
        document = {
            "source": "synthetic curriculum",
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {
                    "code": "C100",
                    "name_th": "Normal course",
                    "name_en": "NORMAL COURSE",
                    "credits": "3(3-0-6)",
                    "category": "core",
                    "type": "required",
                    "year": 1,
                    "semester": 1,
                    "prerequisite": "ไม่มี",
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
                    "name_th": "Normal course",
                    "name_en": "NORMAL COURSE",
                    "credits": "3(3-0-6)",
                    "category": "core",
                    "type": "required",
                    "year": 2,
                    "semester": 1,
                    "prerequisite": "ไม่มี",
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
                    "code": "C200",
                    "name_th": "Advanced course",
                    "credits": "3(3-0-6)",
                    "year": 2,
                    "semester": 2,
                    "prerequisite": "C300",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-12.png",
                            "source_page": 12,
                            "document_category": "description",
                        }
                    ],
                },
                {
                    "code": "C300",
                    "name_th": "Prerequisite course",
                    "credits": "3(3-0-6)",
                    "year": 1,
                    "semester": 2,
                    "prerequisite": "ไม่มี",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-13.png",
                            "source_page": 13,
                            "document_category": "plan",
                        }
                    ],
                },
                {
                    "code": "C400 หรือ C500",
                    "name_th": "Choice one\nChoice two",
                    "name_en": "CHOICE ONE\nCHOICE TWO",
                    "credits": "3(3-0-6)",
                    "category": "elective",
                    "type": "choose-one",
                    "year": 3,
                    "semester": 1,
                    "prerequisite": "ไม่มี",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-14.png",
                            "source_page": 14,
                            "document_category": "plan",
                        }
                    ],
                },
            ],
        }
        original = json.loads(json.dumps(document))

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            self.assertEqual(document, original)
            with closing(sqlite3.connect(database_path)) as connection:
                repeated = connection.execute(
                    """
                    SELECT COUNT(*), COUNT(DISTINCT placement_id)
                    FROM plan_placements
                    JOIN courses ON courses.course_id = plan_placements.course_id
                    WHERE courses.course_code = 'C100'
                    """
                ).fetchone()
                self.assertEqual(repeated, (2, 2))

                prerequisite = connection.execute(
                    """
                    SELECT target.course_code
                    FROM prerequisites AS prereq
                    JOIN courses AS course ON course.course_id = prereq.course_id
                    JOIN courses AS target
                        ON target.course_id = prereq.prerequisite_course_id
                    WHERE course.course_code = 'C200'
                    """
                ).fetchone()
                self.assertEqual(prerequisite, ("C300",))

                alternatives = connection.execute(
                    """
                    SELECT courses.course_code
                    FROM alternative_course_group_members AS members
                    JOIN courses ON courses.course_id = members.course_id
                    ORDER BY members.member_order
                    """
                ).fetchall()
                self.assertEqual(alternatives, [("C400",), ("C500",)])

                source_pages = connection.execute(
                    """
                    SELECT DISTINCT provenance.source_page
                    FROM plan_placement_provenance AS links
                    JOIN provenance
                        ON provenance.provenance_id = links.provenance_id
                    ORDER BY provenance.source_page
                    """
                ).fetchall()
                self.assertEqual(
                    source_pages,
                    [(10,), (11,), (12,), (13,), (14,)],
                )

    def test_loads_multiple_documents_into_one_database_with_views(self):
        documents = (
            {
                "program": "IT",
                "plan": "coop",
                "courses": [{"code": "C100", "year": 1, "semester": 1}],
                "source_provenance": [
                    {
                        "source_filename": "it-coop.pdf",
                        "source_page": 10,
                        "document_category": "plan",
                    }
                ],
            },
            {
                "program": "IT",
                "plan": "no_coop",
                "courses": [{"code": "C100", "year": 1, "semester": 1}],
                "source_provenance": [
                    {
                        "source_filename": "it-no-coop.pdf",
                        "source_page": 20,
                        "document_category": "plan",
                    }
                ],
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_paths = []
            for index, document in enumerate(documents):
                input_path = directory_path / f"curriculum-{index}.json"
                input_path.write_text(json.dumps(document), encoding="utf-8")
                input_paths.append(input_path)
            database_path = directory_path / "curriculum.db"

            catalog_ids = load_jsons_to_sqlite(input_paths, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                counts = connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM catalogs),
                        (SELECT COUNT(*) FROM courses),
                        (SELECT COUNT(*) FROM curriculum_plans),
                        (SELECT COUNT(*) FROM plan_placements)
                    """
                ).fetchone()
                view_rows = connection.execute(
                    """
                    SELECT program, plan, course_code
                    FROM v_plan_courses
                    ORDER BY plan
                    """
                ).fetchall()
                pages = connection.execute(
                    "SELECT source_page FROM provenance ORDER BY source_page"
                ).fetchall()

        self.assertEqual(len(catalog_ids), 2)
        self.assertEqual(counts, (2, 2, 2, 2))
        self.assertEqual(
            view_rows,
            [("IT", "coop", "C100"), ("IT", "no_coop", "C100")],
        )
        self.assertEqual(pages, [(10,), (20,)])


if __name__ == "__main__":
    unittest.main()
