import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from rag.structured.qa import ask_structured


class RagStructuredQaTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.directory.name) / "structured.db"
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("CREATE TABLE courses (course_id INTEGER, code TEXT)")
            connection.execute("INSERT INTO courses VALUES (1, 'C101')")
            connection.commit()

    def tearDown(self):
        self.directory.cleanup()

    def _create_plan_database(self):
        database_path = Path(self.directory.name) / "curriculum.db"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE courses (
                    course_id INTEGER,
                    course_code TEXT,
                    catalog_id INTEGER
                );
                CREATE TABLE v_plan_courses (
                    program TEXT,
                    plan TEXT,
                    year INTEGER,
                    semester INTEGER,
                    course_id INTEGER,
                    course_code TEXT
                );
                """
            )
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?, ?)",
                [(1, "C101", 4), (2, "C101", 5), (3, "C102", 4)],
            )
            connection.executemany(
                "INSERT INTO v_plan_courses VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("DSBA", "coop", 1, 1, 1, "C101"),
                    ("DSBA", "coop", 1, 1, 3, "C102"),
                ],
            )
            connection.commit()
        return database_path

    def test_runs_fake_model_sql_and_returns_rows(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return "SELECT course_id, code FROM courses"

        result = ask_structured(
            self.database_path,
            "รหัสวิชาอะไร",
            "courses(course_id, code)",
            fake_model,
        )

        self.assertEqual(result["sql"], "SELECT course_id, code FROM courses LIMIT 100")
        self.assertEqual(result["columns"], ["course_id", "code"])
        self.assertEqual(result["rows"], [(1, "C101")])
        self.assertEqual(len(calls), 1)

    def test_empty_rows_are_returned_without_an_answer(self):
        def fake_model(_prompt):
            return "SELECT course_id, code FROM courses WHERE course_id = 99"

        result = ask_structured(
            self.database_path,
            "วิชาที่ไม่มี",
            "courses(course_id, code)",
            fake_model,
        )

        self.assertEqual(result["columns"], ["course_id", "code"])
        self.assertEqual(result["rows"], [])
        self.assertNotIn("answer", result)

    def test_course_code_join_is_repaired_to_course_id_join(self):
        database_path = self._create_plan_database()
        calls = []
        outputs = [
            """
            SELECT c.course_code, v.year, v.semester
            FROM v_plan_courses AS v
            JOIN courses AS c ON c.course_code = v.course_code
            WHERE v.program = 'DSBA'
              AND v.plan = 'coop'
              AND v.year = 1
              AND v.semester = 1
            ORDER BY v.course_id
            """,
            """
            SELECT c.course_code, v.year, v.semester
            FROM v_plan_courses AS v
            JOIN courses AS c ON c.course_id = v.course_id
            WHERE v.program = 'DSBA'
              AND v.plan = 'coop'
              AND v.year = 1
              AND v.semester = 1
            ORDER BY v.course_id
            """,
        ]

        def fake_model(_prompt):
            calls.append(True)
            return outputs[len(calls) - 1]

        result = ask_structured(
            database_path,
            "ปี 1 เทอม 1 เรียนไรบ้าง ของหลักสูตร DSBA ของcoop",
            "v_plan_courses(program, plan, year, semester, course_id, course_code) "
            "courses(course_id, course_code)",
            fake_model,
        )

        self.assertEqual(result["rows"], [("C101", 1, 1), ("C102", 1, 1)])
        self.assertIn("c.course_id = v.course_id", result["sql"])
        self.assertEqual(len(calls), 2)

    def test_course_id_join_passes_without_repair(self):
        database_path = self._create_plan_database()
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            return """
                SELECT c.course_code, v.year, v.semester
                FROM v_plan_courses AS v
                JOIN courses AS c ON c.course_id = v.course_id
                WHERE v.program = 'DSBA'
                  AND v.plan = 'coop'
                  AND v.year = 1
                  AND v.semester = 1
                ORDER BY v.course_id
            """

        result = ask_structured(
            database_path,
            "ปี 1 เทอม 1 เรียนไรบ้าง ของหลักสูตร DSBA ของcoop",
            "v_plan_courses(program, plan, year, semester, course_id, course_code) "
            "courses(course_id, course_code)",
            fake_model,
        )

        self.assertEqual(result["rows"], [("C101", 1, 1), ("C102", 1, 1)])
        self.assertEqual(len(calls), 1)

    def test_invalid_column_is_repaired_once(self):
        calls = []
        outputs = [
            "SELECT missing_code FROM courses",
            "SELECT course_id, code FROM courses",
        ]

        def fake_model(prompt):
            calls.append(prompt)
            return outputs[len(calls) - 1]

        result = ask_structured(
            self.database_path,
            "รหัสวิชาอะไร",
            "courses(course_id, code)",
            fake_model,
        )

        self.assertEqual(result["rows"], [(1, "C101")])
        self.assertEqual(result["sql"], "SELECT course_id, code FROM courses LIMIT 100")
        self.assertEqual(len(calls), 2)

    def test_guard_rejected_sql_is_repaired_once(self):
        calls = []
        outputs = [
            "DELETE FROM courses",
            "SELECT course_id, code FROM courses",
        ]

        def fake_model(prompt):
            calls.append(prompt)
            return outputs[len(calls) - 1]

        result = ask_structured(
            self.database_path,
            "รหัสวิชาอะไร",
            "courses(course_id, code)",
            fake_model,
        )

        self.assertEqual(result["rows"], [(1, "C101")])
        self.assertEqual(len(calls), 2)

    def test_invalid_repair_fails_after_two_calls(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return "SELECT missing_code FROM courses"

        with self.assertRaisesRegex(ValueError, "SQL repair failed validation"):
            ask_structured(
                self.database_path,
                "รหัสวิชาอะไร",
                "courses(course_id, code)",
                fake_model,
            )

        self.assertEqual(len(calls), 2)

    def test_missing_database_does_not_trigger_repair(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return "SELECT course_id, code FROM courses"

        with self.assertRaises(FileNotFoundError):
            ask_structured(
                self.database_path.with_name("missing.db"),
                "รหัสวิชาอะไร",
                "courses(course_id, code)",
                fake_model,
            )

        self.assertEqual(len(calls), 1)

    def test_infrastructure_validation_error_does_not_trigger_repair(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return "SELECT course_id, code FROM courses"

        with patch(
            "rag.structured.qa.validate_readonly_sql",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "database is locked"):
                ask_structured(
                    self.database_path,
                    "รหัสวิชาอะไร",
                    "courses(course_id, code)",
                    fake_model,
                )

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
