import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.structured.execute import execute_readonly, validate_readonly_sql


class RagExecuteTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.directory.name) / "query.db"
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("CREATE TABLE courses (course_id INTEGER, code TEXT)")
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?)",
                [(1, "C101"), (2, "C102")],
            )
            connection.commit()

    def tearDown(self):
        self.directory.cleanup()

    def test_returns_columns_and_rows(self):
        columns, rows = execute_readonly(
            self.database_path,
            "SELECT course_id, code FROM courses ORDER BY course_id",
        )
        self.assertEqual(columns, ["course_id", "code"])
        self.assertEqual(rows, [(1, "C101"), (2, "C102")])

    def test_returns_columns_and_empty_rows(self):
        columns, rows = execute_readonly(
            self.database_path,
            "SELECT course_id, code FROM courses WHERE course_id = 99",
        )
        self.assertEqual(columns, ["course_id", "code"])
        self.assertEqual(rows, [])

    def test_guard_rejects_write_queries(self):
        with self.assertRaises(ValueError):
            execute_readonly(self.database_path, "DELETE FROM courses")

    def test_valid_select_and_with_queries_validate(self):
        validate_readonly_sql(
            self.database_path,
            "SELECT course_id, code FROM courses",
        )
        validate_readonly_sql(
            self.database_path,
            "WITH selected AS (SELECT code FROM courses) "
            "SELECT code FROM selected",
        )

    def test_missing_table_or_column_fails_validation(self):
        for sql in (
            "SELECT code FROM missing_courses",
            "SELECT missing_code FROM courses",
        ):
            with self.subTest(sql=sql), self.assertRaises(sqlite3.OperationalError):
                validate_readonly_sql(self.database_path, sql)

    def test_validation_rejects_write_queries(self):
        with self.assertRaises(ValueError):
            validate_readonly_sql(self.database_path, "DELETE FROM courses")

    def test_validation_reports_missing_database(self):
        with self.assertRaises(FileNotFoundError):
            validate_readonly_sql(
                self.database_path.with_name("missing.db"),
                "SELECT 1",
            )


if __name__ == "__main__":
    unittest.main()
