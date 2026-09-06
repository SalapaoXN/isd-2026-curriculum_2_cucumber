import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.structured.execute import execute_readonly


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


if __name__ == "__main__":
    unittest.main()
