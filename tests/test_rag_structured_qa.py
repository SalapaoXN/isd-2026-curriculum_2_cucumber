import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

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

    def test_runs_fake_model_sql_and_returns_rows(self):
        def fake_model(_prompt):
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


if __name__ == "__main__":
    unittest.main()
