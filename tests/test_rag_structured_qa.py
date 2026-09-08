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
