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
                    plan_key TEXT,
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
                "INSERT INTO v_plan_courses VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ("DSBA", "coop", "coop", 1, 1, 1, "C101"),
                    ("DSBA", "coop", "coop", 1, 1, 3, "C102"),
                ],
            )
            connection.commit()
        return database_path

    def _create_canonical_plan_database(self):
        database_path = Path(self.directory.name) / "canonical_plans.db"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                "CREATE TABLE curriculum_plans "
                "(plan_key TEXT, plan_name TEXT, plan TEXT)"
            )
            connection.executemany(
                "INSERT INTO curriculum_plans VALUES (?, ?, ?)",
                [
                    ("coop", "สหกิจศึกษา", "coop"),
                    ("no_coop", "ไม่สหกิจ", "no_coop"),
                    ("default", None, "default"),
                    ("gened", "GENED", "gened"),
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
              AND v.plan_key = 'coop'
              AND v.year = 1
              AND v.semester = 1
            ORDER BY v.course_id
            """,
            """
            SELECT c.course_code, v.year, v.semester
            FROM v_plan_courses AS v
            JOIN courses AS c ON c.course_id = v.course_id
            WHERE v.program = 'DSBA'
              AND v.plan_key = 'coop'
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
            "v_plan_courses(program, plan, plan_key, year, semester, course_id, course_code) "
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
                  AND v.plan_key = 'coop'
                  AND v.year = 1
                  AND v.semester = 1
                ORDER BY v.course_id
            """

        result = ask_structured(
            database_path,
            "ปี 1 เทอม 1 เรียนไรบ้าง ของหลักสูตร DSBA ของcoop",
            "v_plan_courses(program, plan, plan_key, year, semester, course_id, course_code) "
            "courses(course_id, course_code)",
            fake_model,
        )

        self.assertEqual(result["rows"], [("C101", 1, 1), ("C102", 1, 1)])
        self.assertEqual(len(calls), 1)

    def test_no_coop_unsafe_plan_filters_are_repaired_to_plan_key(self):
        database_path = self._create_canonical_plan_database()
        for bad_sql in (
            "SELECT plan_key FROM curriculum_plans "
            "WHERE plan_key NOT LIKE '%coop%'",
            "SELECT plan_key FROM curriculum_plans "
            "WHERE plan_name LIKE '%ไม่สหกิจ%'",
        ):
            with self.subTest(bad_sql=bad_sql):
                calls = []

                def fake_model(prompt):
                    calls.append(prompt)
                    return (
                        bad_sql
                        if len(calls) == 1
                        else "SELECT plan_key FROM curriculum_plans "
                        "WHERE plan_key = 'no_coop'"
                    )

                result = ask_structured(
                    database_path,
                    "แผนไม่สหกิจเป็นอย่างไร",
                    "curriculum_plans(plan_key, plan_name, plan)",
                    fake_model,
                )

                self.assertEqual(result["rows"], [("no_coop",)])
                self.assertEqual(len(calls), 2)
                self.assertIn("unsafe plan filter", calls[1])

    def test_coop_unsafe_thai_plan_filter_is_repaired_to_plan_key(self):
        database_path = self._create_canonical_plan_database()
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return (
                "SELECT plan_key FROM curriculum_plans "
                "WHERE plan LIKE '%สหกิจ%'"
                if len(calls) == 1
                else "SELECT plan_key FROM curriculum_plans "
                "WHERE plan_key = 'coop'"
            )

        result = ask_structured(
            database_path,
            "แผนสหกิจมีอะไรบ้าง",
            "curriculum_plans(plan_key, plan_name, plan)",
            fake_model,
        )

        self.assertEqual(result["rows"], [("coop",)])
        self.assertEqual(len(calls), 2)
        self.assertIn("unsafe plan filter", calls[1])

    def test_exact_no_coop_plan_key_passes_without_repair(self):
        database_path = self._create_canonical_plan_database()
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            return "SELECT plan_key FROM curriculum_plans WHERE plan_key = 'no_coop'"

        result = ask_structured(
            database_path,
            "แผนไม่สหกิจ",
            "curriculum_plans(plan_key, plan_name, plan)",
            fake_model,
        )

        self.assertEqual(result["rows"], [("no_coop",)])
        self.assertEqual(len(calls), 1)

    def test_exact_coop_plan_key_passes_without_repair(self):
        database_path = self._create_canonical_plan_database()
        calls = []

        def fake_model(_prompt):
            calls.append(True)
            return "SELECT plan_key FROM curriculum_plans WHERE plan_key = 'coop'"

        result = ask_structured(
            database_path,
            "แผนสหกิจ",
            "curriculum_plans(plan_key, plan_name, plan)",
            fake_model,
        )

        self.assertEqual(result["rows"], [("coop",)])
        self.assertEqual(len(calls), 1)

    def test_plan_filter_repair_still_fails_after_two_bad_sql_calls(self):
        database_path = self._create_canonical_plan_database()
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return "SELECT plan_key FROM curriculum_plans WHERE plan_name LIKE '%สหกิจ%'"

        with self.assertRaisesRegex(ValueError, "SQL repair failed validation"):
            ask_structured(
                database_path,
                "แผนสหกิจ",
                "curriculum_plans(plan_key, plan_name, plan)",
                fake_model,
            )

        self.assertEqual(len(calls), 2)

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
