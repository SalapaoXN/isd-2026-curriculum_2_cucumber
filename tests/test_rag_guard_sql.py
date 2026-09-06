import unittest

from rag.structured.guard_sql import guard_sql


class RagGuardSqlTest(unittest.TestCase):
    def test_allows_read_queries_and_bounds_limits(self):
        self.assertEqual(
            guard_sql("SELECT course_code FROM courses"),
            "SELECT course_code FROM courses LIMIT 100",
        )
        self.assertEqual(
            guard_sql("WITH selected AS (SELECT 1) SELECT * FROM selected"),
            "WITH selected AS (SELECT 1) SELECT * FROM selected LIMIT 100",
        )
        self.assertEqual(
            guard_sql("SELECT * FROM courses LIMIT 5"),
            "SELECT * FROM courses LIMIT 5",
        )
        self.assertEqual(
            guard_sql("SELECT * FROM courses LIMIT 500"),
            "SELECT * FROM courses LIMIT 100",
        )
        self.assertEqual(
            guard_sql("SELECT * FROM courses;"),
            "SELECT * FROM courses LIMIT 100;",
        )

    def test_rejects_writes_and_multiple_statements(self):
        for sql in (
            "INSERT INTO courses VALUES (1)",
            "UPDATE courses SET course_code = 'x'",
            "DELETE FROM courses",
            "DROP TABLE courses",
            "ALTER TABLE courses ADD COLUMN x TEXT",
            "PRAGMA foreign_keys",
            "ATTACH DATABASE 'other.db' AS other",
            "SELECT 1; SELECT 2",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(ValueError):
                    guard_sql(sql)

    def test_ignores_keywords_in_literals_and_comments(self):
        sql = "SELECT 'DROP; UPDATE' AS value /* ATTACH */ -- DELETE\n"
        self.assertEqual(
            guard_sql(sql),
            "SELECT 'DROP; UPDATE' AS value LIMIT 100 /* ATTACH */ -- DELETE\n",
        )


if __name__ == "__main__":
    unittest.main()
