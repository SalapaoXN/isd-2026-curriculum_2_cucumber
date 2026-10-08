"""Resource controls at the real read-only SQLite boundary."""

from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from backend.llm_sql_qa import ask_sql
from rag.structured.execute import execute_readonly
from rag.structured.guard_sql import guard_sql


class SqlResourceBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "test.db"
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("CREATE TABLE courses(course_id INTEGER, course_code TEXT)")
            conn.executemany("INSERT INTO courses VALUES (?, ?)",
                             [(i, f"{i:08d}") for i in range(1000)])
            conn.commit()

    def test_normal_query_and_nested_query(self):
        for query in ("SELECT course_code FROM courses ORDER BY course_code LIMIT 5",
                      "SELECT course_code FROM (SELECT course_code FROM courses) LIMIT 5"):
            columns, rows = execute_readonly(self.db, query)
            self.assertEqual(columns, ["course_code"])
            self.assertEqual(len(rows), 5)

    def test_cross_join_aborts(self):
        with self.assertRaisesRegex(sqlite3.OperationalError, "interrupted"):
            execute_readonly(self.db, "SELECT COUNT(*) FROM courses a "
                             "JOIN courses b ON 1=1 JOIN courses c ON 1=1")

    def test_heavy_order_by_aborts(self):
        with self.assertRaisesRegex(sqlite3.OperationalError, "interrupted"):
            execute_readonly(self.db, "SELECT a.course_code FROM courses a "
                             "JOIN courses b ON 1=1 ORDER BY random() LIMIT 1")

    def test_allocator_functions_rejected_including_quoted_names(self):
        for function in ("randomblob", "zeroblob", "repeat", '"randomblob"',
                         "[zeroblob]", "`randomblob`"):
            with self.subTest(function=function), self.assertRaises(ValueError):
                guard_sql(f"SELECT {function}(1000000000)")

    def test_single_expression_allocation_is_bounded(self):
        self.assertEqual(execute_readonly(self.db, "SELECT printf('%2000000s', 'x')")[1], [(None,)])
        with self.assertRaises(sqlite3.DataError):
            execute_readonly(self.db, "SELECT printf('%500000s', 'x') || printf('%500000s', 'x') || printf('%500000s', 'x')")

    def test_recursive_cte_rejected(self):
        with self.assertRaisesRegex(ValueError, "RECURSIVE"):
            guard_sql("WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT * FROM x")

    def test_length_bound_precedes_tokenization(self):
        with self.assertRaisesRegex(ValueError, "maximum length"):
            guard_sql("SELECT 1 " + " " * 20000)

    def test_query_remains_read_only(self):
        with self.assertRaises(ValueError):
            execute_readonly(self.db, "DELETE FROM courses")
        self.assertEqual(execute_readonly(self.db, "SELECT COUNT(*) FROM courses")[1], [(1000,)])

    def test_resource_abort_cannot_reach_answer_provider(self):
        runtime = Path(__file__).resolve().parents[2] / "cucumber_outputs/runtime/curriculum.db"
        with patch("backend.llm_sql_qa.execute_readonly", side_effect=sqlite3.OperationalError("interrupted")):
            calls = []
            result = ask_sql(runtime, "IT มีวิชาอะไรบ้าง", "IT",
                             lambda prompt: "SELECT course_code FROM courses",
                             lambda prompt: calls.append(prompt) or "invented",
                             conversation_context={"program": "IT", "catalog_key": "it-2565"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "sqlite_error")
        self.assertEqual(calls, [])
