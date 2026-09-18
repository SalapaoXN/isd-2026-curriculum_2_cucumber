import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from rag.structured.fallback import (
    GroundedCourseListResult,
    StructuredFallbackResult,
    StructuredFallbackScope,
    ground_course_list,
    run_structured_fallback,
)


class RagStructuredFallbackTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "fallback.sqlite"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("BEGIN")
            connection.execute(
                "CREATE TABLE courses (course_id INTEGER, course_code TEXT)"
            )
            connection.execute(
                "CREATE TABLE v_plan_courses ("
                "course_id INTEGER, program TEXT, plan_key TEXT, "
                "year INTEGER, semester INTEGER, requirement_type TEXT, credits REAL"
                ")"
            )
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?)",
                [(1, "06016420"), (2, "06016421")],
            )
            connection.execute(
                "INSERT INTO v_plan_courses VALUES "
                "(1, 'IT', 'no_coop', 2, 2, 'required', 3)"
            )
            connection.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def scope():
        return StructuredFallbackScope(
            program="IT",
            plans=("no_coop",),
            years=(2,),
            semesters=(2,),
            course_ids=(1,),
            course_codes=("06016420",),
        )

    def run_with_sql(self, sql, *, model=None):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return sql

        result = run_structured_fallback(
            self.db_path,
            "วิชา 06016420 มีกี่หน่วยกิต?",
            self.scope(),
            model or fake_model,
        )
        return result, calls

    def test_success_uses_one_model_call_and_executes_allowed_select(self):
        result, calls = self.run_with_sql(
            "SELECT DISTINCT course_id AS course_id FROM courses WHERE course_id = 1"
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.columns, ("course_id",))
        self.assertEqual(result.rows, ((1,),))
        self.assertEqual(len(calls), 1)

    def test_course_list_prompt_requires_canonical_course_id_selector(self):
        sql = (
            "SELECT DISTINCT p.course_id AS course_id "
            "FROM v_plan_courses AS p "
            "WHERE p.program = 'IT' AND p.plan_key = 'no_coop' "
            "AND p.year = 2 AND p.semester = 2"
        )
        result, calls = self.run_with_sql(sql)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.columns, ("course_id",))
        self.assertEqual(result.rows, ((1,),))
        grounded, _, _ = self.ground(
            result,
            [self.canonical_record()],
        )
        self.assertEqual(grounded.status, "complete")
        prompt = calls[0]
        self.assertIn("COURSE LIST/FILTER FALLBACK CONTRACT", prompt)
        self.assertIn("MUST contain the canonical course_id column", prompt)
        self.assertIn("SELECT DISTINCT p.course_id AS course_id", prompt)
        self.assertIn("candidate selector, not a presentation query", prompt)
        self.assertIn("do not return course_code, names, credits, or provenance", prompt)

    def test_prompt_marks_all_scope_values_authoritative(self):
        result, calls = self.run_with_sql("SELECT course_code FROM courses")

        self.assertEqual(result.status, "success")
        self.assertEqual(len(calls), 1)
        prompt = calls[0]
        self.assertIn("AUTHORITATIVE DETERMINISTIC SCOPE", prompt)
        for value in ("IT", "no_coop", "2", "06016420"):
            self.assertIn(value, prompt)
        self.assertIn("MUST be preserved", prompt)
        self.assertIn("Never infer a program from a course-code prefix", prompt)

    def test_disallowed_relation_fails_before_execution(self):
        with patch("rag.structured.fallback.execute_readonly") as execute:
            result, calls = self.run_with_sql("SELECT * FROM secret_table")

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_category, "relation_guard")
        self.assertEqual(len(calls), 1)
        execute.assert_not_called()

    def test_write_and_pragma_sql_fail_closed(self):
        for sql in ("UPDATE courses SET course_code = 'x'", "PRAGMA user_version"):
            with self.subTest(sql=sql):
                result, calls = self.run_with_sql(sql)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.error_category, "generation")
                self.assertEqual(len(calls), 1)

    def test_model_exception_fails_closed_without_retry(self):
        calls = []

        def failing_model(prompt):
            calls.append(prompt)
            raise RuntimeError("provider unavailable")

        result = run_structured_fallback(
            self.db_path,
            "course list",
            self.scope(),
            failing_model,
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_category, "generation")
        self.assertIn("provider unavailable", result.error)
        self.assertEqual(len(calls), 1)

    def test_invalid_sql_fails_closed_without_retry(self):
        result, calls = self.run_with_sql("SELECT missing_column FROM courses")

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_category, "execution")
        self.assertEqual(len(calls), 1)

    def test_no_repair_call_and_rows_are_internal_result(self):
        with patch("rag.structured.fallback.question_to_sql") as question_to_sql:
            question_to_sql.return_value = "SELECT course_id FROM courses"
            with patch("rag.structured.fallback.execute_readonly") as execute:
                execute.return_value = (["course_code"], [("06016420",)])
                with patch("rag.structured.fallback.guard_sql", wraps=None) as guard:
                    guard.return_value = "SELECT course_code FROM courses LIMIT 100"
                    result = run_structured_fallback(
                        self.db_path,
                        "course list",
                        self.scope(),
                        lambda prompt: "unused",
                    )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.rows, (("06016420",),))
        question_to_sql.assert_called_once()
        guard.assert_called_once()
        execute.assert_called_once()

    def test_invalid_scope_is_rejected_before_model_call(self):
        with self.assertRaises(ValueError):
            StructuredFallbackScope(program="")

    @staticmethod
    def successful_selector(*rows, columns=("course_id",)):
        return StructuredFallbackResult(
            status="success",
            columns=tuple(columns),
            rows=tuple(rows),
        )

    @staticmethod
    def canonical_record(course_id=1, *, provenance=None, course_code="06016420"):
        return {
            "course_id": course_id,
            "course_code": course_code,
            "name_en": "CANONICAL NAME",
            "credits": "3(3-0-6)",
            "provenance": (
                ({"provenance_id": 7},) if provenance is None else provenance
            ),
            "placement_id": 11,
            "alternative_group_id": None,
            "is_alternative": False,
        }

    def ground(self, selector, canonical_records, *, scope=None):
        with patch(
            "rag.structured.fallback.applicable_plan_keys",
            return_value=("no_coop",),
        ) as plans, patch(
            "rag.structured.fallback.scoped_course_set",
            return_value={"status": "ok", "courses": canonical_records},
        ) as scoped:
            grounded = ground_course_list(
                self.db_path,
                selector,
                scope or self.scope(),
            )
        return grounded, plans, scoped

    def test_grounding_returns_canonical_in_scope_records(self):
        selector = self.successful_selector((1,))
        grounded, _, scoped = self.ground(
            selector,
            [self.canonical_record()],
        )

        self.assertIsInstance(grounded, GroundedCourseListResult)
        self.assertEqual(grounded.status, "complete")
        self.assertEqual(grounded.records[0]["course_id"], 1)
        scoped.assert_called_once()

    def test_grounding_ignores_misleading_sql_fields(self):
        selector = self.successful_selector(
            (1, "SQL FALSE NAME", "99 credits"),
            columns=("course_id", "name_en", "credits"),
        )
        grounded, _, _ = self.ground(
            selector,
            [self.canonical_record()],
        )

        self.assertEqual(grounded.status, "complete")
        self.assertEqual(grounded.records[0]["name_en"], "CANONICAL NAME")
        self.assertEqual(grounded.records[0]["credits"], "3(3-0-6)")

    def test_out_of_scope_selected_id_fails_closed(self):
        selector = self.successful_selector((2,))
        grounded, _, _ = self.ground(
            selector,
            [self.canonical_record(course_id=2, course_code="06016421")],
        )

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_missing_course_id_column_fails_closed(self):
        selector = self.successful_selector(
            (1, "CANONICAL NAME"),
            columns=("name_en", "credits"),
        )
        grounded, _, _ = self.ground(selector, [self.canonical_record()])

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_unknown_course_id_fails_closed(self):
        selector = self.successful_selector((99,))
        grounded, _, _ = self.ground(selector, [])

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_duplicate_selected_ids_are_deduplicated(self):
        selector = self.successful_selector((1,), (1,))
        grounded, _, scoped = self.ground(
            selector,
            [self.canonical_record()],
        )

        self.assertEqual(grounded.status, "complete")
        self.assertEqual(len(grounded.records), 1)
        targets = scoped.call_args.kwargs["course_targets"]
        self.assertEqual(targets, [{"course_id": 1}])

    def test_empty_selector_rows_are_valid_empty(self):
        selector = self.successful_selector()
        grounded, _, scoped = self.ground(selector, [])

        self.assertEqual(grounded.status, "valid_empty")
        self.assertEqual(grounded.records, ())
        scoped.assert_called_once()

    def test_grounded_records_require_canonical_provenance(self):
        selector = self.successful_selector((1,))
        grounded, _, _ = self.ground(
            selector,
            [self.canonical_record(provenance=())],
        )

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_scope_is_forwarded_without_widening(self):
        selector = self.successful_selector((1,))
        scope = StructuredFallbackScope(
            program="IT",
            plans=("coop",),
            years=(3,),
            semesters=(1,),
            course_ids=(1,),
            course_codes=("06016420",),
        )
        grounded, _, scoped = self.ground(
            selector,
            [self.canonical_record()],
            scope=scope,
        )

        self.assertEqual(grounded.status, "complete")
        scoped.assert_called_once_with(
            self.db_path,
            "IT",
            ("coop",),
            years=(3,),
            semesters=(1,),
            course_targets=[{"course_id": 1}],
        )


if __name__ == "__main__":
    unittest.main()
