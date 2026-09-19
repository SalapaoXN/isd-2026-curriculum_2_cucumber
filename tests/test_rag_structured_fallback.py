import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from rag.structured.fallback import (
    GroundedCourseListResult,
    GroundedCourseCreditResult,
    GroundedPlacementResult,
    StructuredFallbackResult,
    StructuredFallbackScope,
    ground_course_list,
    ground_course_credit,
    ground_placement,
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

    def run_with_placement_sql(self, sql):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return sql

        result = run_structured_fallback(
            self.db_path,
            "วิชา 06016420 อยู่ช่วงไหนของหลักสูตร?",
            self.scope(),
            fake_model,
            selector_mode="placement",
        )
        return result, calls

    def run_with_credit_sql(self, sql):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return sql

        result = run_structured_fallback(
            self.db_path,
            "วิชา 06016420 มีกี่หน่วยกิต?",
            self.scope(),
            fake_model,
            selector_mode="course_credit",
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

    def test_placement_prompt_requires_canonical_placement_id_selector(self):
        result, calls = self.run_with_placement_sql(
            "SELECT 11 AS placement_id"
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.columns, ("placement_id",))
        self.assertEqual(result.rows, ((11,),))
        self.assertEqual(len(calls), 1)
        prompt = calls[0]
        self.assertIn("PLACEMENT FALLBACK CONTRACT", prompt)
        self.assertIn("MUST contain the canonical placement_id column", prompt)
        self.assertIn("SELECT DISTINCT p.placement_id AS placement_id", prompt)
        self.assertIn("do not return course names, credits, year, semester", prompt)

    def test_course_credit_prompt_requires_canonical_course_id_selector(self):
        result, calls = self.run_with_credit_sql(
            "SELECT DISTINCT course_id AS course_id FROM courses WHERE course_id = 1"
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.columns, ("course_id",))
        self.assertEqual(result.rows, ((1,),))
        self.assertEqual(len(calls), 1)
        prompt = calls[0]
        self.assertIn("COURSE CREDIT FALLBACK CONTRACT", prompt)
        self.assertIn("MUST contain the canonical course_id column", prompt)
        self.assertIn("not a presentation query", prompt)
        self.assertIn("do not return course names, credits, or provenance", prompt)

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

    @staticmethod
    def canonical_credit_record(
        course_id=1,
        *,
        plan_key="no_coop",
        credit_units=3,
        credits="3(3-0-6)",
        provenance=None,
    ):
        return {
            "course_id": course_id,
            "course_code": "06016420",
            "credit_units": credit_units,
            "credits": credits,
            "credits_raw": credits,
            "programs": ["IT"],
            "placements": [
                {
                    "plan_key": plan_key,
                    "year": 2,
                    "semester": 2,
                    "flexible_year_semester_raw": None,
                }
            ],
            "provenance": (
                ({"provenance_id": course_id + 100},)
                if provenance is None
                else provenance
            ),
        }

    @staticmethod
    def canonical_placement_record(**overrides):
        record = {
            "placement_id": 11,
            "program": "IT",
            "plan_key": "no_coop",
            "year_number": 2,
            "semester_number": 2,
            "year_semester_choices": ((2, 2),),
            "course_id": 1,
            "course_code": "06016420",
            "name_en": "CANONICAL NAME",
            "credits": "3(3-0-6)",
            "provenance": ({"provenance_id": 7},),
            "alternative_group_id": None,
            "is_alternative": False,
        }
        record.update(overrides)
        return record

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

    def ground_placements(self, selector, canonical_records, *, scope=None):
        with patch(
            "rag.structured.fallback.applicable_plan_keys",
            return_value=("no_coop",),
        ) as plans, patch(
            "rag.structured.fallback.scoped_course_set",
            return_value={"status": "ok", "courses": canonical_records},
        ) as scoped:
            grounded = ground_placement(
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

    def test_grounding_exposes_only_selected_canonical_targets(self):
        selector = self.successful_selector((1,))
        alternative_record = self.canonical_record(course_id=None)
        alternative_record.update(
            {
                "program": "IT",
                "catalog_id": 4,
                "alternative_courses": (
                    {"course_id": 1, "course_code": "06016420"},
                    {"course_id": 2, "course_code": "06016421"},
                ),
            }
        )

        grounded, _, _ = self.ground(selector, [alternative_record])

        self.assertEqual(grounded.status, "complete")
        self.assertEqual(len(grounded.selected_targets), 1)
        self.assertEqual(
            dict(grounded.selected_targets[0]),
            {
                "program": "IT",
                "course_id": 1,
                "course_code": "06016420",
                "catalog_id": 4,
            },
        )

    def test_course_credit_grounding_returns_canonical_fact_and_provenance(self):
        selector = self.successful_selector((1,))
        fact = self.canonical_credit_record()
        with patch(
            "rag.structured.fallback.course_facts",
            return_value={"status": "ok", "courses": [fact]},
        ) as facts:
            grounded = ground_course_credit(
                self.db_path,
                selector,
                self.scope(),
            )

        self.assertIsInstance(grounded, GroundedCourseCreditResult)
        self.assertEqual(grounded.status, "complete")
        self.assertEqual(grounded.credit_units, 3)
        self.assertEqual(grounded.credits, "3(3-0-6)")
        self.assertEqual(grounded.records, (fact,))
        self.assertEqual(grounded.provenance, ({"provenance_id": 101},))
        facts.assert_called_once_with(self.db_path, "06016420", "IT")

    def test_course_credit_ignores_misleading_sql_credit_columns(self):
        selector = self.successful_selector(
            (1, "99(9-9-9)"),
            columns=("course_id", "credits"),
        )
        fact = self.canonical_credit_record()
        with patch(
            "rag.structured.fallback.course_facts",
            return_value={"status": "ok", "courses": [fact]},
        ):
            grounded = ground_course_credit(
                self.db_path,
                selector,
                self.scope(),
            )

        self.assertEqual(grounded.status, "complete")
        self.assertEqual(grounded.credit_units, 3)
        self.assertEqual(grounded.credits, "3(3-0-6)")

    def test_course_credit_same_code_across_catalogs_is_resolved_together(self):
        scope = StructuredFallbackScope(
            program="IT",
            course_codes=("06016420",),
        )
        selector = self.successful_selector((1,))
        facts = [
            self.canonical_credit_record(course_id=1, plan_key="coop"),
            self.canonical_credit_record(course_id=2, plan_key="no_coop"),
        ]
        with patch(
            "rag.structured.fallback.course_facts",
            return_value={"status": "ok", "courses": facts},
        ):
            grounded = ground_course_credit(self.db_path, selector, scope)

        self.assertEqual(grounded.status, "complete")
        self.assertEqual(
            [record["course_id"] for record in grounded.records],
            [1, 2],
        )

    def test_course_credit_conflicting_canonical_values_fail_closed(self):
        scope = StructuredFallbackScope(
            program="IT",
            course_codes=("06016420",),
        )
        selector = self.successful_selector((1,))
        facts = [
            self.canonical_credit_record(course_id=1, plan_key="coop"),
            self.canonical_credit_record(
                course_id=2,
                plan_key="no_coop",
                credit_units=4,
                credits="4(4-0-8)",
            ),
        ]
        with patch(
            "rag.structured.fallback.course_facts",
            return_value={"status": "ok", "courses": facts},
        ):
            grounded = ground_course_credit(self.db_path, selector, scope)

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_course_credit_explicit_plan_restricts_canonical_identity(self):
        scope = StructuredFallbackScope(
            program="IT",
            plans=("no_coop",),
            course_codes=("06016420",),
        )
        selector = self.successful_selector((2,))
        facts = [
            self.canonical_credit_record(course_id=1, plan_key="coop"),
            self.canonical_credit_record(course_id=2, plan_key="no_coop"),
        ]
        with patch(
            "rag.structured.fallback.course_facts",
            return_value={"status": "ok", "courses": facts},
        ):
            grounded = ground_course_credit(self.db_path, selector, scope)

        self.assertEqual(grounded.status, "complete")
        self.assertEqual([r["course_id"] for r in grounded.records], [2])

    def test_course_credit_empty_selector_is_valid_empty_only_without_canonical_match(self):
        selector = self.successful_selector()
        with patch(
            "rag.structured.fallback.course_facts",
            return_value={"status": "no_data", "courses": []},
        ):
            grounded = ground_course_credit(
                self.db_path,
                selector,
                self.scope(),
            )

        self.assertEqual(grounded.status, "valid_empty")

    def test_course_credit_empty_selector_with_canonical_match_fails_closed(self):
        selector = self.successful_selector()
        with patch(
            "rag.structured.fallback.course_facts",
            return_value={
                "status": "ok",
                "courses": [self.canonical_credit_record()],
            },
        ):
            grounded = ground_course_credit(
                self.db_path,
                selector,
                self.scope(),
            )

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_course_credit_duplicate_ids_are_deduplicated(self):
        selector = self.successful_selector((1,), (1,))
        fact = self.canonical_credit_record()
        with patch(
            "rag.structured.fallback.course_facts",
            return_value={"status": "ok", "courses": [fact]},
        ):
            grounded = ground_course_credit(
                self.db_path,
                selector,
                self.scope(),
            )

        self.assertEqual(grounded.status, "complete")
        self.assertEqual(len(grounded.records), 1)

    def test_course_credit_missing_unknown_or_out_of_scope_id_fails_closed(self):
        cases = (
            ((1,), ("name_en",)),
            ((99,), ("course_id",)),
            ((2,), ("course_id",)),
        )
        fact = self.canonical_credit_record()
        for row, columns in cases:
            with self.subTest(row=row, columns=columns):
                selector = self.successful_selector(row, columns=columns)
                with patch(
                    "rag.structured.fallback.course_facts",
                    return_value={"status": "ok", "courses": [fact]},
                ):
                    grounded = ground_course_credit(
                        self.db_path,
                        selector,
                        self.scope(),
                    )
                self.assertEqual(grounded.status, "insufficient_evidence")

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

    def test_placement_grounding_returns_canonical_in_scope_record(self):
        selector = self.successful_selector((11,), columns=("placement_id",))
        grounded, _, scoped = self.ground_placements(
            selector,
            [self.canonical_placement_record()],
        )

        self.assertIsInstance(grounded, GroundedPlacementResult)
        self.assertEqual(grounded.status, "complete")
        self.assertEqual(grounded.records[0]["year_number"], 2)
        self.assertEqual(grounded.records[0]["placement_id"], 11)
        scoped.assert_called_once()

    def test_placement_grounding_ignores_misleading_sql_fields(self):
        selector = self.successful_selector(
            (11, 99, "SQL FALSE NAME"),
            columns=("placement_id", "year", "name_en"),
        )
        grounded, _, _ = self.ground_placements(
            selector,
            [self.canonical_placement_record()],
        )

        self.assertEqual(grounded.status, "complete")
        self.assertEqual(grounded.records[0]["year_number"], 2)
        self.assertEqual(grounded.records[0]["name_en"], "CANONICAL NAME")

    def test_missing_placement_id_column_fails_closed(self):
        selector = self.successful_selector((11,), columns=("course_id",))
        grounded, _, scoped = self.ground_placements(
            selector,
            [self.canonical_placement_record()],
        )

        self.assertEqual(grounded.status, "insufficient_evidence")
        scoped.assert_not_called()

    def test_malformed_placement_id_fails_closed(self):
        selector = self.successful_selector(("11",), columns=("placement_id",))
        grounded, _, scoped = self.ground_placements(
            selector,
            [self.canonical_placement_record()],
        )

        self.assertEqual(grounded.status, "insufficient_evidence")
        scoped.assert_not_called()

    def test_unknown_placement_id_fails_closed(self):
        selector = self.successful_selector((99,), columns=("placement_id",))
        grounded, _, _ = self.ground_placements(selector, [])

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_out_of_program_placement_fails_closed(self):
        selector = self.successful_selector((11,), columns=("placement_id",))
        grounded, _, _ = self.ground_placements(
            selector,
            [self.canonical_placement_record(program="DSBA")],
        )

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_out_of_plan_placement_fails_closed(self):
        selector = self.successful_selector((11,), columns=("placement_id",))
        grounded, _, _ = self.ground_placements(
            selector,
            [self.canonical_placement_record(plan_key="coop")],
        )

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_out_of_year_or_semester_placement_fails_closed(self):
        for field, value in (("year_number", 3), ("semester_number", 1)):
            with self.subTest(field=field):
                selector = self.successful_selector(
                    (11,), columns=("placement_id",)
                )
                record = self.canonical_placement_record(
                    **{field: value},
                    year_semester_choices=((value, 2),)
                    if field == "year_number"
                    else ((2, value),),
                )
                grounded, _, _ = self.ground_placements(selector, [record])

                self.assertEqual(grounded.status, "insufficient_evidence")

    def test_duplicate_placement_ids_are_deduplicated(self):
        selector = self.successful_selector(
            (11,), (11,), columns=("placement_id",)
        )
        grounded, _, scoped = self.ground_placements(
            selector,
            [self.canonical_placement_record()],
        )

        self.assertEqual(grounded.status, "complete")
        self.assertEqual(len(grounded.records), 1)
        self.assertEqual(
            scoped.call_args.kwargs["course_targets"],
            [{"course_id": 1}],
        )

    def test_empty_placement_selector_is_valid_empty(self):
        selector = self.successful_selector(columns=("placement_id",))
        grounded, _, scoped = self.ground_placements(selector, [])

        self.assertEqual(grounded.status, "valid_empty")
        self.assertEqual(grounded.records, ())
        scoped.assert_called_once()

    def test_placement_grounding_uses_canonical_provenance_only(self):
        selector = self.successful_selector(
            (11, {"provenance_id": "raw-sql"}),
            columns=("placement_id", "provenance"),
        )
        grounded, _, _ = self.ground_placements(
            selector,
            [self.canonical_placement_record()],
        )

        self.assertEqual(grounded.status, "complete")
        self.assertEqual(grounded.records[0]["provenance"], ({"provenance_id": 7},))

    def test_placement_scope_is_forwarded_without_widening(self):
        selector = self.successful_selector((11,), columns=("placement_id",))
        grounded, _, scoped = self.ground_placements(
            selector,
            [self.canonical_placement_record()],
        )

        self.assertEqual(grounded.status, "complete")
        scoped.assert_called_once_with(
            self.db_path,
            "IT",
            ("no_coop",),
            years=(2,),
            semesters=(2,),
            course_targets=[{"course_id": 1}],
        )


    def test_course_list_prompt_forbids_presentation_aggregates(self):
        result, calls = self.run_with_sql(
            "SELECT DISTINCT course_id AS course_id FROM courses WHERE course_id = 1"
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(len(calls), 1)
        prompt = calls[0]
        self.assertIn("ALWAYS return course_id rows", prompt)
        self.assertIn("NEVER use COUNT, SUM, AVG, MIN, MAX", prompt)
        for token in ("กี่วิชา", "มีไหม", "รวมกี่หน่วยกิต"):
            self.assertIn(token, prompt)
        self.assertIn(
            "Downstream deterministic canonical code computes", prompt
        )
        self.assertIn("SELECT COUNT(*)", prompt)
        self.assertIn("SELECT SUM(p.credits)", prompt)

    def test_count_style_question_uses_course_id_selector(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return (
                "SELECT DISTINCT p.course_id AS course_id "
                "FROM v_plan_courses AS p WHERE p.program = 'IT'"
            )

        result = run_structured_fallback(
            self.db_path,
            "IT ปี 3 วิชาบังคับมีกี่วิชา",
            self.scope(),
            fake_model,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.columns, ("course_id",))
        self.assertEqual(len(calls), 1)
        grounded, _, _ = self.ground(result, [self.canonical_record()])
        self.assertEqual(grounded.status, "complete")

    def test_sum_style_question_uses_course_id_selector(self):
        calls = []

        def fake_model(prompt):
            calls.append(prompt)
            return (
                "SELECT DISTINCT p.course_id AS course_id "
                "FROM v_plan_courses AS p WHERE p.program = 'IT'"
            )

        result = run_structured_fallback(
            self.db_path,
            "IT ปี 2 เทอม 1 วิชาบังคับ รวมกี่หน่วยกิต",
            self.scope(),
            fake_model,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.columns, ("course_id",))
        self.assertEqual(len(calls), 1)
        grounded, _, _ = self.ground(result, [self.canonical_record()])
        self.assertEqual(grounded.status, "complete")

    def test_count_aggregate_without_course_id_fails_closed(self):
        selector = StructuredFallbackResult(
            status="success",
            columns=("course_count",),
            rows=((12,),),
        )
        grounded, _, _ = self.ground(selector, [self.canonical_record()])

        self.assertEqual(grounded.status, "insufficient_evidence")

    def test_sum_aggregate_without_course_id_fails_closed(self):
        selector = StructuredFallbackResult(
            status="success",
            columns=("SUM(v_plan_courses.credits)",),
            rows=((36,),),
        )
        grounded, _, _ = self.ground(selector, [self.canonical_record()])

        self.assertEqual(grounded.status, "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
