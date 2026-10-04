import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend import llm_sql_qa
from backend.llm_sql_qa import ask_sql


class LlmSqlQaTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "curriculum.db"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "CREATE TABLE courses (course_id INTEGER, course_code TEXT, name_th TEXT)"
            )
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?, ?)",
                [(1, "C101", "แคลคูลัส"), (2, "C102", "ฟิสิกส์")],
            )
            connection.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _enable_program_course_scope(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("ALTER TABLE courses ADD COLUMN program TEXT")
            connection.execute("ALTER TABLE courses ADD COLUMN catalog_id INTEGER DEFAULT 1")
            connection.execute("UPDATE courses SET program = 'IT'")
            connection.execute(
                "CREATE TABLE catalogs (catalog_id INTEGER PRIMARY KEY, catalog_key TEXT)"
            )
            connection.execute("INSERT INTO catalogs VALUES (1, 'it-2560')")
            connection.execute(
                "CREATE VIEW v_plan_courses AS "
                "SELECT course_id, program, course_code, course_code AS course "
                "FROM courses"
            )
            connection.commit()

    def _build_edition_scope_db(self):
        db_path = Path(self.temp_dir.name) / "edition-curriculum.db"
        with closing(sqlite3.connect(db_path)) as connection:
            connection.executescript(
                "CREATE TABLE catalogs (catalog_id INTEGER PRIMARY KEY, catalog_key TEXT, "
                "academic_year TEXT);"
                "CREATE TABLE programs (program_id INTEGER PRIMARY KEY, catalog_id INTEGER, "
                "program_code TEXT, program_code_normalized TEXT);"
                "CREATE TABLE courses (course_id INTEGER PRIMARY KEY, catalog_id INTEGER, "
                "course_code TEXT, course_code_normalized TEXT, name_th TEXT);"
                "CREATE TABLE curriculum_plans (plan_id INTEGER PRIMARY KEY, catalog_id INTEGER, "
                "program_id INTEGER, plan_key TEXT);"
                "INSERT INTO catalogs VALUES "
                "(1, 'dsba-2560', '2560'), (2, 'dsba-2565', '2565');"
                "INSERT INTO programs VALUES (1, 1, 'DSBA', 'dsba'), (2, 2, 'DSBA', 'dsba');"
                "INSERT INTO courses VALUES "
                "(1, 1, 'C101', 'c101', 'legacy course'), "
                "(2, 2, 'C101', 'c101', 'current course'), "
                "(3, 1, 'C202', 'c202', 'unique course');"
                "INSERT INTO curriculum_plans VALUES (1, 1, 1, 'default'), (2, 2, 2, 'default');"
                "CREATE VIEW v_plan_courses AS "
                "SELECT plans.plan_id, plans.program_id, programs.program_code AS program, "
                "plans.plan_key AS plan, courses.course_id, courses.course_code, "
                "courses.course_code AS course FROM curriculum_plans AS plans "
                "JOIN programs ON programs.program_id = plans.program_id "
                "JOIN courses ON courses.catalog_id = plans.catalog_id;"
            )
        return db_path

    def test_previous_result_catalog_scope_blocks_cross_edition_rows(self):
        db_path = self._build_edition_scope_db()
        context = {
            "result_courses": [
                {"catalog_key": "dsba-2560", "program": "DSBA", "course_code": "C101"}
            ],
            "result_scope_program": "DSBA",
        }

        result = ask_sql(
            db_path,
            "ในวิชาเหล่านี้มีอะไรบ้าง",
            "DSBA",
            lambda _prompt: "SELECT course_id, course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context=context,
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"], [{"course_id": 1, "course_code": "C101"}])

    def test_grounded_answer_without_source_provenance_fails_closed(self):
        result = ask_sql(
            self.db_path,
            "IT มีกี่วิชา",
            "IT",
            lambda _prompt: "SELECT course_id, course_code FROM courses",
            lambda _prompt: "unverified model answer",
            grounding_callable=lambda _question: {
                "status": "answer",
                "final_answer": "unverified answer",
                "provenance": [],
            },
        )

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("unverified", result["answer"])
        self.assertEqual(result["provenance"], [])

    def test_previous_result_catalog_scope_bounds_aggregate(self):
        db_path = self._build_edition_scope_db()
        context = {
            "result_courses": [
                {"catalog_key": "dsba-2560", "program": "DSBA", "course_code": "C101"}
            ],
            "result_scope_program": "DSBA",
        }

        result = ask_sql(
            db_path,
            "ในวิชาเหล่านี้มีกี่วิชา",
            "DSBA",
            lambda _prompt: "SELECT COUNT(*) AS total FROM courses",
            lambda _prompt: "พบ 1 วิชา",
            conversation_context=context,
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"], [{"total": 1}])

    def test_ambiguous_legacy_previous_result_context_fails_closed(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "ในวิชาเหล่านี้มีอะไรบ้าง",
            "DSBA",
            lambda _prompt: "SELECT course_id, course_code FROM courses",
            lambda _prompt: "ไม่ควรถูกเรียก",
            conversation_context={
                "result_courses": [{"program": "DSBA", "course_code": "C101"}],
                "result_scope_program": "DSBA",
            },
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["action"], "catalog_required")

    def test_legacy_previous_result_without_catalog_fails_closed_for_multi_edition_program(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "ในวิชาเหล่านี้มีอะไรบ้าง",
            "DSBA",
            lambda _prompt: "SELECT course_id, course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context={
                "result_courses": [{"program": "DSBA", "course_code": "C202"}],
                "result_scope_program": "DSBA",
            },
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["action"], "catalog_required")

    def test_edition_aware_focus_context_blocks_cross_edition_rows(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "วิชานี้มีอะไรบ้าง",
            "DSBA",
            lambda _prompt: "SELECT course_id, course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context={
                "focus_course": {
                    "catalog_key": "dsba-2560",
                    "program": "DSBA",
                    "course_code": "C101",
                }
            },
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"], [{"course_id": 1, "course_code": "C101"}])

    def test_ambiguous_legacy_focus_context_fails_closed(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "วิชานี้มีอะไรบ้าง",
            "DSBA",
            lambda _prompt: "SELECT course_id, course_code FROM courses",
            lambda _prompt: self.fail("ambiguous legacy focus must not reach answer model"),
            conversation_context={
                "focus_course": {"program": "DSBA", "course_code": "C101"}
            },
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["action"], "catalog_required")

    def test_unscoped_multi_edition_program_fails_closed_before_sql_model(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "แสดงรายวิชา",
            "DSBA",
            lambda _prompt: self.fail("ambiguous edition must fail before SQL generation"),
            lambda _prompt: self.fail("ambiguous edition must not reach answer model"),
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["action"], "catalog_required")
        self.assertEqual(result["catalog_keys"], ["dsba-2560", "dsba-2565"])

    def test_selected_catalog_scope_filters_rows_and_aggregates(self):
        db_path = self._build_edition_scope_db()
        context = {"catalog_key": "dsba-2560"}
        rows_result = ask_sql(
            db_path,
            "แสดงรายวิชา",
            "DSBA",
            lambda _prompt: "SELECT course_id, course_code FROM courses ORDER BY course_id",
            lambda _prompt: "พบข้อมูล",
            conversation_context=context,
        )
        count_result = ask_sql(
            db_path,
            "มีวิชากี่วิชา",
            "DSBA",
            lambda _prompt: "SELECT COUNT(*) AS total FROM courses",
            lambda _prompt: "พบ 2 วิชา",
            conversation_context=context,
        )

        self.assertEqual(rows_result["rows"], [
            {"course_id": 1, "course_code": "C101"},
            {"course_id": 3, "course_code": "C202"},
        ])
        self.assertEqual(count_result["rows"], [{"total": 2}])

    def test_canonical_dsba_editions_scope_course_aggregate_and_followup_queries(self):
        db_path = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"

        for catalog_key, academic_year in (
            ("dsba-2560", "2560"),
            ("dsba-2565", "2565"),
        ):
            for plan in ("coop", "no_coop"):
                with self.subTest(catalog_key=catalog_key, plan=plan):
                    course_sql = (
                        "SELECT catalogs.catalog_key, catalogs.academic_year, "
                        "plan_rows.program, plan_rows.plan, plan_rows.course_code, "
                        "plan_rows.year, plan_rows.semester "
                        "FROM v_plan_courses AS plan_rows "
                        "JOIN courses ON courses.course_id = plan_rows.course_id "
                        "JOIN catalogs ON catalogs.catalog_id = courses.catalog_id "
                        f"WHERE plan_rows.program = 'DSBA' AND plan_rows.plan = '{plan}' "
                        "ORDER BY plan_rows.course_code LIMIT 5"
                    )
                    first = ask_sql(
                        db_path,
                        f"แสดงรายวิชา DSBA {plan}",
                        "DSBA",
                        lambda _prompt, sql=course_sql: sql,
                        lambda _prompt: "พบรายวิชา",
                        conversation_context={"catalog_key": catalog_key},
                    )

                    self.assertEqual(first["status"], "answer")
                    self.assertTrue(first["rows"])
                    self.assertTrue(
                        all(
                            row["catalog_key"] == catalog_key
                            and row["academic_year"] == academic_year
                            and row["plan"] == plan
                            and row["year"] is not None
                            and row["semester"] is not None
                            for row in first["rows"]
                        )
                    )
                    self.assertEqual(first["next_context"]["catalog_key"], catalog_key)
                    self.assertTrue(
                        all(
                            item["catalog_key"] == catalog_key
                            for item in first["next_context"]["result_courses"]
                        )
                    )

                    aggregate_sql = (
                        "SELECT catalogs.catalog_key, catalogs.academic_year, "
                        "COUNT(DISTINCT plan_rows.course_code) AS course_count "
                        "FROM v_plan_courses AS plan_rows "
                        "JOIN courses ON courses.course_id = plan_rows.course_id "
                        "JOIN catalogs ON catalogs.catalog_id = courses.catalog_id "
                        f"WHERE plan_rows.program = 'DSBA' AND plan_rows.plan = '{plan}' "
                        "GROUP BY catalogs.catalog_id, catalogs.catalog_key, catalogs.academic_year"
                    )
                    aggregate = ask_sql(
                        db_path,
                        f"DSBA {plan} มีวิชากี่วิชา",
                        "DSBA",
                        lambda _prompt, sql=aggregate_sql: sql,
                        lambda _prompt: "นับรายวิชาแล้ว",
                        conversation_context={"catalog_key": catalog_key},
                    )
                    self.assertEqual(aggregate["status"], "answer")
                    self.assertEqual(len(aggregate["rows"]), 1)
                    self.assertEqual(aggregate["rows"][0]["catalog_key"], catalog_key)
                    self.assertEqual(aggregate["rows"][0]["academic_year"], academic_year)

                    follow_up_sql = (
                        "SELECT catalogs.catalog_key, plan_rows.program, plan_rows.course_code "
                        "FROM v_plan_courses AS plan_rows "
                        "JOIN courses ON courses.course_id = plan_rows.course_id "
                        "JOIN catalogs ON catalogs.catalog_id = courses.catalog_id "
                        "ORDER BY plan_rows.course_code"
                    )
                    follow_up = ask_sql(
                        db_path,
                        "ในวิชาเหล่านี้มีอะไรบ้าง",
                        "DSBA",
                        lambda _prompt, sql=follow_up_sql: sql,
                        lambda _prompt: "พบวิชาจากผลก่อนหน้า",
                        conversation_context={
                            key: value
                            for key, value in first["next_context"].items()
                            if key in {
                                "catalog_key", "focus_course", "result_courses",
                                "result_scope_program", "result_set_empty",
                            }
                        },
                    )
                    self.assertEqual(follow_up["status"], "answer")
                    self.assertTrue(follow_up["rows"])
                    self.assertTrue(
                        all(row["catalog_key"] == catalog_key for row in follow_up["rows"])
                    )

    def test_canonical_course_in_both_dsba_plans_has_single_edition_identity(self):
        db_path = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"
        sql = (
            "SELECT catalogs.catalog_key, plan_rows.program, plan_rows.plan, "
            "plan_rows.course_code FROM v_plan_courses AS plan_rows "
            "JOIN courses ON courses.course_id = plan_rows.course_id "
            "JOIN catalogs ON catalogs.catalog_id = courses.catalog_id "
            "WHERE plan_rows.program = 'DSBA' AND plan_rows.course_code = '06026100' "
            "ORDER BY plan_rows.plan"
        )

        result = ask_sql(
            db_path,
            "DSBA course 06026100 is in which selected-edition plans?",
            "DSBA",
            lambda _prompt: sql,
            lambda _prompt: "พบในแผนของฉบับที่เลือก",
            conversation_context={"catalog_key": "dsba-2560"},
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            {row["plan"] for row in result["rows"]}, {"coop", "no_coop"}
        )
        self.assertEqual(result["next_context"]["catalog_key"], "dsba-2560")
        self.assertEqual(
            result["next_context"]["focus_course"],
            {
                "catalog_key": "dsba-2560",
                "program": "DSBA",
                "course_code": "06026100",
            },
        )

    def test_canonical_catalog_change_invalidates_prior_edition_result_rows(self):
        db_path = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"
        query = (
            "SELECT catalogs.catalog_key, catalogs.academic_year, "
            "plan_rows.program, plan_rows.course_code "
            "FROM v_plan_courses AS plan_rows "
            "JOIN courses ON courses.course_id = plan_rows.course_id "
            "JOIN catalogs ON catalogs.catalog_id = courses.catalog_id "
            "WHERE plan_rows.program = 'DSBA' AND plan_rows.plan = 'coop' "
            "ORDER BY plan_rows.course_code LIMIT 5"
        )

        for selected_catalog, selected_year, prior_catalog in (
            ("dsba-2560", "2560", "dsba-2565"),
            ("dsba-2565", "2565", "dsba-2560"),
        ):
            context = {
                "catalog_key": selected_catalog,
                "result_courses": [
                    {
                        "catalog_key": prior_catalog,
                        "program": "DSBA",
                        "course_code": "06026100" if prior_catalog == "dsba-2560" else "06066300",
                    }
                ],
                "result_scope_program": "DSBA",
            }
            result = ask_sql(
                db_path,
                "ในวิชาเหล่านี้มีอะไรบ้าง",
                "DSBA",
                lambda _prompt: query,
                lambda _prompt: "พบรายวิชาในฉบับที่เลือก",
                conversation_context=context,
            )

            self.assertEqual(result["status"], "answer")
            self.assertTrue(result["rows"])
            self.assertTrue(
                all(
                    row["catalog_key"] == selected_catalog
                    and row["academic_year"] == selected_year
                    for row in result["rows"]
                )
            )
            self.assertEqual(result["next_context"]["catalog_key"], selected_catalog)

    def test_canonical_unique_dsba_courses_stay_inside_selected_edition(self):
        db_path = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"
        cases = (
            ("06026160", "dsba-2560", "dsba-2565"),
            ("90641002", "dsba-2565", "dsba-2560"),
            ("06026106", "dsba-2560", "dsba-2565"),
            ("06066300", "dsba-2565", "dsba-2560"),
        )
        sql_template = (
            "SELECT catalogs.catalog_key, plan_rows.course_code "
            "FROM v_plan_courses AS plan_rows "
            "JOIN courses ON courses.course_id = plan_rows.course_id "
            "JOIN catalogs ON catalogs.catalog_id = courses.catalog_id "
            "WHERE plan_rows.program = 'DSBA' "
            "AND plan_rows.course_code = '{course_code}'"
        )

        for course_code, source_catalog, other_catalog in cases:
            for selected_catalog, should_exist in (
                (source_catalog, True),
                (other_catalog, False),
            ):
                with self.subTest(course_code=course_code, selected_catalog=selected_catalog):
                    result = ask_sql(
                        db_path,
                        f"แสดงรายวิชา DSBA {course_code}",
                        "DSBA",
                        lambda _prompt, code=course_code: sql_template.format(course_code=code),
                        lambda _prompt: "พบรายวิชา",
                        conversation_context={"catalog_key": selected_catalog},
                    )

                    self.assertEqual(bool(result["rows"]), should_exist)
                    self.assertTrue(
                        all(row["catalog_key"] == selected_catalog for row in result["rows"])
                    )
                    self.assertEqual(result["next_context"]["catalog_key"], selected_catalog)

    def test_selected_catalog_invalidates_stale_focus_and_persists_scope(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "วิชานี้มีอะไรบ้าง",
            "DSBA",
            lambda _prompt: "SELECT course_id, course_code FROM courses ORDER BY course_id",
            lambda _prompt: "พบข้อมูล",
            conversation_context={
                "catalog_key": "dsba-2565",
                "focus_course": {
                    "catalog_key": "dsba-2560",
                    "program": "DSBA",
                    "course_code": "C101",
                },
            },
        )

        self.assertEqual(result["rows"], [{"course_id": 2, "course_code": "C101"}])
        self.assertEqual(result["next_context"]["catalog_key"], "dsba-2565")

    def test_selected_catalog_invalidates_stale_result_set(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "ในวิชาเหล่านี้มีอะไรบ้าง",
            "DSBA",
            lambda _prompt: "SELECT course_id, course_code FROM courses ORDER BY course_id",
            lambda _prompt: "พบข้อมูล",
            conversation_context={
                "catalog_key": "dsba-2565",
                "result_courses": [
                    {"catalog_key": "dsba-2560", "program": "DSBA", "course_code": "C101"}
                ],
                "result_scope_program": "DSBA",
            },
        )

        self.assertEqual(result["rows"], [{"course_id": 2, "course_code": "C101"}])
        self.assertEqual(result["next_context"]["catalog_key"], "dsba-2565")

    def test_nonexistent_catalog_key_fails_before_sql_model(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "แสดงรายวิชา",
            "DSBA",
            lambda _prompt: self.fail("invalid catalog must fail before SQL generation"),
            lambda _prompt: self.fail("invalid catalog must not reach answer model"),
            conversation_context={"catalog_key": "dsba-9999"},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_context")

    def test_selected_catalog_key_is_added_to_next_focus_context(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "DSBA C202 คืออะไร",
            "DSBA",
            lambda _prompt: (
                "SELECT plan_rows.program, plan_rows.course_code "
                "FROM v_plan_courses AS plan_rows WHERE plan_rows.course_id = 3"
            ),
            lambda _prompt: "พบข้อมูล",
            conversation_context={"catalog_key": "dsba-2560"},
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            result["next_context"]["focus_course"]["catalog_key"], "dsba-2560"
        )

    def test_year_semester_focus_survives_course_list_followup_in_each_edition(self):
        db_path = Path(__file__).parents[1] / "cucumber_outputs" / "runtime" / "curriculum.db"
        credit_sql = (
            "SELECT catalogs.catalog_key, plan_rows.program, plan_rows.plan, "
            "plan_rows.year, plan_rows.semester, SUM(plan_rows.credit_units) AS total_credits "
            "FROM v_plan_courses AS plan_rows "
            "JOIN curriculum_plans AS plans ON plans.plan_id = plan_rows.plan_id "
            "JOIN catalogs ON catalogs.catalog_id = plans.catalog_id "
            "WHERE plan_rows.program = 'DSBA' AND plan_rows.year = 2 "
            "AND plan_rows.semester = 1 "
            "GROUP BY catalogs.catalog_key, plan_rows.program, plan_rows.plan, "
            "plan_rows.year, plan_rows.semester"
        )
        course_sql = (
            "SELECT catalogs.catalog_key, plan_rows.program, plan_rows.plan, "
            "plan_rows.year, plan_rows.semester, plan_rows.course_code "
            "FROM v_plan_courses AS plan_rows "
            "JOIN curriculum_plans AS plans ON plans.plan_id = plan_rows.plan_id "
            "JOIN catalogs ON catalogs.catalog_id = plans.catalog_id "
            "ORDER BY plan_rows.course_code"
        )

        for catalog_key, expected_credits in (("dsba-2565", 15), ("dsba-2560", 19)):
            with self.subTest(catalog_key=catalog_key):
                first = ask_sql(
                    db_path,
                    "ปี 2 เทอม 1 เรียนกี่หน่วยกิต",
                    "DSBA",
                    lambda _prompt: credit_sql,
                    lambda _prompt: "พบยอดหน่วยกิต",
                    conversation_context={"catalog_key": catalog_key},
                )
                self.assertEqual(first["status"], "answer")
                self.assertTrue(first["rows"])
                self.assertEqual({row["total_credits"] for row in first["rows"]}, {expected_credits})
                context = first["next_context"]
                self.assertEqual(context["catalog_key"], catalog_key)
                self.assertEqual(context["years"], [2])
                self.assertEqual(context["semesters"], [1])

                second = ask_sql(
                    db_path,
                    "แล้วมีวิชาอะไรบ้าง",
                    "DSBA",
                    lambda prompt: course_sql,
                    lambda _prompt: "พบรายวิชาในช่วงที่เลือก",
                    conversation_context=context,
                )
                self.assertEqual(second["status"], "answer")
                self.assertTrue(second["rows"])
                self.assertTrue(
                    all(
                        row["catalog_key"] == catalog_key
                        and row["year"] == 2
                        and row["semester"] == 1
                        for row in second["rows"]
                    )
                )
                self.assertEqual(second["next_context"]["catalog_key"], catalog_key)

    def test_catalog_scope_keeps_read_only_sql_guard(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "ในวิชาเหล่านี้มีอะไรบ้าง",
            "DSBA",
            lambda _prompt: "DELETE FROM courses",
            lambda _prompt: self.fail("mutating SQL must not reach the answer model"),
            conversation_context={
                "result_courses": [
                    {"catalog_key": "dsba-2560", "program": "DSBA", "course_code": "C101"}
                ],
                "result_scope_program": "DSBA",
            },
        )

        self.assertEqual(result["error"]["code"], "invalid_sql")
        with closing(sqlite3.connect(db_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM courses").fetchone()[0], 3)

    def test_next_context_preserves_catalog_key_from_canonical_query(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "DSBA C202 คือวิชาอะไร",
            "DSBA",
            lambda _prompt: (
                "SELECT plan_rows.program, plan_rows.course_code, catalogs.catalog_key "
                "FROM v_plan_courses AS plan_rows "
                "JOIN courses ON courses.course_id = plan_rows.course_id "
                "JOIN catalogs ON catalogs.catalog_id = courses.catalog_id "
                "WHERE courses.catalog_id = 1 AND courses.course_id = 3"
            ),
            lambda _prompt: "C202 เป็นรายวิชาในหลักสูตร",
            conversation_context={"catalog_key": "dsba-2560"},
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            result["next_context"]["focus_course"]["catalog_key"], "dsba-2560"
        )

    def test_unscoped_multi_edition_listing_does_not_mix_catalogs(self):
        db_path = self._build_edition_scope_db()
        result = ask_sql(
            db_path,
            "แสดงรายวิชา C101",
            "DSBA",
            lambda _prompt: self.fail("ambiguous edition must not generate SQL"),
            lambda _prompt: self.fail("ambiguous edition must not reach answer model"),
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["action"], "catalog_required")

    def test_previous_result_scope_blocks_row_leak_before_answer_model(self):
        self._enable_program_course_scope()
        context = {
            "result_courses": [{"program": "IT", "course_code": "C101"}],
            "result_scope_program": "IT",
        }

        result = ask_sql(
            self.db_path,
            "ในวิชาเหล่านี้มีอะไรบ้าง",
            "IT",
            lambda _prompt: "SELECT course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context=context,
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"], [{"course_code": "C101"}])

    def test_previous_result_scope_bounds_aggregate_before_count(self):
        self._enable_program_course_scope()
        context = {
            "result_courses": [{"program": "IT", "course_code": "C101"}],
            "result_scope_program": "IT",
        }

        result = ask_sql(
            self.db_path,
            "ในวิชาเหล่านี้มีกี่วิชา",
            "IT",
            lambda _prompt: "SELECT COUNT(*) AS total FROM courses",
            lambda _prompt: "พบข้อมูล 1 วิชา",
            conversation_context=context,
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"], [{"total": 1}])

    def test_no_previous_result_context_keeps_unscoped_query_behavior(self):
        self._enable_program_course_scope()

        result = ask_sql(
            self.db_path,
            "มีวิชาอะไรบ้าง",
            "IT",
            lambda _prompt: "SELECT course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(
            result["rows"],
            [{"course_code": "C101"}, {"course_code": "C102"}],
        )

    def test_previous_result_scope_allows_multiple_and_matches_program_identity(self):
        self._enable_program_course_scope()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO courses(course_id, course_code, name_th, program) "
                "VALUES (3, 'C101', 'หลักสูตรอื่น', 'DSBA')"
            )
            connection.commit()

        context = {
            "result_courses": [
                {"program": "IT", "course_code": "C101"},
                {"program": "IT", "course_code": "C102"},
            ],
            "result_scope_program": "IT",
        }
        result = ask_sql(
            self.db_path,
            "ในวิชาเหล่านี้มีอะไรบ้าง",
            "IT",
            lambda _prompt: "SELECT course_id, course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context=context,
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            result["rows"],
            [
                {"course_id": 1, "course_code": "C101"},
                {"course_id": 2, "course_code": "C102"},
            ],
        )

    def test_previous_result_scope_does_not_bypass_read_only_sql_guard(self):
        self._enable_program_course_scope()
        context = {
            "result_courses": [{"program": "IT", "course_code": "C101"}],
            "result_scope_program": "IT",
        }

        result = ask_sql(
            self.db_path,
            "ในวิชาเหล่านี้มีอะไรบ้าง",
            "IT",
            lambda _prompt: "DELETE FROM courses",
            lambda _prompt: self.fail("mutating SQL must not reach the answer model"),
            conversation_context=context,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_sql")
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM courses").fetchone()[0],
                2,
            )

    def test_valid_question_returns_bounded_rows_and_thai_answer(self):
        prompts = []

        def sql_model(prompt):
            prompts.append(prompt)
            return "SELECT course_code, name_th FROM courses ORDER BY course_id"

        answer_prompts = []

        def answer_model(prompt):
            answer_prompts.append(prompt)
            return "พบวิชา C101 และ C102 ตามข้อมูลที่ค้นพบ"

        result = ask_sql(
            self.db_path,
            "มีวิชาอะไรบ้าง",
            None,
            sql_model,
            answer_model,
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["answer"], "พบวิชา C101 และ C102 ตามข้อมูลที่ค้นพบ")
        self.assertEqual(result["columns"], ["course_code", "name_th"])
        self.assertEqual(
            result["rows"],
            [
                {"course_code": "C101", "name_th": "แคลคูลัส"},
                {"course_code": "C102", "name_th": "ฟิสิกส์"},
            ],
        )
        self.assertIn("LIMIT 100", result["sql"].upper())
        self.assertEqual(len(prompts), 1)
        self.assertEqual(len(answer_prompts), 1)
        self.assertTrue(json.dumps(result, ensure_ascii=False))

    def test_selected_program_is_marked_as_active_scope(self):
        prompts = []

        def sql_model(prompt):
            prompts.append(prompt)
            return "SELECT course_code FROM courses"

        result = ask_sql(
            self.db_path,
            "แสดงรายวิชา",
            "IT",
            sql_model,
            lambda _prompt: "มีข้อมูลรายวิชา",
        )

        self.assertEqual(result["status"], "answer")
        self.assertIn("IT", prompts[0])
        self.assertRegex(prompts[0], r"(?i)active curriculum scope")
        self.assertIn("scope results to that program only", prompts[0])

    def test_distinct_course_count_prompt_uses_logical_identity_per_program(self):
        prompts = []

        result = ask_sql(
            self.db_path,
            "แต่ละหลักสูตรมีรายวิชาไม่ซ้ำกี่วิชา",
            None,
            lambda prompt: prompts.append(prompt)
            or "SELECT COUNT(DISTINCT course_code) AS unique_courses FROM courses",
            lambda _prompt: "พบจำนวนรายวิชา",
        )

        self.assertEqual(result["status"], "answer")
        guidance = prompts[0].casefold()
        self.assertIn("course_id identifies a physical course row", guidance)
        self.assertIn("course_code identifies the logical course", guidance)
        self.assertIn("count(distinct course_code)", guidance)
        self.assertIn("group independently by program", guidance)
        self.assertIn("same course_code may count once in each program", guidance)
        self.assertIn("course_id remains correct for physical-row joins", guidance)
        self.assertIn("prerequisite-edge identity", guidance)

    def test_distinct_course_count_deduplicates_plans_within_each_program(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("ALTER TABLE courses ADD COLUMN program TEXT")
            connection.execute("ALTER TABLE courses ADD COLUMN plan_key TEXT")
            connection.execute(
                "UPDATE courses SET program = 'IT', plan_key = 'coop' "
                "WHERE course_id = 1"
            )
            connection.execute(
                "UPDATE courses SET program = 'IT', plan_key = 'no_coop' "
                "WHERE course_id = 2"
            )
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?, ?, ?, ?)",
                [
                    (3, "C103", "Course C103", "IT", "coop"),
                    (4, "C103", "Course C103", "IT", "no_coop"),
                    (5, "C101", "Course C101", "DSBA", "default"),
                ],
            )
            connection.commit()

        result = ask_sql(
            self.db_path,
            "แต่ละหลักสูตรมีรายวิชาไม่ซ้ำกี่วิชา",
            None,
            lambda _prompt: (
                "SELECT program, COUNT(DISTINCT course_code) AS unique_courses "
                "FROM courses GROUP BY program ORDER BY program"
            ),
            lambda _prompt: "นับรายวิชาแยกตามหลักสูตร",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            result["rows"],
            [
                {"program": "DSBA", "unique_courses": 1},
                {"program": "IT", "unique_courses": 3},
            ],
        )

    def test_sql_prompt_requires_unique_output_columns_and_clear_entity_aliases(self):
        prompts = []

        result = ask_sql(
            self.db_path,
            "มีวิชาบังคับก่อนหรือไม่",
            None,
            lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(result["status"], "answer")
        self.assertIn("unique output column names", prompts[0].casefold())
        self.assertIn("same-named fields", prompts[0].casefold())
        self.assertIn("source_course_code", prompts[0])
        self.assertIn("source_name_th", prompts[0])
        self.assertIn("prerequisite_course_code", prompts[0])
        self.assertIn("prerequisite_name_th", prompts[0])

    def test_sql_prompt_projects_requested_factual_attributes(self):
        prompts = []

        result = ask_sql(
            self.db_path,
            "ตัวไหน 3 หน่วยกิต",
            "IT",
            lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(result["status"], "answer")
        guidance = prompts[0].casefold()
        self.assertIn("answer-relevant factual attribute", guidance)
        self.assertIn(
            "include that answer-relevant factual attribute in the select list",
            guidance,
        )
        self.assertIn("credit_units", guidance)
        self.assertIn("year", guidance)
        self.assertIn("semester", guidance)
        self.assertIn("do not require every where column", guidance)

    def test_generation_and_repair_receive_verified_semester_view_semantics(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "CREATE TABLE v_semester_credits "
                "(plan_id INTEGER, program TEXT, plan TEXT, year INTEGER, "
                "semester INTEGER, total_credits INTEGER)"
            )
            connection.execute(
                "INSERT INTO v_semester_credits VALUES (1, 'AIT', 'default', 2, 2, 16)"
            )
            connection.commit()

        sql_prompts = []

        def sql_model(prompt):
            sql_prompts.append(prompt)
            if len(sql_prompts) == 1:
                return (
                    "SELECT v.credit_units FROM v_semester_credits AS v "
                    "WHERE v.program = 'AIT' AND v.year = 2 AND v.semester = 2"
                )
            return (
                "SELECT v.total_credits FROM v_semester_credits AS v "
                "WHERE v.program = 'AIT' AND v.year = 2 AND v.semester = 2"
            )

        result = ask_sql(
            self.db_path,
            "ปี 2 เทอม 2 มีหน่วยกิตรวมเท่าไร",
            "AIT",
            sql_model,
            lambda _prompt: "ปี 2 เทอม 2 มี 16 หน่วยกิต",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"], [{"total_credits": 16}])
        self.assertEqual(len(sql_prompts), 2)
        for prompt in sql_prompts:
            guidance = prompt.casefold()
            self.assertIn("canonical view semantics for v_semester_credits", guidance)
            self.assertIn(
                "output columns: plan_id, program, plan, year, semester, total_credits",
                guidance,
            )
            self.assertIn("pre-aggregated semester-level view", guidance)
            self.assertIn("total_credits is the semester-level aggregate measure", guidance)
            self.assertIn("credit_units is not an output column", guidance)
            self.assertIn("do not aggregate it again when one row", guidance)
            self.assertIn("sum total_credits only when intentionally combining", guidance)

        self.assertIn("failed sql:", sql_prompts[1].casefold())
        self.assertIn("v.credit_units", sql_prompts[1])

    def test_sql_prompt_requires_explicit_comparison_sides_and_zero_materialization(self):
        prompts = []
        result = ask_sql(
            self.db_path,
            "เทอมไหนมีหน่วยกิตรวมมากกว่า ระหว่างเทอม 1 กับเทอม 2",
            "AIT",
            lambda prompt: prompts.append(prompt) or "SELECT 1 AS total_credits",
            lambda _prompt: "เปรียบเทียบได้",
        )

        self.assertEqual(result["status"], "answer")
        guidance = prompts[0].casefold()
        self.assertIn("every explicitly requested comparison side", guidance)
        self.assertIn("left join", guidance)
        self.assertIn("coalesce", guidance)
        self.assertIn("explicit requested-bucket cte", guidance)
        self.assertIn("equal", guidance)
        self.assertIn("do not choose a comparison winner with limit 1", guidance)
        self.assertIn("do not invent unspecified categories", guidance)
        self.assertIn("canonical placement and credit-grain semantics", guidance)
        self.assertIn("duplicate catalog copies", guidance)

    def test_explicit_semester_comparison_materializes_missing_side_as_zero(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "CREATE TABLE v_semester_credits "
                "(plan_id INTEGER, program TEXT, plan TEXT, year INTEGER, "
                "semester INTEGER, total_credits INTEGER)"
            )
            connection.execute(
                "INSERT INTO v_semester_credits VALUES (1, 'AIT', 'default', 2, 2, 16)"
            )
            connection.commit()

        answer_prompts = []
        query = (
            "WITH requested_semesters(semester) AS ("
            "SELECT 1 UNION ALL SELECT 2) "
            "SELECT requested_semesters.semester, "
            "COALESCE(v.total_credits, 0) AS total_credits "
            "FROM requested_semesters LEFT JOIN v_semester_credits AS v "
            "ON v.program = 'AIT' AND v.year = 2 "
            "AND v.semester = requested_semesters.semester "
            "ORDER BY requested_semesters.semester"
        )
        result = ask_sql(
            self.db_path,
            "ปี 2 เทอม 1 กับปี 2 เทอม 2 เทอมไหนมีหน่วยกิตรวมมากกว่า",
            "AIT",
            lambda _prompt: query,
            lambda prompt: answer_prompts.append(prompt)
            or "เทอม 2 มี 16 หน่วยกิต มากกว่าเทอม 1 ที่มี 0 หน่วยกิต",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            result["rows"],
            [
                {"semester": 1, "total_credits": 0},
                {"semester": 2, "total_credits": 16},
            ],
        )
        answer_payload = json.loads(
            answer_prompts[0].split("ข้อมูลสำหรับตอบ (JSON):\n", 1)[1]
        )
        self.assertEqual(
            answer_payload["rows"],
            [
                {"semester": 1, "total_credits": 0},
                {"semester": 2, "total_credits": 16},
            ],
        )

    def test_explicit_equal_comparison_values_remain_available_as_a_tie(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "CREATE TABLE v_semester_credits "
                "(plan_id INTEGER, program TEXT, plan TEXT, year INTEGER, "
                "semester INTEGER, total_credits INTEGER)"
            )
            connection.executemany(
                "INSERT INTO v_semester_credits VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (1, "AIT", "default", 2, 1, 16),
                    (1, "AIT", "default", 2, 2, 16),
                ],
            )
            connection.commit()

        comparison_sql = (
            "WITH requested_semesters(semester) AS (SELECT 1 UNION ALL SELECT 2) "
            "SELECT requested_semesters.semester, "
            "COALESCE(v_semester_credits.total_credits, 0) AS total_credits "
            "FROM requested_semesters LEFT JOIN v_semester_credits "
            "ON requested_semesters.semester = v_semester_credits.semester "
            "AND v_semester_credits.program = 'AIT' "
            "AND v_semester_credits.year = 2 "
            "ORDER BY requested_semesters.semester"
        )
        result = ask_sql(
            self.db_path,
            "ปี 2 เทอม 1 กับปี 2 เทอม 2 เทอมไหนมีหน่วยกิตรวมมากกว่า",
            "AIT",
            lambda _prompt: comparison_sql,
            lambda _prompt: "ทั้งสองเทอมมีหน่วยกิตเท่ากัน คือ 16",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            result["rows"],
            [
                {"semester": 1, "total_credits": 16},
                {"semester": 2, "total_credits": 16},
            ],
        )

    def test_ordinary_semester_aggregate_does_not_invent_missing_groups(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "CREATE TABLE v_semester_credits "
                "(plan_id INTEGER, program TEXT, plan TEXT, year INTEGER, "
                "semester INTEGER, total_credits INTEGER)"
            )
            connection.execute(
                "INSERT INTO v_semester_credits VALUES (1, 'AIT', 'default', 2, 2, 16)"
            )
            connection.commit()

        result = ask_sql(
            self.db_path,
            "ปี 2 แต่ละเทอมมีหน่วยกิตรวมเท่าไร",
            "AIT",
            lambda _prompt: (
                "SELECT semester, total_credits FROM v_semester_credits "
                "WHERE program = 'AIT' AND year = 2 ORDER BY semester"
            ),
            lambda _prompt: "เทอม 2 มี 16 หน่วยกิต",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"], [{"semester": 2, "total_credits": 16}])

    def test_projected_credit_evidence_reaches_answer_model_without_sql(self):
        answer_prompts = []

        result = ask_sql(
            self.db_path,
            "ตัวไหน 3 หน่วยกิต",
            "IT",
            lambda _prompt: (
                "SELECT course_code, name_th AS course_name, 3 AS credit_units "
                "FROM courses WHERE course_id = 1"
            ),
            lambda prompt: answer_prompts.append(prompt) or "C101 มี 3 หน่วยกิต",
        )

        self.assertEqual(result["status"], "answer")
        payload = json.loads(
            answer_prompts[0].split("ข้อมูลสำหรับตอบ (JSON):\n", 1)[1]
        )
        self.assertEqual(
            payload["columns"], ["course_code", "course_name", "credit_units"]
        )
        self.assertEqual(payload["rows"][0]["credit_units"], 3)
        self.assertNotIn("sql", payload)
        self.assertNotIn("WHERE", answer_prompts[0].upper())

    def test_aliased_prerequisite_style_result_is_processed(self):
        result = ask_sql(
            self.db_path,
            "มีวิชาบังคับก่อนหรือไม่",
            None,
            lambda _prompt: (
                "SELECT source.course_code AS source_course_code, "
                "source.name_th AS source_name_th, "
                "prerequisite.course_code AS prerequisite_course_code, "
                "prerequisite.name_th AS prerequisite_name_th "
                "FROM courses AS source JOIN courses AS prerequisite "
                "ON source.course_id <> prerequisite.course_id "
                "ORDER BY source.course_id, prerequisite.course_id LIMIT 1"
            ),
            lambda _prompt: "มีรายวิชาบังคับก่อน",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            result["columns"],
            [
                "source_course_code",
                "source_name_th",
                "prerequisite_course_code",
                "prerequisite_name_th",
            ],
        )
        self.assertEqual(result["rows"][0]["source_course_code"], "C101")
        self.assertEqual(result["rows"][0]["prerequisite_course_code"], "C102")

    def test_duplicate_output_columns_remain_rejected(self):
        answer_calls = []

        result = ask_sql(
            self.db_path,
            "มีวิชาบังคับก่อนหรือไม่",
            None,
            lambda _prompt: (
                "SELECT source.name_th, prerequisite.name_th "
                "FROM courses AS source JOIN courses AS prerequisite "
                "ON source.course_id <> prerequisite.course_id LIMIT 1"
            ),
            lambda prompt: answer_calls.append(prompt) or "ไม่ควรถูกเรียก",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_result")
        self.assertEqual(answer_calls, [])

    def test_single_canonical_course_result_adds_bounded_focus_context(self):
        result = ask_sql(
            self.db_path,
            "Calculus 2 รหัสอะไร",
            "DSBA",
            lambda _prompt: (
                "SELECT course_code, name_th, 'CALCULUS 2' AS name_en FROM courses "
                "WHERE course_id = 1"
            ),
            lambda _prompt: "พบ C101",
        )

        self.assertEqual(
            result["next_context"],
            {
                "program": "DSBA",
                "focus_course": {
                    "course_code": "C101",
                    "course_name": "CALCULUS 2",
                    "program": "DSBA",
                },
            },
        )

    def test_unscoped_single_course_result_keeps_existing_focus_behavior(self):
        result = ask_sql(
            self.db_path,
            "C101 คือวิชาอะไร",
            None,
            lambda _prompt: (
                "SELECT course_code, name_th AS course_name FROM courses "
                "WHERE course_id = 1"
            ),
            lambda _prompt: "พบ C101",
        )

        self.assertEqual(
            result["next_context"]["focus_course"],
            {"course_code": "C101", "course_name": "แคลคูลัส", "program": None},
        )

    def test_follow_up_sql_prompt_receives_prior_focused_course(self):
        prompts = []
        prior_context = {
            "focus_course": {
                "course_code": "C101",
                "course_name": "แคลคูลัส",
                "program": "DSBA",
            }
        }

        result = ask_sql(
            self.db_path,
            "แล้วเรียนปีไหน",
            "DSBA",
            lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses WHERE course_code = 'C101'",
            lambda _prompt: "มีข้อมูล",
            conversation_context=prior_context,
        )

        self.assertEqual(result["status"], "answer")
        self.assertIn("Conversation focus", prompts[0])
        self.assertIn("C101", prompts[0])
        self.assertIn("แคลคูลัส", prompts[0])
        self.assertIn("Explicit entities in the current question override", prompts[0])
        self.assertIn("Do not force the focused course into unrelated questions", prompts[0])

    def test_current_explicit_course_result_replaces_prior_focus(self):
        prior_context = {
            "focus_course": {
                "course_code": "C101",
                "course_name": "แคลคูลัส",
                "program": "DSBA",
            }
        }

        result = ask_sql(
            self.db_path,
            "06000002 เรียนปีไหน",
            "DSBA",
            lambda _prompt: (
                "SELECT course_code, name_th FROM courses "
                "WHERE course_id = 2"
            ),
            lambda _prompt: "พบ C102",
            conversation_context=prior_context,
        )

        self.assertEqual(
            result["next_context"]["focus_course"]["course_code"], "C102"
        )

    def test_multiple_returned_courses_do_not_create_ambiguous_focus(self):
        result = ask_sql(
            self.db_path,
            "มีวิชาอะไรบ้าง",
            "DSBA",
            lambda _prompt: "SELECT course_code, name_th FROM courses ORDER BY course_id",
            lambda _prompt: "พบหลายวิชา",
        )

        self.assertEqual(result["status"], "answer")
        self.assertNotIn("focus_course", result["next_context"])
        self.assertEqual(
            [item["course_code"] for item in result["next_context"]["result_courses"]],
            ["C101", "C102"],
        )

    def test_multiple_course_rows_create_a_bounded_result_set(self):
        result = ask_sql(
            self.db_path,
            "ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
            "IT",
            lambda _prompt: (
                "SELECT course_code, name_th AS course_name FROM courses "
                "ORDER BY course_id"
            ),
            lambda _prompt: "พบสองวิชา",
        )

        self.assertEqual(
            result["next_context"],
            {
                "program": "IT",
                "operations": ["list"],
                "years": [1],
                "semesters": [1],
                "result_courses": [
                    {"program": "IT", "course_code": "C101", "course_name": "แคลคูลัส"},
                    {"program": "IT", "course_code": "C102", "course_name": "ฟิสิกส์"},
                ],
                "result_scope_program": "IT",
            },
        )

    def test_duplicate_plan_rows_deduplicate_by_program_and_course_code(self):
        sql = (
            "SELECT 'IT' AS program, course_code, name_th AS course_name, "
            "'coop' AS plan_key FROM courses WHERE course_id = 1 "
            "UNION ALL SELECT 'IT', course_code, name_th, 'no_coop' "
            "FROM courses WHERE course_id = 1"
        )
        result = ask_sql(
            self.db_path,
            "แสดงวิชาที่พบ",
            None,
            lambda _prompt: sql,
            lambda _prompt: "พบวิชาเดียว",
        )

        self.assertEqual(result["next_context"]["focus_course"]["course_code"], "C101")
        self.assertNotIn("result_courses", result["next_context"])

    def test_same_course_code_in_distinct_programs_is_preserved(self):
        sql = (
            "SELECT 'AIT' AS program, course_code, name_th AS course_name "
            "FROM courses WHERE course_id = 1 UNION ALL "
            "SELECT 'DSBA', course_code, name_th FROM courses WHERE course_id = 1"
        )
        result = ask_sql(
            self.db_path,
            "Calculus 2 รหัสอะไร",
            None,
            lambda _prompt: sql,
            lambda _prompt: "พบสองหลักสูตร",
        )

        self.assertEqual(
            [(item["program"], item["course_code"]) for item in result["next_context"]["result_courses"]],
            [("AIT", "C101"), ("DSBA", "C101")],
        )

    def test_prerequisite_follow_up_prompt_includes_previous_result_set(self):
        prompts = []
        prior_context = {
            "result_courses": [
                {"program": "IT", "course_code": "C101", "course_name": "Calculus"},
                {"program": "IT", "course_code": "C102", "course_name": "Physics"},
            ],
            "result_scope_program": "IT",
        }
        result = ask_sql(
            self.db_path,
            "ตัวไหนมีวิชาบังคับก่อน",
            "IT",
            lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses WHERE course_code = 'C101'",
            lambda _prompt: "พบข้อมูล",
            conversation_context=prior_context,
        )

        self.assertEqual(result["status"], "answer")
        self.assertIn("Previous result-set courses", prompts[0])
        self.assertIn("[legacy catalog key omitted] | IT | C101 | Calculus", prompts[0])
        self.assertIn("[legacy catalog key omitted] | IT | C102 | Physics", prompts[0])
        self.assertIn(
            "restrict SQL to these canonical (catalog_key, program, course_code) identities",
            prompts[0],
        )
        self.assertIn("Always query canonical SQLite again", prompts[0])

    def test_credit_follow_up_prompt_has_prior_result_set_as_available_scope(self):
        prompts = []
        prior_context = {
            "result_courses": [
                {"program": "IT", "course_code": "C101"},
                {"program": "IT", "course_code": "C102"},
            ],
            "result_scope_program": "IT",
        }
        ask_sql(
            self.db_path,
            "ตัวไหน 3 หน่วยกิต",
            "IT",
            lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses WHERE course_code = 'C101'",
            lambda _prompt: "พบข้อมูล",
            conversation_context=prior_context,
        )

        self.assertIn("ตัวไหน 3 หน่วยกิต", prompts[0])
        self.assertIn("IT | C101", prompts[0])
        self.assertIn("Use this set only when the current question needs that reference", prompts[0])

    def test_explicit_current_course_overrides_prior_result_set(self):
        prompts = []
        prior_context = {
            "result_courses": [
                {"program": "IT", "course_code": "C101"},
                {"program": "IT", "course_code": "C102"},
            ],
            "result_scope_program": "IT",
        }
        result = ask_sql(
            self.db_path,
            "วิชา C102 มี prerequisite ไหม",
            "IT",
            lambda prompt: prompts.append(prompt) or (
                "SELECT course_code, name_th FROM courses WHERE course_code = 'C102'"
            ),
            lambda _prompt: "พบ C102",
            conversation_context=prior_context,
        )

        self.assertIn("Explicit course or program entities in the current question override", prompts[0])
        self.assertEqual(
            result["next_context"]["focus_course"]["course_code"], "C102"
        )
        self.assertNotIn("result_courses", result["next_context"])

    def test_unrelated_question_is_not_deterministically_forced_to_prior_set(self):
        prompts = []
        prior_context = {
            "result_courses": [{"program": "IT", "course_code": "C101"}],
            "result_scope_program": "IT",
        }
        question = "Calculus 2 รหัสอะไร"
        ask_sql(
            self.db_path,
            question,
            "IT",
            lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses WHERE course_code = 'C101'",
            lambda _prompt: "พบข้อมูล",
            conversation_context=prior_context,
        )

        self.assertIn(f"Question:\n{question}", prompts[0])
        self.assertIn("Do not force this set into an unrelated new question", prompts[0])

    def test_program_change_invalidates_stale_result_set(self):
        prompts = []
        prior_context = {
            "result_courses": [
                {"program": "IT", "course_code": "C101"},
                {"program": "IT", "course_code": "C102"},
            ],
            "result_scope_program": "IT",
        }
        result = ask_sql(
            self.db_path,
            "นับจำนวนวิชา",
            "DSBA",
            lambda prompt: prompts.append(prompt) or "SELECT COUNT(*) AS total FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context=prior_context,
        )

        self.assertNotIn("Previous result-set courses", prompts[0])
        self.assertEqual(result["next_context"], {"program": "DSBA"})

    def test_result_set_over_context_bound_is_not_partially_retained(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?, ?)",
                [(course_id, f"C{course_id:03d}", f"Course {course_id}") for course_id in range(3, 52)],
            )
            connection.commit()

        result = ask_sql(
            self.db_path,
            "แสดงวิชาทั้งหมด",
            "IT",
            lambda _prompt: "SELECT course_code, name_th AS course_name FROM courses",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(result["next_context"], {"program": "IT"})

    def test_result_set_context_contains_only_bounded_identity_metadata(self):
        result = ask_sql(
            self.db_path,
            "ปี 1 มีวิชาอะไรบ้าง",
            "IT",
            lambda _prompt: (
                "SELECT course_code, name_th AS course_name, 'IT' AS program "
                "FROM courses ORDER BY course_id"
            ),
            lambda _prompt: "พบข้อมูล",
        )

        context = result["next_context"]
        self.assertEqual(
            set(context), {"program", "years", "result_courses", "result_scope_program"}
        )
        self.assertEqual(context["years"], [1])
        self.assertEqual(len(context["result_courses"]), 2)
        self.assertTrue(
            all(set(item) <= {"program", "course_code", "course_name"} for item in context["result_courses"])
        )
        encoded = json.dumps(context, ensure_ascii=False)
        self.assertNotIn("sql", encoded.casefold())
        self.assertNotIn("rows", encoded.casefold())
        self.assertNotIn("provenance", encoded.casefold())
        self.assertNotIn("claims", encoded.casefold())

    def test_empty_result_does_not_create_focus_course(self):
        result = ask_sql(
            self.db_path,
            "รหัส 99999999 คืออะไร",
            "DSBA",
            lambda _prompt: (
                "SELECT course_code, name_th FROM courses "
                "WHERE course_code = '99999999'"
            ),
            lambda _prompt: "ไม่ควรถูกเรียก",
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["next_context"], {"program": "DSBA"})

    def test_fresh_empty_course_list_creates_bounded_known_empty_context(self):
        answer_calls = []
        result = ask_sql(
            self.db_path,
            "ปี 2 เทอม 1 มีวิชาอะไรบ้าง",
            "AIT",
            lambda _prompt: (
                "SELECT course_code, name_th FROM courses WHERE course_id = 999"
            ),
            lambda prompt: answer_calls.append(prompt) or "ไม่ควรถูกเรียก",
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(
            result["next_context"],
            {
                "program": "AIT",
                "operations": ["list"],
                "years": [2],
                "semesters": [1],
                "result_courses": [],
                "result_set_empty": True,
                "result_scope_program": "AIT",
            },
        )
        self.assertEqual(answer_calls, [])
        serialized = json.dumps(result["next_context"], ensure_ascii=False).casefold()
        for forbidden in ("sql", "rows", "provenance", "answer", "999"):
            self.assertNotIn(forbidden, serialized)

    def test_known_empty_referential_followups_return_scoped_empty_without_model_calls(self):
        empty_context = {
            "result_courses": [],
            "result_set_empty": True,
            "result_scope_program": "AIT",
        }
        for question in (
            "ในวิชาเหล่านี้รวมกี่หน่วยกิต",
            "ในวิชาเหล่านี้ตัวไหนมีวิชาบังคับก่อน",
        ):
            with self.subTest(question=question):
                sql_calls = []
                answer_calls = []
                result = ask_sql(
                    self.db_path.parent / "must-not-be-opened.db",
                    question,
                    "AIT",
                    lambda prompt: sql_calls.append(prompt) or "SELECT 1",
                    lambda prompt: answer_calls.append(prompt) or "must not be called",
                    conversation_context=empty_context,
                )

                self.assertEqual(result["status"], "no_data")
                self.assertEqual(
                    result["answer"],
                    "จากรายการก่อนหน้า ไม่พบรายการที่ตรงกับเงื่อนไขนี้",
                )
                self.assertEqual(result["next_context"]["result_courses"], [])
                self.assertTrue(result["next_context"]["result_set_empty"])
                self.assertEqual(sql_calls, [])
                self.assertEqual(answer_calls, [])

    def test_known_empty_referential_follow_ups_short_circuit_models_and_sqlite(self):
        sql_calls = []
        answer_calls = []
        empty_db_path = self.db_path.parent / "must-not-be-opened.db"
        empty_context = {
            "result_courses": [],
            "result_set_empty": True,
            "result_scope_program": "AIT",
        }
        for question in (
            "ในวิชาเหล่านี้รวมกี่หน่วยกิต",
            "ในวิชาเหล่านี้ตัวไหนมีวิชาบังคับก่อน",
        ):
            with self.subTest(question=question):
                result = ask_sql(
                    empty_db_path,
                    question,
                    "AIT",
                    lambda prompt: sql_calls.append(prompt) or "SELECT 1",
                    lambda prompt: answer_calls.append(prompt) or "ไม่ควรถูกเรียก",
                    conversation_context=empty_context,
                )

                self.assertEqual(result["status"], "no_data")
                self.assertEqual(
                    result["answer"],
                    "จากรายการก่อนหน้า ไม่พบรายการที่ตรงกับเงื่อนไขนี้",
                )
                self.assertIsNone(result["sql"])
                self.assertEqual(result["columns"], [])
                self.assertEqual(result["rows"], [])
        self.assertEqual(sql_calls, [])
        self.assertEqual(answer_calls, [])

    def test_known_empty_reference_yields_to_explicit_course_code(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO courses VALUES (?, ?, ?)",
                (3, "06046401", "Calculus 2"),
            )
            connection.commit()
        sql_calls = []
        answer_calls = []

        result = ask_sql(
            self.db_path,
            "ในวิชาเหล่านี้ แล้ว 06046401 กี่หน่วยกิต",
            "AIT",
            lambda prompt: sql_calls.append(prompt)
            or "SELECT course_code, name_th FROM courses WHERE course_code = '06046401'",
            lambda prompt: answer_calls.append(prompt) or "06046401 มีข้อมูล",
            conversation_context={
                "result_courses": [],
                "result_set_empty": True,
                "result_scope_program": "AIT",
            },
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(sql_calls), 1)
        self.assertEqual(len(answer_calls), 1)
        self.assertEqual(result["rows"][0]["course_code"], "06046401")

    def test_known_empty_reference_yields_to_explicit_year_semester_and_plan(self):
        cases = (
            "จากวิชาเหล่านี้ แต่ปี 3 เทอม 1 มีอะไรบ้าง",
            "จากรายการก่อนหน้า แต่แผน default ปี 2 เทอม 2 มีอะไรบ้าง",
            "จากวิชาเหล่านี้ แต่ IT ปี 3 เทอม 1 มีอะไรบ้าง",
        )
        for question in cases:
            with self.subTest(question=question):
                sql_calls = []
                result = ask_sql(
                    self.db_path,
                    question,
                    "AIT",
                    lambda prompt: sql_calls.append(prompt)
                    or "SELECT course_code, name_th FROM courses WHERE course_id = 1",
                    lambda _prompt: "พบข้อมูลจากขอบเขตใหม่",
                    conversation_context={
                        "result_courses": [],
                        "result_set_empty": True,
                        "result_scope_program": "AIT",
                    },
                )

                self.assertEqual(result["status"], "answer")
                self.assertEqual(len(sql_calls), 1)

    def test_known_empty_is_not_applied_without_a_prior_empty_result_set(self):
        sql_calls = []
        result = ask_sql(
            self.db_path,
            "ในวิชาเหล่านี้รวมกี่หน่วยกิต",
            "AIT",
            lambda prompt: sql_calls.append(prompt)
            or "SELECT course_code FROM courses WHERE course_id = 1",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(sql_calls), 1)

    def test_nonempty_result_set_reference_keeps_normal_sql_generation_path(self):
        self._enable_program_course_scope()
        sql_calls = []
        result = ask_sql(
            self.db_path,
            "ในวิชาเหล่านี้ตัวไหนมี prerequisite",
            "IT",
            lambda prompt: sql_calls.append(prompt)
            or "SELECT course_code FROM courses WHERE course_code = 'C101'",
            lambda _prompt: "พบ prerequisite result",
            conversation_context={
                "result_courses": [
                    {"program": "IT", "course_code": "C101"},
                    {"program": "IT", "course_code": "C102"},
                ],
                "result_scope_program": "IT",
            },
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(sql_calls), 1)
        self.assertIn("Previous result-set courses", sql_calls[0])

    def test_zero_row_filter_retains_seven_course_set_after_explicit_reference(self):
        self._enable_program_course_scope()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executemany(
                "INSERT INTO courses(course_id, course_code, name_th, program) "
                "VALUES (?, ?, ?, 'IT')",
                [
                    (course_id, f"C10{course_id}", f"course {course_id}")
                    for course_id in range(3, 8)
                ],
            )
            connection.commit()
        seven_courses = [
            {"program": "IT", "course_code": f"C10{number}"}
            for number in range(1, 8)
        ]
        result = ask_sql(
            self.db_path,
            "ในวิชาเหล่านี้ตัวไหนมี prerequisite",
            "IT",
            lambda _prompt: "SELECT course_code FROM courses WHERE 1 = 0",
            lambda _prompt: "ไม่ควรถูกเรียก",
            conversation_context={
                "result_courses": seven_courses,
                "result_scope_program": "IT",
            },
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["next_context"]["result_courses"], seven_courses)
        self.assertNotIn("result_set_empty", result["next_context"])

    def test_known_empty_context_is_invalidated_when_program_changes(self):
        prompts = []
        result = ask_sql(
            self.db_path,
            "จากรายการก่อนหน้า รวมกี่หน่วยกิต",
            "IT",
            lambda prompt: prompts.append(prompt)
            or "SELECT COUNT(*) AS total FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context={
                "result_courses": [],
                "result_set_empty": True,
                "result_scope_program": "AIT",
            },
        )

        self.assertNotIn("KNOWN EMPTY", prompts[0])
        self.assertEqual(result["next_context"], {"program": "IT"})

    def test_zero_row_filter_preserves_prior_nonempty_result_set(self):
        prior_courses = [
            {"program": "IT", "course_code": "C101"},
            {"program": "IT", "course_code": "C102"},
        ]
        result = ask_sql(
            self.db_path,
            "ตัวไหนมี prerequisite",
            "IT",
            lambda _prompt: "SELECT course_code FROM courses WHERE course_id = 999",
            lambda _prompt: "ไม่ควรถูกเรียก",
            conversation_context={
                "result_courses": prior_courses,
                "result_scope_program": "IT",
            },
        )

        self.assertEqual(result["next_context"]["result_courses"], prior_courses)
        self.assertNotIn("result_set_empty", result["next_context"])

    def test_known_empty_context_is_not_forced_on_unrelated_current_question(self):
        prompts = []
        result = ask_sql(
            self.db_path,
            "มีหลักสูตรอะไรบ้างในฐานข้อมูล",
            "AIT",
            lambda prompt: prompts.append(prompt)
            or "SELECT course_code FROM courses WHERE 1 = 0",
            lambda _prompt: "ไม่ควรถูกเรียก",
            conversation_context={
                "result_courses": [],
                "result_set_empty": True,
                "result_scope_program": "AIT",
            },
        )

        self.assertEqual(result["status"], "no_data")
        self.assertIn("Do not force this empty set into an unrelated new question", prompts[0])

    def test_known_empty_context_is_invalid_when_marker_shape_is_ambiguous(self):
        result = ask_sql(
            self.db_path,
            "นับจำนวนวิชา",
            "AIT",
            lambda _prompt: "SELECT COUNT(*) AS total FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context={
                "result_courses": [],
                "result_scope_program": "AIT",
            },
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_context")

    def test_program_change_invalidates_stale_focus(self):
        prior_context = {
            "focus_course": {
                "course_code": "C101",
                "course_name": "แคลคูลัส",
                "program": "DSBA",
            }
        }

        result = ask_sql(
            self.db_path,
            "นับจำนวนวิชา",
            "IT",
            lambda _prompt: "SELECT COUNT(*) AS total FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context=prior_context,
        )

        self.assertEqual(result["next_context"], {"program": "IT"})

    def test_existing_focus_survives_result_without_course_identity(self):
        prior_focus = {
            "course_code": "C101",
            "course_name": "แคลคูลัส",
            "program": "DSBA",
        }
        result = ask_sql(
            self.db_path,
            "แล้วล่ะ",
            "DSBA",
            lambda _prompt: "SELECT COUNT(*) AS total FROM courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context={"focus_course": prior_focus},
        )

        self.assertEqual(
            result["next_context"],
            {"program": "DSBA", "focus_course": prior_focus},
        )

    def test_next_context_never_stores_rows_sql_or_provenance(self):
        result = ask_sql(
            self.db_path,
            "แสดง C101",
            "DSBA",
            lambda _prompt: (
                "SELECT course_code, name_th FROM courses "
                "WHERE course_id = 1"
            ),
            lambda _prompt: "พบข้อมูล",
        )

        context = result["next_context"]
        self.assertEqual(set(context), {"program", "focus_course"})
        self.assertEqual(
            set(context["focus_course"]), {"course_code", "course_name", "program"}
        )
        encoded = json.dumps(context, ensure_ascii=False)
        self.assertNotIn("sql", encoded.casefold())
        self.assertNotIn("provenance", encoded.casefold())
        self.assertNotIn("name_th", encoded)
        self.assertNotIn("rows", encoded.casefold())

    def test_no_program_prompt_keeps_all_programs_eligible(self):
        prompts = []

        result = ask_sql(
            self.db_path,
            "Calculus 2 รหัสอะไร",
            None,
            lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(result["status"], "answer")
        self.assertIn("all programs remain eligible", prompts[0])

    def test_no_program_prompt_forbids_inference_from_course_title(self):
        prompts = []

        result = ask_sql(
            self.db_path,
            "Calculus 2 รหัสอะไร",
            None,
            lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(result["status"], "answer")
        self.assertIn("Never infer or choose a program from a course title", prompts[0])

    def test_unscoped_title_prompt_preserves_all_cross_program_matches(self):
        prompts = []

        result = ask_sql(
            self.db_path,
            "Calculus 2 รหัสอะไร",
            None,
            lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(result["status"], "answer")
        self.assertIn("preserve all canonical matches across programs", prompts[0])
        self.assertIn("do not use limit 1", prompts[0].casefold())
        self.assertIn("include program identity in the result columns", prompts[0].casefold())

    def test_unscoped_results_keep_distinct_program_rows(self):
        sql = (
            "SELECT 'AIT' AS program, '06046401' AS course_code "
            "UNION ALL SELECT 'DSBA', '06026201'"
        )
        result = ask_sql(
            self.db_path,
            "Calculus 2 รหัสอะไร",
            None,
            lambda _prompt: sql,
            lambda _prompt: "พบสองรายการใน AIT และ DSBA",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            result["rows"],
            [
                {"program": "AIT", "course_code": "06046401"},
                {"program": "DSBA", "course_code": "06026201"},
            ],
        )

    def test_answer_prompt_receives_program_identity_for_ambiguous_rows(self):
        prompts = []
        sql = (
            "SELECT 'AIT' AS program, '06046401' AS course_code "
            "UNION ALL SELECT 'DSBA', '06026201'"
        )

        result = ask_sql(
            self.db_path,
            "Calculus 2 รหัสอะไร",
            None,
            lambda _prompt: sql,
            lambda prompt: prompts.append(prompt) or "พบสองรายการ",
        )

        self.assertEqual(result["status"], "answer")
        answer_payload = json.loads(
            prompts[0].split("ข้อมูลสำหรับตอบ (JSON):\n", 1)[1]
        )
        self.assertIn("program", answer_payload["columns"])
        self.assertEqual(
            [row["program"] for row in answer_payload["rows"]], ["AIT", "DSBA"]
        )

    def test_sql_generation_guidance_never_infers_plan_from_program(self):
        def capture_prompt(question, program):
            prompts = []
            result = ask_sql(
                self.db_path,
                question,
                program,
                lambda prompt: prompts.append(prompt) or "SELECT course_code FROM courses",
                lambda _prompt: "พบข้อมูล",
            )
            self.assertEqual(result["status"], "answer")
            self.assertEqual(len(prompts), 1)
            return prompts[0]

        unplanned_prompt = capture_prompt("ปี 1 เทอม 1 มีวิชาอะไรบ้าง", "IT")
        explicit_plan_prompt = capture_prompt("แสดงวิชาในแผนสหกิจ", "IT")

        for prompt in (unplanned_prompt, explicit_plan_prompt):
            self.assertIn("selected program scopes the program only", prompt)
            self.assertIn("Never infer or invent a plan_key from a program", prompt)
            self.assertIn("only when the user explicitly names a plan", prompt)
            self.assertIn("application context explicitly contains a selected plan", prompt)
            self.assertIn("all applicable plans", prompt)
            self.assertIn("DISTINCT", prompt)
            self.assertIn("coop", prompt)
            self.assertIn("no_coop", prompt)
            self.assertIn("default", prompt)
            self.assertIn("gened", prompt)
        self.assertIn("แผนสหกิจ", explicit_plan_prompt)

    def test_answer_model_receives_only_question_program_columns_and_bounded_rows(self):
        prompts = []
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?, ?)",
                [
                    (course_id, f"C{course_id:03d}", "ชื่อ" + "ย" * 600)
                    for course_id in range(3, 28)
                ],
            )
            connection.commit()

        def answer_model(prompt):
            prompts.append(prompt)
            return "พบข้อมูล"

        result = ask_sql(
            self.db_path,
            "แสดงรายวิชา",
            "IT",
            lambda _prompt: "SELECT course_code, name_th FROM courses",
            answer_model,
        )

        self.assertEqual(result["status"], "answer")
        payload = json.loads(prompts[0].split("ข้อมูลสำหรับตอบ (JSON):\n", 1)[1])
        self.assertEqual(
            set(payload), {"question", "selected_program", "columns", "rows"}
        )
        self.assertEqual(payload["question"], "แสดงรายวิชา")
        self.assertEqual(payload["selected_program"], "IT")
        self.assertEqual(payload["columns"], ["course_code", "name_th"])
        self.assertEqual(len(payload["rows"]), 20)
        self.assertLessEqual(len(payload["rows"][2]["name_th"]), 501)
        self.assertNotIn("sql", payload)
        self.assertNotIn("schema", payload)
        self.assertNotIn("provenance", prompts[0])

    def test_nonempty_rows_prompt_forbids_no_data_claims(self):
        prompts = []

        result = ask_sql(
            self.db_path,
            "แสดงรายวิชา",
            "IT",
            lambda _prompt: "SELECT course_code FROM courses",
            lambda prompt: prompts.append(prompt) or "มีข้อมูล",
        )

        self.assertEqual(result["status"], "answer")
        self.assertIn("matching database rows exist", prompts[0].casefold())
        self.assertIn("MUST answer from the returned rows", prompts[0])
        self.assertIn("MUST NOT say that no data was found", prompts[0])
        self.assertIn("MUST NOT invent facts outside the returned rows", prompts[0])

    def test_nonempty_rows_reject_no_data_answer_with_grounded_fallback(self):
        answer_calls = []

        result = ask_sql(
            self.db_path,
            "แสดงรายวิชา",
            None,
            lambda _prompt: "SELECT course_code, name_th FROM courses ORDER BY course_id",
            lambda prompt: answer_calls.append(prompt) or "ไม่พบข้อมูลที่ตรงกัน",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(answer_calls), 1)
        self.assertNotIn("ไม่พบข้อมูล", result["answer"])
        self.assertIn("C101", result["answer"])
        self.assertIn("แคลคูลัส", result["answer"])
        self.assertIn("C102", result["answer"])
        self.assertIn("ฟิสิกส์", result["answer"])

    def test_mutation_sql_is_rejected(self):
        answer_calls = []
        sql_calls = []
        result = ask_sql(
            self.db_path,
            "ลบข้อมูล",
            None,
            lambda prompt: sql_calls.append(prompt) or "DELETE FROM courses",
            lambda prompt: answer_calls.append(prompt) or "ไม่ควรถูกเรียก",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_sql")
        self.assertEqual(answer_calls, [])
        self.assertEqual(len(sql_calls), 1)

    def test_unknown_relation_is_rejected_by_closed_allowlist(self):
        sql_calls = []
        result = ask_sql(
            self.db_path,
            "อ่านตารางลับ",
            None,
            lambda prompt: sql_calls.append(prompt) or "SELECT value FROM secret_table",
            lambda _prompt: "ไม่ควรถูกเรียก",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "disallowed_relation")
        self.assertEqual(len(sql_calls), 1)

    def test_sql_model_failure_is_controlled(self):
        sql_calls = []
        result = ask_sql(
            self.db_path,
            "แสดงรายวิชา",
            None,
            lambda prompt: sql_calls.append(prompt)
            or (_ for _ in ()).throw(RuntimeError("provider down")),
            lambda _prompt: "ไม่ควรถูกเรียก",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "sql_model_failure")
        self.assertEqual(len(sql_calls), 1)

    def test_sqlite_error_is_controlled(self):
        sql_calls = []
        result = ask_sql(
            self.db_path,
            "แสดงข้อมูล",
            None,
            lambda prompt: sql_calls.append(prompt)
            or "SELECT missing_column FROM courses",
            lambda _prompt: "ไม่ควรถูกเรียก",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "sqlite_error")
        self.assertEqual(len(sql_calls), 2)

    def test_current_question_course_code_is_allowed(self):
        sql_calls = []

        result = ask_sql(
            self.db_path,
            "แล้ว 06026200 กี่หน่วยกิต",
            None,
            lambda prompt: sql_calls.append(prompt)
            or "SELECT course_code FROM courses AS c WHERE c.course_code = '06026200'",
            lambda _prompt: "ไม่ควรถูกเรียกเมื่อไม่มีข้อมูล",
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(len(sql_calls), 1)

    def test_focus_course_code_is_allowed_in_follow_up(self):
        sql_calls = []

        result = ask_sql(
            self.db_path,
            "แล้วเรียนปีไหน",
            "DSBA",
            lambda prompt: sql_calls.append(prompt)
            or "SELECT course_code FROM courses AS c WHERE c.course_code = '06026201'",
            lambda _prompt: "ไม่ควรถูกเรียกเมื่อไม่มีข้อมูล",
            conversation_context={
                "focus_course": {
                    "course_code": "06026201",
                    "course_name": "CALCULUS 2",
                    "program": "DSBA",
                }
            },
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(len(sql_calls), 1)

    def test_explicit_current_course_code_takes_precedence_over_focus(self):
        sql_prompts = []
        queries = iter(
            [
                "SELECT course_code FROM courses "
                "WHERE course_code = '06026201'",
                "SELECT course_code FROM courses "
                "WHERE course_code = '06026200'",
            ]
        )

        result = ask_sql(
            self.db_path,
            "แล้ว 06026200 กี่หน่วยกิต",
            "DSBA",
            lambda prompt: sql_prompts.append(prompt) or next(queries),
            lambda _prompt: "ไม่ควรถูกเรียกเมื่อไม่มีข้อมูล",
            conversation_context={
                "focus_course": {
                    "course_code": "06026201",
                    "course_name": "CALCULUS 2",
                    "program": "DSBA",
                }
            },
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(len(sql_prompts), 2)

    def test_result_set_course_codes_are_allowed_in_in_predicate(self):
        sql_calls = []

        result = ask_sql(
            self.db_path,
            "ตัวไหนมี prerequisite",
            "IT",
            lambda prompt: sql_calls.append(prompt)
            or "SELECT course_code FROM courses AS c "
            "WHERE c.course_code IN ('06016401', '06016402')",
            lambda _prompt: "ไม่ควรถูกเรียกเมื่อไม่มีข้อมูล",
            conversation_context={
                "result_courses": [
                    {"program": "IT", "course_code": "06016401"},
                    {"program": "IT", "course_code": "06016402"},
                ],
                "result_scope_program": "IT",
            },
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(len(sql_calls), 1)

    def test_unsupported_course_code_is_repaired_to_name_query_before_execution(self):
        sql_prompts = []
        queries = iter(
            [
                "SELECT course_code, name_th FROM courses "
                "WHERE course_code_normalized = 'MATH102'",
                "SELECT course_code, name_th FROM courses "
                "WHERE name_th LIKE '%แคลคูลัส%'",
            ]
        )
        answer_prompts = []

        with patch(
            "backend.llm_sql_qa.execute_readonly",
            wraps=llm_sql_qa.execute_readonly,
        ) as execute:
            result = ask_sql(
                self.db_path,
                "Calculus 2 อยู่หลักสูตรไหนบ้าง",
                None,
                lambda prompt: sql_prompts.append(prompt) or next(queries),
                lambda prompt: answer_prompts.append(prompt) or "พบข้อมูล",
            )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(sql_prompts), 2)
        self.assertIn("unsupported course-code literal", sql_prompts[1])
        self.assertIn("MATH102", sql_prompts[1])
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(result["rows"][0]["course_code"], "C101")
        self.assertEqual(len(answer_prompts), 1)

    def test_repaired_unsupported_or_branch_fails_without_third_model_call(self):
        sql_prompts = []
        queries = iter(
            [
                "SELECT course_code FROM courses AS c "
                "WHERE c.course_code = '06026200' "
                "OR c.course_code_normalized = 'MATH102'",
                "SELECT course_code FROM courses AS c "
                "WHERE c.course_code = '06026200' "
                "OR c.course_code_normalized = 'MATH112'",
            ]
        )

        with patch("backend.llm_sql_qa.execute_readonly") as execute:
            result = ask_sql(
                self.db_path,
                "แล้ว 06026200 กี่หน่วยกิต",
                None,
                lambda prompt: sql_prompts.append(prompt) or next(queries),
                lambda _prompt: "ไม่ควรถูกเรียก",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["error"]["code"], "unsupported_course_code_literal"
        )
        self.assertEqual(len(sql_prompts), 2)
        self.assertIn("MATH112", result["sql"])
        execute.assert_not_called()

    def test_name_like_predicate_is_not_course_code_guarded(self):
        sql_calls = []
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("ALTER TABLE courses ADD COLUMN name_en TEXT")
            connection.execute(
                "UPDATE courses SET name_en = 'Calculus 2' WHERE course_id = 1"
            )
            connection.commit()

        result = ask_sql(
            self.db_path,
            "Calculus 2 อยู่หลักสูตรไหนบ้าง",
            None,
            lambda prompt: sql_calls.append(prompt)
            or "SELECT course_code FROM courses WHERE name_en LIKE '%Calculus 2%'",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"][0]["course_code"], "C101")
        self.assertEqual(len(sql_calls), 1)

    def test_program_and_plan_literals_are_outside_course_code_guard(self):
        sql_calls = []
        self._enable_program_course_scope()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("ALTER TABLE courses ADD COLUMN plan_key TEXT")
            connection.execute(
                "UPDATE courses SET plan_key = 'coop' "
                "WHERE course_id = 1"
            )
            connection.commit()

        result = ask_sql(
            self.db_path,
            "แล้ววิชานี้คืออะไร",
            "IT",
            lambda prompt: sql_calls.append(prompt)
            or "SELECT course_code FROM courses WHERE course_code = 'C101' "
            "AND program = 'IT' AND plan_key = 'coop'",
            lambda _prompt: "พบข้อมูล",
            conversation_context={
                "focus_course": {"course_code": "C101", "program": "IT"}
            },
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"][0]["course_code"], "C101")
        self.assertEqual(len(sql_calls), 1)

    def test_repaired_query_is_rechecked_against_relation_allowlist(self):
        sql_calls = []
        queries = iter(
            [
                "SELECT nonexistent_column FROM courses",
                "SELECT value FROM secret_table",
            ]
        )

        result = ask_sql(
            self.db_path,
            "แสดงข้อมูล",
            None,
            lambda prompt: sql_calls.append(prompt) or next(queries),
            lambda _prompt: "ไม่ควรถูกเรียก",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "disallowed_relation")
        self.assertEqual(len(sql_calls), 2)

    def test_invalid_column_is_repaired_once_and_repaired_query_executes(self):
        sql_prompts = []
        generated_queries = iter(
            [
                "SELECT nonexistent_column FROM courses",
                "SELECT course_code, name_th FROM courses WHERE course_id = 1",
            ]
        )
        answer_prompts = []

        result = ask_sql(
            self.db_path,
            "แสดงข้อมูลวิชา C101",
            "IT",
            lambda prompt: sql_prompts.append(prompt) or next(generated_queries),
            lambda prompt: answer_prompts.append(prompt) or "พบข้อมูล C101",
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(sql_prompts), 2)
        self.assertIn("Repair the failed SQLite query", sql_prompts[1])
        self.assertIn("แสดงข้อมูลวิชา C101", sql_prompts[1])
        self.assertIn("Selected program: IT", sql_prompts[1])
        self.assertIn("Schema:", sql_prompts[1])
        self.assertIn("Failed SQL:", sql_prompts[1])
        self.assertIn(
            'column "nonexistent_column" does not exist',
            sql_prompts[1].casefold(),
        )
        self.assertEqual(result["rows"][0]["course_code"], "C101")
        self.assertEqual(len(answer_prompts), 1)

    def test_v_plan_courses_contract_and_selected_catalog_prompt_guidance(self):
        schema = llm_sql_qa._canonical_schema()
        view_body = schema.split("CREATE VIEW v_plan_courses AS", 1)[1].split(";", 1)[0]
        self.assertNotIn("catalog_id", view_body.casefold())
        self.assertIn("v_plan_courses has no catalog_id column", schema.casefold())
        self.assertIn("never reference v_plan_courses.catalog_id", schema.casefold())
        self.assertIn("courses.catalog_id", schema.casefold())
        self.assertIn("catalogs.catalog_id", schema.casefold())

        db_path = self._build_edition_scope_db()
        prompts = []
        result = ask_sql(
            db_path,
            "แสดงรหัสวิชา",
            "DSBA",
            lambda prompt: prompts.append(prompt)
            or "SELECT course_code FROM v_plan_courses",
            lambda _prompt: "พบข้อมูล",
            conversation_context={"catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["status"], "answer")
        generation_prompt = prompts[0].casefold()
        self.assertIn(
            "execution environment already restricts canonical relations",
            generation_prompt,
        )
        self.assertIn("never query outside the scoped relation set", generation_prompt)
        self.assertNotIn(
            "selected catalog_key: dsba-2565. this is the active curriculum edition "
            "and must constrain the sql",
            generation_prompt,
        )

    def test_scoped_v_plan_courses_missing_catalog_id_is_repaired_with_identifier(self):
        db_path = self._build_edition_scope_db()
        prompts = []
        queries = iter(
            [
                "SELECT v_plan_courses_alias.catalog_id "
                "FROM v_plan_courses AS v_plan_courses_alias",
                "SELECT v_plan_courses_alias.course_code "
                "FROM v_plan_courses AS v_plan_courses_alias",
            ]
        )

        result = ask_sql(
            db_path,
            "แสดงรายวิชา DSBA",
            "DSBA",
            lambda prompt: prompts.append(prompt) or next(queries),
            lambda _prompt: "พบข้อมูล",
            conversation_context={"catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(prompts), 2)
        repair_prompt = prompts[1].casefold()
        self.assertIn(
            'column "v_plan_courses_alias.catalog_id" does not exist',
            repair_prompt,
        )
        self.assertIn("only columns exposed by the supplied relations", repair_prompt)
        self.assertTrue(result["rows"])
        self.assertEqual(result["columns"], ["course_code"])

    def test_sqlite_missing_column_detail_accepts_only_bounded_identifiers(self):
        self.assertIn(
            'column "t1.catalog_id" does not exist',
            llm_sql_qa._repairable_sqlite_error(
                sqlite3.OperationalError("no such column: t1.catalog_id")
            ),
        )
        self.assertEqual(
            llm_sql_qa._repairable_sqlite_error(
                sqlite3.OperationalError("no such column: x; DROP TABLE courses")
            ),
            "SQLite schema validation error: no such column",
        )

    def test_identical_invalid_sql_is_still_repaired_only_once(self):
        db_path = self._build_edition_scope_db()
        prompts = []
        broken_sql = (
            "SELECT v_plan_courses_alias.catalog_id "
            "FROM v_plan_courses AS v_plan_courses_alias"
        )

        result = ask_sql(
            db_path,
            "แสดงรายวิชา DSBA",
            "DSBA",
            lambda prompt: prompts.append(prompt) or broken_sql,
            lambda _prompt: "ไม่ควรถูกเรียก",
            conversation_context={"catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "sqlite_error")
        self.assertEqual(len(prompts), 2)

    def test_catalog_key_projection_uses_scoped_course_catalog_join(self):
        db_path = self._build_edition_scope_db()
        sql = (
            "SELECT catalogs.catalog_key, plan_rows.course_code "
            "FROM v_plan_courses AS plan_rows "
            "JOIN courses ON courses.course_id = plan_rows.course_id "
            "JOIN catalogs ON catalogs.catalog_id = courses.catalog_id"
        )

        result = ask_sql(
            db_path,
            "แสดงรหัสวิชาและฉบับหลักสูตร DSBA",
            "DSBA",
            lambda _prompt: sql,
            lambda _prompt: "พบข้อมูล",
            conversation_context={"catalog_key": "dsba-2565"},
        )

        self.assertEqual(result["status"], "answer")
        self.assertTrue(result["rows"])
        self.assertEqual(
            {row["catalog_key"] for row in result["rows"]}, {"dsba-2565"}
        )

    def test_answer_model_failure_is_controlled(self):
        result = ask_sql(
            self.db_path,
            "แสดงรายวิชา",
            None,
            lambda _prompt: "SELECT course_code FROM courses",
            lambda _prompt: (_ for _ in ()).throw(RuntimeError("provider down")),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "answer_model_failure")

    def test_result_limit_is_capped(self):
        result = ask_sql(
            self.db_path,
            "แสดงรายวิชา",
            None,
            lambda _prompt: "SELECT course_code FROM courses LIMIT 5000",
            lambda _prompt: "พบข้อมูล",
        )

        self.assertEqual(result["status"], "answer")
        self.assertRegex(result["sql"], r"(?i)LIMIT\s+100")
        self.assertNotRegex(result["sql"], r"(?i)LIMIT\s+5000")

    def test_empty_rows_cannot_be_overridden_by_answer_model(self):
        answer_calls = []
        sql_calls = []
        result = ask_sql(
            self.db_path,
            "มีวิชารหัส 99999999 ไหม",
            None,
            lambda prompt: sql_calls.append(prompt)
            or "SELECT course_code FROM courses WHERE course_code = '99999999'",
            lambda prompt: answer_calls.append(prompt) or "พบ X999 แน่นอน",
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["answer"], "ไม่พบข้อมูลที่ตรงกับคำถาม")
        self.assertEqual(result["rows"], [])
        self.assertEqual(answer_calls, [])
        self.assertEqual(len(sql_calls), 1)

    def test_zero_rows_with_prior_result_set_use_scoped_valid_empty_wording(self):
        answer_calls = []
        prior_context = {
            "result_courses": [
                {"program": "IT", "course_code": "C101"},
                {"program": "IT", "course_code": "C102"},
            ],
            "result_scope_program": "IT",
        }
        result = ask_sql(
            self.db_path,
            "ตัวไหนมีวิชาบังคับก่อน",
            "IT",
            lambda _prompt: (
                "SELECT course_code FROM courses "
                "WHERE course_code = 'C101' AND course_id = 999"
            ),
            lambda prompt: answer_calls.append(prompt) or "ไม่ควรถูกเรียก",
            conversation_context=prior_context,
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(
            result["answer"],
            "จากรายการก่อนหน้า ไม่พบรายการที่ตรงกับเงื่อนไขนี้",
        )
        self.assertEqual(result["rows"], [])
        self.assertEqual(answer_calls, [])

    def test_zero_rows_with_single_course_focus_keep_generic_wording(self):
        answer_calls = []
        result = ask_sql(
            self.db_path,
            "แล้วเรียนปีไหน",
            "DSBA",
            lambda _prompt: (
                "SELECT course_code FROM courses "
                "WHERE course_code = 'C101' AND course_id = 999"
            ),
            lambda prompt: answer_calls.append(prompt) or "ไม่ควรถูกเรียก",
            conversation_context={
                "focus_course": {
                    "course_code": "C101",
                    "course_name": "Calculus",
                    "program": "DSBA",
                }
            },
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["answer"], "ไม่พบข้อมูลที่ตรงกับคำถาม")
        self.assertEqual(answer_calls, [])

    def test_execution_uses_existing_read_only_sqlite_executor(self):
        import rag.structured.execute as safe_execute

        real_connect = sqlite3.connect
        seen_uris = []

        def connect_spy(database, *args, **kwargs):
            seen_uris.append(str(database))
            return real_connect(database, *args, **kwargs)

        with patch.object(safe_execute.sqlite3, "connect", side_effect=connect_spy):
            result = ask_sql(
                self.db_path,
                "แสดงรหัสวิชา",
                None,
                lambda _prompt: "SELECT course_code FROM courses",
                lambda _prompt: "มีข้อมูล",
            )

        self.assertEqual(result["status"], "answer")
        self.assertTrue(any("mode=ro" in uri for uri in seen_uris))

    def test_dsba_plan_difference_cte_executes_without_sql_repair(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "CREATE TABLE v_plan_courses "
                "(program TEXT, plan_key TEXT, course_code TEXT)"
            )
            connection.executemany(
                "INSERT INTO v_plan_courses VALUES (?, ?, ?)",
                [
                    ("DSBA", "coop", "C101"),
                    ("DSBA", "coop", "C102"),
                    ("DSBA", "no_coop", "C102"),
                ],
            )
            connection.commit()

        generated = []

        def sql_model(prompt):
            generated.append(prompt)
            return (
                "WITH coop_courses AS ("
                "SELECT course_code FROM v_plan_courses "
                "WHERE program = 'DSBA' AND plan_key = 'coop'"
                "), no_coop_courses AS ("
                "SELECT course_code FROM v_plan_courses "
                "WHERE program = 'DSBA' AND plan_key = 'no_coop'"
                ") SELECT course_code FROM coop_courses "
                "EXCEPT SELECT course_code FROM no_coop_courses"
            )

        with patch(
            "backend.llm_sql_qa.repair_sql",
            wraps=llm_sql_qa.repair_sql,
        ) as repair_spy:
            result = ask_sql(
                self.db_path,
                "แผน coop กับ no_coop ต่างกันที่วิชาไหนบ้าง",
                "DSBA",
                sql_model,
                lambda _prompt: "แผน coop มีวิชาที่ต่างจาก no_coop",
            )

        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["rows"], [{"course_code": "C101"}])
        self.assertEqual(len(generated), 1)
        repair_spy.assert_not_called()

    def test_existing_rag_is_not_called(self):
        with patch("rag.qa.ask") as rag_ask:
            result = ask_sql(
                self.db_path,
                "แสดงรหัสวิชา",
                None,
                lambda _prompt: "SELECT course_code FROM courses",
                lambda _prompt: "พบรายวิชา",
            )

        self.assertEqual(result["status"], "answer")
        rag_ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
