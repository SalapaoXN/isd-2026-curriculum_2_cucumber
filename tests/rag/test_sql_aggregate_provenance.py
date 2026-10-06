"""Focused tests for verified aggregate provenance (HSQL-6)."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend.llm_sql_qa import ask_sql
from rag.query_spec import parse_query_spec
from rag.structured.aggregate_provenance import (
    VerifiedAggregateEvidence,
    verify_aggregate_evidence,
)


V_PLAN_COURSES = """
CREATE VIEW v_plan_courses AS
SELECT
    plans.plan_id,
    plans.program_id,
    programs.program_code AS program,
    COALESCE(plans.plan_code, plans.plan_key) AS plan,
    plans.plan_code,
    plans.plan_key,
    placements.placement_id,
    placements.placement_order,
    placements.year_number AS year,
    placements.semester_number AS semester,
    placements.flexible_year_number,
    placements.flexible_semester_number,
    placements.flexible_year_semester_raw,
    COALESCE(placements.course_id, members.course_id) AS course_id,
    COALESCE(courses.course_code, member_courses.course_code) AS course,
    COALESCE(courses.course_code, member_courses.course_code) AS course_code,
    COALESCE(courses.credit_units, member_courses.credit_units) AS credits,
    COALESCE(courses.credit_units, member_courses.credit_units) AS credit_units,
    COALESCE(courses.credits_raw, member_courses.credits_raw) AS credits_raw,
    placements.alternative_group_id,
    members.member_order AS alternative_member_order,
    groups.minimum_choices,
    groups.maximum_choices,
    CASE WHEN placements.alternative_group_id IS NULL THEN 0 ELSE 1 END
        AS is_alternative,
    placements.category,
    placements.requirement_type,
    placements.credits_override,
    placements.raw_text,
    placements.notes
FROM plan_placements AS placements
JOIN curriculum_plans AS plans
    ON plans.plan_id = placements.plan_id
JOIN programs
    ON programs.program_id = plans.program_id
LEFT JOIN courses
    ON courses.course_id = placements.course_id
LEFT JOIN alternative_course_groups AS groups
    ON groups.alternative_group_id = placements.alternative_group_id
LEFT JOIN alternative_course_group_members AS members
    ON members.alternative_group_id = placements.alternative_group_id
LEFT JOIN courses AS member_courses
    ON member_courses.course_id = members.course_id;
"""

V_SEMESTER_CREDITS = """
CREATE VIEW v_semester_credits AS
WITH counted_courses AS (
    SELECT *
    FROM v_plan_courses
    WHERE alternative_group_id IS NULL
       OR (
           alternative_member_order IS NOT NULL
           AND alternative_member_order <= minimum_choices
       )
)
SELECT
    plan_id,
    program,
    plan,
    year,
    semester,
    COALESCE(SUM(credit_units), 0) AS total_credits
FROM counted_courses
GROUP BY plan_id, program, plan, year, semester;
"""


def _build_fixture(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE catalogs (
                catalog_id INTEGER PRIMARY KEY,
                catalog_key TEXT,
                academic_year TEXT
            );
            CREATE TABLE programs (
                program_id INTEGER PRIMARY KEY,
                catalog_id INTEGER,
                program_code TEXT,
                program_code_normalized TEXT
            );
            CREATE TABLE curriculum_plans (
                plan_id INTEGER PRIMARY KEY,
                catalog_id INTEGER,
                program_id INTEGER,
                program_code TEXT,
                plan_key TEXT,
                plan_code TEXT
            );
            CREATE TABLE courses (
                course_id INTEGER PRIMARY KEY,
                catalog_id INTEGER,
                course_code TEXT,
                course_code_normalized TEXT,
                name_th TEXT,
                name_en TEXT,
                credits TEXT,
                credit_units INTEGER,
                credits_raw TEXT,
                description_th TEXT,
                description_en TEXT,
                category TEXT,
                course_type TEXT,
                prerequisite_text TEXT,
                notes TEXT
            );
            CREATE TABLE plan_placements (
                placement_id INTEGER PRIMARY KEY,
                plan_id INTEGER,
                course_id INTEGER,
                alternative_group_id INTEGER,
                year_number INTEGER,
                semester_number INTEGER,
                flexible_year_number INTEGER,
                flexible_semester_number INTEGER,
                flexible_year_semester_raw TEXT,
                placement_order INTEGER,
                category TEXT,
                requirement_type TEXT,
                credits_override TEXT,
                raw_text TEXT,
                notes TEXT
            );
            CREATE TABLE alternative_course_groups (
                alternative_group_id INTEGER PRIMARY KEY,
                catalog_id INTEGER,
                plan_id INTEGER,
                group_key TEXT,
                label TEXT,
                minimum_choices INTEGER,
                maximum_choices INTEGER,
                notes TEXT
            );
            CREATE TABLE alternative_course_group_members (
                alternative_group_member_id INTEGER PRIMARY KEY,
                alternative_group_id INTEGER,
                course_id INTEGER,
                member_order INTEGER,
                notes TEXT
            );
            CREATE TABLE provenance (
                provenance_id INTEGER PRIMARY KEY,
                source_document_key TEXT NOT NULL,
                program TEXT,
                source_filename TEXT,
                source_page INTEGER,
                document_page INTEGER,
                document_category TEXT NOT NULL DEFAULT 'unknown',
                source_uri TEXT,
                source_locator TEXT,
                excerpt TEXT
            );
            CREATE TABLE course_provenance (
                course_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (course_id, provenance_id)
            );
            CREATE TABLE plan_placement_provenance (
                placement_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (placement_id, provenance_id)
            );
            CREATE TABLE alternative_group_provenance (
                alternative_group_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (alternative_group_id, provenance_id)
            );
            CREATE TABLE alternative_group_member_provenance (
                alternative_group_member_id INTEGER NOT NULL,
                provenance_id INTEGER NOT NULL,
                PRIMARY KEY (alternative_group_member_id, provenance_id)
            );
            """
        )
        connection.executescript(V_PLAN_COURSES)
        connection.executescript(V_SEMESTER_CREDITS)
        connection.executemany(
            "INSERT INTO catalogs VALUES (?, ?, ?)",
            [(1, "t-2565", "2565"), (2, "t-2560", "2560")],
        )
        connection.executemany(
            "INSERT INTO programs VALUES (?, ?, ?, ?)",
            [(1, 1, "T", "t"), (2, 2, "T", "t")],
        )
        connection.executemany(
            "INSERT INTO curriculum_plans VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 1, "T", "coop", None),
                (2, 1, 1, "T", "no_coop", None),
                (3, 2, 2, "T", "coop", None),
            ],
        )
        connection.executemany(
            "INSERT INTO courses (course_id, catalog_id, course_code,"
            " name_th, credits, credit_units, credits_raw)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (11, 1, "A101", "วิชาเอ", "3(3-0-6)", 3, "3(3-0-6)"),
                (12, 1, "B101", "วิชาบี", "3(3-0-6)", 3, "3(3-0-6)"),
                (13, 1, "C101", "วิชาซี", "4(4-0-8)", 4, "4(4-0-8)"),
                (14, 1, "D101", "วิชาดี", "5(5-0-10)", 5, "5(5-0-10)"),
                # Unknown credits: contributor without determinable value.
                (15, 1, "E101", "วิชาอี", None, None, None),
                # Known credits but no provenance links at all.
                (16, 1, "F101", "วิชาเอฟ", "3(3-0-6)", 3, "3(3-0-6)"),
                (17, 1, "G101", "วิชาจี", "2(2-0-4)", 2, "2(2-0-4)"),
                (18, 2, "H101", "วิชาเอช", "5(5-0-8)", 5, "5(5-0-8)"),
            ],
        )
        connection.executemany(
            "INSERT INTO plan_placements (placement_id, plan_id, course_id,"
            " alternative_group_id, year_number, semester_number,"
            " placement_order) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                # coop Y1S1: 3 + 3 + min1(C=4, D=5) = 10.
                (101, 1, 11, None, 1, 1, 1),
                (102, 1, 12, None, 1, 1, 2),
                (103, 1, None, 201, 1, 1, 3),
                # coop Y1S2: unknown-credit course.
                (104, 1, 15, None, 1, 2, 1),
                # coop Y2S1: known credits but zero provenance.
                (105, 1, 16, None, 2, 1, 1),
                # no_coop Y1S1: 2.
                (106, 2, 17, None, 1, 1, 1),
                # t-2560 coop Y1S1: 5.
                (107, 3, 18, None, 1, 1, 1),
            ],
        )
        connection.execute(
            "INSERT INTO alternative_course_groups VALUES"
            " (201, 1, 1, 'g1', 'group', 1, 2, NULL)"
        )
        connection.executemany(
            "INSERT INTO alternative_course_group_members VALUES"
            " (?, ?, ?, ?, NULL)",
            [(301, 201, 13, 1), (302, 201, 14, 2)],
        )
        connection.executemany(
            "INSERT INTO provenance (provenance_id, source_document_key,"
            " program, source_filename, source_page)"
            " VALUES (?, ?, ?, ?, ?)",
            [(11, "d", "T", "p11.png", 11),
             (12, "d", "T", "p12.png", 12),
             (13, "d", "T", "p13.png", 13),
             (14, "d", "T", "p14.png", 14),
             (15, "d", "T", "p15.png", 15),
             (16, "d", "T", "p16.png", 16),
             (17, "d", "T", "p17.png", 17),
             (18, "d", "T", "p18.png", 18),
             (19, "d", "T", "p19.png", 19),
             (20, "d", "T", "p20.png", 20),
             (21, "d", "T", "p21.png", 21),
             (22, "d", "T", "p22.png", 22)],
        )
        connection.executemany(
            "INSERT INTO course_provenance VALUES (?, ?)",
            [(11, 11), (12, 12), (13, 13), (14, 14), (17, 21), (18, 22)],
        )
        connection.executemany(
            "INSERT INTO plan_placement_provenance VALUES (?, ?)",
            [(101, 18), (102, 19), (103, 20)],
        )
        connection.execute(
            "INSERT INTO alternative_group_provenance VALUES (201, 15)"
        )
        connection.executemany(
            "INSERT INTO alternative_group_member_provenance VALUES (?, ?)",
            [(301, 16), (302, 17)],
        )
        connection.commit()


def _credit_spec():
    return parse_query_spec("ปี 1 เทอม 1 รวมกี่หน่วยกิต")


class SqlAggregateProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = str(Path(cls.temp_dir.name) / "curriculum.db")
        _build_fixture(Path(cls.db_path))

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_valid_semester_total(self):
        result = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 10}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan="coop",
            years=(1,),
            semesters=(1,),
            query_spec=_credit_spec(),
        )
        self.assertIsInstance(result, VerifiedAggregateEvidence)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.aggregate_kind, "semester_credit_total")
        self.assertEqual(result.value, 10)
        self.assertEqual(result.contributing_rows, 3)
        self.assertEqual(
            sorted(
                reference["provenance_id"] for reference in result.provenance
            ),
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        )

    def test_wrong_sql_aggregate_value_fails_closed(self):
        result = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 21}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan="coop",
            years=(1,),
            semesters=(1,),
            query_spec=_credit_spec(),
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.provenance, ())

    def test_missing_plan_fails_closed(self):
        result = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 10}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan=None,
            years=(1,),
            semesters=(1,),
            query_spec=_credit_spec(),
        )
        self.assertEqual(result.status, "insufficient_evidence")

    def test_multiple_years_or_semesters_unsupported(self):
        for years, semesters in (((1, 2), (1,)), ((1,), (1, 2))):
            with self.subTest(years=years, semesters=semesters):
                result = verify_aggregate_evidence(
                    self.db_path,
                    rows=[{"total_credits": 10}],
                    columns=["total_credits"],
                    program="T",
                    catalog_key="t-2565",
                    plan="coop",
                    years=years,
                    semesters=semesters,
                    query_spec=_credit_spec(),
                )
                self.assertEqual(result.status, "insufficient_evidence")

    def test_empty_result_not_complete(self):
        result = verify_aggregate_evidence(
            self.db_path,
            rows=[],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan="coop",
            years=(1,),
            semesters=(1,),
            query_spec=_credit_spec(),
        )
        self.assertNotEqual(result.status, "complete")

    def test_multiple_aggregate_rows_not_supported(self):
        result = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 10}, {"total_credits": 10}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan="coop",
            years=(1,),
            semesters=(1,),
            query_spec=_credit_spec(),
        )
        self.assertNotEqual(result.status, "complete")

    def test_non_numeric_total_rejected(self):
        for bad in ("10", True, None):
            with self.subTest(bad=bad):
                result = verify_aggregate_evidence(
                    self.db_path,
                    rows=[{"total_credits": bad}],
                    columns=["total_credits"],
                    program="T",
                    catalog_key="t-2565",
                    plan="coop",
                    years=(1,),
                    semesters=(1,),
                    query_spec=_credit_spec(),
                )
                self.assertNotEqual(result.status, "complete")

    def test_contributor_without_provenance_fails_closed(self):
        result = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 3}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan="coop",
            years=(2,),
            semesters=(1,),
            query_spec=_credit_spec(),
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.provenance, ())

    def test_unknown_credit_contributor_fails_closed(self):
        result = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 0}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan="coop",
            years=(1,),
            semesters=(2,),
            query_spec=_credit_spec(),
        )
        self.assertEqual(result.status, "insufficient_evidence")

    def test_plan_isolation(self):
        coop = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 10}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan="coop",
            years=(1,),
            semesters=(1,),
            query_spec=_credit_spec(),
        )
        no_coop = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 2}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan="no_coop",
            years=(1,),
            semesters=(1,),
            query_spec=_credit_spec(),
        )
        self.assertEqual((coop.status, coop.value), ("complete", 10))
        self.assertEqual((no_coop.status, no_coop.value), ("complete", 2))
        coop_ids = {
            reference["provenance_id"] for reference in coop.provenance
        }
        no_coop_ids = {
            reference["provenance_id"] for reference in no_coop.provenance
        }
        self.assertTrue(coop_ids.isdisjoint(no_coop_ids))

    def test_catalog_isolation(self):
        other = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 5}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2560",
            plan="coop",
            years=(1,),
            semesters=(1,),
            query_spec=_credit_spec(),
        )
        self.assertEqual((other.status, other.value), ("complete", 5))
        self.assertEqual(
            [reference["provenance_id"] for reference in other.provenance],
            [22],
        )

    def test_non_credit_intent_refused(self):
        result = verify_aggregate_evidence(
            self.db_path,
            rows=[{"total_credits": 10}],
            columns=["total_credits"],
            program="T",
            catalog_key="t-2565",
            plan="coop",
            years=(1,),
            semesters=(1,),
            query_spec=parse_query_spec("ปี 1 เทอม 1 มีวิชาอะไรบ้าง"),
        )
        self.assertNotEqual(result.status, "complete")

    def test_aggregate_rescue_end_to_end(self):
        sql_calls = []
        answer_calls = []

        def sql_model(prompt):
            sql_calls.append(prompt)
            return (
                "SELECT total_credits FROM v_semester_credits"
                " WHERE program = 'T' AND plan = 'coop'"
                " AND year = 1 AND semester = 1"
            )

        def answer_model(prompt):
            answer_calls.append(prompt)
            return "row grounded summary"

        result = ask_sql(
            self.db_path,
            "ปี 1 เทอม 1 รวมกี่หน่วยกิต",
            "T",
            sql_model,
            answer_model,
            conversation_context={"catalog_key": "t-2565", "plan": "coop"},
            grounding_callable=lambda _q: {
                "status": "insufficient_evidence",
                "final_answer": "",
                "provenance": [],
            },
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            sorted(
                reference["provenance_id"] for reference in result["provenance"]
            ),
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        )
        self.assertEqual(len(sql_calls), 1)

    def test_aggregate_mismatch_fails_closed_without_answer_model(self):
        answer_calls = []

        def answer_model(prompt):
            answer_calls.append(prompt)
            return "row grounded summary"

        result = ask_sql(
            self.db_path,
            "ปี 1 เทอม 1 รวมกี่หน่วยกิต",
            "T",
            lambda _prompt: (
                "SELECT 21 AS total_credits FROM v_semester_credits"
                " WHERE program = 'T' AND plan = 'coop'"
                " AND year = 1 AND semester = 1 LIMIT 1"
            ),
            answer_model,
            conversation_context={"catalog_key": "t-2565", "plan": "coop"},
            grounding_callable=lambda _q: {
                "status": "insufficient_evidence",
                "final_answer": "",
                "provenance": [],
            },
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])
        self.assertEqual(answer_calls, [])

    def test_scopeless_aggregate_denied_before_verification(self):
        with patch(
            "backend.llm_sql_qa.verify_aggregate_evidence",
            side_effect=AssertionError("gate must deny first"),
        ):
            result = ask_sql(
                self.db_path,
                "ปี 1 เทอม 1 รวมกี่หน่วยกิต",
                None,
                lambda _prompt: "SELECT total_credits FROM v_semester_credits",
                lambda _prompt: "row grounded summary",
                grounding_callable=lambda _q: {
                    "status": "insufficient_evidence",
                    "final_answer": "",
                    "provenance": [],
                },
                allow_grounded_row_rescue=True,
            )
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_row_hydration_wins_over_aggregate_verifier(self):
        with patch(
            "backend.llm_sql_qa.verify_aggregate_evidence",
            side_effect=AssertionError("hydration precedence violated"),
        ):
            result = ask_sql(
                self.db_path,
                "ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
                "T",
                lambda _prompt: (
                    "SELECT course_id, course_code FROM courses"
                    " WHERE course_id = 11"
                ),
                lambda _prompt: "row grounded summary",
                conversation_context={"catalog_key": "t-2565"},
                grounding_callable=lambda _q: {
                    "status": "insufficient_evidence",
                    "final_answer": "",
                    "provenance": [],
                },
                allow_grounded_row_rescue=True,
            )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [11],
        )


if __name__ == "__main__":
    unittest.main()
