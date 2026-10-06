"""Focused tests for opt-in grounded SQL-row rescue (HSQL-2)."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend.llm_sql_qa import ask_sql


def _failed_grounding(_question):
    return {"status": "insufficient_evidence", "final_answer": "", "provenance": []}


def _grounded_answer(_question):
    return {
        "status": "answer",
        "final_answer": "grounded deterministic answer",
        "provenance": [
            {"source_filename": "canon.png", "source_page": 1},
        ],
        "next_context": {"program": "IT"},
    }


class LlmSqlRescueTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "curriculum.db"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE courses (
                    course_id INTEGER,
                    course_code TEXT,
                    name_th TEXT,
                    placement_id INTEGER,
                    prerequisite_id INTEGER,
                    requirement_id INTEGER,
                    fact_id INTEGER,
                    total_credits INTEGER
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
                """
            )
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (101, "C101", "แคลคูลัส", 201, None, None, None, None),
                    (102, "C102", "ฟิสิกส์", None, None, None, None, None),
                    (103, "C103", "เคมี", None, None, None, None, 3),
                ],
            )
            connection.executemany(
                "INSERT INTO provenance (provenance_id, source_document_key,"
                " program, source_filename, source_page)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (1, "doc-a", "IT", "it_page_001.png", 1),
                    (2, "doc-a", "IT", "it_page_002.png", 2),
                ],
            )
            connection.executemany(
                "INSERT INTO course_provenance (course_id, provenance_id)"
                " VALUES (?, ?)",
                [(101, 1), (101, 2), (102, 2)],
            )
            connection.execute(
                "INSERT INTO plan_placement_provenance (placement_id, provenance_id)"
                " VALUES (201, 2)"
            )
            connection.commit()
        self.sql_calls = []
        self.answer_calls = []

    def tearDown(self):
        self.temp_dir.cleanup()

    def _ask(self, sql, **kwargs):
        def sql_model(prompt):
            self.sql_calls.append(prompt)
            return sql

        def answer_model(prompt):
            self.answer_calls.append(prompt)
            return kwargs.pop("answer_text", "row grounded summary")

        grounding = kwargs.pop("grounding_callable", _failed_grounding)
        return ask_sql(
            self.db_path,
            kwargs.pop("question", "IT มีวิชาอะไรบ้าง"),
            "IT",
            sql_model,
            answer_model,
            grounding_callable=grounding,
            **kwargs,
        )

    def test_default_off_preserves_insufficient_evidence(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("hydrator must not run when rescue is off"),
        ):
            result = self._ask("SELECT course_id, course_code FROM courses")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])
        self.assertEqual(result["answer"], "ไม่พบหลักฐานที่มีแหล่งอ้างอิงเพียงพอสำหรับคำตอบนี้")

    def test_rescue_flag_defaults_to_off(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("hydrator must not run by default"),
        ):
            result = self._ask(
                "SELECT course_id, course_code FROM courses WHERE course_id = 101"
            )
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_opt_in_complete_course_row_rescue(self):
        result = self._ask(
            "SELECT course_id, course_code FROM courses WHERE course_id = 101",
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["answer"], "row grounded summary")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [1, 2],
        )
        self.assertEqual(result["provenance"][0]["source_filename"], "it_page_001.png")
        self.assertEqual(len(self.sql_calls), 1)

    def test_deterministic_grounder_still_wins(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("hydrator must not run after grounded answer"),
        ):
            answer_calls_before = len(self.answer_calls)
            result = self._ask(
                "SELECT course_id, course_code FROM courses WHERE course_id = 101",
                allow_grounded_row_rescue=True,
                grounding_callable=_grounded_answer,
            )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["answer"], "grounded deterministic answer")
        self.assertEqual(
            result["provenance"],
            [{"source_filename": "canon.png", "source_page": 1}],
        )
        self.assertEqual(len(self.answer_calls), answer_calls_before)

    def test_partial_row_coverage_never_rescues(self):
        answer_calls_before = len(self.answer_calls)
        result = self._ask(
            "SELECT course_id, course_code FROM courses",
            allow_grounded_row_rescue=True,
            question="IT aggregate",
        )
        # Row for C103 carries a dangling course_id with no provenance.
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])
        self.assertEqual(len(self.answer_calls), answer_calls_before)

    def test_dangling_canonical_id_never_rescues(self):
        result = self._ask(
            "SELECT course_id, course_code FROM courses WHERE course_id = 999",
            allow_grounded_row_rescue=True,
        )
        # Empty result set keeps existing grounded-product failure behavior.
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_dangling_id_in_nonempty_rows_never_rescues(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO courses VALUES (777, 'C777', 'ลอย', NULL, NULL, NULL, NULL, NULL)"
            )
            connection.commit()
        result = self._ask(
            "SELECT course_id, course_code FROM courses WHERE course_id = 777",
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])

    def test_empty_rows_never_rescue(self):
        result = self._ask(
            "SELECT course_id FROM courses WHERE 1 = 0",
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_aggregate_without_entity_id_never_rescues(self):
        result = self._ask(
            "SELECT total_credits FROM courses WHERE course_id = 103",
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])

    def test_hydrator_exception_fails_closed(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=sqlite3.Error("boom"),
        ):
            result = self._ask(
                "SELECT course_id, course_code FROM courses WHERE course_id = 101",
                allow_grounded_row_rescue=True,
            )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("error", result)
        self.assertEqual(len(self.answer_calls), 0)

    def test_answer_model_failure_fails_closed(self):
        def failing_answer(_prompt):
            raise RuntimeError("provider down")

        result = ask_sql(
            self.db_path,
            "IT มีวิชาอะไรบ้าง",
            "IT",
            lambda _prompt: "SELECT course_id, course_code FROM courses"
            " WHERE course_id = 101",
            failing_answer,
            grounding_callable=_failed_grounding,
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])

    def test_answer_model_blank_fails_closed(self):
        result = ask_sql(
            self.db_path,
            "IT มีวิชาอะไรบ้าง",
            "IT",
            lambda _prompt: "SELECT course_id, course_code FROM courses"
            " WHERE course_id = 101",
            lambda _prompt: "   ",
            grounding_callable=_failed_grounding,
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_answer_model_false_no_data_uses_deterministic_fallback(self):
        result = ask_sql(
            self.db_path,
            "IT มีวิชาอะไรบ้าง",
            "IT",
            lambda _prompt: "SELECT course_id, course_code FROM courses"
            " WHERE course_id = 101",
            lambda _prompt: "ไม่พบข้อมูล",
            grounding_callable=_failed_grounding,
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "answer")
        self.assertIn("C101", result["answer"])
        self.assertNotIn("ไม่พบข้อมูล", result["answer"])
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [1, 2],
        )

    def test_fake_row_provenance_ignored_in_rescue(self):
        result = ask_sql(
            self.db_path,
            "IT มีวิชาอะไรบ้าง",
            "IT",
            lambda _prompt: "SELECT course_id, course_code FROM courses"
            " WHERE course_id = 101",
            lambda _prompt: "row grounded summary",
            grounding_callable=_failed_grounding,
            allow_grounded_row_rescue=True,
        )
        pages = [reference.get("source_page") for reference in result["provenance"]]
        self.assertEqual(result["status"], "answer")
        self.assertNotIn(999, pages)

    def test_rescue_context_follows_existing_row_context_behavior(self):
        result = self._ask(
            "SELECT course_id, course_code FROM courses WHERE course_id = 101",
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "answer")
        next_context = result["next_context"]
        self.assertIsNotNone(next_context)
        self.assertEqual(next_context.get("program"), "IT")
        self.assertEqual(
            (next_context.get("focus_course") or {}).get("course_code"), "C101"
        )

    def test_non_bool_rescue_flag_rejected(self):
        result = self._ask(
            "SELECT course_id, course_code FROM courses WHERE course_id = 101",
            allow_grounded_row_rescue="yes",
        )
        self.assertEqual(result["status"], "error")

    def test_rescue_reuses_executed_rows_without_new_sql(self):
        self._ask(
            "SELECT course_id, course_code FROM courses WHERE course_id = 101",
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(len(self.sql_calls), 1)


RESCUE_GUIDANCE_MARKER = "Rescue evidence-ID projection"


def _prompt_capturing_ask(test_case, sql, **kwargs):
    prompts = []

    def sql_model(prompt):
        prompts.append(prompt)
        test_case.sql_calls.append(prompt)
        return sql

    def answer_model(prompt):
        test_case.answer_calls.append(prompt)
        return "row grounded summary"

    result = ask_sql(
        test_case.db_path,
        "IT มีวิชาอะไรบ้าง",
        "IT",
        sql_model,
        answer_model,
        grounding_callable=_failed_grounding,
        **kwargs,
    )
    return result, prompts


class LlmSqlRescuePromptTest(unittest.TestCase):
    def setUp(self):
        self.inner = LlmSqlRescueTest("test_opt_in_complete_course_row_rescue")
        self.inner.setUp()
        self.db_path = self.inner.db_path
        self.sql_calls = self.inner.sql_calls
        self.answer_calls = self.inner.answer_calls

    def tearDown(self):
        self.inner.tearDown()

    def test_default_off_prompt_has_no_rescue_guidance(self):
        _, prompts = _prompt_capturing_ask(
            self, "SELECT course_id, course_code FROM courses"
        )
        self.assertEqual(len(prompts), 1)
        self.assertNotIn(RESCUE_GUIDANCE_MARKER, prompts[0])

    def test_rescue_on_prompt_requires_course_id(self):
        _, prompts = _prompt_capturing_ask(
            self,
            "SELECT course_id, course_code FROM courses",
            allow_grounded_row_rescue=True,
        )
        self.assertIn(RESCUE_GUIDANCE_MARKER, prompts[0])
        self.assertIn("course_id", prompts[0])

    def test_rescue_on_prompt_requires_placement_id(self):
        _, prompts = _prompt_capturing_ask(
            self,
            "SELECT course_id, course_code FROM courses",
            allow_grounded_row_rescue=True,
        )
        self.assertIn("placement_id", prompts[0])

    def test_rescue_on_prompt_requires_prerequisite_id(self):
        _, prompts = _prompt_capturing_ask(
            self,
            "SELECT course_id, course_code FROM courses",
            allow_grounded_row_rescue=True,
        )
        self.assertIn("prerequisite_id", prompts[0])

    def test_rescue_on_prompt_requires_requirement_and_fact_ids(self):
        _, prompts = _prompt_capturing_ask(
            self,
            "SELECT course_id, course_code FROM courses",
            allow_grounded_row_rescue=True,
        )
        self.assertIn("requirement_id", prompts[0])
        self.assertIn("fact_id", prompts[0])

    def test_rescue_prompt_forbids_provenance_table_citation(self):
        _, prompts = _prompt_capturing_ask(
            self,
            "SELECT course_id, course_code FROM courses",
            allow_grounded_row_rescue=True,
        )
        self.assertIn("Do not query provenance", prompts[0])

    def test_rescue_prompt_protects_result_grain(self):
        _, prompts = _prompt_capturing_ask(
            self,
            "SELECT course_id, course_code FROM courses",
            allow_grounded_row_rescue=True,
        )
        self.assertIn("DISTINCT", prompts[0])
        self.assertIn("grain", prompts[0])

    def test_rescue_prompt_defers_id_less_aggregates(self):
        _, prompts = _prompt_capturing_ask(
            self,
            "SELECT course_id, course_code FROM courses",
            allow_grounded_row_rescue=True,
        )
        self.assertIn("aggregate", prompts[0])

    def test_guided_course_sql_rescues_end_to_end(self):
        def sql_model(prompt):
            self.sql_calls.append(prompt)
            assert RESCUE_GUIDANCE_MARKER in prompt
            return (
                "SELECT course_id, course_code, name_th FROM courses"
                " WHERE course_id = 101"
            )

        def answer_model(prompt):
            self.answer_calls.append(prompt)
            return "row grounded summary"

        result = ask_sql(
            self.db_path,
            "IT มีวิชาอะไรบ้าง",
            "IT",
            sql_model,
            answer_model,
            grounding_callable=_failed_grounding,
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [1, 2],
        )

    def test_guided_no_id_sql_still_fails_closed(self):
        result, _ = _prompt_capturing_ask(
            self,
            "SELECT course_code, name_th FROM courses WHERE course_id = 101",
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])

    def test_projection_adds_no_extra_sql_call(self):
        _prompt_capturing_ask(
            self,
            "SELECT course_id, course_code FROM courses WHERE course_id = 101",
            allow_grounded_row_rescue=True,
        )
        self.assertEqual(len(self.sql_calls), 1)


class LlmSqlRescueAdmissionTest(unittest.TestCase):
    """HSQL-5A: deterministic admission gate before provenance hydration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "curriculum.db"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE courses (
                    course_id INTEGER,
                    course_code TEXT,
                    name_th TEXT,
                    catalog_id INTEGER,
                    placement_id INTEGER,
                    prerequisite_id INTEGER,
                    requirement_id INTEGER,
                    fact_id INTEGER
                );
                CREATE TABLE catalogs (
                    catalog_id INTEGER PRIMARY KEY,
                    catalog_key TEXT
                );
                CREATE VIEW v_plan_courses AS
                    SELECT course_id, course_id AS plan_id, course_id AS program_id,
                        'IT' AS program, 'coop' AS plan, course_code
                    FROM courses;
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
                CREATE TABLE prerequisite_provenance (
                    prerequisite_id INTEGER NOT NULL,
                    provenance_id INTEGER NOT NULL,
                    PRIMARY KEY (prerequisite_id, provenance_id)
                );
                CREATE TABLE program_requirements (
                    requirement_id INTEGER PRIMARY KEY
                );
                CREATE TABLE program_requirement_provenance (
                    requirement_id INTEGER NOT NULL,
                    provenance_id INTEGER NOT NULL,
                    PRIMARY KEY (requirement_id, provenance_id)
                );
                CREATE TABLE policy_facts (
                    fact_id INTEGER PRIMARY KEY,
                    fact_key TEXT,
                    value NUMERIC,
                    unit TEXT
                );
                CREATE TABLE policy_fact_provenance (
                    fact_id INTEGER NOT NULL,
                    provenance_id INTEGER NOT NULL,
                    PRIMARY KEY (fact_id, provenance_id)
                );
                """
            )
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (101, "C101", "แคลคูลัส", 1, 201, None, None, None),
                    (102, "C102", "ฟิสิกส์", 1, None, 301, None, None),
                    (103, "C103", "เคมี", 1, None, None, None, 501),
                    (104, "C104", "ชีวะ", 1, None, None, 401, None),
                    (105, "06016414", "ฐานข้อมูล", 1, None, None, None, None),
                ],
            )
            connection.execute("INSERT INTO catalogs VALUES (1, 'it-2565')");
            connection.executemany(
                "INSERT INTO provenance (provenance_id, source_document_key,"
                " program, source_filename, source_page)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (1, "doc-a", "IT", "it_page_001.png", 1),
                    (2, "doc-a", "IT", "it_page_002.png", 2),
                    (3, "doc-b", "DSBA", "dsba_page_010.png", 10),
                ],
            )
            connection.executemany(
                "INSERT INTO course_provenance (course_id, provenance_id)"
                " VALUES (?, ?)",
                [(101, 1), (101, 2), (102, 2), (105, 1)],
            )
            connection.execute(
                "INSERT INTO plan_placement_provenance (placement_id, provenance_id)"
                " VALUES (201, 2)"
            )
            connection.execute(
                "INSERT INTO prerequisite_provenance (prerequisite_id, provenance_id)"
                " VALUES (301, 3)"
            )
            connection.execute("INSERT INTO program_requirements VALUES (401)")
            connection.execute(
                "INSERT INTO program_requirement_provenance"
                " (requirement_id, provenance_id) VALUES (401, 1)"
            )
            connection.execute(
                "INSERT INTO policy_facts VALUES (501, 'honors', 3.5, 'GPA')"
            )
            connection.execute(
                "INSERT INTO policy_fact_provenance (fact_id, provenance_id)"
                " VALUES (501, 3)"
            )
            connection.commit()
        self.sql_calls = []
        self.answer_calls = []

    def tearDown(self):
        self.temp_dir.cleanup()

    def _ask(self, sql, question, program, context=None, answer_text="row grounded summary"):
        def sql_model(prompt):
            self.sql_calls.append(prompt)
            return sql

        def answer_model(prompt):
            self.answer_calls.append(prompt)
            return answer_text

        return ask_sql(
            self.db_path,
            question,
            program,
            sql_model,
            answer_model,
            conversation_context=context,
            grounding_callable=_failed_grounding,
            allow_grounded_row_rescue=True,
        )

    def _assert_denied(self, result):
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])
        self.assertEqual(len(self.answer_calls), 0)

    def test_scopeless_targetless_judgement_denied_without_hydration(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("gate must run before hydration"),
        ):
            result = self._ask(
                "SELECT course_id, course_code FROM courses",
                "วิชานี้ดีไหม",
                None,
            )
        self._assert_denied(result)

    def test_unbounded_dump_request_denied_without_hydration(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("gate must run before hydration"),
        ):
            result = self._ask(
                "SELECT course_id, course_code FROM courses",
                "ขอทุกอย่างในฐานข้อมูล",
                "IT",
            )
        self._assert_denied(result)

    def test_global_policy_fact_rescue_survives_without_program(self):
        result = self._ask(
            "SELECT fact_id, fact_key, value, unit FROM policy_facts",
            "เกรดเท่าไหร่จะได้เกียรตินิยม",
            None,
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [3],
        )
        self.assertEqual(len(self.sql_calls), 1)

    def test_mixed_fact_and_curriculum_row_denied_without_scope(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("ambiguous rows must not widen scope"),
        ):
            result = self._ask(
                "SELECT course_id, fact_id, course_code FROM courses"
                " WHERE course_id = 103",
                "เกรดเท่าไหร่จะได้เกียรตินิยม",
                None,
            )
        self._assert_denied(result)

    def test_mixed_fact_and_curriculum_row_admitted_with_scope(self):
        result = self._ask(
            "SELECT course_id, fact_id, course_code FROM courses"
            " WHERE course_id = 103",
            "IT มีวิชาอะไรบ้าง",
            "IT",
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [3],
        )

    def test_requirement_id_without_scope_denied(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("requirement needs curriculum scope"),
        ):
            result = self._ask(
                "SELECT requirement_id FROM courses WHERE requirement_id = 401",
                "มีวิชาอะไรบ้าง",
                None,
            )
        self._assert_denied(result)

    def test_requirement_id_with_scope_admitted(self):
        result = self._ask(
            "SELECT requirement_id FROM courses WHERE requirement_id = 401",
            "IT มีวิชาอะไรบ้าง",
            "IT",
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [1],
        )

    def test_scoped_exact_course_rescue_still_works(self):
        result = self._ask(
            "SELECT course_id, course_code FROM courses"
            " WHERE course_code = '06016414'",
            "06016414 กี่หน่วย",
            "IT",
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [1],
        )

    def test_scoped_placement_rescue_still_works(self):
        result = self._ask(
            "SELECT placement_id, course_code FROM courses"
            " WHERE placement_id = 201",
            "IT มีวิชาอะไรบ้าง",
            "IT",
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [2],
        )

    def test_scoped_prerequisite_rescue_still_works(self):
        result = self._ask(
            "SELECT prerequisite_id, course_code FROM courses"
            " WHERE prerequisite_id = 301",
            "IT มีวิชาอะไรบ้าง",
            "IT",
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [3],
        )

    def test_program_only_without_substance_denied(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("program alone must not authorize rescue"),
        ):
            result = self._ask(
                "SELECT course_id, course_code FROM courses",
                "สวัสดีครับ",
                "IT",
            )
        self._assert_denied(result)

    def test_prior_focus_course_keeps_bounded_followup_eligible(self):
        result = self._ask(
            "SELECT course_id, course_code FROM courses WHERE course_id = 101",
            "แล้วตัวนี้กี่หน่วย",
            "IT",
            context={"focus_course": {"course_code": "C101", "program": "IT"}},
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [1, 2],
        )

    def test_prior_result_set_ordinal_stays_eligible(self):
        result = self._ask(
            "SELECT course_id, course_code FROM courses WHERE course_id = 101",
            "ตัวแรกกี่หน่วย",
            "IT",
            context={
                "result_courses": [{"program": "IT", "course_code": "C101"}],
                "result_scope_program": "IT",
            },
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [1, 2],
        )

    def test_scoped_semantic_topic_stays_eligible(self):
        result = self._ask(
            "SELECT course_id, course_code FROM courses WHERE course_id = 101",
            "มีวิชาเกี่ยวกับ cyber อะไรบ้าง",
            "IT",
            context={"program": "IT", "semantic_topic": "cyber"},
        )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(
            [reference["provenance_id"] for reference in result["provenance"]],
            [1, 2],
        )

    def test_malformed_ids_do_not_qualify(self):
        result = self._ask(
            "SELECT 'abc' AS course_id, course_code FROM courses",
            "IT มีวิชาอะไรบ้าง",
            "IT",
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])
        self.assertEqual(len(self.answer_calls), 0)

    def test_malformed_fact_id_does_not_qualify(self):
        result = self._ask(
            "SELECT 'x' AS fact_id, fact_key FROM policy_facts",
            "เกรดเท่าไหร่จะได้เกียรตินิยม",
            None,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["provenance"], [])
        self.assertEqual(len(self.answer_calls), 0)

    def test_deterministic_grounder_still_wins(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("grounder precedence unchanged"),
        ):
            result = ask_sql(
                self.db_path,
                "IT มีวิชาอะไรบ้าง",
                "IT",
                lambda _prompt: "SELECT course_id FROM courses",
                lambda _prompt: "row grounded summary",
                grounding_callable=_grounded_answer,
                allow_grounded_row_rescue=True,
            )
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["answer"], "grounded deterministic answer")

    def test_default_off_has_zero_gate_effect(self):
        with patch(
            "backend.llm_sql_qa.hydrate_sql_row_provenance",
            side_effect=AssertionError("gate must not run when rescue is off"),
        ):
            result = ask_sql(
                self.db_path,
                "IT มีวิชาอะไรบ้าง",
                "IT",
                lambda _prompt: "SELECT course_id, course_code FROM courses",
                lambda _prompt: "row grounded summary",
                grounding_callable=_failed_grounding,
            )
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_no_sentence_specific_markers_in_gate(self):
        source = Path("backend/llm_sql_qa.py").read_text(encoding="utf-8")
        for marker in ("เกียรตินิยม", "วิชานี้ดีไหม", "ขอทุกอย่างในฐานข้อมูล"):
            self.assertNotIn(marker, source)


if __name__ == "__main__":
    unittest.main()
