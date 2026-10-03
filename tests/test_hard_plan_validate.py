import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend.hard_plan_validate import validate_curriculum_plan_structure


class HardPlanValidateTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "curriculum.db"
        schema_path = Path(__file__).parents[1] / "rag" / "structured" / "schema.sql"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(schema_path.read_text(encoding="utf-8"))
            self._seed_fixture(connection)
            connection.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _seed_fixture(connection):
        connection.execute("INSERT INTO catalogs(catalog_id, catalog_key) VALUES (1, 'fixture')")
        connection.execute(
            "INSERT INTO programs VALUES (1, 1, 'TST', 'tst')"
        )
        connection.execute(
            """INSERT INTO curriculum_plans
               (plan_id, catalog_id, program_id, program_code, plan_key)
               VALUES (1, 1, 1, 'TST', 'default')"""
        )
        connection.executemany(
            """INSERT INTO courses
               (course_id, catalog_id, course_code, course_code_normalized,
                name_th, name_en, credit_units)
               VALUES (?, 1, ?, ?, ?, ?, ?)""",
            [
                (1, "10000001", "10000001", "วิชาบังคับ", "Required", 3),
                (2, "10000002", "10000002", "วิชาเลือก", "Elective", 4),
                (3, "10000003", "10000003", "ตัวเลือก ก", "Option A", 2),
                (4, "10000004", "10000004", "ตัวเลือก ข", "Option B", 5),
            ],
        )
        connection.execute(
            """INSERT INTO alternative_course_groups
               (alternative_group_id, catalog_id, plan_id, group_key, label,
                minimum_choices, maximum_choices)
               VALUES (10, 1, 1, 'choose-one', 'Choose one', 1, 1)"""
        )
        connection.executemany(
            """INSERT INTO alternative_course_group_members
               (alternative_group_member_id, alternative_group_id, course_id, member_order)
               VALUES (?, 10, ?, ?)""",
            [(301, 3, 1), (302, 4, 2)],
        )
        connection.executemany(
            """INSERT INTO plan_placements
               (placement_id, plan_id, course_id, alternative_group_id,
                year_number, semester_number, flexible_year_number,
                flexible_semester_number, flexible_year_semester_raw,
                category, requirement_type, raw_text)
               VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (101, 1, None, 1, 1, None, None, None, "core", "บังคับ", None),
                (102, 1, None, 1, 1, None, None, None, "core", "บังคับ", None),
                (103, 2, None, None, None, 2, 1, "1/2", "elective", "เลือก", None),
                (104, None, 10, None, None, None, None, None, "elective", "บังคับ", None),
                (105, 2, None, None, None, None, None, None, "elective", "เลือก", "elective candidate"),
            ],
        )
        connection.execute(
            """INSERT INTO program_requirements
               (requirement_id, catalog_id, program_code, requirement_type,
                operator, value, unit)
               VALUES (1, 1, 'TST', 'total_program_credits', '=', 20, 'credits')"""
        )
        connection.executemany(
            """INSERT INTO provenance
               (provenance_id, source_document_key, program, source_filename,
                source_page, document_page, document_category)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (500, "plan", "TST", "plan.pdf", 10, None, "plan"),
                (501, "placement-101", "TST", "plan.pdf", 11, 1, "plan"),
                (502, "placement-102", "TST", "plan.pdf", 12, 2, "plan"),
                (503, "placement-103", "TST", "plan.pdf", 13, 3, "plan"),
                (504, "placement-104", "TST", "plan.pdf", 14, 4, "plan"),
                (505, "placement-105", "TST", "plan.pdf", 15, 5, "plan"),
                (511, "course-1", "TST", "courses.pdf", 20, 6, "description"),
                (512, "course-2", "TST", "courses.pdf", 21, None, "description"),
                (513, "course-3", "TST", "courses.pdf", 22, 7, "description"),
                (514, "course-4", "TST", "courses.pdf", 23, 8, "description"),
                (520, "alt-group", "TST", "plan.pdf", 16, 9, "plan"),
                (521, "alt-member-1", "TST", "plan.pdf", 17, None, "plan"),
                (522, "alt-member-2", "TST", "plan.pdf", 18, 10, "plan"),
                (530, "program-requirement", "TST", "requirements.pdf", 30, 11, "program_requirement"),
                (540, "policy", "RULE", "rules.pdf", 40, 20, "rule"),
                (541, "rule-reference", "RULE", "rules.pdf", 41, None, "rule"),
            ],
        )
        connection.execute("INSERT INTO curriculum_plan_provenance VALUES (1, 500)")
        connection.executemany(
            "INSERT INTO plan_placement_provenance VALUES (?, ?)",
            [(101, 501), (102, 502), (103, 503), (104, 504), (105, 505)],
        )
        connection.executemany(
            "INSERT INTO course_provenance(course_id, provenance_id) VALUES (?, ?)",
            [(1, 511), (2, 512), (3, 513), (4, 514)],
        )
        connection.execute("INSERT INTO alternative_group_provenance VALUES (10, 520)")
        connection.executemany(
            "INSERT INTO alternative_group_member_provenance VALUES (?, ?)",
            [(301, 521), (302, 522)],
        )
        connection.execute("INSERT INTO program_requirement_provenance VALUES (1, 530)")
        connection.executemany(
            """INSERT INTO regulation_rules
               (rule_id, section_number, category, rule_text, references_json)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    "rule:25.2", "25.2", "เกณฑ์การสำเร็จการศึกษา", "Complete structure and exam.",
                    json.dumps({"source_provenance": [{"source_filename": "rules.pdf", "source_page": 41}]}),
                ),
                (
                    "rule:25.4", "25.4", "เกณฑ์การสำเร็จการศึกษา", "No institutional debt.",
                    json.dumps({"source_provenance": [{"source_filename": "rules.pdf", "source_page": 41}]}),
                ),
            ],
        )
        connection.executemany(
            """INSERT INTO policy_facts
               (fact_id, category, fact_key, operator, value, unit, context,
                source_rule_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (1, "เกณฑ์การสำเร็จการศึกษา", "GPA สะสม", "at_least", 2, "GPA", "cumulative", "rule:25.2"),
                (2, "เกณฑ์การสำเร็จการศึกษา", "English Exit Exam", "required", "English Exit Exam", None, None, "rule:25.2"),
                (3, "เกณฑ์การสำเร็จการศึกษา", "ไม่มีหนี้สิน", "required", "ไม่มีหนี้สิน", None, None, "rule:25.4"),
            ],
        )
        connection.executemany(
            "INSERT INTO policy_fact_provenance VALUES (?, ?)", [(1, 540), (2, 540), (3, 540)]
        )

    def test_explicit_scope_returns_structural_checks_and_requirement_evidence(self):
        result = validate_curriculum_plan_structure(self.db_path, "TST", "default")

        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertEqual(result["assessment_scope"], "curriculum_plan_structure_only")
        self.assertEqual((result["program"], result["plan"]), ("TST", "default"))
        requirement = result["credit_requirements"][0]
        self.assertEqual(requirement["requirement_type"], "total_program_credits")
        self.assertEqual(requirement["required_value"], 20)
        self.assertIsNone(requirement["measured_value"])
        self.assertEqual(requirement["status"], "incomplete_evidence")
        self.assertEqual(requirement["provenance"][0]["source_filename"], "requirements.pdf")

    def test_elective_credits_are_not_summed_as_required_total(self):
        result = validate_curriculum_plan_structure(self.db_path, "TST", "default")
        requirement = result["credit_requirements"][0]

        self.assertIsNone(requirement["measured_value"])
        self.assertIn("elective", requirement["limitation"].lower())
        self.assertEqual(result["status"], "incomplete_evidence")

    def test_alternative_candidates_are_one_choice_constraint_not_mandatory_courses(self):
        result = validate_curriculum_plan_structure(self.db_path, "TST", "default")

        self.assertEqual(result["mandatory_courses"]["count"], 1)
        self.assertEqual(result["mandatory_courses"]["status"], "complete")
        self.assertEqual(
            [course["course_code_normalized"] for course in result["mandatory_courses"]["courses"]],
            ["10000001"],
        )
        group = result["alternative_groups"][0]
        self.assertEqual((group["minimum_choices"], group["maximum_choices"]), (1, 1))
        self.assertEqual(
            [candidate["course_code_normalized"] for candidate in group["candidates"]],
            ["10000003", "10000004"],
        )
        self.assertTrue(group["structurally_complete"])
        self.assertEqual(group["status"], "complete")

    def test_repeated_placements_collapse_mandatory_course_identity(self):
        result = validate_curriculum_plan_structure(self.db_path, "TST", "default")
        required = result["mandatory_courses"]["courses"][0]

        self.assertEqual(required["course_code"], "10000001")
        self.assertEqual(required["placement_ids"], [101, 102])
        self.assertEqual(result["mandatory_courses"]["count"], 1)

    def test_fixed_flexible_and_untimed_placements_are_distinguished(self):
        result = validate_curriculum_plan_structure(self.db_path, "TST", "default")
        quality = result["placement_quality"]

        self.assertEqual(quality["fixed_count"], 2)
        self.assertEqual(quality["flexible_count"], 1)
        self.assertEqual(quality["untimed_count"], 2)
        self.assertEqual(quality["unresolved_raw_count"], 0)
        self.assertEqual(
            quality["duplicate_logical_placements"]["10000001"]["count"], 2
        )

    def test_missing_category_requirement_and_student_conditions_are_unassessable(self):
        result = validate_curriculum_plan_structure(self.db_path, "TST", "default")
        unassessable = result["unassessable_requirements"]

        self.assertTrue(
            any(item["requirement_type"] == "category_specific_minimum_credits" for item in unassessable)
        )
        canonical_keys = {item.get("fact_key") for item in unassessable if item.get("kind") == "policy_fact"}
        self.assertIn("English Exit Exam", canonical_keys)
        self.assertIn("GPA สะสม", canonical_keys)
        self.assertIn("ไม่มีหนี้สิน", canonical_keys)
        for requirement_type in (
            "student_completed_courses",
            "student_grades_or_gpa",
            "english_exit_exam_result",
            "debt_status",
        ):
            self.assertTrue(
                any(item["requirement_type"] == requirement_type for item in unassessable)
            )
        self.assertTrue(any(item["requirement_type"] == "actual_enrollment_history" for item in unassessable))
        self.assertTrue(
            any(item["requirement_type"] == "elective_selection_semantics" for item in unassessable)
        )
        for item in unassessable:
            if item.get("kind") in ("policy_fact", "regulation_rule"):
                self.assertTrue(item["provenance"])
                self.assertEqual(item["status"], "unassessable")
        self.assertFalse(result.get("graduation_verdict"))

    def test_provenance_pages_are_stored_values_and_missing_document_page_stays_null(self):
        result = validate_curriculum_plan_structure(self.db_path, "TST", "default")
        required = result["mandatory_courses"]["courses"][0]
        plan_ref = next(ref for ref in result["provenance"] if ref["provenance_id"] == 500)
        missing_page = next(ref for ref in required["provenance"] if ref["provenance_id"] == 511)

        self.assertEqual(plan_ref["source_page"], 10)
        self.assertIsNone(plan_ref["document_page"])
        self.assertEqual(missing_page["source_page"], 20)
        self.assertEqual(missing_page["document_page"], 6)
        policy_fact = next(
            item for item in result["unassessable_requirements"]
            if item.get("fact_key") == "English Exit Exam"
        )
        self.assertIn(
            {"source_filename": "rules.pdf", "source_page": 41},
            policy_fact["provenance"],
        )

    def test_missing_and_ambiguous_scope_return_controlled_status(self):
        missing = validate_curriculum_plan_structure(self.db_path, "TST", "coop")
        self.assertEqual(missing["status"], "plan_not_found")
        invalid = validate_curriculum_plan_structure(self.db_path, "", "default")
        self.assertEqual(invalid["status"], "invalid_scope")

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("INSERT INTO catalogs(catalog_id, catalog_key) VALUES (2, 'copy')")
            connection.execute("INSERT INTO programs VALUES (2, 2, 'TST', 'tst')")
            connection.execute(
                """INSERT INTO curriculum_plans
                   (plan_id, catalog_id, program_id, program_code, plan_key)
                   VALUES (2, 2, 2, 'TST', 'default')"""
            )
            connection.commit()
        ambiguous = validate_curriculum_plan_structure(self.db_path, "TST", "default")
        self.assertEqual(ambiguous["status"], "ambiguous_plan")
        selected = validate_curriculum_plan_structure(
            self.db_path, "TST", "default", catalog_key="fixture"
        )
        self.assertNotEqual(selected["status"], "ambiguous_plan")

    def test_database_is_opened_read_only(self):
        original_connect = sqlite3.connect
        seen = []

        def capture_connect(database, *args, **kwargs):
            seen.append((str(database), kwargs.get("uri")))
            return original_connect(database, *args, **kwargs)

        with patch("backend.hard_plan_validate.sqlite3.connect", side_effect=capture_connect):
            result = validate_curriculum_plan_structure(self.db_path, "TST", "default")

        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertEqual(len(seen), 1)
        self.assertIn("mode=ro", seen[0][0])
        self.assertTrue(seen[0][1])

    def test_current_dsba_plan_has_no_definitive_graduation_verdict(self):
        db_path = Path("cucumber_outputs/runtime/curriculum.db")
        if not db_path.exists():
            self.skipTest("runtime curriculum DB is not present")

        result = validate_curriculum_plan_structure(
            db_path, "DSBA", "coop", catalog_key="dsba-2565"
        )

        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertEqual(result["program"], "DSBA")
        self.assertEqual(result["plan"], "coop")
        self.assertTrue(result["mandatory_courses"]["courses"])
        self.assertEqual(result["credit_requirements"][0]["required_value"], 132)
        self.assertIsNone(result["credit_requirements"][0]["measured_value"])
        self.assertFalse(result.get("graduation_verdict"))

    def test_dsba_editions_use_their_own_program_credit_requirements(self):
        db_path = Path("cucumber_outputs/runtime/curriculum.db")
        if not db_path.exists():
            self.skipTest("runtime curriculum DB is not present")

        old_edition = validate_curriculum_plan_structure(
            db_path, "DSBA", "coop", catalog_key="dsba-2560"
        )
        current_edition = validate_curriculum_plan_structure(
            db_path, "DSBA", "coop", catalog_key="dsba-2565"
        )

        old_total = next(
            requirement for requirement in old_edition["credit_requirements"]
            if requirement["requirement_type"] == "total_program_credits"
        )
        self.assertEqual(old_total["required_value"], 126)
        self.assertTrue(old_total["provenance"])
        self.assertNotEqual(old_total["required_value"], 132)
        self.assertTrue(
            all(
                reference["source_filename"].startswith("dsba2560_page_")
                for reference in old_total["provenance"]
            )
        )
        old_credit_check = next(
            check for check in old_edition["checks"]
            if check["check"] == "program_credit_requirements"
        )
        self.assertEqual(old_credit_check["status"], "incomplete_evidence")
        current_total = next(
            requirement for requirement in current_edition["credit_requirements"]
            if requirement["requirement_type"] == "total_program_credits"
        )
        self.assertEqual(current_total["required_value"], 132)
        self.assertTrue(
            any(
                reference.get("source_page") in {32, 39}
                for reference in current_total["provenance"]
            )
        )

    def test_degree_program_totals_are_catalog_scoped_and_provenanced(self):
        db_path = Path("cucumber_outputs/runtime/curriculum.db")
        if not db_path.exists():
            self.skipTest("runtime curriculum DB is not present")

        expected = {
            ("DSBA", "dsba-2560"): 126,
            ("DSBA", "dsba-2565"): 132,
            ("IT", "it-2565"): 129,
            ("BIT", "bit-2565"): 126,
            ("AIT", "ait-2566"): 120,
        }
        with closing(sqlite3.connect(db_path)) as connection:
            rows = connection.execute(
                """SELECT catalog_key, program_code, value, requirement_id
                   FROM program_requirements
                   JOIN catalogs USING (catalog_id)
                   WHERE requirement_type = 'total_program_credits'
                   ORDER BY catalog_key"""
            ).fetchall()
            self.assertEqual(
                {(program, catalog): value for catalog, program, value, _ in rows},
                expected,
            )
            for _, _, _, requirement_id in rows:
                provenance = connection.execute(
                    """SELECT document_category, source_page
                       FROM program_requirement_provenance
                       JOIN provenance USING (provenance_id)
                       WHERE requirement_id = ?""",
                    (requirement_id,),
                ).fetchall()
                self.assertTrue(provenance)
                self.assertTrue(all(category == "plan" and page for category, page in provenance))

            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) FROM program_requirements
                       WHERE program_code = 'GENED'
                         AND requirement_type = 'total_program_credits'"""
                ).fetchone()[0],
                0,
            )

        plans = {
            ("DSBA", "dsba-2560"): "coop",
            ("DSBA", "dsba-2565"): "coop",
            ("IT", "it-2565"): "no_coop",
            ("BIT", "bit-2565"): "no_coop",
            ("AIT", "ait-2566"): "default",
        }
        for (program, catalog_key), plan in plans.items():
            with self.subTest(program=program, catalog_key=catalog_key):
                result = validate_curriculum_plan_structure(
                    db_path, program, plan, catalog_key=catalog_key
                )
                total = next(
                    requirement for requirement in result["credit_requirements"]
                    if requirement["requirement_type"] == "total_program_credits"
                )
                self.assertEqual(total["required_value"], expected[(program, catalog_key)])
                self.assertTrue(total["provenance"])

    def test_requirement_selection_is_catalog_scoped_and_unscoped_is_ambiguous(self):
        from backend.hard_plan_validate import _program_requirements

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                "INSERT INTO catalogs(catalog_id, catalog_key) VALUES (2, 'fixture-2')"
            )
            connection.execute("INSERT INTO programs VALUES (2, 2, 'TST', 'tst')")
            connection.execute(
                """INSERT INTO program_requirements
                   (catalog_id, program_code, requirement_type, operator, value, unit)
                   VALUES (2, 'TST', 'total_program_credits', '=', 30, 'credits')"""
            )
            connection.execute(
                """INSERT INTO provenance
                   (provenance_id, source_document_key, program, source_filename,
                    source_page, document_category)
                   VALUES (542, 'fixture-2-requirement', 'TST', 'requirements-2.pdf',
                           31, 'program_requirement')"""
            )
            connection.execute("INSERT INTO program_requirement_provenance VALUES (2, 542)")
            connection.commit()
            unscoped, unscoped_complete = _program_requirements(connection, "TST")
            first, first_complete = _program_requirements(
                connection, "TST", "fixture"
            )
            second, second_complete = _program_requirements(
                connection, "TST", "fixture-2"
            )

        self.assertEqual((unscoped, unscoped_complete), ([], False))
        self.assertEqual((first[0]["required_value"], first_complete), (20, True))
        self.assertEqual((second[0]["required_value"], second_complete), (30, True))


if __name__ == "__main__":
    unittest.main()
