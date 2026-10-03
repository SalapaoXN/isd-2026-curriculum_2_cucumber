import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend.hard_sequence_planner import plan_curriculum_sequence


class HardSequencePlannerTest(unittest.TestCase):
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
        connection.execute("INSERT INTO programs VALUES (1, 1, 'TST', 'tst')")
        connection.execute(
            """INSERT INTO curriculum_plans
               (plan_id, catalog_id, program_id, program_code, plan_key)
               VALUES (1, 1, 1, 'TST', 'default')"""
        )
        connection.executemany(
            """INSERT INTO courses
               (course_id, catalog_id, course_code, course_code_normalized,
                name_en, credit_units)
               VALUES (?, 1, ?, ?, ?, ?)""",
            [
                (1, "10000001", "10000001", "A", 3),
                (2, "10000002", "10000002", "B", 3),
                (3, "10000003", "10000003", "C", 3),
                (4, "10000004", "10000004", "D", 5),
                (5, "10000005", "10000005", "Option E", 2),
                (6, "10000006", "10000006", "Option F", 4),
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
               (alternative_group_member_id, alternative_group_id, course_id,
                member_order)
               VALUES (?, 10, ?, ?)""",
            [(401, 5, 1), (402, 6, 2)],
        )
        connection.executemany(
            """INSERT INTO plan_placements
               (placement_id, plan_id, course_id, alternative_group_id,
                year_number, semester_number, requirement_type, placement_order)
               VALUES (?, 1, ?, ?, ?, ?, ?, ?)""",
            [
                (101, 1, None, 4, 1, "\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a", 1),
                (102, 2, None, 4, 1, "\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a", 2),
                (103, 3, None, 4, 1, "\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a", 3),
                (104, 4, None, 1, 2, "\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a", 4),
                (105, None, 10, 2, 2, "\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a", 5),
            ],
        )
        connection.executemany(
            """INSERT INTO prerequisites
               (prerequisite_id, course_id, prerequisite_course_id,
                requirement_type)
               VALUES (?, ?, ?, 'required')""",
            [(201, 2, 1), (202, 3, 2)],
        )
        connection.executemany(
            """INSERT INTO provenance
               (provenance_id, source_document_key, program, source_filename,
                source_page, document_page, document_category)
               VALUES (?, ?, 'TST', ?, ?, ?, ?)""",
            [
                (301, "plan", "plan.pdf", 1, None, "plan"),
                (302, "placement-101", "plan.pdf", 2, 1, "plan"),
                (303, "placement-102", "plan.pdf", 3, 2, "plan"),
                (304, "placement-103", "plan.pdf", 4, 3, "plan"),
                (305, "placement-104", "plan.pdf", 5, 4, "plan"),
                (306, "placement-105", "plan.pdf", 6, 5, "plan"),
                (311, "course-1", "courses.pdf", 11, 1, "description"),
                (312, "course-2", "courses.pdf", 12, 2, "description"),
                (313, "course-3", "courses.pdf", 13, 3, "description"),
                (314, "course-4", "courses.pdf", 14, 4, "description"),
                (315, "course-5", "courses.pdf", 15, 5, "description"),
                (316, "course-6", "courses.pdf", 16, 6, "description"),
                (320, "edge-201", "plan.pdf", 20, 10, "plan"),
                (321, "edge-202", "plan.pdf", 21, 11, "plan"),
                (322, "group", "plan.pdf", 22, 12, "plan"),
                (323, "member-1", "plan.pdf", 23, 13, "plan"),
                (324, "member-2", "plan.pdf", 24, 14, "plan"),
                (330, "credit-policy", "rules.pdf", 30, 20, "rule"),
                (331, "credit-requirement", "requirements.pdf", 31, 21, "program_requirement"),
            ],
        )
        connection.execute("INSERT INTO curriculum_plan_provenance VALUES (1, 301)")
        connection.executemany(
            "INSERT INTO plan_placement_provenance VALUES (?, ?)",
            [(101, 302), (102, 303), (103, 304), (104, 305), (105, 306)],
        )
        connection.executemany(
            "INSERT INTO course_provenance(course_id, provenance_id) VALUES (?, ?)",
            [(1, 311), (2, 312), (3, 313), (4, 314), (5, 315), (6, 316)],
        )
        connection.executemany(
            "INSERT INTO prerequisite_provenance VALUES (?, ?)", [(201, 320), (202, 321)]
        )
        connection.execute("INSERT INTO alternative_group_provenance VALUES (10, 322)")
        connection.executemany(
            "INSERT INTO alternative_group_member_provenance VALUES (?, ?)",
            [(401, 323), (402, 324)],
        )
        connection.execute(
            """INSERT INTO program_requirements
               (requirement_id, program_code, requirement_type, operator, value, unit)
               VALUES (1, 'TST', 'total_program_credits', '=', 15, 'credits')"""
        )
        connection.execute("INSERT INTO program_requirement_provenance VALUES (1, 331)")
        connection.execute(
            """INSERT INTO regulation_rules
               (rule_id, section_number, category, rule_text, references_json)
               VALUES ('rule:11', '11', 'enrollment', 'Regular semester credits.', '{}')"""
        )
        connection.executemany(
            """INSERT INTO policy_facts
               (fact_id, category, fact_key, operator, value, unit, condition,
                context, source_rule_id)
               VALUES (?, 'enrollment', ?, ?, ?, 'credits', ?, ?, 'rule:11')""",
            [
                (1, "regular_maximum", "<=", 22, "at_most", "regular_semester"),
                (2, "regular_minimum", ">=", 9, "at_least", "regular_semester"),
                (3, "graduation_exception_maximum", "<=", 27, "at_most", "graduation_exception"),
            ],
        )
        connection.executemany(
            "INSERT INTO policy_fact_provenance VALUES (?, 330)", [(1,), (2,), (3,)]
        )

    @staticmethod
    def _terms(result):
        return {course["course_code"]: term["term_index"] for term in result["terms"] for course in term["courses"]}

    def test_explicit_program_plan_and_seven_term_horizon_are_required(self):
        self.assertEqual(plan_curriculum_sequence(self.db_path, "", "default")["status"], "invalid_scope")
        self.assertEqual(plan_curriculum_sequence(self.db_path, "TST", "")["status"], "invalid_scope")
        self.assertEqual(plan_curriculum_sequence(self.db_path, "TST", "default", horizon_terms=6)["status"], "unsupported_horizon")

    def test_prerequisite_chain_is_ordered_and_final_sequence_is_h3_validated(self):
        result = plan_curriculum_sequence(self.db_path, "TST", "default")
        terms = self._terms(result)

        self.assertEqual(result["sequence_feasible"], True)
        self.assertLess(terms["10000001"], terms["10000002"])
        self.assertLess(terms["10000002"], terms["10000003"])
        self.assertEqual(result["prerequisite_validation"]["status"], "satisfied")
        self.assertTrue(result["prerequisite_validation"]["validations"])

    def test_same_term_prerequisite_is_never_emitted_for_required_edges(self):
        result = plan_curriculum_sequence(self.db_path, "TST", "default")
        terms = self._terms(result)
        self.assertNotEqual(terms["10000001"], terms["10000002"])
        self.assertNotEqual(terms["10000002"], terms["10000003"])

    def test_regular_credit_cap_is_hard_and_conditional_overload_is_not_applied(self):
        result = plan_curriculum_sequence(self.db_path, "TST", "default")
        self.assertEqual(result["credit_constraints"]["regular_maximum"]["value"], 22)
        self.assertEqual(result["credit_constraints"]["conditional_overload"]["value"], 27)
        self.assertFalse(result["credit_constraints"]["conditional_overload"]["applied"])
        self.assertTrue(all(term["maximum_possible_credits"] <= 22 for term in result["terms"]))

    def test_conditional_overload_is_not_used_to_fit_an_over_cap_mandatory_course(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("UPDATE courses SET credit_units=23 WHERE course_id=4")
            connection.commit()
        result = plan_curriculum_sequence(self.db_path, "TST", "default")

        self.assertEqual(result["credit_constraints"]["regular_maximum"]["value"], 22)
        self.assertEqual(result["credit_constraints"]["conditional_overload"]["value"], 27)
        self.assertFalse(result["credit_constraints"]["conditional_overload"]["applied"])
        self.assertFalse(result["sequence_feasible"])
        self.assertIn("a mandatory course exceeds the documented normal regular-term maximum", result["limitations"])

    def test_known_mandatory_credit_lower_bound_over_seven_terms_is_infeasible(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            for offset in range(8):
                course_id = 10 + offset
                code = f"100000{10 + offset}"
                placement_id = 110 + offset
                connection.execute(
                    "INSERT INTO courses(course_id,catalog_id,course_code,course_code_normalized,name_en,credit_units) VALUES (?,1,?,?,?,22)",
                    (course_id, code, code, f"Required {code}"),
                )
                connection.execute(
                    "INSERT INTO plan_placements(placement_id,plan_id,course_id,year_number,semester_number,requirement_type,placement_order) VALUES (?,1,?,1,1,'\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a',?)",
                    (placement_id, course_id, 10 + offset),
                )
                connection.execute("INSERT INTO plan_placement_provenance VALUES (?,302)", (placement_id,))
                connection.execute("INSERT INTO course_provenance(course_id,provenance_id) VALUES (?,311)", (course_id,))
            connection.commit()
        result = plan_curriculum_sequence(self.db_path, "TST", "default")

        self.assertFalse(result["sequence_feasible"])
        self.assertEqual(result["status"], "infeasible")
        self.assertIn("known mandatory credits exceed seven regular-term capacity", result["limitations"])

    def test_baseline_is_preserved_when_possible_and_prerequisite_can_move_earlier(self):
        baseline = plan_curriculum_sequence(self.db_path, "TST", "default")
        baseline_terms = self._terms(baseline)
        self.assertEqual(baseline_terms["10000004"], 2)
        self.assertEqual(baseline_terms["10000003"], 7)

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("UPDATE plan_placements SET year_number=4, semester_number=1 WHERE placement_id IN (101, 102, 103)")
            connection.commit()
        moved = plan_curriculum_sequence(self.db_path, "TST", "default")
        moved_terms = self._terms(moved)
        self.assertEqual([moved_terms[f"1000000{i}"] for i in (1, 2, 3)], [5, 6, 7])
        self.assertTrue(any(item["course_code"] == "10000001" and item["from_term_index"] == 7 and item["to_term_index"] == 5 for item in moved["moved_from_baseline"]))

    def test_flexible_baseline_is_not_guessed_but_represented_sequence_can_be_built(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("UPDATE plan_placements SET flexible_year_semester_raw='flexible' WHERE placement_id=104")
            connection.commit()
        result = plan_curriculum_sequence(self.db_path, "TST", "default")

        course = next(course for term in result["terms"] for course in term["courses"] if course["course_code"] == "10000004")
        self.assertIsNone(course["baseline_term_index"])
        self.assertTrue(result["sequence_feasible"])
        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertTrue(any("no single fixed baseline term" in item for item in result["limitations"]))

    def test_output_is_deterministic_and_duplicate_placements_do_not_duplicate_courses(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO plan_placements(placement_id,plan_id,course_id,year_number,semester_number,requirement_type,placement_order) VALUES (106,1,1,4,1,'\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a',1)"
            )
            connection.execute("INSERT INTO plan_placement_provenance VALUES (106, 302)")
            connection.commit()
        first = plan_curriculum_sequence(self.db_path, "TST", "default")
        second = plan_curriculum_sequence(self.db_path, "TST", "default")

        self.assertEqual(first, second)
        codes = [course["course_code"] for term in first["terms"] for course in term["courses"]]
        self.assertEqual(codes.count("10000001"), 1)

    def test_choice_slot_preserves_candidates_without_scheduling_all_as_required(self):
        result = plan_curriculum_sequence(self.db_path, "TST", "default")
        slots = [slot for term in result["terms"] for slot in term["choice_slots"]]
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["minimum_choices"], 1)
        self.assertEqual({c["course_code"] for c in slots[0]["candidates"]}, {"10000005", "10000006"})
        courses = {c["course_code"] for term in result["terms"] for c in term["courses"]}
        self.assertNotIn("10000005", courses)
        self.assertNotIn("10000006", courses)
        self.assertEqual(result["actual_course_offering_unverified"], True)

    def test_unknown_elective_semantics_and_raw_prerequisites_prevent_complete_claim(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("INSERT INTO courses(course_id,catalog_id,course_code,course_code_normalized,name_en,credit_units) VALUES (7,1,'10000007','10000007','Elective candidate',3)")
            connection.execute("INSERT INTO plan_placements(placement_id,plan_id,course_id,year_number,semester_number,requirement_type) VALUES (107,1,7,2,1,'เลือก')")
            connection.execute("INSERT INTO plan_placement_provenance VALUES (107,305)")
            connection.execute("INSERT INTO course_provenance(course_id,provenance_id) VALUES (7,315)")
            connection.execute("INSERT INTO prerequisites(prerequisite_id,course_id,raw_text,requirement_type) VALUES (203,3,'unresolved raw condition','required')")
            connection.execute("INSERT INTO prerequisite_provenance VALUES (203,320)")
            connection.commit()
        result = plan_curriculum_sequence(self.db_path, "TST", "default")

        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertIn("elective_selection_semantics", {item["type"] for item in result["unresolved_requirements"]})
        self.assertIsNone(result["sequence_feasible"])
        h3_incomplete = [
            item
            for validation in result["prerequisite_validation"]["validations"]
            for item in (validation.get("h3") or {}).get("incomplete", [])
        ]
        self.assertTrue(any(item.get("prerequisite_condition", {}).get("kind") == "raw_only" for item in h3_incomplete))

    def test_unknown_credit_value_fails_closed_for_sequence_feasibility(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("UPDATE courses SET credit_units=NULL WHERE course_id=2")
            connection.commit()
        result = plan_curriculum_sequence(self.db_path, "TST", "default")

        self.assertIsNone(result["sequence_feasible"])
        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertIn("10000002", result["unknown_credit_courses"])

    def test_stored_provenance_and_null_document_page_are_preserved(self):
        result = plan_curriculum_sequence(self.db_path, "TST", "default")
        evidence = {item["provenance_id"]: item for item in result["evidence"]}
        self.assertIn(301, evidence)
        self.assertIn(320, evidence)
        self.assertIn(330, evidence)
        self.assertIsNone(evidence[301]["document_page"])

    def test_selected_catalog_resolves_plan_while_unscoped_lookup_is_ambiguous(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("INSERT INTO catalogs(catalog_id, catalog_key) VALUES (2, 'copy')")
            connection.execute("INSERT INTO programs VALUES (2, 2, 'TST', 'tst')")
            connection.execute(
                """INSERT INTO curriculum_plans
                   (plan_id, catalog_id, program_id, program_code, plan_key)
                   VALUES (2, 2, 2, 'TST', 'default')"""
            )
            connection.commit()

        unscoped = plan_curriculum_sequence(self.db_path, "TST", "default")
        selected = plan_curriculum_sequence(
            self.db_path, "TST", "default", catalog_key="fixture"
        )

        self.assertEqual(unscoped["status"], "ambiguous_plan")
        self.assertNotEqual(selected["status"], "ambiguous_plan")

    def test_read_only_database_access(self):
        original_connect = sqlite3.connect
        seen = []

        def capture(database, *args, **kwargs):
            seen.append((str(database), kwargs.get("uri")))
            return original_connect(database, *args, **kwargs)

        with patch("backend.hard_sequence_planner.sqlite3.connect", side_effect=capture):
            result = plan_curriculum_sequence(self.db_path, "TST", "default")
        self.assertEqual(result["sequence_feasible"], True)
        self.assertTrue(seen)
        self.assertTrue(all("mode=ro" in uri and use_uri for uri, use_uri in seen))

    def test_h3_violation_is_never_reported_as_feasible(self):
        real_validator = __import__("backend.hard_sequence_planner", fromlist=["validate_candidate_sequence"]).validate_candidate_sequence

        def return_violation(*args, **kwargs):
            result = real_validator(*args, **kwargs)
            result["status"] = "violation"
            result["violations"] = [{"prerequisite_id": 201, "status": "violation"}]
            return result

        with patch("backend.hard_sequence_planner.validate_candidate_sequence", side_effect=return_violation):
            result = plan_curriculum_sequence(self.db_path, "TST", "default")
        self.assertIsNone(result["sequence_feasible"])
        self.assertEqual(result["status"], "incomplete_evidence")

    def test_runtime_plan_returns_structure_without_guaranteeing_graduation(self):
        db_path = Path("cucumber_outputs/runtime/curriculum.db")
        if not db_path.exists():
            self.skipTest("runtime curriculum DB is not present")
        result = plan_curriculum_sequence(
            db_path, "DSBA", "coop", catalog_key="dsba-2565"
        )

        self.assertNotEqual(result["status"], "ambiguous_plan")
        self.assertEqual(result["terms"], [])
        self.assertEqual(result["status"], "incomplete_evidence")
        self.assertFalse(result.get("graduation_guaranteed", False))

    def test_runtime_sequence_never_uses_2565_credits_for_2560(self):
        db_path = Path("cucumber_outputs/runtime/curriculum.db")
        if not db_path.exists():
            self.skipTest("runtime curriculum DB is not present")

        for catalog_key in ("dsba-2560", "dsba-2565"):
            for plan in ("coop", "no_coop"):
                with self.subTest(catalog_key=catalog_key, plan=plan):
                    result = plan_curriculum_sequence(
                        db_path, "DSBA", plan, catalog_key=catalog_key
                    )
                    value = result.get("required_program_credits")
                    if catalog_key == "dsba-2560":
                        self.assertNotEqual(value, 132)
                    else:
                        self.assertIn(value, (None, 132))


if __name__ == "__main__":
    unittest.main()
