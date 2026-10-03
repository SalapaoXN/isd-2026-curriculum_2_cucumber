import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend.hard_prerequisite_validate import (
    validate_candidate_sequence,
    validate_plan_prerequisite_sequence,
)


class HardPrerequisiteValidateTest(unittest.TestCase):
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
                name_en)
               VALUES (?, 1, ?, ?, ?)""",
            [(i, f"1000000{i}", f"1000000{i}", f"Course {i}") for i in range(1, 10)],
        )
        connection.executemany(
            """INSERT INTO plan_placements
               (placement_id, plan_id, course_id, year_number, semester_number,
                flexible_year_number, flexible_semester_number,
                flexible_year_semester_raw)
               VALUES (?, 1, ?, ?, ?, ?, ?, ?)""",
            [
                (101, 1, 1, 1, None, None, None),
                (102, 2, 1, 2, None, None, None),
                (103, 3, 2, 1, None, None, None),
                (104, 4, 2, 2, None, None, None),
                (105, 5, 3, 1, None, None, None),
                (106, 6, None, None, 1, 2, "flexible"),
                (107, 1, 1, 1, None, None, None),  # duplicate logical placement
                (108, 7, 4, 1, None, None, None),
                (109, 8, 4, 2, None, None, None),
                (110, 9, 3, 1, None, None, None),
            ],
        )
        connection.executemany(
            """INSERT INTO alternative_course_groups
               (alternative_group_id, catalog_id, plan_id, group_key,
                minimum_choices, maximum_choices)
               VALUES (?, 1, 1, ?, 1, 1)""",
            [(10, "one-before"), (11, "none-before")],
        )
        connection.executemany(
            """INSERT INTO alternative_course_group_members
               (alternative_group_member_id, alternative_group_id,
                course_id, member_order)
               VALUES (?, ?, ?, ?)""",
            [(401, 10, 1, 1), (402, 10, 5, 2), (411, 11, 7, 1), (412, 11, 8, 2)],
        )
        connection.executemany(
            "INSERT INTO prerequisites(prerequisite_id, course_id, prerequisite_course_id, alternative_group_id, requirement_type) VALUES (?, ?, ?, ?, 'required')",
            [(201, 2, 1, None), (202, 3, 2, None), (204, 2, 6, None), (205, 6, 1, None), (206, 4, None, 10), (207, 9, None, 11)],
        )
        connection.execute(
            "INSERT INTO prerequisites(prerequisite_id, course_id, raw_text, requirement_type) VALUES (203, 5, 'prior study unclear', 'required')"
        )
        connection.executemany(
            """INSERT INTO provenance
               (provenance_id, source_document_key, program, source_filename,
                source_page, document_page, document_category)
               VALUES (?, ?, 'TST', ?, ?, ?, 'plan')""",
            [
                (i, f"source-{i}", "fixture.pdf", i, None if i == 301 else i - 300)
                for i in range(301, 331)
            ],
        )
        connection.execute("INSERT INTO curriculum_plan_provenance VALUES (1, 301)")
        connection.executemany(
            "INSERT INTO plan_placement_provenance VALUES (?, ?)",
            [(101 + i, 302 + i) for i in range(10)],
        )
        connection.executemany(
            "INSERT INTO course_provenance(course_id, provenance_id) VALUES (?, ?)",
            [(i, 310 + i) for i in range(1, 10)],
        )
        connection.executemany(
            "INSERT INTO prerequisite_provenance VALUES (?, ?)",
            [(201, 320), (202, 321), (203, 322), (204, 323), (205, 324), (206, 325), (207, 326)],
        )
        connection.execute("INSERT INTO alternative_group_provenance VALUES (10, 327)")
        connection.execute("INSERT INTO alternative_group_provenance VALUES (11, 328)")
        connection.executemany(
            "INSERT INTO alternative_group_member_provenance VALUES (?, ?)",
            [(401, 329), (402, 330), (411, 329), (412, 330)],
        )

    def test_explicit_program_and_plan_resolve_and_direct_order_is_satisfied(self):
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")

        self.assertEqual(result["status"], "incomplete_evidence")  # raw-only record remains unresolved
        self.assertEqual((result["program"], result["plan"]), ("TST", "default"))
        relationship = next(item for item in result["relationships"] if item["prerequisite_id"] == 201)
        self.assertEqual(relationship["status"], "satisfied")
        self.assertEqual(relationship["course_term"]["ordinal"], 1)
        self.assertEqual(relationship["prerequisite_terms"][0]["term"]["ordinal"], 0)

    def test_same_term_and_reversed_direct_prerequisites_are_violations(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("UPDATE plan_placements SET year_number=1, semester_number=1 WHERE placement_id=102")
            connection.commit()
        same_term = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        self.assertEqual(next(item for item in same_term["relationships"] if item["prerequisite_id"] == 201)["status"], "violation")

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("UPDATE plan_placements SET year_number=1, semester_number=1 WHERE placement_id=102")
            connection.execute("UPDATE plan_placements SET year_number=2, semester_number=1 WHERE placement_id IN (101, 107)")
            connection.commit()
        reversed_order = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        self.assertEqual(next(item for item in reversed_order["relationships"] if item["prerequisite_id"] == 201)["status"], "violation")

    def test_explicit_corequisite_semantics_allow_same_term(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("UPDATE plan_placements SET year_number=1, semester_number=1 WHERE placement_id=102")
            connection.execute("UPDATE prerequisites SET requirement_type='co_requisite' WHERE prerequisite_id=201")
            connection.commit()
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        relationship = next(item for item in result["relationships"] if item["prerequisite_id"] == 201)

        self.assertEqual(relationship["status"], "satisfied")
        self.assertFalse(relationship["prerequisite_terms"][0]["before_dependent"])
        self.assertTrue(relationship["prerequisite_terms"][0]["satisfies_order"])

    def test_candidate_sequence_checks_explicit_term_indices(self):
        result = validate_candidate_sequence(
            self.db_path,
            "TST",
            "default",
            [
                {"course_code": "10000001", "term_index": 1},
                {"course_code": "10000002", "term_index": 2},
                {"course_code": "10000003", "term_index": 3},
                {"course_code": "10000004", "term_index": 4},
                {"course_code": "10000005", "term_index": 5},
                {"course_code": "10000006", "term_index": 6},
            ],
        )

        self.assertEqual(next(item for item in result["relationships"] if item["prerequisite_id"] == 201)["status"], "satisfied")
        self.assertEqual(next(item for item in result["relationships"] if item["prerequisite_id"] == 202)["status"], "satisfied")
        candidate_term = next(item for item in result["relationships"] if item["prerequisite_id"] == 201)["course_term"]
        self.assertEqual(candidate_term["term_index"], 2)
        self.assertEqual(candidate_term["basis"], "candidate_sequence_input")
        self.assertIsNone(candidate_term["year"])

    def test_unknown_candidate_course_returns_controlled_failure(self):
        result = validate_candidate_sequence(
            self.db_path, "TST", "default", [{"course_code": "99999999", "term_index": 1}]
        )

        self.assertEqual(result["status"], "unknown_course")
        self.assertEqual(result["relationships"], [])

    def test_missing_and_ambiguous_explicit_plan_scopes_fail_closed(self):
        missing = validate_plan_prerequisite_sequence(self.db_path, "TST", "coop")
        self.assertEqual(missing["status"], "plan_not_found")

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("INSERT INTO catalogs(catalog_id, catalog_key) VALUES (2, 'copy')")
            connection.execute("INSERT INTO programs VALUES (2, 2, 'TST', 'tst')")
            connection.execute(
                """INSERT INTO curriculum_plans
                   (plan_id, catalog_id, program_id, program_code, plan_key)
                   VALUES (2, 2, 2, 'TST', 'default')"""
            )
            connection.commit()
        ambiguous = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        self.assertEqual(ambiguous["status"], "ambiguous_plan")

    def test_missing_prerequisite_source_link_makes_judgment_incomplete(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("DELETE FROM prerequisite_provenance WHERE prerequisite_id=201")
            connection.commit()
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        relationship = next(item for item in result["relationships"] if item["prerequisite_id"] == 201)

        self.assertEqual(relationship["status"], "incomplete_evidence")
        self.assertIn("prerequisite relationship", relationship["reason"])

    def test_transitive_closure_exposes_chain_and_upstream_violation(self):
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        chain = next(path for path in result["transitive_paths"] if path["course_codes"] == ["10000001", "10000002", "10000003"])
        self.assertEqual(chain["status"], "satisfied")
        self.assertEqual(chain["prerequisite_ids"], [201, 202])

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("UPDATE plan_placements SET year_number=2, semester_number=1 WHERE placement_id IN (101, 107)")
            connection.commit()
        broken = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        upstream = next(path for path in broken["transitive_paths"] if path["course_codes"] == ["10000001", "10000002", "10000003"])
        self.assertEqual(upstream["status"], "violation")
        self.assertTrue(any(item["prerequisite_id"] == 201 for item in broken["violations"]))

    def test_alternative_prerequisite_minimum_is_a_choice_not_all_candidates(self):
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        relationship = next(item for item in result["relationships"] if item["prerequisite_id"] == 206)

        self.assertEqual(relationship["status"], "satisfied")
        self.assertEqual(relationship["prerequisite_condition"]["minimum_choices"], 1)
        self.assertEqual(len(relationship["prerequisite_terms"]), 2)
        self.assertEqual(sum(item["before_dependent"] for item in relationship["prerequisite_terms"]), 1)
        provenance_ids = {item["provenance_id"] for item in relationship["provenance"]}
        self.assertIn(327, provenance_ids)
        self.assertIn(329, provenance_ids)

    def test_alternative_group_with_no_earlier_candidate_is_a_violation(self):
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        relationship = next(item for item in result["relationships"] if item["prerequisite_id"] == 207)

        self.assertEqual(relationship["status"], "violation")
        self.assertEqual(sum(item["before_dependent"] for item in relationship["prerequisite_terms"]), 0)

    def test_flexible_prerequisite_or_dependent_placement_is_indeterminate(self):
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")

        flexible_prereq = next(item for item in result["relationships"] if item["prerequisite_id"] == 204)
        flexible_dependent = next(item for item in result["relationships"] if item["prerequisite_id"] == 205)
        self.assertEqual(flexible_prereq["status"], "incomplete_evidence")
        self.assertEqual(flexible_dependent["status"], "incomplete_evidence")

    def test_duplicate_course_placements_do_not_duplicate_judgments(self):
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")

        direct = [item for item in result["relationships"] if item["prerequisite_id"] == 201]
        self.assertEqual(len(direct), 1)
        self.assertEqual(len(direct[0]["prerequisite_terms"]), 1)

    def test_raw_only_prerequisite_is_incomplete_and_preserves_raw_text(self):
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        raw = next(item for item in result["relationships"] if item["prerequisite_id"] == 203)

        self.assertEqual(raw["status"], "incomplete_evidence")
        self.assertEqual(raw["prerequisite_condition"]["raw_text"], "prior study unclear")

    def test_cycles_terminate_and_mark_affected_relations_incomplete(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO prerequisites(prerequisite_id, course_id, prerequisite_course_id, requirement_type) VALUES (208, 1, 3, 'required')"
            )
            connection.execute("INSERT INTO prerequisite_provenance VALUES (208, 320)")
            connection.commit()
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")

        self.assertTrue(result["cycles"])
        self.assertEqual(result["summary"]["cycles"], 1)
        self.assertTrue(all(item["status"] == "incomplete_evidence" for item in result["relationships"] if item["dependent_course_code"] in {"10000001", "10000002", "10000003"}))

    def test_relationship_provenance_uses_stored_prerequisite_and_placements(self):
        result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")
        relationship = next(item for item in result["relationships"] if item["prerequisite_id"] == 201)
        ids = {item["provenance_id"] for item in relationship["provenance"]}

        self.assertIn(320, ids)  # prerequisite edge
        self.assertIn(303, ids)  # dependent placement
        self.assertIn(311, ids)  # canonical course source
        self.assertTrue(result["provenance"])
        missing_page = next(item for item in result["provenance"] if item["document_page"] is None)
        self.assertIsNone(missing_page["document_page"])

    def test_current_runtime_has_a_real_resolved_prerequisite_order(self):
        db_path = Path("cucumber_outputs/runtime/curriculum.db")
        if not db_path.exists():
            self.skipTest("runtime curriculum DB is not present")
        result = validate_plan_prerequisite_sequence(db_path, "DSBA", "coop")
        relationship = next(
            item for item in result["relationships"]
            if item["dependent_course_code"] == "06026201"
            and item["prerequisite_condition"].get("course_code") == "06026200"
        )

        self.assertEqual(relationship["status"], "satisfied")
        self.assertEqual(relationship["course_term"]["ordinal"], 1)
        self.assertEqual(relationship["prerequisite_terms"][0]["term"]["ordinal"], 0)
        self.assertTrue(relationship["provenance"])

    def test_database_is_opened_read_only(self):
        original_connect = sqlite3.connect
        seen = []

        def capture_connect(database, *args, **kwargs):
            seen.append((str(database), kwargs.get("uri")))
            return original_connect(database, *args, **kwargs)

        with patch("backend.hard_prerequisite_validate.sqlite3.connect", side_effect=capture_connect):
            result = validate_plan_prerequisite_sequence(self.db_path, "TST", "default")

        self.assertEqual(result["program"], "TST")
        self.assertEqual(len(seen), 1)
        self.assertIn("mode=ro", seen[0][0])
        self.assertTrue(seen[0][1])


if __name__ == "__main__":
    unittest.main()
