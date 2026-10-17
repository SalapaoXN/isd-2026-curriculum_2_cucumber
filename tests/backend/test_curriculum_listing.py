import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend import rag_service
from backend.curriculum import (
    course_prerequisite_requirements, courses_requiring, filters, list_courses, program_requirements,
    verified_course_name_in_text,
)


class CurriculumListingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "curriculum.db"
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript("""
                CREATE TABLE catalogs(catalog_id INTEGER, catalog_key TEXT, academic_year TEXT);
                CREATE TABLE programs(program_id INTEGER, catalog_id INTEGER, program_code TEXT);
                CREATE TABLE courses(course_id INTEGER, catalog_id INTEGER, course_code TEXT,
                    name_th TEXT, name_en TEXT, credits TEXT, credit_units INTEGER,
                    description_th TEXT, description_en TEXT, category TEXT, course_type TEXT,
                    prerequisite_text TEXT, notes TEXT);
                CREATE TABLE curriculum_plans(plan_id INTEGER, catalog_id INTEGER, program_id INTEGER,
                    plan_key TEXT, plan_code TEXT);
                CREATE TABLE plan_placements(placement_id INTEGER, plan_id INTEGER, course_id INTEGER,
                    year_number INTEGER, semester_number INTEGER, flexible_year_number INTEGER,
                    flexible_semester_number INTEGER,
                    flexible_year_semester_raw TEXT);
                CREATE TABLE provenance(provenance_id INTEGER, document_category TEXT,
                    source_filename TEXT, source_page INTEGER, source_uri TEXT, source_locator TEXT);
                CREATE TABLE course_provenance(course_id INTEGER, provenance_id INTEGER);
                CREATE TABLE prerequisites(prerequisite_id INTEGER, course_id INTEGER,
                    prerequisite_course_id INTEGER, alternative_group_id INTEGER, requirement_type TEXT,
                    raw_text TEXT);
                CREATE TABLE alternative_course_groups(alternative_group_id INTEGER, catalog_id INTEGER,
                    plan_id INTEGER, group_key TEXT, label TEXT, minimum_choices INTEGER,
                    maximum_choices INTEGER, notes TEXT);
                CREATE TABLE alternative_course_group_members(alternative_group_member_id INTEGER PRIMARY KEY,
                    alternative_group_id INTEGER, course_id INTEGER, member_order INTEGER, notes TEXT);
                CREATE TABLE program_requirements(requirement_id INTEGER, catalog_id INTEGER,
                    program_code TEXT, requirement_type TEXT, operator TEXT, value NUMERIC, unit TEXT);
                CREATE TABLE program_requirement_provenance(requirement_id INTEGER, provenance_id INTEGER);
            """)
            db.executemany("INSERT INTO catalogs VALUES (?,?,?)", [
                (1, "it-2565", "2565"), (2, "it-2560", "2560"), (3, "dsba-2568", "2568")])
            db.executemany("INSERT INTO programs VALUES (?,?,?)", [(1, 1, "IT"), (2, 2, "IT"), (3, 3, "DSBA")])
            db.executemany("INSERT INTO courses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (1, 1, "IT100", "พื้นฐาน", "INTRODUCTION", "3(3-0-6)", 3, "คำอธิบาย", "Intro course", "Core", "Required", "None", None),
                (2, 1, "IT200", "เลือก", "ELECTIVE", "3(3-0-6)", 3, None, "Elective topics", "Elective", "Elective", "IT100", None),
                (3, 2, "IT060", None, "OLD CATALOG COURSE", "3", 3, None, None, None, None, None, None),
                (4, 3, "DS100", None, "DSBA COURSE", "3", 3, None, None, None, None, None, None),
            ])
            db.executemany("INSERT INTO curriculum_plans VALUES (?,?,?,?,?)", [
                (1, 1, 1, "coop", "coop"), (2, 1, 1, "no_coop", "no_coop")])
            db.executemany("INSERT INTO plan_placements VALUES (?,?,?,?,?,?,?,?)", [
                (1, 1, 1, 1, 1, None, None, None), (2, 2, 1, 1, 1, None, None, None),
                (3, 1, 2, 2, 2, None, None, None)])
            db.executemany("INSERT INTO provenance VALUES (?,?,?,?,?,?)", [
                (1, "plan", "it-plan.pdf", 3, None, None),
                (2, "description", "it-desc.pdf", 12, None, None),
                (3, "program_requirement", "it-overview.pdf", 1, None, "page-1")])
            db.executemany("INSERT INTO course_provenance VALUES (?,?)", [(1, 1), (1, 2), (2, 1)])
            db.execute("INSERT INTO prerequisites VALUES (1, 2, 1, NULL, 'required', 'IT100')")
            db.execute("INSERT INTO program_requirements VALUES (1, 1, 'IT', 'total_program_credits', '=', 129, 'credits')")
            db.execute("INSERT INTO program_requirement_provenance VALUES (1, 3)")
            db.commit()
        self.path_patch = patch.object(rag_service, "INDEX_PATH", self.path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def test_filters_return_program_scoped_edition_years_semesters_and_other_options(self):
        data = filters()
        self.assertEqual(data["programs"], ["DSBA", "IT"])
        self.assertEqual(data["academic_years"], ["2568", "2565", "2560"])
        self.assertEqual(filters(program="IT")["academic_years"], ["2565", "2560"])
        self.assertEqual(filters(program="DSBA")["academic_years"], ["2568"])
        self.assertEqual(filters(program="IT", academic_year="2565", plan="coop")["study_years"], ["1", "2"])
        self.assertEqual(filters(program="IT", academic_year="2565", plan="coop", study_year="1")["semesters"], ["1"])
        self.assertEqual(data["semesters"], ["1", "2"])
        self.assertEqual(data["categories"], ["Core", "Elective"])
        self.assertEqual(data["course_types"], ["Elective", "Required"])
        self.assertEqual(data["plans"], ["coop", "no_coop"])

    def test_lists_subjects_with_plan_placement_and_provenance(self):
        result = list_courses(limit=10)
        self.assertEqual(result["total"], 4)
        first = next(item for item in result["items"] if item["course_code"] == "IT100")
        self.assertEqual(first["plans"], ["coop", "no_coop"])
        self.assertIn("Year 1 · Semester 1", first["placements"])
        self.assertEqual(first["document_types"], ["plan", "description"])

    def test_source_verified_course_name_alias_is_exact_and_catalog_scoped(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("ALTER TABLE courses ADD COLUMN course_code_normalized TEXT")
            db.execute("UPDATE courses SET course_code_normalized=lower(course_code)")
            db.commit()

        entry = {
            "catalog_key": "it-2565", "program": "IT", "academic_year": "2565",
            "course_code": "IT100", "canonical_name": "INTRODUCTION",
            "observed_aliases": ["INTRODUCT1ON"],
        }
        with patch("backend.curriculum._course_name_alias_entries", return_value=(entry,)):
            result = list_courses(search="INTRODUCT1ON", program="IT", academic_year="2565")
            phrase_result = list_courses(search="INTRODUCT1ON TYPO", program="IT", academic_year="2565")
            other_edition = list_courses(search="INTRODUCT1ON", program="IT", academic_year="2560")
            course = {
                "course_code": "IT100", "program_code": "IT", "academic_year": "2565",
                "plans": ["coop", "no_coop"],
            }
            extracted_name = verified_course_name_in_text(
                "How many credits for INTRODUCT1ON?", course
            )

        self.assertEqual([row["course_code"] for row in result["items"]], ["IT100"])
        self.assertEqual(phrase_result["total"], 0)
        self.assertEqual(other_edition["total"], 0)
        self.assertEqual(extracted_name, "INTRODUCTION")

    def test_filters_apply_program_edition_year_plan_category_type_semester_and_search(self):
        result = list_courses(program="IT", academic_year="2565", plan="no_coop",
                              category="Core", course_type="Required", study_year="1", semester="1",
                              search="IT100", limit=10)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["course_code"], "IT100")
        self.assertEqual(list_courses(plan="missing")["total"], 0)
        self.assertEqual(list_courses(program="it")["total"], list_courses(program="IT")["total"])
        self.assertEqual(list_courses(search="not-present")["total"], 0)
        self.assertEqual(list_courses(semester="2")["items"][0]["course_code"], "IT200")

    def test_plan_filter_returns_only_the_course_variant_placed_in_that_plan(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.executemany("INSERT INTO courses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (5, 1, "IT300", "แผนสหกิจ", "COOP VERSION", "3", 3, None, None,
                 "Core", "Required", None, None),
                (6, 1, "IT300", "แผนปกติ", "NO COOP VERSION", "6", 6, None, None,
                 "Core", "Required", None, None),
            ])
            db.executemany("INSERT INTO plan_placements VALUES (?,?,?,?,?,?,?,?)", [
                (4, 1, 5, 1, 1, None, None, None),
                (5, 2, 6, 1, 1, None, None, None),
            ])
            db.commit()

        coop = list_courses(course_code="IT300", program="IT", plan="coop", limit=10)
        no_coop = list_courses(course_code="IT300", program="IT", plan="no_coop", limit=10)
        self.assertEqual(coop["total"], 1)
        self.assertEqual((coop["items"][0]["name_en"], coop["items"][0]["credits"]),
                         ("COOP VERSION", "3"))
        self.assertEqual(coop["items"][0]["plans"], ["coop"])
        self.assertEqual(no_coop["total"], 1)
        self.assertEqual((no_coop["items"][0]["name_en"], no_coop["items"][0]["credits"]),
                         ("NO COOP VERSION", "6"))
        self.assertEqual(no_coop["items"][0]["plans"], ["no_coop"])

    def test_page_limit_and_offset(self):
        self.assertEqual(list_courses(limit=1, offset=0)["total"], 4)
        self.assertEqual(len(list_courses(limit=1, offset=0)["items"]), 1)
        self.assertEqual(len(list_courses(limit=1, offset=1)["items"]), 1)

    def test_exact_course_code_filter_scopes_results_and_keeps_credits(self):
        result = list_courses(course_code="it100", program="IT", academic_year="2565", limit=10)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["credits"], "3(3-0-6)")
        self.assertEqual(list_courses(course_code="IT10", program="IT")["total"], 0)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("INSERT INTO courses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                5, 2, "IT100", "قديم", "OLD INTRODUCTION", "3", 3, None, None,
                None, None, None, None))
            db.commit()
        self.assertEqual(list_courses(course_code="IT100", program="IT", academic_year="2565")["total"], 1)
        self.assertEqual(list_courses(course_code="IT100", program="IT")["total"], 2)

    def test_inverse_prerequisite_lookup_returns_dependent_course(self):
        result = courses_requiring(prerequisite_code="IT100", program="IT", academic_year="2565")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["course_code"], "IT200")

    def test_inverse_prerequisite_lookup_includes_alternative_group_members(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("INSERT INTO courses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                5, 2, "IT100", "พื้นฐาน", "INTRODUCTION", "3", 3, None, None,
                None, None, None, None))
            db.execute("INSERT INTO curriculum_plans VALUES (?,?,?,?,?)", (3, 2, 2, "coop", "coop"))
            db.executemany("INSERT INTO plan_placements VALUES (?,?,?,?,?,?,?,?)", [
                (6, 3, 3, None, None, None, None, None),
                (7, 3, 5, None, None, None, None, None),
            ])
            db.execute("INSERT INTO prerequisites VALUES (2, 3, NULL, 22, 'required', 'IT100 or IT060')")
            db.execute("INSERT INTO alternative_course_group_members (alternative_group_id, course_id) VALUES (22, 5)")
            db.commit()
        result = courses_requiring(prerequisite_code="IT100", program="IT", academic_year="2560")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["course_code"], "IT060")

    def test_prerequisite_lookup_preserves_alternative_choice_bounds_and_members(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.executemany("INSERT INTO courses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (5, 1, "IT110", None, "INTRODUCTION TO NETWORKS", "3", 3, None, None,
                 None, None, None, None),
                (6, 1, "IT120", None, "INTRODUCTION TO SYSTEMS", "3", 3, None, None,
                 None, None, None, None),
            ])
            db.execute("DELETE FROM prerequisites WHERE course_id=2")
            db.execute("UPDATE courses SET prerequisite_text=? WHERE course_id=2", ("IT110 or IT120",))
            db.execute(
                "INSERT INTO alternative_course_groups VALUES (22, 1, 1, 'prereq-choice', 'Choose one', 1, 1, NULL)"
            )
            db.execute(
                "INSERT INTO prerequisites VALUES (2, 2, NULL, 22, 'required', 'IT110 or IT120')"
            )
            db.executemany(
                "INSERT INTO alternative_course_group_members "
                "(alternative_group_id, course_id, member_order) VALUES (22, ?, ?)",
                [(5, 1), (6, 2)],
            )
            db.commit()

        requirements = course_prerequisite_requirements(course_id=2)
        choice = next(item for item in requirements if item["kind"] == "alternative")
        self.assertEqual(choice["minimum_choices"], 1)
        self.assertEqual(choice["maximum_choices"], 1)
        self.assertEqual(
            [(member["course_id"], member["course_code"]) for member in choice["members"]],
            [(5, "IT110"), (6, "IT120")],
        )
        self.assertEqual(choice["plans"], ["coop"])
        self.assertFalse(any(item["kind"] == "alternative" for item in
                             course_prerequisite_requirements(course_id=2, plan="no_coop")))

    def test_program_requirement_lookup_is_scoped_and_returns_provenance(self):
        result = program_requirements(program="it", academic_year="2565")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["value"], 129)
        self.assertEqual(result["items"][0]["sources"][0]["source_filename"], "it-overview.pdf")
        self.assertEqual(program_requirements(program="IT", academic_year="2560")["total"], 0)


if __name__ == "__main__":
    unittest.main()
