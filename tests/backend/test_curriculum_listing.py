import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend import rag_service
from backend.curriculum import filters, list_courses


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
                    source_filename TEXT, source_page INTEGER);
                CREATE TABLE course_provenance(course_id INTEGER, provenance_id INTEGER);
            """)
            db.executemany("INSERT INTO catalogs VALUES (?,?,?)", [
                (1, "it-2565", "2565"), (2, "it-2560", "2560"), (3, "dsba-2568", "2568")])
            db.executemany("INSERT INTO programs VALUES (?,?,?)", [(1, 1, "IT"), (2, 2, "IT"), (3, 3, "DSBA")])
            db.executemany("INSERT INTO courses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (1, 1, "IT100", "พื้นฐาน", "INTRODUCTION", "3(3-0-6)", 3, "คำอธิบาย", "Intro course", "Core", "Required", "None", None),
                (2, 1, "IT200", "เลือก", "ELECTIVE", "3(3-0-6)", 3, None, "Elective topics", "Elective", "Elective", None, None),
                (3, 2, "IT060", None, "OLD CATALOG COURSE", "3", 3, None, None, None, None, None, None),
                (4, 3, "DS100", None, "DSBA COURSE", "3", 3, None, None, None, None, None, None),
            ])
            db.executemany("INSERT INTO curriculum_plans VALUES (?,?,?,?,?)", [
                (1, 1, 1, "coop", "coop"), (2, 1, 1, "no_coop", "no_coop")])
            db.executemany("INSERT INTO plan_placements VALUES (?,?,?,?,?,?,?,?)", [
                (1, 1, 1, 1, 1, None, None, None), (2, 2, 1, 1, 1, None, None, None),
                (3, 1, 2, 2, 2, None, None, None)])
            db.executemany("INSERT INTO provenance VALUES (?,?,?,?)", [
                (1, "plan", "it-plan.pdf", 3), (2, "description", "it-desc.pdf", 12)])
            db.executemany("INSERT INTO course_provenance VALUES (?,?)", [(1, 1), (1, 2), (2, 1)])
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

    def test_filters_apply_program_edition_year_plan_category_type_semester_and_search(self):
        result = list_courses(program="IT", academic_year="2565", plan="no_coop",
                              category="Core", course_type="Required", study_year="1", semester="1",
                              search="IT100", limit=10)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["course_code"], "IT100")
        self.assertEqual(list_courses(plan="missing")["total"], 0)
        self.assertEqual(list_courses(search="not-present")["total"], 0)
        self.assertEqual(list_courses(semester="2")["items"][0]["course_code"], "IT200")

    def test_page_limit_and_offset(self):
        self.assertEqual(list_courses(limit=1, offset=0)["total"], 4)
        self.assertEqual(len(list_courses(limit=1, offset=0)["items"]), 1)
        self.assertEqual(len(list_courses(limit=1, offset=1)["items"]), 1)


if __name__ == "__main__":
    unittest.main()
