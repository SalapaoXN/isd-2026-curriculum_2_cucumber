import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag.structured.loader import load_json_to_sqlite, load_jsons_to_sqlite


class RagLoaderTest(unittest.TestCase):
    def test_dsba2560_course_06026127_is_shared_with_plan_specific_provenance(self):
        root = Path(__file__).resolve().parents[2]
        artifacts = [
            root / "data" / "output" / "final" / filename
            for filename in (
                "dsba2560_coop_final.json",
                "dsba2560_no_coop_final.json",
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            inputs = []
            for index, artifact in enumerate(artifacts):
                document = json.loads(artifact.read_text(encoding="utf-8-sig"))
                courses = [course for course in document["courses"] if course.get("code") == "06026127"]
                self.assertEqual(len(courses), 1)
                focused = {
                    "program": document["program"],
                    "catalog": document["catalog"],
                    "plan": document["plan"],
                    "courses": courses,
                }
                source = temp_root / f"dsba-plan-{index}.json"
                source.write_text(json.dumps(focused), encoding="utf-8")
                inputs.append(source)

            database = temp_root / "curriculum.db"
            load_jsons_to_sqlite(inputs, database)

            with closing(sqlite3.connect(database)) as connection:
                courses = connection.execute(
                    """SELECT catalogs.catalog_key, courses.course_id, courses.name_en,
                              courses.credits
                       FROM courses JOIN catalogs USING (catalog_id)
                       WHERE courses.course_code = '06026127'"""
                ).fetchall()
                placements = connection.execute(
                    """SELECT curriculum_plans.plan_code, plan_placements.course_id
                       FROM plan_placements
                       JOIN curriculum_plans USING (plan_id)
                       JOIN courses USING (course_id)
                       WHERE courses.course_code = '06026127'
                       ORDER BY curriculum_plans.plan_code"""
                ).fetchall()
                provenance = connection.execute(
                    """SELECT DISTINCT curriculum_plans.plan_code, provenance.source_page
                       FROM plan_placement_provenance AS link
                       JOIN plan_placements USING (placement_id)
                       JOIN curriculum_plans USING (plan_id)
                       JOIN provenance USING (provenance_id)
                       JOIN courses USING (course_id)
                       WHERE courses.course_code = '06026127'
                       ORDER BY curriculum_plans.plan_code, provenance.source_page"""
                ).fetchall()

        self.assertEqual(len(courses), 1)
        catalog_key, course_id, name_en, credits = courses[0]
        self.assertEqual((catalog_key, name_en, credits), ("dsba-2560", "PROJECT IN DATA SCIENCE AND BUSINESS ANALYTICS 1", "3(0-9-0)"))
        self.assertEqual(placements, [("coop", course_id), ("no_coop", course_id)])
        self.assertEqual(
            provenance,
            [("coop", 33), ("coop", 188), ("no_coop", 28), ("no_coop", 188)],
        )

    def test_dsba2560_course_type_is_plan_specific_placement_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            inputs = []
            for index, (plan, requirement_type) in enumerate(
                (("coop", "เลือก"), ("no_coop", "บังคับ"))
            ):
                focused = {
                    "program": "DSBA",
                    "catalog": {"catalog_key": "dsba-2560", "academic_year": "2560"},
                    "plan": plan,
                    "courses": [
                        {
                            "code": "06026129",
                            "name_en": "DATA SCIENCE AND BUSINESS ANALYTICS PROFESSIONAL PRACTICES",
                            "credits": "3(0-18-0)",
                            "category": "วิชาเฉพาะ",
                            "type": requirement_type,
                            "year": 3,
                            "semester": index + 1,
                        }
                    ],
                }
                source = temp_root / f"dsba-plan-{index}.json"
                source.write_text(json.dumps(focused), encoding="utf-8")
                inputs.append(source)

            database = temp_root / "curriculum.db"
            load_jsons_to_sqlite(inputs, database)

            with closing(sqlite3.connect(database)) as connection:
                courses = connection.execute(
                    "SELECT course_id, course_type FROM courses WHERE course_code = '06026129'"
                ).fetchall()
                placements = connection.execute(
                    """SELECT curriculum_plans.plan_code, plan_placements.category,
                              plan_placements.requirement_type, plan_placements.course_id
                       FROM plan_placements
                       JOIN curriculum_plans USING (plan_id)
                       JOIN courses USING (course_id)
                       WHERE courses.course_code = '06026129'
                       ORDER BY curriculum_plans.plan_code"""
                ).fetchall()

        self.assertEqual(len(courses), 1)
        course_id, global_course_type = courses[0]
        self.assertIsNone(global_course_type)
        self.assertEqual(
            [(row[0], row[2], row[3]) for row in placements],
            [("coop", "เลือก", course_id), ("no_coop", "บังคับ", course_id)],
        )
        self.assertEqual([row[1] for row in placements], ["วิชาเฉพาะ", "วิชาเฉพาะ"])

    def test_ait_elective_placeholder_repeated_on_source_page_loads_as_one_course(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "output"
            / "final"
            / "ait2566_final.json"
        )
        document = json.loads(source.read_text(encoding="utf-8-sig"))
        matching = [course for course in document["courses"] if course.get("code") == "060464xx"]
        self.assertEqual(len(matching), 2)
        self.assertEqual([course["credits"] for course in matching], ["6(3-0-6)", "6(3-0-6)"])
        self.assertEqual(
            [course["source_provenance"][0]["source_page"] for course in matching],
            [25, 25],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            focused_source = root / "ait-placeholder.json"
            focused_source.write_text(
                json.dumps(
                    {
                        "program": document["program"],
                        "plan": "regular",
                        "courses": matching,
                    }
                ),
                encoding="utf-8",
            )
            database = root / "curriculum.db"
            load_json_to_sqlite(focused_source, database)

            with closing(sqlite3.connect(database)) as connection:
                course = connection.execute(
                    "SELECT credits_raw FROM courses WHERE course_code = '060464xx'"
                ).fetchone()
                placements = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM plan_placements AS placements
                    JOIN courses ON courses.course_id = placements.course_id
                    WHERE courses.course_code = '060464xx'
                    """
                ).fetchone()

        self.assertEqual(course, ("6(3-0-6)",))
        self.assertEqual(placements, (2,))

    def test_ait_distinct_free_elective_placeholders_keep_separate_provenance(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "output"
            / "final"
            / "ait2566_final.json"
        )
        document = json.loads(source.read_text(encoding="utf-8-sig"))
        matching = [course for course in document["courses"] if course.get("code") == "xxxxxxxx"]
        self.assertEqual(
            [course["name_en"] for course in matching],
            ["FREE ELECTIVE COURSE", "FREE ELECTIVE COURSE 2"],
        )
        self.assertEqual(
            [course["source_provenance"][0]["source_page"] for course in matching],
            [25, 26],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            focused_source = root / "ait-elective-placeholders.json"
            focused_source.write_text(
                json.dumps(
                    {"program": "AIT", "plan": "regular", "courses": matching}
                ),
                encoding="utf-8",
            )
            database = root / "curriculum.db"
            load_json_to_sqlite(focused_source, database)

            with closing(sqlite3.connect(database)) as connection:
                courses = connection.execute(
                    """SELECT course_id, name_en, course_identity_discriminator
                       FROM courses WHERE course_code = 'xxxxxxxx' ORDER BY name_en"""
                ).fetchall()
                placements = connection.execute(
                    """SELECT courses.name_en, provenance.source_page
                       FROM plan_placements AS placements
                       JOIN courses ON courses.course_id = placements.course_id
                       JOIN plan_placement_provenance AS link
                         ON link.placement_id = placements.placement_id
                       JOIN provenance ON provenance.provenance_id = link.provenance_id
                       WHERE courses.course_code = 'xxxxxxxx'
                       ORDER BY courses.name_en"""
                ).fetchall()

        self.assertEqual(
            [(name, identity) for _, name, identity in courses],
            [
                ("FREE ELECTIVE COURSE", "placeholder:free elective course"),
                ("FREE ELECTIVE COURSE 2", "placeholder:free elective course 2"),
            ],
        )
        self.assertEqual(placements, [("FREE ELECTIVE COURSE", 25), ("FREE ELECTIVE COURSE 2", 26)])

    def test_same_placeholder_identity_reuses_course_across_plans(self):
        documents = [
            {
                "program": "TEST",
                "catalog": {"catalog_key": "test-1", "academic_year": "1"},
                "plan": plan,
                "source_provenance": [
                    {"source_document_key": plan, "source_filename": f"{plan}.pdf", "source_page": page}
                ],
                "courses": [
                    {
                        "code": "xxxxxxxx",
                        "name_en": "FREE ELECTIVE COURSE",
                        "credits": "3(x-x-x)",
                    }
                ],
            }
            for plan, page in (("coop", 25), ("no_coop", 40))
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, document in enumerate(documents):
                path = root / f"plan-{index}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                inputs.append(path)
            database = root / "curriculum.db"
            load_jsons_to_sqlite(inputs, database)

            with closing(sqlite3.connect(database)) as connection:
                course_count = connection.execute(
                    "SELECT COUNT(*) FROM courses WHERE course_code = 'xxxxxxxx'"
                ).fetchone()
                placement_count = connection.execute(
                    """SELECT COUNT(*) FROM plan_placements AS placements
                       JOIN courses ON courses.course_id = placements.course_id
                       WHERE courses.course_code = 'xxxxxxxx'"""
                ).fetchone()
                provenance_pages = connection.execute(
                    """SELECT provenance.source_page FROM course_provenance AS link
                       JOIN courses ON courses.course_id = link.course_id
                       JOIN provenance ON provenance.provenance_id = link.provenance_id
                       WHERE courses.course_code = 'xxxxxxxx' ORDER BY provenance.source_page"""
                ).fetchall()

        self.assertEqual(course_count, (1,))
        self.assertEqual(placement_count, (2,))
        self.assertEqual(provenance_pages, [(25,), (40,)])

    def test_same_edition_plans_share_catalog_program_and_course_with_provenance(self):
        documents = [
            {
                "program": "DSBA",
                "catalog": {"catalog_key": "dsba-2560", "academic_year": 2560},
                "plan": {"plan_code": "coop", "version": "2560"},
                "source_provenance": [
                    {"source_document_key": "dsba-coop", "source_filename": "coop.pdf", "source_page": 10, "document_category": "plan"}
                ],
                "courses": [
                    {"code": "C101", "name_en": "Shared course", "credits": "3", "year": 1, "semester": 1},
                    {"code": "C102", "name_en": "Coop course", "credits": "3", "prerequisite": "C101", "year": 2, "semester": 1},
                ],
            },
            {
                "program": "DSBA",
                "catalog": {"catalog_key": "dsba-2560", "academic_year": "2560"},
                "plan": {"plan_code": "no_coop", "version": "2560"},
                "source_provenance": [
                    {"source_document_key": "dsba-no-coop", "source_filename": "no-coop.pdf", "source_page": 20, "document_category": "plan"}
                ],
                "courses": [
                    {"code": "C101", "name_en": "Shared course", "credits": "3", "year": 1, "semester": 1},
                    {"code": "C103", "name_en": "No-coop course", "credits": "3", "prerequisite": "C101", "year": 2, "semester": 1},
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, document in enumerate(documents):
                path = root / f"document-{index}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                inputs.append(path)
            database = root / "curriculum.db"

            catalog_ids = load_jsons_to_sqlite(inputs, database)

            with closing(sqlite3.connect(database)) as connection:
                catalogs = connection.execute(
                    "SELECT catalog_key, academic_year FROM catalogs"
                ).fetchall()
                programs = connection.execute(
                    "SELECT catalog_id, program_code FROM programs"
                ).fetchall()
                plans = connection.execute(
                    "SELECT plan_code FROM curriculum_plans ORDER BY plan_code"
                ).fetchall()
                shared_courses = connection.execute(
                    "SELECT course_id, catalog_id FROM courses WHERE course_code = 'C101'"
                ).fetchall()
                placements = connection.execute(
                    """
                    SELECT plans.plan_code, courses.course_code
                    FROM plan_placements AS placements
                    JOIN curriculum_plans AS plans ON plans.plan_id = placements.plan_id
                    JOIN courses ON courses.course_id = placements.course_id
                    WHERE courses.course_code = 'C101'
                    ORDER BY plans.plan_code
                    """
                ).fetchall()
                prerequisites = connection.execute(
                    """
                    SELECT dependent.course_code, target.course_code,
                           dependent.catalog_id, target.catalog_id
                    FROM prerequisites AS edge
                    JOIN courses AS dependent ON dependent.course_id = edge.course_id
                    JOIN courses AS target ON target.course_id = edge.prerequisite_course_id
                    WHERE dependent.course_code IN ('C102', 'C103')
                    ORDER BY dependent.course_code
                    """
                ).fetchall()
                shared_sources = connection.execute(
                    """
                    SELECT provenance.source_document_key, provenance.source_filename
                    FROM course_provenance AS link
                    JOIN courses ON courses.course_id = link.course_id
                    JOIN provenance ON provenance.provenance_id = link.provenance_id
                    WHERE courses.course_code = 'C101'
                    ORDER BY provenance.source_document_key
                    """
                ).fetchall()
                placement_sources = connection.execute(
                    """
                    SELECT plans.plan_code, provenance.source_document_key
                    FROM plan_placement_provenance AS link
                    JOIN plan_placements AS placements ON placements.placement_id = link.placement_id
                    JOIN curriculum_plans AS plans ON plans.plan_id = placements.plan_id
                    JOIN provenance ON provenance.provenance_id = link.provenance_id
                    WHERE placements.course_id = (SELECT course_id FROM courses WHERE course_code = 'C101')
                    ORDER BY plans.plan_code
                    """
                ).fetchall()

        self.assertEqual(catalog_ids, [catalog_ids[0], catalog_ids[0]])
        self.assertEqual(catalogs, [("dsba-2560", "2560")])
        self.assertEqual(programs, [(catalog_ids[0], "DSBA")])
        self.assertEqual(plans, [("coop",), ("no_coop",)])
        self.assertEqual(len(shared_courses), 1)
        self.assertEqual(shared_courses[0][1], catalog_ids[0])
        self.assertEqual(placements, [("coop", "C101"), ("no_coop", "C101")])
        self.assertEqual(
            prerequisites,
            [("C102", "C101", catalog_ids[0], catalog_ids[0]), ("C103", "C101", catalog_ids[0], catalog_ids[0])],
        )
        self.assertEqual(
            shared_sources,
            [("dsba-coop", "coop.pdf"), ("dsba-no-coop", "no-coop.pdf")],
        )
        self.assertEqual(
            placement_sources,
            [("coop", "dsba-coop"), ("no_coop", "dsba-no-coop")],
        )

    def test_same_course_code_remains_isolated_between_editions(self):
        documents = [
            {"program": "DSBA", "catalog": {"catalog_key": key, "academic_year": year}, "plan": "coop", "courses": [{"code": "C101", "name_en": name}]}
            for key, year, name in (("dsba-2560", "2560", "Edition 2560"), ("dsba-2565", "2565", "Edition 2565"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, document in enumerate(documents):
                path = root / f"edition-{index}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                inputs.append(path)
            database = root / "curriculum.db"
            catalog_ids = load_jsons_to_sqlite(inputs, database)
            with closing(sqlite3.connect(database)) as connection:
                rows = connection.execute(
                    "SELECT catalog_id, course_code, name_en FROM courses ORDER BY catalog_id"
                ).fetchall()
        self.assertEqual(len(set(catalog_ids)), 2)
        self.assertEqual(rows, [(catalog_ids[0], "C101", "Edition 2560"), (catalog_ids[1], "C101", "Edition 2565")])

    def test_same_catalog_conflicting_year_or_shared_course_facts_fails_closed(self):
        base = {
            "program": "DSBA",
            "catalog": {"catalog_key": "dsba-2560", "academic_year": "2560"},
            "plan": "coop",
            "courses": [{"code": "C101", "name_en": "Shared course", "credits": "3"}],
        }
        conflicting_year = {**base, "catalog": {"catalog_key": "dsba-2560", "academic_year": "2565"}, "plan": "no_coop"}
        conflicting_course = {**base, "plan": "no_coop", "courses": [{"code": "C101", "name_en": "Shared course", "credits": "6"}]}

        for second, expected in ((conflicting_year, "academic_year"), (conflicting_course, "C101")):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = []
                for index, document in enumerate((base, second)):
                    path = root / f"document-{index}.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    inputs.append(path)
                with self.assertRaisesRegex(ValueError, expected):
                    load_jsons_to_sqlite(inputs, root / "curriculum.db")

    def test_shared_thai_course_name_whitespace_is_compatible(self):
        base = {
            "program": "BIT",
            "catalog": {"catalog_key": "bit-2560", "academic_year": "2560"},
            "courses": [
                {
                    "code": "06036018",
                    "name_th": "สื่อสังคม และเครือข่ายสังคม",
                    "name_en": "SOCIAL MEDIA AND SOCIAL NETWORK",
                    "credits": "3(3-0-6)",
                }
            ],
        }
        second = {
            **base,
            "plan": "no_coop",
            "courses": [
                {
                    "code": "06036018",
                    "name_th": "สื่อสังคมและเครือข่ายสังคม",
                    "name_en": "SOCIAL MEDIA AND SOCIAL NETWORK",
                    "credits": "3(3-0-6)",
                }
            ],
        }
        first = {**base, "plan": "coop"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, document in enumerate((first, second)):
                path = root / f"document-{index}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                inputs.append(path)
            database = root / "curriculum.db"
            load_jsons_to_sqlite(inputs, database)
            with closing(sqlite3.connect(database)) as connection:
                rows = connection.execute(
                    "SELECT course_code, name_th FROM courses WHERE course_code = '06036018'"
                ).fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "06036018")

    def _load_shared_course_pair(self, first_course, second_course):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, (plan, course) in enumerate(
                (("coop", first_course), ("no_coop", second_course))
            ):
                path = root / f"document-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "program": "DSBA",
                            "catalog": {
                                "catalog_key": "dsba-2560",
                                "academic_year": "2560",
                            },
                            "plan": plan,
                            "courses": [course],
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(path)
            database = root / "curriculum.db"
            load_jsons_to_sqlite(inputs, database)
            with closing(sqlite3.connect(database)) as connection:
                return connection.execute(
                    "SELECT name_en, credits FROM courses WHERE course_code = 'C101'"
                ).fetchall()

    def _load_description_records(self, records):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, (plan, courses) in enumerate(records):
                path = root / f"document-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "program": "DSBA",
                            "catalog": {
                                "catalog_key": "dsba-2565",
                                "academic_year": "2565",
                            },
                            "plan": plan,
                            "courses": courses,
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(path)
            database = root / "curriculum.db"
            load_jsons_to_sqlite(inputs, database)
            with closing(sqlite3.connect(database)) as connection:
                course_rows = connection.execute(
                    """SELECT course_id, description_th, name_th, name_en, credits, notes
                       FROM courses WHERE course_code = 'C101'"""
                ).fetchall()
                if not course_rows:
                    return course_rows, 0, []
                provenance_count = connection.execute(
                    "SELECT COUNT(*) FROM course_provenance WHERE course_id = ?",
                    (course_rows[0][0],),
                ).fetchone()[0]
                prerequisites = connection.execute(
                    """SELECT prerequisite_course.course_code
                       FROM prerequisites
                       LEFT JOIN courses AS prerequisite_course
                         ON prerequisite_course.course_id = prerequisites.prerequisite_course_id
                       WHERE prerequisites.course_id = ?
                       ORDER BY prerequisite_course.course_code""",
                    (course_rows[0][0],),
                ).fetchall()
                return course_rows, provenance_count, [row[0] for row in prerequisites]

    def test_shared_course_same_description_is_reused(self):
        description = "Course description"
        rows, _, _ = self._load_description_records(
            [
                ("coop", [{"code": "C101", "name_en": "Course", "desc_th": description}]),
                ("no_coop", [{"code": "C101", "name_en": "Course", "desc_th": description}]),
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], description)

    def test_shared_course_missing_description_retains_available_value(self):
        description = "Available description"
        rows, _, _ = self._load_description_records(
            [
                ("coop", [{"code": "C101", "name_en": "Course"}]),
                ("no_coop", [{"code": "C101", "name_en": "Course", "desc_th": description}]),
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], description)

    def test_shared_course_conflicting_description_is_cleared_with_provenance(self):
        first_description = "Description for course one"
        bundled_course = {
            "code": "C101 or C102",
            "name_en": "COURSE ONE\nCOURSE TWO",
            "credits": "6",
            "desc_th": f"{first_description}\nDescription for course two",
            "source_provenance": [
                {"source_filename": "plan.png", "source_page": 39, "document_category": "plan"}
            ],
        }
        single_course = {
            "code": "C101",
            "name_en": "COURSE ONE",
            "credits": "6",
            "desc_th": first_description,
            "source_provenance": [
                {"source_filename": "description.png", "source_page": 344, "document_category": "description"}
            ],
        }
        later_duplicate = {
            **single_course,
            "source_provenance": [
                {"source_filename": "other-plan.png", "source_page": 40, "document_category": "plan"}
            ],
        }
        rows, provenance_count, _ = self._load_description_records(
            [
                ("coop", [bundled_course]),
                ("no_coop", [single_course]),
                ("third", [later_duplicate]),
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0][1])
        self.assertEqual(provenance_count, 3)

    def test_plan_table_name_and_credits_win_and_description_prerequisite_enriches(self):
        plan_course = {
            "code": "C101",
            "name_th": "Plan Thai name",
            "name_en": "PLAN COURSE NAME",
            "credits": "3",
            "source_provenance": [
                {"source_filename": "plan.png", "source_page": 10, "document_category": "plan"}
            ],
        }
        prerequisite_course = {
            "code": "C200",
            "name_en": "PREREQUISITE COURSE",
            "credits": "3",
            "source_provenance": [
                {"source_filename": "plan.png", "source_page": 10, "document_category": "plan"}
            ],
        }
        description_course = {
            "code": "C101",
            "name_th": "Description Thai name",
            "name_en": "DESCRIPTION COURSE NAME",
            "credits": "6",
            "desc_th": "Enriched description",
            "prerequisite": "C200",
            "source_provenance": [
                {
                    "source_filename": "description.png",
                    "source_page": 20,
                    "document_category": "description",
                }
            ],
        }
        course_orders = (
            [plan_course, prerequisite_course, description_course],
            [description_course, plan_course, prerequisite_course],
        )
        for courses in course_orders:
            with self.subTest(description_loaded_first=courses[0] is description_course):
                rows, provenance_count, prerequisites = self._load_description_records(
                    [("coop", courses)]
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(
                    rows[0][2:5], ("Plan Thai name", "PLAN COURSE NAME", "3")
                )
                self.assertEqual(rows[0][1], "Enriched description")
                self.assertEqual(prerequisites, ["C200"])
                self.assertEqual(provenance_count, 2)

    def test_conflicting_notes_are_cleared_without_aborting(self):
        plan_course = {
            "code": "C101",
            "name_en": "COURSE",
            "credits": "3",
            "note": "Plan note",
            "source_provenance": [
                {"source_filename": "plan.png", "source_page": 10, "document_category": "plan"}
            ],
        }
        description_course = {
            "code": "C101",
            "name_en": "COURSE",
            "credits": "3",
            "note": "Description note",
            "source_provenance": [
                {
                    "source_filename": "description.png",
                    "source_page": 20,
                    "document_category": "description",
                }
            ],
        }
        rows, provenance_count, _ = self._load_description_records(
            [("coop", [plan_course]), ("coop", [description_course])]
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0][5])
        self.assertEqual(provenance_count, 2)

    def test_conflicting_plan_table_name_or_credits_still_fails_closed(self):
        for field, values in (
            ("name_en", ("COURSE ONE", "COURSE TWO")),
            ("credits", ("3", "6")),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = []
                for index, value in enumerate(values):
                    course = {
                        "code": "C101",
                        "name_en": "COURSE ONE",
                        "credits": "3",
                        field: value,
                        "source_provenance": [
                            {
                                "source_filename": f"plan-{index}.png",
                                "source_page": 10 + index,
                                "document_category": "plan",
                            }
                        ],
                    }
                    path = root / f"document-{index}.json"
                    path.write_text(
                        json.dumps(
                            {
                                "program": "DSBA",
                                "catalog": {
                                    "catalog_key": "dsba-2565",
                                    "academic_year": "2565",
                                },
                                "plan": "coop" if index == 0 else "no_coop",
                                "courses": [course],
                            }
                        ),
                        encoding="utf-8",
                    )
                    inputs.append(path)
                with self.assertRaisesRegex(ValueError, f"conflicting {field}"):
                    load_jsons_to_sqlite(inputs, root / "curriculum.db")

    def test_shared_course_english_capitalization_difference_is_compatible(self):
        rows = self._load_shared_course_pair(
            {"code": "C101", "name_en": "PRACTICAL NoSQL DATABASE", "credits": "3"},
            {"code": "C101", "name_en": "PRACTICAL NOSQL DATABASE", "credits": "3"},
        )
        self.assertEqual(rows, [("PRACTICAL NoSQL DATABASE", "3")])

    def test_shared_course_whitespace_name_difference_is_compatible(self):
        rows = self._load_shared_course_pair(
            {"code": "C101", "name_en": "  COMPUTER   PROGRAMMING  ", "credits": "3"},
            {"code": "C101", "name_en": "COMPUTER PROGRAMMING", "credits": "3"},
        )
        self.assertEqual(rows, [("  COMPUTER   PROGRAMMING  ", "3")])

    def test_shared_course_missing_english_word_separator_is_compatible(self):
        rows = self._load_shared_course_pair(
            {"code": "C101", "name_en": "VISUAL COMMUNICATION FORBUSINESS", "credits": "3"},
            {"code": "C101", "name_en": "VISUAL COMMUNICATION FOR BUSINESS", "credits": "3"},
        )
        self.assertEqual(rows, [("VISUAL COMMUNICATION FORBUSINESS", "3")])

    def test_shared_course_substantive_name_difference_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, name in enumerate(
                ("COMPUTER PROGRAMMING", "COMPUTER PROGRAMMING 2")
            ):
                path = root / f"document-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "program": "DSBA",
                            "catalog": {
                                "catalog_key": "dsba-2560",
                                "academic_year": "2560",
                            },
                            "plan": "coop" if index == 0 else "no_coop",
                            "courses": [
                                {"code": "C101", "name_en": name, "credits": "3"}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(path)
            with self.assertRaisesRegex(ValueError, "conflicting name_en"):
                load_jsons_to_sqlite(inputs, root / "curriculum.db")

    def test_shared_course_credit_difference_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, credits in enumerate(("3", "6")):
                path = root / f"document-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "program": "DSBA",
                            "catalog": {
                                "catalog_key": "dsba-2560",
                                "academic_year": "2560",
                            },
                            "plan": "coop" if index == 0 else "no_coop",
                            "courses": [
                                {
                                    "code": "C101",
                                    "name_en": "COMPUTER PROGRAMMING",
                                    "credits": credits,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(path)
            with self.assertRaisesRegex(ValueError, "conflicting credits"):
                load_jsons_to_sqlite(inputs, root / "curriculum.db")

    def test_program_and_plan_identity_are_normalized_and_unique(self):
        document = {
            "program": "  TEST-PROGRAM  ",
            "plan": "  Regular  ",
            "courses": [{"code": "C100"}],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                identity = connection.execute(
                    """
                    SELECT programs.program_code,
                           programs.program_code_normalized,
                           curriculum_plans.program_id,
                           curriculum_plans.plan_code,
                           curriculum_plans.plan_key
                    FROM programs
                    JOIN curriculum_plans
                        ON curriculum_plans.program_id = programs.program_id
                    """
                ).fetchone()
                catalog_id, program_id = connection.execute(
                    "SELECT catalog_id, program_id FROM curriculum_plans"
                ).fetchone()

                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """
                        INSERT INTO programs (
                            catalog_id, program_code, program_code_normalized
                        ) VALUES (?, ?, ?)
                        """,
                        (catalog_id, "OTHER", "test-program"),
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """
                        INSERT INTO curriculum_plans (
                            catalog_id, program_id, program_code, plan_key
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (catalog_id, program_id, "TEST-PROGRAM", "regular"),
                    )

            self.assertEqual(
                identity,
                ("  TEST-PROGRAM  ", "test-program", program_id, "  Regular  ", "regular"),
            )

    def test_provenance_has_document_key_and_preserves_source_fields(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [{"code": "C100"}],
            "source_provenance": [
                {
                    "source_filename": "curriculum.pdf",
                    "source_page": 7,
                    "document_page": 3,
                    "document_category": "plan",
                    "source_uri": "file:///curriculum.pdf",
                    "source_locator": "page=7",
                    "excerpt": "C100",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                provenance = connection.execute(
                    """
                    SELECT source_document_key, source_filename, source_page,
                           document_page, document_category, source_uri,
                           source_locator, excerpt
                    FROM provenance
                    """
                ).fetchone()

            self.assertEqual(
                provenance,
                (
                    "curriculum.pdf",
                    "curriculum.pdf",
                    7,
                    3,
                    "plan",
                    "file:///curriculum.pdf",
                    "page=7",
                    "C100",
                ),
            )

    def test_production_provenance_without_source_identity_fails_clearly(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [{"code": "C100"}],
            "source_provenance": [
                {"source_page": 7, "document_category": "plan"}
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no usable source identity"):
                load_json_to_sqlite(input_path, database_path)

    def test_unknown_provenance_without_source_identity_is_retained(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [{"code": "C100"}],
            "source_provenance": [{"source_page": 7}],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                provenance = connection.execute(
                    "SELECT source_document_key, source_page FROM provenance"
                ).fetchone()

            self.assertEqual(provenance, ("unknown", 7))

    def test_repeated_course_code_uses_one_course_row_and_two_placements(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {"code": "C100", "year": 1, "semester": 1},
                {"code": " c100 ", "year": 2, "semester": 1},
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                course = connection.execute(
                    """
                    SELECT course_code, course_code_normalized
                    FROM courses
                    """
                ).fetchone()
                course_count = connection.execute(
                    "SELECT COUNT(*) FROM courses WHERE course_code = 'C100'"
                ).fetchone()
                placement_count = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM plan_placements
                    JOIN courses ON courses.course_id = plan_placements.course_id
                    WHERE courses.course_code = 'C100'
                    """
                ).fetchone()
                catalog_id = connection.execute(
                    "SELECT catalog_id FROM catalogs"
                ).fetchone()[0]
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """
                        INSERT INTO courses (catalog_id, course_code, course_code_normalized)
                        VALUES (?, ?, ?)
                        """,
                        (catalog_id, "C101", "c100"),
                    )

            self.assertEqual(course, ("C100", "c100"))
            self.assertEqual(course_count, (1,))
            self.assertEqual(placement_count, (2,))

    def test_prerequisite_resolves_to_repeated_course(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {"code": "C100", "year": 1, "semester": 1},
                {"code": "C100", "year": 2, "semester": 1},
                {"code": "C200", "prerequisite": "C100", "year": 2, "semester": 2},
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                prerequisite = connection.execute(
                    """
                    SELECT COUNT(*), target.course_code
                    FROM prerequisites AS prereq
                    JOIN courses AS course ON course.course_id = prereq.course_id
                    JOIN courses AS target
                        ON target.course_id = prereq.prerequisite_course_id
                    WHERE course.course_code = 'C200'
                    GROUP BY target.course_code
                    """
                ).fetchone()

            self.assertEqual(prerequisite, (1, "C100"))

    def test_alternative_course_members_remain_distinct(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [{"code": "C400 or C500", "type": "choose-one"}],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                alternatives = connection.execute(
                    """
                    SELECT courses.course_code
                    FROM alternative_course_group_members AS members
                    JOIN courses ON courses.course_id = members.course_id
                    ORDER BY members.member_order
                    """
                ).fetchall()

            self.assertEqual(alternatives, [("C400",), ("C500",)])

    def test_credit_units_preserve_raw_credit_values(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {"code": "C100", "credits": "3(2-2-5)"},
                {"code": "C200", "credits": "credit varies"},
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                credits = connection.execute(
                    """
                    SELECT course_code, credit_units, credits_raw
                    FROM courses
                    ORDER BY course_code
                    """
                ).fetchall()

            self.assertEqual(
                credits,
                [("C100", 3, "3(2-2-5)"), ("C200", None, "credit varies")],
            )

    def test_zero_year_and_semester_are_stored_as_null(self):
        document = {
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {
                    "code": "C000",
                    "name_th": "Description-only course",
                    "year": 0,
                    "semester": 0,
                    "flexible_year_semester": "4/1",
                    "notes": "Placement note",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                placement = connection.execute(
                    """
                    SELECT year_number, semester_number,
                           flexible_year_number, flexible_semester_number,
                           flexible_year_semester_raw, notes
                    FROM plan_placements
                    """
                ).fetchone()
            self.assertEqual(placement, (None, None, 4, 1, "4/1", "Placement note"))

    def test_loads_structured_records_without_mutating_source(self):
        document = {
            "source": "synthetic curriculum",
            "program": "TEST",
            "plan": "regular",
            "courses": [
                {
                    "code": "C100",
                    "name_th": "Normal course",
                    "name_en": "NORMAL COURSE",
                    "credits": "3(3-0-6)",
                    "category": "core",
                    "type": "required",
                    "year": 1,
                    "semester": 1,
                    "prerequisite": "ไม่มี",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-10.png",
                            "source_page": 10,
                            "document_category": "plan",
                        }
                    ],
                },
                {
                    "code": "C100",
                    "name_th": "Normal course",
                    "name_en": "NORMAL COURSE",
                    "credits": "3(3-0-6)",
                    "category": "core",
                    "type": "required",
                    "year": 2,
                    "semester": 1,
                    "prerequisite": "ไม่มี",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-11.png",
                            "source_page": 11,
                            "document_category": "plan",
                        }
                    ],
                },
                {
                    "code": "C200",
                    "name_th": "Advanced course",
                    "credits": "3(3-0-6)",
                    "year": 2,
                    "semester": 2,
                    "prerequisite": "C300",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-12.png",
                            "source_page": 12,
                            "document_category": "description",
                        }
                    ],
                },
                {
                    "code": "C300",
                    "name_th": "Prerequisite course",
                    "credits": "3(3-0-6)",
                    "year": 1,
                    "semester": 2,
                    "prerequisite": "ไม่มี",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-13.png",
                            "source_page": 13,
                            "document_category": "plan",
                        }
                    ],
                },
                {
                    "code": "C400 หรือ C500",
                    "name_th": "Choice one\nChoice two",
                    "name_en": "CHOICE ONE\nCHOICE TWO",
                    "credits": "3(3-0-6)",
                    "category": "elective",
                    "type": "choose-one",
                    "year": 3,
                    "semester": 1,
                    "prerequisite": "ไม่มี",
                    "source_provenance": [
                        {
                            "program": "TEST",
                            "source_filename": "page-14.png",
                            "source_page": 14,
                            "document_category": "plan",
                        }
                    ],
                },
            ],
        }
        original = json.loads(json.dumps(document))

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "curriculum.json"
            database_path = directory_path / "curriculum.db"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            load_json_to_sqlite(input_path, database_path)

            self.assertEqual(document, original)
            with closing(sqlite3.connect(database_path)) as connection:
                repeated = connection.execute(
                    """
                    SELECT COUNT(*), COUNT(DISTINCT placement_id)
                    FROM plan_placements
                    JOIN courses ON courses.course_id = plan_placements.course_id
                    WHERE courses.course_code = 'C100'
                    """
                ).fetchone()
                self.assertEqual(repeated, (2, 2))

                prerequisite = connection.execute(
                    """
                    SELECT target.course_code
                    FROM prerequisites AS prereq
                    JOIN courses AS course ON course.course_id = prereq.course_id
                    JOIN courses AS target
                        ON target.course_id = prereq.prerequisite_course_id
                    WHERE course.course_code = 'C200'
                    """
                ).fetchone()
                self.assertEqual(prerequisite, ("C300",))

                alternatives = connection.execute(
                    """
                    SELECT courses.course_code
                    FROM alternative_course_group_members AS members
                    JOIN courses ON courses.course_id = members.course_id
                    ORDER BY members.member_order
                    """
                ).fetchall()
                self.assertEqual(alternatives, [("C400",), ("C500",)])

                source_pages = connection.execute(
                    """
                    SELECT DISTINCT provenance.source_page
                    FROM plan_placement_provenance AS links
                    JOIN provenance
                        ON provenance.provenance_id = links.provenance_id
                    ORDER BY provenance.source_page
                    """
                ).fetchall()
                self.assertEqual(
                    source_pages,
                    [(10,), (11,), (12,), (13,), (14,)],
                )

    def test_loads_multiple_documents_into_one_database_with_views(self):
        documents = (
            {
                "program": "IT",
                "plan": "coop",
                "courses": [{"code": "C100", "year": 1, "semester": 1}],
                "source_provenance": [
                    {
                        "source_filename": "it-coop.pdf",
                        "source_page": 10,
                        "document_category": "plan",
                    }
                ],
            },
            {
                "program": "IT",
                "plan": "no_coop",
                "courses": [{"code": "C100", "year": 1, "semester": 1}],
                "source_provenance": [
                    {
                        "source_filename": "it-no-coop.pdf",
                        "source_page": 20,
                        "document_category": "plan",
                    }
                ],
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_paths = []
            for index, document in enumerate(documents):
                input_path = directory_path / f"curriculum-{index}.json"
                input_path.write_text(json.dumps(document), encoding="utf-8")
                input_paths.append(input_path)
            database_path = directory_path / "curriculum.db"

            catalog_ids = load_jsons_to_sqlite(input_paths, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                counts = connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM catalogs),
                        (SELECT COUNT(*) FROM courses),
                        (SELECT COUNT(*) FROM curriculum_plans),
                        (SELECT COUNT(*) FROM plan_placements)
                    """
                ).fetchone()
                view_rows = connection.execute(
                    """
                    SELECT program, plan, course_code
                    FROM v_plan_courses
                    ORDER BY plan
                    """
                ).fetchall()
                pages = connection.execute(
                    "SELECT source_page FROM provenance ORDER BY source_page"
                ).fetchall()

        self.assertEqual(len(catalog_ids), 2)
        self.assertEqual(counts, (2, 2, 2, 2))
        self.assertEqual(
            view_rows,
            [("IT", "coop", "C100"), ("IT", "no_coop", "C100")],
        )
        self.assertEqual(pages, [(10,), (20,)])

    def test_same_program_and_plan_are_isolated_across_catalog_editions(self):
        documents = [
            {
                "program": "DSBA",
                "catalog": {
                    "catalog_key": "dsba-2565-coop",
                    "academic_year": "2565",
                },
                "plan": {"plan_code": "coop", "version": "2565"},
                "source_provenance": [
                    {
                        "source_document_key": "dsba-2565",
                        "source_filename": "DSBA_2565.pdf",
                        "source_page": 10,
                        "document_category": "plan",
                    }
                ],
                "courses": [
                    {"code": "C100"},
                    {"code": "C101", "prerequisite": "C100"},
                ],
            },
            {
                "program": "DSBA",
                "catalog": {
                    "catalog_key": "dsba-2568-coop",
                    "academic_year": "2568",
                },
                "plan": {"plan_code": "coop", "version": "2568"},
                "source_provenance": [
                    {
                        "source_document_key": "dsba-2568",
                        "source_filename": "DSBA_2568.pdf",
                        "source_page": 10,
                        "document_category": "plan",
                    }
                ],
                "courses": [
                    {"code": "C100"},
                    {"code": "C102", "prerequisite": "C100"},
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_paths = []
            for index, document in enumerate(documents):
                input_path = directory_path / f"edition-{index}.json"
                input_path.write_text(json.dumps(document), encoding="utf-8")
                input_paths.append(input_path)
            database_path = directory_path / "curriculum.db"

            catalog_ids = load_jsons_to_sqlite(input_paths, database_path)

            with closing(sqlite3.connect(database_path)) as connection:
                catalogs = connection.execute(
                    """
                    SELECT catalog_id, catalog_key, academic_year
                    FROM catalogs
                    ORDER BY academic_year
                    """
                ).fetchall()
                plans = connection.execute(
                    """
                    SELECT cp.catalog_id, cp.program_code, cp.plan_key, cp.version
                    FROM curriculum_plans AS cp
                    JOIN catalogs AS c ON c.catalog_id = cp.catalog_id
                    ORDER BY c.academic_year
                    """
                ).fetchall()
                courses = connection.execute(
                    """
                    SELECT catalog_id, course_code, course_id
                    FROM courses
                    ORDER BY catalog_id, course_code
                    """
                ).fetchall()
                prerequisite_edges = connection.execute(
                    """
                    SELECT dependent.catalog_id, prerequisite.catalog_id,
                           dependent.course_id, prerequisite.course_id,
                           dependent.course_code, prerequisite.course_code
                    FROM prerequisites AS edge
                    JOIN courses AS dependent
                        ON dependent.course_id = edge.course_id
                    JOIN courses AS prerequisite
                        ON prerequisite.course_id = edge.prerequisite_course_id
                    ORDER BY dependent.catalog_id
                    """
                ).fetchall()
                course_sources = connection.execute(
                    """
                    SELECT course.catalog_id, provenance.source_document_key,
                           provenance.source_filename, provenance.source_page
                    FROM course_provenance AS link
                    JOIN courses AS course ON course.course_id = link.course_id
                    JOIN provenance ON provenance.provenance_id = link.provenance_id
                    WHERE course.course_code = 'C100'
                    ORDER BY course.catalog_id
                    """
                ).fetchall()
                plan_sources = connection.execute(
                    """
                    SELECT plan.catalog_id, provenance.source_document_key,
                           provenance.source_filename, provenance.source_page
                    FROM curriculum_plan_provenance AS link
                    JOIN curriculum_plans AS plan ON plan.plan_id = link.plan_id
                    JOIN provenance ON provenance.provenance_id = link.provenance_id
                    ORDER BY plan.catalog_id
                    """
                ).fetchall()
                same_page_sources = connection.execute(
                    """
                    SELECT source_document_key, source_filename, source_page
                    FROM provenance
                    WHERE source_page = 10
                    ORDER BY source_document_key
                    """
                ).fetchall()

        self.assertEqual(len(catalog_ids), 2)
        self.assertEqual(
            [(key, year) for _, key, year in catalogs],
            [("dsba-2565-coop", "2565"), ("dsba-2568-coop", "2568")],
        )
        self.assertEqual([catalog_id for catalog_id, _, _ in catalogs], catalog_ids)
        self.assertEqual(
            plans,
            [
                (catalog_ids[0], "DSBA", "coop", "2565"),
                (catalog_ids[1], "DSBA", "coop", "2568"),
            ],
        )

        course_rows = {
            (catalog_id, code): course_id
            for catalog_id, code, course_id in courses
        }
        self.assertEqual(sum(code == "C100" for _, code, _ in courses), 2)
        self.assertIn((catalog_ids[0], "C101"), course_rows)
        self.assertNotIn((catalog_ids[1], "C101"), course_rows)
        self.assertIn((catalog_ids[1], "C102"), course_rows)
        self.assertNotIn((catalog_ids[0], "C102"), course_rows)
        self.assertNotEqual(
            course_rows[(catalog_ids[0], "C100")],
            course_rows[(catalog_ids[1], "C100")],
        )
        self.assertEqual(
            prerequisite_edges,
            [
                (
                    catalog_ids[0], catalog_ids[0],
                    course_rows[(catalog_ids[0], "C101")],
                    course_rows[(catalog_ids[0], "C100")],
                    "C101", "C100",
                ),
                (
                    catalog_ids[1], catalog_ids[1],
                    course_rows[(catalog_ids[1], "C102")],
                    course_rows[(catalog_ids[1], "C100")],
                    "C102", "C100",
                ),
            ],
        )
        self.assertEqual(
            course_sources,
            [
                (catalog_ids[0], "dsba-2565", "DSBA_2565.pdf", 10),
                (catalog_ids[1], "dsba-2568", "DSBA_2568.pdf", 10),
            ],
        )
        self.assertEqual(
            plan_sources,
            [
                (catalog_ids[0], "dsba-2565", "DSBA_2565.pdf", 10),
                (catalog_ids[1], "dsba-2568", "DSBA_2568.pdf", 10),
            ],
        )
        self.assertEqual(
            same_page_sources,
            [
                ("dsba-2565", "DSBA_2565.pdf", 10),
                ("dsba-2568", "DSBA_2568.pdf", 10),
            ],
        )


if __name__ == "__main__":
    unittest.main()
