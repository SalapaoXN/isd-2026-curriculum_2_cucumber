import unittest
from pathlib import Path

from rag.structured.queries import course_placement


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


class CoursePlacementTest(unittest.TestCase):
    def test_it_fixed_and_flexible_placements_are_preserved(self):
        result = course_placement(
            DB_PATH, "IT", "06016481", ["coop", "no_coop"]
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["missing_plan_keys"], [])
        by_plan = {row["plan_key"]: row for row in result["placements"]}
        self.assertEqual(
            (by_plan["coop"]["course_id"], by_plan["coop"]["placement_id"]),
            (649, 669),
        )
        self.assertEqual(
            (by_plan["coop"]["year"], by_plan["coop"]["semester"]),
            (3, 2),
        )
        self.assertEqual(
            (
                by_plan["no_coop"]["course_id"],
                by_plan["no_coop"]["placement_id"],
            ),
            (815, 841),
        )
        self.assertIsNone(by_plan["no_coop"]["year"])
        self.assertIsNone(by_plan["no_coop"]["semester"])
        self.assertEqual(
            by_plan["no_coop"]["flexible_year_semester_raw"],
            "3/1, 3/2, 4/1",
        )
        required_fields = {
            "placement_id",
            "plan_key",
            "plan_id",
            "catalog_id",
            "course_id",
            "year",
            "semester",
            "flexible_year_semester_raw",
            "credits_raw",
            "alternative_group_id",
            "provenance",
        }
        self.assertTrue(required_fields.issubset(by_plan["coop"]))

    def test_bit_cross_plan_course_ids_remain_independent(self):
        result = course_placement(
            DB_PATH, "BIT", "06036103", ["coop", "no_coop"]
        )

        self.assertEqual(result["status"], "ok")
        by_plan = {row["plan_key"]: row for row in result["placements"]}
        self.assertEqual(by_plan["coop"]["course_id"], 68)
        self.assertEqual(by_plan["no_coop"]["course_id"], 129)
        self.assertEqual(
            (by_plan["coop"]["year"], by_plan["coop"]["semester"]),
            (2, 1),
        )
        self.assertEqual(
            (by_plan["no_coop"]["year"], by_plan["no_coop"]["semester"]),
            (2, 1),
        )

    def test_single_plan_does_not_leak_other_plans(self):
        result = course_placement(DB_PATH, "IT", "06016481", "coop")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            [row["plan_key"] for row in result["placements"]], ["coop"]
        )

    def test_full_miss_returns_no_data(self):
        result = course_placement(
            DB_PATH, "IT", "99999999", ["coop", "no_coop"]
        )

        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["placements"], [])
        self.assertEqual(result["missing_plan_keys"], ["coop", "no_coop"])

    def test_one_found_and_one_missing_returns_partial(self):
        result = course_placement(
            DB_PATH, "IT", "06016481", ["coop", "default"]
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["missing_plan_keys"], ["default"])
        self.assertEqual(
            {row["plan_key"] for row in result["placements"]}, {"coop"}
        )

    def test_invalid_course_code_and_plan_key_fail_before_database_access(self):
        missing_database = DB_PATH.with_name("does-not-exist.db")

        with self.assertRaisesRegex(ValueError, "8 ASCII digits"):
            course_placement(missing_database, "IT", "0601648", ["coop"])
        with self.assertRaisesRegex(ValueError, "canonical plan_key"):
            course_placement(missing_database, "IT", "06016481", ["regular"])
        with self.assertRaises(FileNotFoundError):
            course_placement(missing_database, "IT", "06016481", ["coop"])

    def test_multiple_provenance_rows_do_not_duplicate_placement(self):
        result = course_placement(DB_PATH, "IT", "06016481", ["coop"])

        self.assertEqual(len(result["placements"]), 1)
        provenance = result["placements"][0]["provenance"]
        provenance_ids = [row["provenance_id"] for row in provenance]
        self.assertGreater(len(provenance_ids), 1)
        self.assertEqual(len(provenance_ids), len(set(provenance_ids)))


if __name__ == "__main__":
    unittest.main()
