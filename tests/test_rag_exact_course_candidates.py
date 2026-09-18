import unittest
from pathlib import Path

from rag.structured.queries import exact_course_candidates


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


class ExactCourseCandidatesTest(unittest.TestCase):
    def test_exact_code_returns_one_logical_candidate(self):
        candidates = exact_course_candidates(DB_PATH, course_code="06016420")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["program"], "IT")
        self.assertEqual(candidates[0]["course_code"], "06016420")
        self.assertEqual(
            set(candidates[0]),
            {
                "course_id",
                "catalog_id",
                "program",
                "course_code",
                "name_th",
                "name_en",
                "name_th_variants",
                "name_en_variants",
                "provenance",
            },
        )

    def test_unknown_and_partial_codes_do_not_match(self):
        self.assertEqual(
            exact_course_candidates(DB_PATH, course_code="99999999"), []
        )
        self.assertEqual(
            exact_course_candidates(DB_PATH, course_code="0601642"), []
        )

    def test_explicit_program_scopes_exact_code_lookup(self):
        self.assertEqual(
            exact_course_candidates(
                DB_PATH, course_code="06016420", program="it"
            )[0]["program"],
            "IT",
        )
        self.assertEqual(
            exact_course_candidates(
                DB_PATH, course_code="06016420", program="DSBA"
            ),
            [],
        )

    def test_plan_duplicates_are_not_separate_candidates(self):
        candidates = exact_course_candidates(DB_PATH, course_code="06016420")

        self.assertEqual(
            [
                (candidate["program"], candidate["course_code"])
                for candidate in candidates
            ],
            [("IT", "06016420")],
        )

    def test_name_equality_case_and_whitespace_normalization(self):
        for reference in (
            "INFRASTRUCTURE SYSTEMS AND SERVICES",
            "  infrastructure   systems and services  ",
        ):
            with self.subTest(reference=reference):
                candidates = exact_course_candidates(DB_PATH, course_name=reference)
                self.assertEqual(
                    {
                        (item["program"], item["course_code"])
                        for item in candidates
                    },
                    {("IT", "06016420")},
                )

    def test_whole_token_phrase_containment_and_cross_program_name_candidates(self):
        candidates = exact_course_candidates(DB_PATH, course_name="NOSQL")

        self.assertEqual(
            {(item["program"], item["course_code"]) for item in candidates},
            {("IT", "06016414"), ("DSBA", "06026207")},
        )

        self.assertEqual(
            exact_course_candidates(DB_PATH, course_name="SQL"),
            [],
        )

    def test_collapsed_identity_preserves_name_variants_and_provenance(self):
        candidates = exact_course_candidates(DB_PATH, course_code="06026200")

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual((candidate["program"], candidate["course_code"]), ("DSBA", "06026200"))
        self.assertIsNone(candidate["name_th"])
        self.assertEqual(
            set(candidate["name_th_variants"]),
            {"ไม่ระบุ 1", "แคลคูลัส 1"},
        )
        self.assertEqual(candidate["name_en"], "CALCULUS 1")
        self.assertEqual(candidate["name_en_variants"], ["CALCULUS 1"])
        self.assertTrue(candidate["provenance"])

    def test_default_name_lookup_keeps_token_subsequence_match(self):
        candidates = exact_course_candidates(DB_PATH, course_name="PROJECT 1")

        self.assertEqual(
            {(item["program"], item["course_code"]) for item in candidates},
            {("IT", "06016406"), ("AIT", "90641004")},
        )

    def test_strict_title_lookup_skips_token_subsequence_match(self):
        self.assertEqual(
            {
                (item["program"], item["course_code"])
                for item in exact_course_candidates(
                    DB_PATH, course_name="PROJECT 1", exact_title=True
                )
            },
            {("IT", "06016406")},
        )
        self.assertEqual(
            {
                (item["program"], item["course_code"])
                for item in exact_course_candidates(
                    DB_PATH, course_name="TEAM-PROJECT 1", exact_title=True
                )
            },
            {("AIT", "90641004")},
        )

    def test_strict_title_lookup_returns_every_exact_program(self):
        self.assertEqual(
            {
                (item["program"], item["course_code"])
                for item in exact_course_candidates(
                    DB_PATH, course_name="CHARM SCHOOL", exact_title=True
                )
            },
            {
                ("BIT", "96641001"),
                ("DSBA", "90641001"),
                ("GENED", "90641001"),
                ("IT", "90641001"),
            },
        )
        self.assertEqual(
            {
                (item["program"], item["course_code"])
                for item in exact_course_candidates(
                    DB_PATH, course_name="CALCULUS 1", exact_title=True
                )
            },
            {("AIT", "06046400"), ("DSBA", "06026200")},
        )

    def test_strict_title_lookup_preserves_collapsing_and_provenance(self):
        candidates = exact_course_candidates(
            DB_PATH, course_name="CALCULUS 1", exact_title=True
        )

        self.assertEqual(len(candidates), 2)
        candidate = next(
            item for item in candidates if item["course_code"] == "06026200"
        )
        self.assertEqual(candidate["program"], "DSBA")
        self.assertIsNone(candidate["name_th"])
        self.assertEqual(candidate["name_en"], "CALCULUS 1")
        self.assertTrue(candidate["provenance"])

    def test_strict_mode_leaves_code_lookup_unchanged(self):
        for program in (None, "IT"):
            with self.subTest(program=program):
                kwargs = {"program": program} if program else {}
                self.assertEqual(
                    exact_course_candidates(
                        DB_PATH, course_code="06016420", exact_title=True, **kwargs
                    ),
                    exact_course_candidates(
                        DB_PATH, course_code="06016420", **kwargs
                    ),
                )


if __name__ == "__main__":
    unittest.main()
