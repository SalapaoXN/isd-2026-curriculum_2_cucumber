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


if __name__ == "__main__":
    unittest.main()
