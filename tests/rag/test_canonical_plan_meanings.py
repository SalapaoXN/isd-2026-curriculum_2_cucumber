"""Bounded single-phrase normalization over the existing plan vocabulary."""

import unittest

from rag.query_spec import _extract_plans, canonical_plan_meanings


class CanonicalPlanMeaningsTests(unittest.TestCase):
    def test_supported_thai_plan_phrases(self):
        cases = {
            "แบบสหกิจ": ("coop",),
            "แผนสหกิจ": ("coop",),
            "สหกิจ": ("coop",),
            "แบบไม่สหกิจ": ("no_coop",),
            "แผนไม่สหกิจ": ("no_coop",),
            "ไม่สหกิจ": ("no_coop",),
            "แผนปกติ": ("no_coop",),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(canonical_plan_meanings(phrase), expected)

    def test_canonical_and_existing_english_variants(self):
        cases = {
            "coop": ("coop",),
            "no_coop": ("no_coop",),
            "no coop": ("no_coop",),
            "no-coop": ("no_coop",),
            "non-coop": ("no_coop",),
            "default": ("default",),
            "gened": ("gened",),
            "ไม่coop": ("no_coop",),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(canonical_plan_meanings(phrase), expected)

    def test_unsupported_ambiguous_and_contradictory_phrases_are_untrusted(self):
        for phrase in (
            "cooperative learning route",
            "แบบทั่วไป",
            "สหกิจ และไม่สหกิจ",
            "สหกิจ / no_coop",
            "not coop",
            "coop แผน",
            "",
            None,
            42,
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(canonical_plan_meanings(phrase), ())

    def test_legacy_whole_question_plan_extraction_is_unchanged(self):
        question = "IT แบบไม่สหกิจ ปี 3 เทอม 1 มีวิชาอะไรบ้าง"
        self.assertEqual(_extract_plans(question), ("no_coop",))
        # The phrase-scoped safety API is stricter; legacy extraction retains
        # its historic substring behavior for compatibility.
        self.assertEqual(_extract_plans("not coop"), ("coop",))
        self.assertEqual(canonical_plan_meanings("not coop"), ())


if __name__ == "__main__":
    unittest.main()
