import copy
import unittest
from pathlib import Path

from src.english_name_enricher import enrich_courses


def box(x: float, y: float, width: float = 100, height: float = 30):
    return [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ]


def detection(text: str, x: float, y: float, confidence: float = 0.9):
    return [box(x, y), text, confidence]


class FakeEnglishEngine:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def extract_text(self, image_path, detail=1):
        self.calls.append((Path(image_path), detail))
        if self.error is not None:
            raise self.error
        return self.results


def course(code: str, name: str, marker: str):
    return {
        "code": code,
        "name_en": name,
        "name_th": f"TH-{marker}",
        "credits": "3(3-0-6)",
        "prerequisite": "NONE",
        "description": f"DESC-{marker}",
        "category": "CATEGORY",
        "year": 1,
        "semester": 1,
        "marker": marker,
    }


def data_with_courses(courses):
    return {"program": "DSBA", "plan": "coop", "courses": courses}


class EnglishNameEnricherTests(unittest.TestCase):
    def test_exact_code_occurrence_matching_and_duplicate_preservation(self):
        first = course("06000001", "OLD ONE", "first")
        second = course("06000001", "OLD TWO", "second")
        data = data_with_courses([first, second])
        results = [
            detection("06000001", 0, 0),
            detection("3(3-0-6)", 120, 0),
            detection("FIRST TITLE", 0, 40),
            detection("06000001", 0, 100),
            detection("3(3-0-6)", 120, 100),
            detection("SECOND TITLE", 0, 140),
        ]

        enrich_courses(data, "dsba_page_001.png", FakeEnglishEngine(results))

        self.assertEqual(first["name_en"], "FIRST TITLE")
        self.assertEqual(second["name_en"], "SECOND TITLE")
        self.assertEqual(first["english_second_pass"]["page"], 1)
        self.assertEqual(second["english_second_pass"]["page"], 1)

    def test_preceding_credit_is_supported(self):
        record = course("06000002", "OLD", "preceding")
        data = data_with_courses([record])
        results = [
            detection("3(3-0-6)", 0, 0),
            detection("06000002", 120, 0),
            detection("PRECEDING CREDIT TITLE", 0, 40),
        ]

        enrich_courses(data, "dsba_page_002.png", FakeEnglishEngine(results))

        self.assertEqual(record["name_en"], "PRECEDING CREDIT TITLE")

    def test_placeholder_boundary_and_non_exact_code_fallback(self):
        accepted = course("06000003", "OLD", "placeholder")
        placeholder = course("06026XXX", "PLACEHOLDER", "placeholder-row")
        data = data_with_courses([accepted, placeholder])
        results = [
            detection("06000003", 0, 0),
            detection("3(3-0-6)", 120, 0),
            detection("SAFE TITLE", 0, 40),
            detection("06026XXX", 0, 90),
            detection("ELECTIVE NOISE", 0, 130),
        ]

        enrich_courses(data, "dsba_page_003.png", FakeEnglishEngine(results))

        self.assertEqual(accepted["name_en"], "SAFE TITLE")
        self.assertEqual(placeholder["name_en"], "PLACEHOLDER")
        self.assertEqual(
            placeholder["english_second_pass"]["fallback_reason"],
            "non_exact_or_composite_code",
        )

    def test_code_prefixed_row_boundary_stops_before_trailing_text(self):
        record = course("06000004", "OLD", "prefix")
        data = data_with_courses([record])
        results = [
            detection("06000004", 0, 0),
            detection("3(3-0-6)", 120, 0),
            detection("SAFE TITLE", 0, 40),
            detection("06066300 UWANASUUZU?AYA", 0, 90, 0.55),
        ]

        enrich_courses(data, "dsba_page_004.png", FakeEnglishEngine(results))

        self.assertEqual(record["name_en"], "SAFE TITLE")

    def test_single_band_acceptance_allows_same_line_title_tokens(self):
        record = course("06000005", "OLD", "band")
        data = data_with_courses([record])
        results = [
            detection("06000005", 0, 0),
            detection("3(3-0-6)", 120, 0),
            detection("LONG TITLE", 0, 40),
            detection("2", 180, 42, 0.8, ),
        ]

        enrich_courses(data, "dsba_page_005.png", FakeEnglishEngine(results))

        self.assertEqual(record["name_en"], "LONG TITLE 2")

    def test_trailing_noise_is_rejected_by_single_band_gate(self):
        record = course("06000006", "CANONICAL", "noise")
        data = data_with_courses([record])
        results = [
            detection("GOOD TITLE", 0, 40),
        ]
        results.insert(0, detection("06000006", 0, 0))
        results.insert(1, detection("3(3-0-6)", 120, 0))
        results.append(detection("YAI3", 0, 90, 0.55))

        enrich_courses(data, "dsba_page_006.png", FakeEnglishEngine(results))

        self.assertEqual(record["name_en"], "CANONICAL")
        self.assertEqual(
            record["english_second_pass"]["fallback_reason"],
            "unsafe_title_band",
        )
        self.assertEqual(
            record["english_second_pass"]["candidate_name_en"],
            "GOOD TITLE YAI3",
        )

    def test_composite_code_falls_back_and_canonical_fields_are_preserved(self):
        record = course("06000007 หรือ 06000008", "CANONICAL", "composite")
        original = copy.deepcopy(record)
        data = data_with_courses([record])
        results = [
            detection("06000007", 0, 0),
            detection("3(3-0-6)", 120, 0),
            detection("SHOULD NOT APPLY", 0, 40),
        ]

        enrich_courses(data, "dsba_page_007.png", FakeEnglishEngine(results))

        self.assertEqual(record["name_en"], original["name_en"])
        for key, value in original.items():
            self.assertEqual(record[key], value)
        self.assertEqual(
            record["english_second_pass"]["fallback_reason"],
            "non_exact_or_composite_code",
        )

    def test_auxiliary_failure_preserves_canonical_records(self):
        record = course("06000009", "CANONICAL", "failure")
        original = copy.deepcopy(record)
        data = data_with_courses([record])

        enrich_courses(
            data,
            "dsba_page_008.png",
            FakeEnglishEngine(error=RuntimeError("OCR failed")),
        )

        self.assertEqual(record["name_en"], original["name_en"])
        for key, value in original.items():
            self.assertEqual(record[key], value)
        self.assertEqual(
            record["english_second_pass"]["association_status"],
            "fallback_auxiliary_ocr_failure",
        )


if __name__ == "__main__":
    unittest.main()
