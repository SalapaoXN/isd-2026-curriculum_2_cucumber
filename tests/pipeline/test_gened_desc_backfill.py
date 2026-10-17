import tempfile
import unittest
from pathlib import Path

from src.pipeline.tools.merge.consolidator import (
    _gened_final_filename,
    backfill_gened_descriptions,
    load_gened_desc_lookup,
)


def _provenance(program="GENED", filename="gened_page_044.png", page=44):
    return [
        {
            "program": program,
            "source_filename": filename,
            "source_page": page,
            "document_category": "description",
        }
    ]


def _gened_course(code="90000001", desc_th="คำอธิบาย", desc_en="DESCRIPTION"):
    return {
        "code": code,
        "desc_th": desc_th,
        "desc_en": desc_en,
        "source_provenance": _provenance(),
    }


def _plan_course(code="90000001"):
    return {
        "code": code,
        "name_th": "ชื่อวิชา",
        "source_provenance": [
            {
                "program": "IT",
                "source_filename": "it_page_032.png",
                "source_page": 32,
                "document_category": "plan",
            }
        ],
    }


class GenEraMappingTests(unittest.TestCase):
    def test_current_era_maps_to_gened2564(self):
        for catalog in ("it-2565", "bit-2565", "dsba-2565", "ait-2566"):
            self.assertEqual(
                _gened_final_filename(catalog), "gened2564_final.json"
            )

    def test_legacy_era_maps_to_gened2557(self):
        for catalog in ("it-2560", "bit-2560", "dsba-2560"):
            self.assertEqual(
                _gened_final_filename(catalog), "gened2557_final.json"
            )

    def test_gened_itself_and_unknown_need_no_backfill(self):
        for catalog in (None, "", "gened-2564", "gened-2557", "unknown-9999"):
            self.assertIsNone(_gened_final_filename(catalog))

    def test_missing_gened_file_raises_loudly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                load_gened_desc_lookup("it-2565", base_dir=temp_dir)

    def test_duplicate_gened_codes_are_excluded_from_lookup(self):
        import json

        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = Path(temp_dir) / "data" / "output" / "final"
            final_dir.mkdir(parents=True)
            (final_dir / "gened2564_final.json").write_text(
                json.dumps(
                    {
                        "courses": [
                            _gened_course("90000001", "หนึ่ง", "ONE"),
                            _gened_course("90000001", "หนึ่งซ้ำ", "ONE DUP"),
                            _gened_course("90000002", "สอง", "TWO"),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            lookup = load_gened_desc_lookup("it-2565", base_dir=temp_dir)

        self.assertNotIn("90000001", lookup)
        self.assertIn("90000002", lookup)


class BackfillBehaviorTests(unittest.TestCase):
    def test_empty_desc_is_filled_with_provenance(self):
        course = _plan_course()
        filled = backfill_gened_descriptions(
            [course], {"90000001": _gened_course()}
        )

        self.assertEqual(filled, 1)
        self.assertEqual(course["desc_th"], "คำอธิบาย")
        self.assertEqual(course["desc_en"], "DESCRIPTION")
        programs = {
            entry["program"] for entry in course["source_provenance"]
        }
        self.assertEqual(programs, {"IT", "GENED"})

    def test_existing_desc_is_preserved_while_missing_language_is_filled(self):
        course = _plan_course()
        course["desc_th"] = "ของเดิม"
        filled = backfill_gened_descriptions(
            [course], {"90000001": _gened_course()}
        )

        self.assertEqual(filled, 1)
        self.assertEqual(course["desc_th"], "ของเดิม")
        self.assertEqual(course["desc_en"], "DESCRIPTION")

    def test_missing_language_is_filled_independently_with_provenance(self):
        course = _plan_course()
        course["desc_th"] = "ของเดิม"
        filled = backfill_gened_descriptions(
            [course], {"90000001": _gened_course()}
        )

        self.assertEqual(filled, 1)
        self.assertEqual(course["desc_th"], "ของเดิม")
        self.assertEqual(course["desc_en"], "DESCRIPTION")
        programs = {entry["program"] for entry in course["source_provenance"]}
        self.assertEqual(programs, {"IT", "GENED"})

    def test_wildcard_alternative_and_unknown_codes_are_skipped(self):
        courses = [
            _plan_course("9000xxxx"),
            _plan_course("90000001 หรือ 90000002"),
            _plan_course("90000099"),
        ]
        filled = backfill_gened_descriptions(
            courses, {"90000001": _gened_course()}
        )

        self.assertEqual(filled, 0)
        for course in courses:
            self.assertNotIn("desc_th", course)
            self.assertNotIn("desc_en", course)

    def test_empty_lookup_is_a_noop(self):
        course = _plan_course()
        self.assertEqual(backfill_gened_descriptions([course], None), 0)
        self.assertEqual(backfill_gened_descriptions([course], {}), 0)
        self.assertNotIn("desc_th", course)


if __name__ == "__main__":
    unittest.main()
