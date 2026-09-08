import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from extract import (
    _filter_files_by_pages,
    _filter_files_by_prefix,
    _parse_pages,
    main as extract_main,
)
from merge_consecutive import merge_consecutive_files
from src.run_pipeline import parse_pages
from src.pipeline_config import (
    discover_page_files,
    discover_pages,
    resolve_plan,
    resolve_program,
)

class CliSemanticsTests(unittest.TestCase):
    def test_page_parser_supports_ranges_and_rejects_invalid_values(self):
        self.assertEqual(parse_pages("10-12,12,14..15"), [10, 11, 12, 14, 15])

        with self.assertRaisesRegex(ValueError, "Invalid page value"):
            parse_pages("10,wrong")

    def test_page_discovery_is_direct_numeric_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "ait"
            input_dir.mkdir()
            (input_dir / "ait_page_010.png").write_bytes(b"")
            (input_dir / "ait_page_002.webp").write_bytes(b"")
            (input_dir / "ait_page_002.jpg").write_bytes(b"")
            (input_dir / "other_page_001.png").write_bytes(b"")
            nested = input_dir / "nested"
            nested.mkdir()
            (nested / "ait_page_001.png").write_bytes(b"")

            self.assertEqual(discover_pages(input_dir), [2, 10])
            self.assertEqual(discover_page_files(input_dir)[2].suffix, ".jpg")

    def test_page_discovery_fails_without_valid_direct_children(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "ait"
            input_dir.mkdir()
            (input_dir / "ait_page_01.png").write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "No valid page images"):
                discover_pages(input_dir)

    def test_program_is_derived_only_from_supported_directory_names(self):
        self.assertEqual(resolve_program(None, Path("inputs/ait")), "AIT")
        self.assertEqual(resolve_program("it", Path("inputs/custom")), "IT")

        with self.assertRaisesRegex(ValueError, "Cannot derive a program"):
            resolve_program(None, Path("inputs/custom"))

    def test_plan_resolution_does_not_guess_variants(self):
        self.assertEqual(resolve_plan("no-coop", "DSBA"), "no_coop")
        self.assertEqual(resolve_plan(None, "AIT"), None)
        self.assertEqual(resolve_plan("gened", "GENED"), "gened")

        with self.assertRaisesRegex(ValueError, "explicit --plan"):
            resolve_plan(None, "IT")
        with self.assertRaisesRegex(ValueError, "AIT has no study-plan"):
            resolve_plan("coop", "AIT")
        with self.assertRaisesRegex(ValueError, "GENED requires"):
            resolve_plan(None, "GENED")

    def test_ait_merge_keeps_null_plan_and_uses_neutral_filename_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "outputs"
            output_dir = Path(temp_dir) / "consolidated"
            input_dir.mkdir()
            (input_dir / "ait_page_023_ocr_extracted.json").write_text(
                json.dumps(
                    {
                        "program": "AIT",
                        "plan": None,
                        "courses": [{"code": "06000001"}],
                    }
                ),
                encoding="utf-8",
            )

            merge_consecutive_files(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                prefix="ait",
            )

            merged_files = list(output_dir.glob("*.json"))
            self.assertEqual(len(merged_files), 1)
            self.assertIn("no_plan", merged_files[0].name)
            result = json.loads(merged_files[0].read_text(encoding="utf-8"))
            self.assertIsNone(result["plan"])

    def test_extract_legacy_summary_uses_legacy_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "ait"
            input_dir.mkdir()
            for page in (23, 24):
                (input_dir / f"ait_page_{page:03d}_ocr.json").write_text(
                    "{}",
                    encoding="utf-8",
                )

            output_root = root / "extracted"
            args = SimpleNamespace(
                input_path=str(input_dir),
                output_dir=str(output_root),
                program="AIT",
                plan=None,
                prefix="ait",
                pages=None,
                source=None,
            )

            class FakeExtractor:
                def __init__(self, **_kwargs):
                    pass

                def process_file(self, _path):
                    return {"source": "test", "courses": []}

            with patch("extract.parse_arguments", return_value=args), patch(
                "extract.CurriculumExtractor", FakeExtractor
            ):
                extract_main()

            program_output = output_root / "ait"
            legacy_output = program_output / "legacy" / "consolidated_summaries"
            summaries = list(legacy_output.glob("consolidated_curriculum_*.json"))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(json.loads(summaries[0].read_text(encoding="utf-8"))["program"], "AIT")
            self.assertTrue(
                (program_output / "ait_page_023_ocr_extracted.json").is_file()
            )
    
    def test_extract_prefix_filters_only_matching_ocr_files(self):
        files = [
            Path("outputs/ocr/gened/gened_page_016_ocr.json"),
            Path("outputs/ocr/gened/gened_page_044_ocr.json"),
            Path("outputs/ocr/dsba/dsba_page_026_ocr.json"),
            Path("outputs/ocr/it/it_page_032_ocr.json"),
        ]

        result = _filter_files_by_prefix(files, "gened")

        self.assertEqual(
            [file.name for file in result],
            [
                "gened_page_016_ocr.json",
                "gened_page_044_ocr.json",
            ],
        )
        self.assertEqual(
            _filter_files_by_prefix(
                [Path("outputs/ocr/gened/GENED_page_016_ocr.json")],
                "gened",
            ),
            [Path("outputs/ocr/gened/GENED_page_016_ocr.json")],
        )
        
    def test_extract_pages_filters_requested_page_range(self):
        files = [
            Path("outputs/ocr/dsba/dsba_page_026_ocr.json"),
            Path("outputs/ocr/dsba/dsba_page_027_ocr.json"),
            Path("outputs/ocr/dsba/dsba_page_032_ocr.json"),
            Path("outputs/ocr/dsba/dsba_page_033_ocr.json"),
        ]

        pages = _parse_pages("26-27,32")

        result = _filter_files_by_pages(
            files,
            pages,
        )

        self.assertEqual(
            [file.name for file in result],
            [
                "dsba_page_026_ocr.json",
                "dsba_page_027_ocr.json",
                "dsba_page_032_ocr.json",
            ],
        )


if __name__ == "__main__":
    unittest.main()
