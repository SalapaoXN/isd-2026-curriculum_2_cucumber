import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline.datasets import DATASET_CONFIG, get_dataset_config
from src.pipeline.tools.extraction.tool import _filter_files_by_prefix, run_extraction
from src.pipeline.tools.merge.consolidator import edition_filename_token, merge_consecutive_files
from src.pipeline.tools.preparation import tool as prepare_data


def _touch_pages(directory: Path, pages, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for page in pages:
        (directory / f"{prefix}_page_{page:03d}_ocr.json").write_text(
            "{}", encoding="utf-8"
        )


class PrepareDataTests(unittest.TestCase):
    def test_current_dataset_configuration_is_year_explicit(self):
        it = prepare_data.PROGRAM_CONFIG["it2565"]
        self.assertEqual(it.program, "IT")
        self.assertEqual(it.catalog_key, "it-2565")
        self.assertEqual(it.academic_year, "2565")
        self.assertEqual([scope.pages for scope in it.scopes], ["32-38", "39-45"])
        self.assertEqual(it.shared_description.pages, "328-371")

        ait = prepare_data.PROGRAM_CONFIG["ait2566"]
        self.assertEqual(ait.program, "AIT")
        self.assertEqual(ait.catalog_key, "ait-2566")
        self.assertEqual(ait.scopes[0].pages, "23-26")
        self.assertEqual(ait.shared_description.pages, "287-302")

        gened = prepare_data.PROGRAM_CONFIG["gened2564"]
        self.assertEqual(gened.scopes[0].pages, "16-30")
        self.assertEqual(gened.shared_description.pages, "44-117")

    def test_legacy_dataset_configuration_remains_explicit(self):
        dsba = prepare_data.PROGRAM_CONFIG["dsba2560"]
        self.assertEqual(dsba.program, "DSBA")
        self.assertEqual(dsba.dataset_key, "dsba2560")
        self.assertEqual(dsba.prefix, "dsba2560")
        self.assertEqual([scope.plan for scope in dsba.scopes], ["no_coop", "coop"])
        self.assertEqual([scope.pages for scope in dsba.scopes], ["25-29", "30-34"])
        self.assertEqual(
            [scope.description_pages for scope in dsba.scopes],
            ["175-207", "175-207"],
        )
        self.assertEqual(dsba.shared_description.pages, "175-207")
        self.assertEqual(dsba.catalog_key, "dsba-2560")
        self.assertEqual(dsba.academic_year, "2560")

        gened = prepare_data.PROGRAM_CONFIG["gened2557"]
        self.assertEqual(gened.program, "GENED")
        self.assertEqual(gened.dataset_key, "gened2557")
        self.assertEqual(gened.scopes[0].plan, "gened")
        self.assertEqual(gened.scopes[0].pages, "11-18")
        self.assertEqual(gened.shared_description.pages, "47-92")
        self.assertEqual(gened.catalog_key, "gened-2557")
        self.assertEqual(gened.academic_year, "2557")

    def test_registry_contains_all_supported_project_datasets(self):
        self.assertEqual(
            set(DATASET_CONFIG),
            {
                "ait2566",
                "bit2565",
                "bit2560",
                "dsba2565",
                "dsba2560",
                "gened2564",
                "gened2557",
                "it2565",
                "it2560",
            },
        )
        self.assertEqual(get_dataset_config("it").key, "it2565")
        self.assertEqual(get_dataset_config("dsba").key, "dsba2565")

    def test_dataset_discovery_prefers_explicit_year_directory_over_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ocr_root = root / "ocr"
            for key in (
                "dsba",
                "dsba2560",
                "dsba2565",
                "gened",
                "gened2557",
                "custom",
            ):
                (ocr_root / key).mkdir(parents=True)

            supported, unknown = prepare_data.discover_supported_corpora(ocr_root)

            self.assertEqual(
                [key for key, _ in supported],
                ["dsba2560", "dsba2565", "gened2557", "gened2564"],
            )
            selected = dict(supported)
            self.assertEqual(selected["dsba2565"].name, "dsba2565")
            self.assertEqual(selected["gened2564"].name, "gened")
            self.assertEqual(unknown, ["custom"])

    def test_extraction_reads_only_exact_dataset_prefix(self):
        class FakeExtractor:
            seen = []

            def __init__(self, **_kwargs):
                pass

            def process_file(self, path):
                self.seen.append(path.name)
                return {"program": "DSBA", "plan": "no_coop", "courses": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ocr_dir = root / "ocr" / "dsba2560"
            ocr_dir.mkdir(parents=True)
            for filename in (
                "dsba2560_page_025_ocr.json",
                "dsba_page_025_ocr.json",
                "dsba2565_page_025_ocr.json",
            ):
                (ocr_dir / filename).write_text("{}", encoding="utf-8")

            with patch(
                "src.pipeline.tools.extraction.tool.CurriculumExtractor", FakeExtractor
            ):
                output = run_extraction(
                    ocr_dir,
                    root / "extracted",
                    program="DSBA",
                    plan="no_coop",
                    prefix="dsba2560",
                    pages="25-29",
                    dataset_key="dsba2560",
                )

            self.assertEqual(output, root / "extracted" / "dsba2560")
            self.assertEqual(FakeExtractor.seen, ["dsba2560_page_025_ocr.json"])

        files = [
            Path("dsba_page_025_ocr.json"),
            Path("dsba2560_page_025_ocr.json"),
            Path("dsba2565_page_025_ocr.json"),
        ]
        self.assertEqual(_filter_files_by_prefix(files, "dsba2560"), [files[1]])

    def test_merge_prefix_keeps_editions_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            extracted = root / "extracted" / "dsba2560"
            extracted.mkdir(parents=True)
            for filename, code, plan in (
                ("dsba2560_page_025_ocr_extracted.json", "C2560", "no_coop"),
                ("dsba2560_page_026_ocr_extracted.json", "CDESC", "coop"),
                ("dsba2565_page_025_ocr_extracted.json", "CCURRENT", "no_coop"),
            ):
                (extracted / filename).write_text(
                    json.dumps(
                        {
                            "program": "DSBA",
                            "plan": plan,
                            "courses": [{"code": code}],
                        }
                    ),
                    encoding="utf-8",
                )

            merge_consecutive_files(
                input_dir=str(extracted),
                output_dir=str(root / "consolidated" / "dsba2560"),
                plan_filter="no_coop",
                pages="25-26",
                prefix="dsba2560",
                desc_pages="26",
                catalog_key="dsba-2560",
            )

            full_files = list(
                (root / "consolidated" / "dsba2560").glob("**/full/*.json")
            )
            self.assertEqual(len(full_files), 1)
            merged = json.loads(full_files[0].read_text(encoding="utf-8"))
            self.assertEqual(
                {course["code"] for course in merged["courses"]},
                {"C2560", "CDESC"},
            )
            self.assertIn("edition-dsba-2560", full_files[0].name)

    def test_required_pages_are_derived_from_dataset_configuration(self):
        dsba = get_dataset_config("dsba2560")
        expected = set(range(25, 35)) | set(range(175, 208))
        self.assertEqual(set(dsba.required_pages()), expected)

        it = get_dataset_config("it2565")
        expected = set(range(32, 46)) | set(range(328, 372))
        self.assertEqual(set(it.required_pages()), expected)

    def test_selected_dataset_preparation_uses_configured_plan_and_description_ranges(self):
        merge_calls = []

        def fake_extract(_root, _ocr_dir, output_root, config, scope):
            output = output_root / config.dataset_key
            output.mkdir(parents=True, exist_ok=True)
            pages = prepare_data._configured_pages(scope.pages) or set()
            for page in pages:
                (output / f"{config.prefix}_page_{page:03d}_ocr_extracted.json").write_text(
                    json.dumps(
                        {
                            "program": config.program,
                            "plan": scope.plan,
                            "courses": [{"code": f"C{page:03d}"}],
                        }
                    ),
                    encoding="utf-8",
                )

        def fake_merge(extracted_dir, output_dir, config, scope, pages, metadata):
            merge_calls.append((config.dataset_key, scope.plan, pages))
            token = edition_filename_token(metadata["catalog_key"])
            full_dir = output_dir / config.program.casefold()
            if config.program not in {"GENED", "AIT"}:
                full_dir /= scope.plan
            full_dir /= "full"
            full_dir.mkdir(parents=True, exist_ok=True)
            (full_dir / f"merged_{config.prefix}_{scope.plan}_{token}_full.json").write_text(
                json.dumps(
                    {
                        "program": config.program,
                        "plan": scope.plan,
                        "courses": [],
                    }
                ),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ocr = root / "data" / "output" / "ocr" / "dsba2560"
            _touch_pages(
                ocr,
                [*range(25, 35), *range(175, 208)],
                "dsba2560",
            )

            with patch(
                "src.pipeline.tools.preparation.tool._run_extract",
                side_effect=fake_extract,
            ), patch(
                "src.pipeline.tools.preparation.tool._run_merge",
                side_effect=fake_merge,
            ):
                result = prepare_data.prepare_data(
                    root,
                    dataset_keys=["dsba2560"],
                )

            self.assertEqual(result["prepared_programs"], ["dsba2560"])
            self.assertEqual(
                merge_calls,
                [
                    ("dsba2560", "no_coop", "25-29,175-207"),
                    ("dsba2560", "coop", "30-34,175-207"),
                ],
            )
            full_files = sorted(
                (root / "data/output/consolidated/dsba2560").glob(
                    "**/full/*_full.json"
                )
            )
            self.assertEqual(len(full_files), 2)
            for path in full_files:
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(document["catalog"]["catalog_key"], "dsba-2560")
                self.assertEqual(document["catalog"]["academic_year"], "2560")

    def test_unknown_only_corpus_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "output" / "ocr" / "other").mkdir(parents=True)
            with self.assertRaisesRegex(
                prepare_data.PreparationError, "No usable supported OCR source"
            ):
                prepare_data.prepare_data(root)

        with patch.object(
            prepare_data,
            "prepare_data",
            side_effect=prepare_data.PreparationError("No usable supported OCR source"),
        ):
            self.assertEqual(prepare_data.main([]), 1)


if __name__ == "__main__":
    unittest.main()
