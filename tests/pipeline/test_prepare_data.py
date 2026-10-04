import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline.tools.extraction.tool import _filter_files_by_prefix, run_extraction
from src.pipeline.tools.merge.consolidator import merge_consecutive_files
from src.pipeline.tools.preparation import tool as prepare_data


def _touch_pages(directory: Path, pages: range, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for page in pages:
        (directory / f"{prefix}_page_{page:03d}_ocr.json").write_text(
            "{}", encoding="utf-8"
        )


class PrepareDataTests(unittest.TestCase):
    def test_explicit_scope_configuration_matches_current_pipeline(self):
        self.assertEqual(prepare_data.PROGRAM_CONFIG["it"].scopes[0].pages, "32-38")
        self.assertEqual(prepare_data.PROGRAM_CONFIG["it"].scopes[1].pages, "39-45")
        self.assertEqual(
            prepare_data.PROGRAM_CONFIG["it"].shared_description.pages,
            "328-371",
        )
        self.assertEqual(prepare_data.PROGRAM_CONFIG["gened"].scopes[0].pages, "16-30,44-117")

    def test_legacy_and_explicit_edition_dataset_configurations(self):
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
        self.assertEqual(gened.prefix, "gened2557")
        self.assertEqual(gened.scopes[0].plan, "gened")
        self.assertEqual(gened.scopes[0].pages, "11-18,47-92")
        self.assertEqual(gened.scopes[0].description_pages, "47-92")
        self.assertEqual(gened.catalog_key, "gened-2557")
        self.assertEqual(gened.academic_year, "2557")
        self.assertEqual(prepare_data.PROGRAM_CONFIG["dsba"].program, "DSBA")
        self.assertIsNone(prepare_data.PROGRAM_CONFIG["dsba"].dataset_key)

    def test_dataset_discovery_and_prefixes_are_exact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ocr_root = root / "ocr"
            for key in ("dsba", "dsba2560", "gened", "gened2557", "dsba2565"):
                (ocr_root / key).mkdir(parents=True)

            supported, unknown = prepare_data.discover_supported_corpora(ocr_root)

            self.assertEqual(
                [key for key, _ in supported],
                ["dsba", "dsba2560", "gened", "gened2557"],
            )
            self.assertEqual(unknown, ["dsba2565"])

        files = [
            Path("dsba_page_025_ocr.json"),
            Path("dsba2560_page_025_ocr.json"),
            Path("dsba2565_page_025_ocr.json"),
        ]
        self.assertEqual(_filter_files_by_prefix(files, "dsba"), [files[0]])
        self.assertEqual(_filter_files_by_prefix(files, "dsba2560"), [files[1]])
        with tempfile.TemporaryDirectory() as temp_dir:
            ocr_dir = Path(temp_dir)
            for file in files:
                (ocr_dir / file.name).write_text("{}", encoding="utf-8")
            self.assertEqual(
                [path.name for path in prepare_data._ocr_files(ocr_dir, "dsba")],
                ["dsba_page_025_ocr.json"],
            )
            self.assertEqual(
                [path.name for path in prepare_data._ocr_files(ocr_dir, "dsba2560")],
                ["dsba2560_page_025_ocr.json"],
            )

    def test_extraction_reads_only_dataset_files_and_keeps_logical_program(self):
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
            extracted = json.loads(
                (output / "dsba2560_page_025_ocr_extracted.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(extracted["program"], "DSBA")
            self.assertFalse((root / "extracted" / "dsba").exists())

    def test_merge_prefix_excludes_canonical_files_from_edition_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            extracted = root / "extracted" / "dsba2560"
            extracted.mkdir(parents=True)
            for filename, code, plan in (
                ("dsba2560_page_025_ocr_extracted.json", "C2560", "no_coop"),
                ("dsba2560_page_026_ocr_extracted.json", "CDESC", "coop"),
                ("dsba_page_025_ocr_extracted.json", "CCURRENT", "no_coop"),
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
            self.assertEqual(merged["program"], "DSBA")
            merged_codes = {course["code"] for course in merged["courses"]}
            self.assertEqual(merged_codes, {"C2560", "CDESC"})
            self.assertNotIn("CCURRENT", merged_codes)
            self.assertIn("edition-dsba-2560", full_files[0].name)

    def test_selected_editions_build_isolated_full_artifacts_with_catalog_metadata(self):
        merge_calls = []

        def write_extracted_pages(_root, _ocr_dir, output_root, config, scope):
            output = output_root / (config.dataset_key or config.program.casefold())
            output.mkdir(parents=True, exist_ok=True)
            for page in prepare_data._configured_pages(scope.pages):
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

        def write_full_artifact(extracted_dir, output_dir, config, scope, pages, metadata):
            merge_calls.append(
                (config.dataset_key, scope.plan, pages, extracted_dir, output_dir)
            )
            page_set = prepare_data._configured_pages(pages)
            description_set = prepare_data._configured_pages(scope.description_pages)
            plan_pages = page_set - (description_set or set())
            codes = sorted(
                [*(f"C{page:03d}" for page in plan_pages),
                 *(f"C{page:03d}" for page in page_set & (description_set or set()))]
            )
            from src.pipeline.tools.merge.consolidator import edition_filename_token

            token = edition_filename_token(metadata["catalog_key"])
            plan_label = scope.plan
            full_dir = output_dir / config.program.casefold()
            if config.program not in {"GENED", "AIT"}:
                full_dir /= plan_label
            full_dir /= "full"
            full_dir.mkdir(parents=True, exist_ok=True)
            (full_dir / f"merged_{config.prefix}_{plan_label}_{token}_full.json").write_text(
                json.dumps(
                    {
                        "program": config.program,
                        "plan": scope.plan,
                        "courses": [{"code": code} for code in codes],
                    }
                ),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ocr_root = root / "data" / "output" / "ocr"
            _touch_pages(
                ocr_root / "dsba2560",
                [1, *range(17, 35), *range(175, 208)],
                "dsba2560",
            )
            _touch_pages(
                ocr_root / "gened2557",
                [3, *range(6, 19), *range(47, 93)],
                "gened2557",
            )

            with patch(
                "src.pipeline.tools.preparation.tool._run_extract",
                side_effect=write_extracted_pages,
            ), patch(
                "src.pipeline.tools.preparation.tool._run_merge",
                side_effect=write_full_artifact,
            ):
                result = prepare_data.prepare_data(
                    root,
                    dataset_keys=["dsba2560", "gened2557"],
                )

            self.assertEqual(result["prepared_programs"], ["dsba2560", "gened2557"])
            dsba_calls = [call for call in merge_calls if call[0] == "dsba2560"]
            gened_calls = [call for call in merge_calls if call[0] == "gened2557"]
            self.assertEqual(
                [(plan, pages) for _, plan, pages, _, _ in dsba_calls],
                [
                    ("no_coop", "25-29,175-207"),
                    ("coop", "30-34,175-207"),
                ],
            )
            self.assertEqual(
                [(plan, pages) for _, plan, pages, _, _ in gened_calls],
                [("gened", "11-18,47-92")],
            )
            self.assertTrue(
                all(
                    call[3] == (root / "data/output/extracted/dsba2560").resolve()
                    for call in dsba_calls
                )
            )
            self.assertTrue(
                all(
                    call[4] == (root / "data/output/consolidated/dsba2560").resolve()
                    for call in dsba_calls
                )
            )
            self.assertTrue(
                (root / "data/output/extracted/dsba2560").is_dir()
            )
            self.assertTrue(
                (root / "data/output/extracted/gened2557").is_dir()
            )
            dsba_full = sorted(
                (root / "data/output/consolidated/dsba2560").glob(
                    "**/full/*_full.json"
                )
            )
            gened_full = sorted(
                (root / "data/output/consolidated/gened2557").glob(
                    "**/full/*_full.json"
                )
            )
            self.assertEqual(len(dsba_full), 2)
            self.assertEqual(len(gened_full), 1)
            self.assertFalse(set(dsba_full) & set(gened_full))

            documents = {
                path: json.loads(path.read_text(encoding="utf-8"))
                for path in [*dsba_full, *gened_full]
            }
            self.assertEqual(
                {documents[path]["catalog"]["catalog_key"] for path in dsba_full},
                {"dsba-2560"},
            )
            self.assertEqual(
                {documents[path]["catalog"]["academic_year"] for path in dsba_full},
                {"2560"},
            )
            self.assertEqual(
                {documents[path]["catalog"]["catalog_key"] for path in gened_full},
                {"gened-2557"},
            )
            self.assertEqual(
                {documents[path]["catalog"]["academic_year"] for path in gened_full},
                {"2557"},
            )
            self.assertEqual({documents[path]["program"] for path in dsba_full}, {"DSBA"})
            self.assertEqual({documents[path]["program"] for path in gened_full}, {"GENED"})

            no_coop = next(path for path in dsba_full if "/no_coop/" in path.as_posix())
            coop = next(path for path in dsba_full if "/coop/" in path.as_posix())
            no_coop_codes = {course["code"] for course in documents[no_coop]["courses"]}
            coop_codes = {course["code"] for course in documents[coop]["courses"]}
            self.assertTrue({f"C{page:03d}" for page in range(25, 30)}.issubset(no_coop_codes))
            self.assertFalse({f"C{page:03d}" for page in range(30, 35)} & no_coop_codes)
            self.assertTrue({f"C{page:03d}" for page in range(30, 35)}.issubset(coop_codes))
            self.assertFalse({f"C{page:03d}" for page in range(25, 30)} & coop_codes)
            shared_codes = {f"C{page:03d}" for page in range(175, 208)}
            self.assertTrue(shared_codes.issubset(no_coop_codes))
            self.assertTrue(shared_codes.issubset(coop_codes))

            dsba_extracted_names = {
                path.name
                for path in (root / "data/output/extracted/dsba2560").glob("**/*_extracted.json")
            }
            gened_extracted_names = {
                path.name
                for path in (root / "data/output/extracted/gened2557").glob("**/*_extracted.json")
            }
            structural_pages = {1, *range(17, 25)}
            gened_context_pages = {3, *range(6, 11)}
            self.assertFalse(
                any(f"page_{page:03d}_" in name for page in structural_pages for name in dsba_extracted_names)
            )
            self.assertFalse(
                any(f"page_{page:03d}_" in name for page in gened_context_pages for name in gened_extracted_names)
            )

    def test_partial_corpus_skips_missing_programs_and_unknown_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ocr = root / "data" / "output" / "ocr"
            _touch_pages(ocr / "it", range(32, 39), "it")
            (ocr / "bit").mkdir(parents=True)
            (ocr / "custom").mkdir()

            with patch("src.pipeline.tools.preparation.tool._run_extract") as run_extract, patch(
                "src.pipeline.tools.preparation.tool._run_merge"
            ) as run_merge:
                result = prepare_data.prepare_data(root)

            self.assertEqual(result["prepared_programs"], ["it"])
            self.assertIn("bit", result["skipped_programs"])
            self.assertEqual(result["unknown_directories"], ["custom"])
            self.assertEqual(run_extract.call_count, 1)
            self.assertEqual(run_merge.call_count, 1)

    def test_shared_description_range_is_extracted_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ocr = root / "data" / "output" / "ocr" / "it"
            _touch_pages(ocr, range(32, 39), "it")
            _touch_pages(ocr, range(328, 372), "it")

            with patch("src.pipeline.tools.preparation.tool._run_extract") as run_extract, patch(
                "src.pipeline.tools.preparation.tool._run_merge"
            ) as run_merge:
                prepare_data.prepare_data(root)

            self.assertEqual(run_extract.call_count, 2)
            extracted_scopes = [call.args[4] for call in run_extract.call_args_list]
            self.assertEqual(
                [scope.pages for scope in extracted_scopes], ["32-38", "328-371"]
            )
            self.assertEqual(run_merge.call_count, 1)
            self.assertEqual(run_merge.call_args.args[4], "32-38,328-371")

    def test_missing_scope_pages_are_omitted_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch_pages(root / "data" / "output" / "ocr" / "it", range(39, 46), "it")

            with patch("src.pipeline.tools.preparation.tool._run_extract") as run_extract, patch(
                "src.pipeline.tools.preparation.tool._run_merge"
            ) as run_merge:
                result = prepare_data.prepare_data(root)

            self.assertEqual(result["prepared_programs"], ["it"])
            self.assertEqual(run_extract.call_count, 1)
            self.assertEqual(run_extract.call_args.args[4].plan, "coop")
            self.assertEqual(run_merge.call_count, 1)

    def test_empty_or_unknown_only_corpus_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "output" / "ocr" / "it").mkdir(parents=True)
            (root / "data" / "output" / "ocr" / "other").mkdir()

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

    def test_only_existing_stage_commands_are_orchestrated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch_pages(root / "data" / "output" / "ocr" / "ait", range(1, 2), "ait")

            with patch("src.pipeline.tools.preparation.tool._run_extract") as run_extract, patch(
                "src.pipeline.tools.preparation.tool._run_merge"
            ) as run_merge:
                prepare_data.prepare_data(root)

            for call in run_extract.call_args_list:
                self.assertNotIn("run_pipeline", repr(call))
                self.assertNotIn("llm_spell_corrector", repr(call))
                self.assertNotIn("src.pipeline.tools.evaluation.evaluate.py", repr(call))
                self.assertNotIn("rag", repr(call))
            self.assertEqual(run_extract.call_count, 1)
            self.assertEqual(run_merge.call_count, 1)


if __name__ == "__main__":
    unittest.main()
