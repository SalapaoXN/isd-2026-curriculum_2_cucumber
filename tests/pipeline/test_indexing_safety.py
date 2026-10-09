import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.pipeline import run as pipeline_run
from src.pipeline.tools.indexing import tool as indexing_tool


class IndexingSafetyTests(unittest.TestCase):
    @staticmethod
    def _write_source(directory: Path, program: str) -> Path:
        path = directory / f"{program.lower()}_corrected.json"
        path.write_text(json.dumps({"program": program, "courses": []}), encoding="utf-8")
        return path

    @staticmethod
    def _write_edition_source(
        directory: Path,
        filename: str,
        *,
        program: str = "DSBA",
        plan: str = "coop",
        catalog_key: str,
    ) -> Path:
        path = directory / filename
        path.write_text(
            json.dumps(
                {
                    "program": program,
                    "plan": plan,
                    "catalog": {"catalog_key": catalog_key},
                    "courses": [],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_expected_edition_resolves_only_matching_catalog_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_edition_source(
                root, "dsba-2565_corrected.json", catalog_key="dsba-2565-coop"
            )
            new = self._write_edition_source(
                root, "dsba-2568_corrected.json", catalog_key="dsba-2568-coop"
            )
            selected = indexing_tool._resolve_expected_edition_sources(
                [old, new],
                [{"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565-coop"}],
            )

        self.assertEqual(selected, [old])

    def test_new_edition_cannot_satisfy_missing_old_edition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            new = self._write_edition_source(
                root, "dsba-2568_corrected.json", catalog_key="dsba-2568-coop"
            )
            with self.assertRaisesRegex(ValueError, "missing edition.*dsba-2565-coop"):
                indexing_tool._resolve_expected_edition_sources(
                    [new],
                    [{"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565-coop"}],
                )

    def test_both_expected_editions_resolve_and_report_exact_missing_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_edition_source(
                root, "old_corrected.json", catalog_key="dsba-2565-coop"
            )
            new = self._write_edition_source(
                root, "new_corrected.json", catalog_key="dsba-2568-coop"
            )
            expected = [
                {"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565-coop"},
                {"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2568-coop"},
            ]
            self.assertEqual(
                indexing_tool._resolve_expected_edition_sources([old, new], expected),
                [old, new],
            )
            with self.assertRaisesRegex(ValueError, "missing edition.*dsba-2568-coop"):
                indexing_tool._resolve_expected_edition_sources([old], expected)

    def test_duplicate_artifacts_for_expected_edition_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            one = self._write_edition_source(
                root, "one_corrected.json", catalog_key="dsba-2565-coop"
            )
            two = self._write_edition_source(
                root, "two_corrected.json", catalog_key="dsba-2565-coop"
            )
            with self.assertRaisesRegex(ValueError, "ambiguous edition.*dsba-2565-coop"):
                indexing_tool._resolve_expected_edition_sources(
                    [one, two],
                    [{"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565-coop"}],
                )

    def test_same_program_and_plan_do_not_collapse_across_expected_editions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_edition_source(
                root, "old_corrected.json", catalog_key="dsba-2565-coop"
            )
            new = self._write_edition_source(
                root, "new_corrected.json", catalog_key="dsba-2568-coop"
            )
            selected = indexing_tool._resolve_expected_edition_sources(
                [old, new],
                [
                    {"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565-coop"},
                    {"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2568-coop"},
                ],
            )

        self.assertEqual(selected, [old, new])

    def test_json_catalog_identity_is_authoritative_over_filename_year(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            misleading = self._write_edition_source(
                root, "dsba_2568_corrected.json", catalog_key="dsba-2565-coop"
            )
            selected = indexing_tool._resolve_expected_edition_sources(
                [misleading],
                [{"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2565-coop"}],
            )
            with self.assertRaisesRegex(ValueError, "missing edition.*dsba-2568-coop"):
                indexing_tool._resolve_expected_edition_sources(
                    [misleading],
                    [{"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2568-coop"}],
                )

        self.assertEqual(selected, [misleading])

    def test_optional_plan_filter_selects_only_requested_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            coop = self._write_edition_source(
                root,
                "coop_corrected.json",
                plan="coop",
                catalog_key="dsba-2568",
            )
            no_coop = self._write_edition_source(
                root,
                "no_coop_corrected.json",
                plan="no_coop",
                catalog_key="dsba-2568",
            )
            selected = indexing_tool._resolve_expected_edition_sources(
                [coop, no_coop],
                [{"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2568"}],
            )

        self.assertEqual(selected, [coop])

    def test_explicit_edition_index_uses_resolved_artifact_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_edition_source(
                root, "old_corrected.json", catalog_key="dsba-2565-coop"
            )
            new = self._write_edition_source(
                root, "new_corrected.json", catalog_key="dsba-2568-coop"
            )
            scoped_index = root / "scoped.db"
            expected = [{"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2568-coop"}]
            with patch.object(indexing_tool, "build_index", return_value=scoped_index) as build:
                result = indexing_tool.run_build_index_stage(
                    [old, new], scoped_index, expected_editions=expected
                )

        self.assertEqual(result, scoped_index)
        build.assert_called_once_with([new], scoped_index)

    def test_shared_default_discovery_reads_canonical_artifacts_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical_dir = root / "canonical"
            canonical_dir.mkdir()
            old = self._write_edition_source(
                canonical_dir, "old_final.json", catalog_key="dsba-2565-coop"
            )
            new = self._write_edition_source(
                canonical_dir, "new_final.json", catalog_key="dsba-2568-coop"
            )
            raw_final = root / "dsba2560_coop_final.json"
            raw_final.write_text("{}", encoding="utf-8")
            with patch("rag.build_index.CANONICAL_DIR", canonical_dir):
                sources = indexing_tool._default_sources()

        self.assertEqual(sources, [new, old])

    def test_shared_runtime_validates_expected_editions_but_indexes_all_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [
                self._write_source(root, program)
                for program in ("AIT", "BIT", "GENED", "IT")
            ]
            old = self._write_edition_source(
                root, "old_corrected.json", catalog_key="dsba-2565-coop"
            )
            new = self._write_edition_source(
                root, "new_corrected.json", catalog_key="dsba-2568-coop"
            )
            sources.extend((old, new))
            artifact_dir = root / "cucumber_outputs" / "runtime"
            expected_path = artifact_dir / indexing_tool.DEFAULT_INDEX_NAME
            expected = [{"program": "DSBA", "plan": "coop", "catalog_key": "dsba-2568-coop"}]
            with patch.object(indexing_tool, "ARTIFACTS_DIR", artifact_dir):
                with patch.object(indexing_tool, "build_index", return_value=expected_path) as build:
                    result = indexing_tool.run_build_index_stage(
                        sources, expected_editions=expected
                    )

        self.assertEqual(result, expected_path)
        build.assert_called_once_with(sources, None)

    def test_shared_runtime_rejects_scoped_sources_before_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source(root, "IT")
            artifact_dir = root / "cucumber_outputs" / "runtime"
            artifact_dir.mkdir(parents=True)
            database = artifact_dir / indexing_tool.DEFAULT_INDEX_NAME
            sentinel = database.with_suffix(".h1-test-sentinel")
            sentinel.write_text("must remain untouched", encoding="utf-8")
            try:
                with patch.object(indexing_tool, "ARTIFACTS_DIR", artifact_dir):
                    with patch.object(indexing_tool, "build_index") as build:
                        with self.assertRaisesRegex(ValueError, "incomplete source set"):
                            indexing_tool.run_build_index_stage([source])
                build.assert_not_called()
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "must remain untouched")
            finally:
                sentinel.unlink(missing_ok=True)

    def test_complete_source_set_is_accepted_for_shared_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [self._write_source(root, program) for program in indexing_tool._UNIFIED_PROGRAMS]
            artifact_dir = root / "cucumber_outputs" / "runtime"
            expected_path = artifact_dir / indexing_tool.DEFAULT_INDEX_NAME
            with patch.object(indexing_tool, "ARTIFACTS_DIR", artifact_dir):
                with patch.object(indexing_tool, "build_index", return_value=expected_path) as build:
                    result = indexing_tool.run_build_index_stage(sources)

            self.assertEqual(result, expected_path)
            build.assert_called_once_with(sources, None)

    def test_shared_runtime_validation_preserves_iterable_sources_for_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [self._write_source(root, program) for program in indexing_tool._UNIFIED_PROGRAMS]
            artifact_dir = root / "cucumber_outputs" / "runtime"
            expected_path = artifact_dir / indexing_tool.DEFAULT_INDEX_NAME
            with patch.object(indexing_tool, "ARTIFACTS_DIR", artifact_dir):
                with patch.object(indexing_tool, "build_index", return_value=expected_path) as build:
                    result = indexing_tool.run_build_index_stage(iter(sources))

            self.assertEqual(result, expected_path)
            build.assert_called_once_with(sources, None)

    def test_scoped_source_set_requires_a_distinct_index_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source(root, "IT")
            artifact_dir = root / "cucumber_outputs" / "runtime"
            scoped_index = artifact_dir / "h1-scoped-test.db"
            with patch.object(indexing_tool, "ARTIFACTS_DIR", artifact_dir):
                with patch.object(indexing_tool, "build_index", return_value=scoped_index) as build:
                    result = indexing_tool.run_build_index_stage([source], scoped_index)

            self.assertEqual(result, scoped_index)
            build.assert_called_once_with([source], scoped_index)

    def test_only_index_default_target_uses_unified_source_discovery(self):
        with patch(
            "src.pipeline.tools.indexing.tool.run_build_index_stage"
        ) as build:
            result = pipeline_run.main(["--dataset", "it2565", "--only-index"])

        self.assertEqual(result, 0)
        build.assert_called_once_with(None, None)

    def test_with_index_runs_evaluation_canonicalization_preflight_and_build_in_order(self):
        events = []
        final_path = Path("data/output/final/it2565_coop_final.json")
        pair = (final_path, Path("ground_truth/IT/IT_academic_plan_coop.json"))
        canonicalization = SimpleNamespace(
            paths=(Path("data/output/canonical/it2565_coop_final.json"),),
            plan_gt_fields_applied=1,
            shared_general_education_fields_applied=1,
            legacy_corrections_applied=0,
        )

        with (
            patch.object(pipeline_run, "_corrected_paths_for_dataset", return_value=[final_path]),
            patch.object(pipeline_run, "_consolidated_paths_for_dataset", return_value=[]),
            patch(
                "src.pipeline.tools.evaluation.evaluate.discover_llm_evaluation_pairs",
                side_effect=lambda *_: events.append("discover") or [pair],
            ),
            patch(
                "src.pipeline.tools.evaluation.tool.run_evaluate_stage",
                side_effect=lambda **_: events.append("evaluate"),
            ),
            patch(
                "src.pipeline.tools.runtime_artifacts.canonicalize_runtime_artifacts",
                side_effect=lambda: events.append("canonicalize") or canonicalization,
            ),
            patch(
                "src.pipeline.tools.runtime_artifacts.preflight_runtime_artifacts",
                side_effect=lambda: events.append("preflight") or [],
            ),
            patch(
                "src.pipeline.tools.indexing.tool.run_build_index_stage",
                side_effect=lambda *args: events.append("build") or Path("curriculum.db"),
            ) as build,
        ):
            result = pipeline_run.main(
                ["--dataset", "it2565", "--from", "final", "--with-index"]
            )

        self.assertEqual(result, 0)
        self.assertEqual(events, ["discover", "evaluate", "canonicalize", "preflight", "build"])
        build.assert_called_once_with(None, None)

    def test_with_index_cannot_skip_evaluation(self):
        with self.assertRaises(SystemExit):
            pipeline_run.parse_args(
                ["--dataset", "it2565", "--with-index", "--skip-eval"]
            )


if __name__ == "__main__":
    unittest.main()
