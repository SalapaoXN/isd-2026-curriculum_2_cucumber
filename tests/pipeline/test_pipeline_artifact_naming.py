import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.datasets import get_dataset_config
from src.pipeline.run import (
    AUTO_EVALUATION_DATASETS,
    _artifact_scope_stem,
    _publish_consolidated_artifacts,
    _publish_corrected_artifacts,
)


class PipelineArtifactNamingTests(unittest.TestCase):
    def test_current_dataset_requires_year_labelled_input_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "input" / "it").mkdir(parents=True)
            config = get_dataset_config("it2565")
            with self.assertRaisesRegex(FileNotFoundError, "year-labelled"):
                config.resolve_input_dir(root)

            expected = root / "data" / "input" / "it2565"
            expected.mkdir()
            self.assertEqual(config.resolve_input_dir(root), expected)

    def test_auto_evaluation_never_compares_legacy_editions_to_current_ground_truth(self):
        self.assertIn("it2565", AUTO_EVALUATION_DATASETS)
        self.assertIn("dsba2565", AUTO_EVALUATION_DATASETS)
        self.assertNotIn("it2560", AUTO_EVALUATION_DATASETS)
        self.assertNotIn("dsba2560", AUTO_EVALUATION_DATASETS)

    def test_scope_stem_omits_redundant_single_plan_labels(self):
        self.assertEqual(_artifact_scope_stem("ait2566", None), "ait2566")
        self.assertEqual(_artifact_scope_stem("gened2564", "gened"), "gened2564")
        self.assertEqual(
            _artifact_scope_stem("it2565", "no_coop"), "it2565_no_coop"
        )
        self.assertEqual(_artifact_scope_stem("it2565", "coop"), "it2565_coop")

    def test_consolidated_outputs_use_flat_readable_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            work = root / "work"
            output = root / "consolidated"
            for plan in ("no_coop", "coop"):
                full_dir = work / "it2565" / "it" / plan / "full"
                full_dir.mkdir(parents=True, exist_ok=True)
                (full_dir / f"merged_it2565_{plan}_full.json").write_text(
                    json.dumps({"program": "IT", "plan": plan, "courses": []}),
                    encoding="utf-8",
                )

            paths = _publish_consolidated_artifacts(work, output, "it2565")
            self.assertEqual(
                [path.name for path in paths],
                [
                    "it2565_coop_consolidated.json",
                    "it2565_no_coop_consolidated.json",
                ],
            )
            self.assertTrue(all(path.parent == output for path in paths))

    def test_final_data_and_correction_logs_are_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            temp_corrected = root / "temp"
            final_root = root / "final"
            corrections_root = root / "corrections"
            temp_corrected.mkdir()

            corrected_paths = []
            for plan in ("no_coop", "coop"):
                corrected = temp_corrected / f"working_{plan}_corrected.json"
                corrected.write_text(
                    json.dumps(
                        {
                            "program": "IT",
                            "plan": plan,
                            "catalog": {
                                "catalog_key": "it-2565",
                                "academic_year": "2565",
                            },
                            "courses": [],
                        }
                    ),
                    encoding="utf-8",
                )
                corrected.with_name(
                    corrected.name.replace("_corrected.json", "_corrections.json")
                ).write_text("[]", encoding="utf-8")
                corrected_paths.append(corrected)

            final_paths = _publish_corrected_artifacts(
                "it2565", corrected_paths, final_root, corrections_root
            )

            self.assertEqual(
                sorted(path.name for path in final_paths),
                ["it2565_coop_final.json", "it2565_no_coop_final.json"],
            )
            self.assertEqual(
                sorted(path.name for path in corrections_root.glob("*.json")),
                [
                    "it2565_coop_corrections.json",
                    "it2565_no_coop_corrections.json",
                ],
            )
            self.assertEqual(
                sorted(path.name for path in final_root.glob("*.json")),
                ["it2565_coop_final.json", "it2565_no_coop_final.json"],
            )


if __name__ == "__main__":
    unittest.main()
