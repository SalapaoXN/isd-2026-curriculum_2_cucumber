import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline import run_rules


class RulesPipelineTests(unittest.TestCase):
    def test_dry_run_manifest_has_thirteen_rule_and_four_program_sources(self):
        manifest = run_rules._source_manifest(
            Path("data/input/rule")
        )

        self.assertEqual(len(manifest["rule_sources"]), 13)
        self.assertEqual(manifest["rule_sources"][0], "rule_page_001.png")
        self.assertEqual(manifest["rule_sources"][-1], "rule_page_013.png")
        self.assertEqual(
            manifest["program_sources"],
            [
                "ait_page_005.png",
                "bit_page_006.png",
                "dsba_page_006.png",
                "it_page_006.png",
            ],
        )

    def test_output_paths_are_deterministic(self):
        paths = run_rules._output_paths(Path("data/output"))

        self.assertEqual(paths["rules_extracted"], Path("data/output/rules_extracted.json"))
        self.assertEqual(
            paths["institution_policy"],
            Path("data/output/final/institution_policy.json"),
        )
        self.assertEqual(
            paths["program_requirements"],
            Path("data/output/final/program_requirements.json"),
        )

    def test_real_run_sends_only_rule_pages_to_rule_extractor_and_four_programs_to_r2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_dir = root / "output"
            source_dir.mkdir()
            for page in range(1, 14):
                (source_dir / f"rule_page_{page:03d}.png").write_bytes(b"image")
            for filename in (
                "ait_page_005.png",
                "bit_page_006.png",
                "dsba_page_006.png",
                "it_page_006.png",
            ):
                (source_dir / filename).write_bytes(b"image")

            ocr_dir = output_dir / "ocr" / "rule"
            ocr_dir.mkdir(parents=True)
            for page in range(1, 14):
                (ocr_dir / f"rule_page_{page:03d}_ocr.json").write_text(
                    json.dumps({"text_lines": ["rule"]}), encoding="utf-8"
                )

            fake_rules = {"source": "Academic Rules", "total_rules": 0, "rules": []}
            fake_policy = {"categories": []}
            fake_program_requirements = [
                {"program": program, "value": value}
                for program, value in {
                    "AIT": 120,
                    "BIT": 126,
                    "DSBA": 132,
                    "IT": 129,
                }.items()
            ]

            with (
                patch("src.pipeline.tools.ocr.pipeline_runner.run_ocr") as ocr,
                patch.object(run_rules.RuleExtractor, "extract_from_files", return_value=fake_rules) as extract_rules,
                patch.object(run_rules.RulesPolicyMapper, "map_data", return_value=fake_policy),
                patch("src.pipeline.run_rules.extract_program_requirements", return_value=fake_program_requirements) as extract_programs,
            ):
                result = run_rules.run_rules(
                    source_dir=source_dir,
                    output_dir=output_dir,
                    no_gpu=True,
                )

            self.assertEqual(ocr.call_args.kwargs["pages"], list(range(1, 14)))
            self.assertEqual(
                [path.name for path in extract_rules.call_args.args[0]],
                [f"rule_page_{page:03d}_ocr.json" for page in range(1, 14)],
            )
            self.assertEqual(extract_programs.call_args.args[0], source_dir)
            self.assertEqual(result["program_requirement_count"], 4)
            self.assertTrue(Path(result["institution_policy"]).is_file())
            self.assertTrue(Path(result["program_requirements"]).is_file())

    def test_skip_ocr_uses_existing_canonical_intermediates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_dir = root / "output"
            source_dir.mkdir()
            for page in range(1, 14):
                (source_dir / f"rule_page_{page:03d}.png").write_bytes(b"image")
            for filename in (
                "ait_page_005.png",
                "bit_page_006.png",
                "dsba_page_006.png",
                "it_page_006.png",
            ):
                (source_dir / filename).write_bytes(b"image")
            output_dir.mkdir()
            (output_dir / "rules_extracted.json").write_text(
                json.dumps({"rules": []}), encoding="utf-8"
            )
            final_dir = output_dir / "final"
            final_dir.mkdir()
            (final_dir / "program_requirements.json").write_text(
                json.dumps([]), encoding="utf-8"
            )

            with patch("src.pipeline.tools.ocr.pipeline_runner.run_ocr") as ocr:
                result = run_rules.run_rules(
                    source_dir=source_dir,
                    output_dir=output_dir,
                    skip_ocr=True,
                )

            ocr.assert_not_called()
            self.assertEqual(result["program_requirement_count"], 0)
            self.assertTrue((final_dir / "institution_policy.json").is_file())


if __name__ == "__main__":
    unittest.main()
