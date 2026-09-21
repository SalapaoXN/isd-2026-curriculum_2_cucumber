"""Run the institution-rules and program-requirement pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.pipeline.tools.extraction.program_requirements import (
    PROGRAM_REQUIREMENT_SOURCES,
    extract_program_requirements,
)
from src.pipeline.tools.extraction.rules import RuleExtractor, discover_rule_ocr_files
from src.pipeline.tools.merge.policy import RulesPolicyMapper


BASE_DIR = Path(__file__).resolve().parents[2]
RULE_PAGE_NUMBERS = tuple(range(1, 14))


def _source_manifest(source_dir: Path) -> dict[str, Any]:
    rule_sources = [f"rule_page_{page:03d}.png" for page in RULE_PAGE_NUMBERS]
    program_sources = [filename for filename, _, _ in PROGRAM_REQUIREMENT_SOURCES.values()]
    missing = [name for name in (*rule_sources, *program_sources) if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing R3 source image(s): " + ", ".join(sorted(missing))
        )
    return {
        "source_dir": str(source_dir),
        "rule_sources": rule_sources,
        "program_sources": program_sources,
    }


def _output_paths(output_dir: Path) -> dict[str, Path]:
    final_dir = output_dir / "final"
    return {
        "rules_extracted": output_dir / "rules_extracted.json",
        "institution_policy": final_dir / "institution_policy.json",
        "program_requirements": final_dir / "program_requirements.json",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rule_ocr_files(output_dir: Path) -> list[Path]:
    ocr_dir = output_dir / "ocr" / "rule"
    files = discover_rule_ocr_files(ocr_dir)
    expected = {f"rule_page_{page:03d}_ocr.json" for page in RULE_PAGE_NUMBERS}
    selected = sorted(
        (path for path in files if path.name in expected),
        key=lambda path: path.name,
    )
    if {path.name for path in selected} != expected:
        missing = sorted(expected - {path.name for path in selected})
        raise FileNotFoundError(
            "Missing OCR output for rule page(s): " + ", ".join(missing)
        )
    return selected


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_rules(
    *,
    output_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    no_gpu: bool = False,
    skip_ocr: bool = False,
) -> dict[str, Any]:
    """Build canonical institution-policy and program-requirement outputs."""
    resolved_source_dir = (
        Path(source_dir) if source_dir is not None else BASE_DIR / "data" / "input" / "rule"
    )
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else BASE_DIR / "data" / "output"
    )
    _source_manifest(resolved_source_dir)
    paths = _output_paths(resolved_output_dir)

    if skip_ocr:
        if not paths["rules_extracted"].is_file():
            raise FileNotFoundError(
                "--skip-ocr requires an existing rules_extracted.json"
            )
        if not paths["program_requirements"].is_file():
            raise FileNotFoundError(
                "--skip-ocr requires an existing final/program_requirements.json"
            )
        extracted_rules = _load_json(paths["rules_extracted"])
        program_requirements = _load_json(paths["program_requirements"])
    else:
        from src.pipeline.tools.ocr.pipeline_runner import run_ocr

        run_ocr(
            input_dir=resolved_source_dir,
            output_dir=resolved_output_dir,
            program="RULE",
            pages=list(RULE_PAGE_NUMBERS),
            no_gpu=no_gpu,
        )
        rule_files = _rule_ocr_files(resolved_output_dir)
        extracted_rules = RuleExtractor().extract_from_files(rule_files)
        _write_json(paths["rules_extracted"], extracted_rules)
        program_requirements = extract_program_requirements(
            resolved_source_dir,
            gpu=not no_gpu,
        )
        _write_json(paths["program_requirements"], program_requirements)

    policy = RulesPolicyMapper().map_data(extracted_rules)
    _write_json(paths["institution_policy"], policy)
    return {
        "rules_extracted": str(paths["rules_extracted"]),
        "institution_policy": str(paths["institution_policy"]),
        "program_requirements": str(paths["program_requirements"]),
        "program_requirement_count": len(program_requirements),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build institution policy and program requirement outputs."
    )
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "data" / "output"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    source_dir = BASE_DIR / "data" / "input" / "rule"
    output_dir = Path(args.output_dir)

    try:
        manifest = _source_manifest(source_dir)
        if args.dry_run:
            payload = {
                **manifest,
                "output_paths": {
                    key: str(path) for key, path in _output_paths(output_dir).items()
                },
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        print(
            json.dumps(
                run_rules(
                    output_dir=output_dir,
                    source_dir=source_dir,
                    no_gpu=args.no_gpu,
                    skip_ocr=args.skip_ocr,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
