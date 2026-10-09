"""Single end-to-end curriculum preprocessing entry point.

Normal usage selects one explicit dataset key and the pipeline resolves all
bundled page/plan/edition metadata automatically:

    python -m src.pipeline.run --dataset it2560
    python -m src.pipeline.run --dataset it2565

Stages:
    OCR -> Extract -> Merge -> Gemini name correction -> (Evaluate) -> (Index)

The public CLI intentionally does not accept manual page, plan, output-folder,
or edition flags for normal bundled datasets. Those values live in the central
``src.pipeline.datasets`` registry so one dataset key is enough.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from src.pipeline.datasets import (
    DATASET_ALIASES,
    DATASET_CONFIG,
    get_dataset_config,
    resolve_dataset_key,
)
from src.pipeline.tools.merge.consolidator import edition_filename_token


BASE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_DATASETS = tuple(DATASET_CONFIG)
# Compatibility name for older imports/tests. Values now represent dataset keys.
SUPPORTED_PROGRAMS = tuple(dict.fromkeys((*SUPPORTED_DATASETS, *DATASET_ALIASES)))
FROM_CHOICES = ("ocr", "extracted", "consolidated", "corrected")
_ACADEMIC_YEAR_RE = re.compile(r"^[1-9]\d{3}(?:[-/]\d{2,4})?$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR -> extract -> merge -> correct for one configured curriculum dataset.",
        epilog="Example: python -m src.pipeline.run --dataset it2560",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        help="Explicit dataset edition, for example it2560 or dsba2565.",
    )
    # Backward compatibility: old commands such as --program it still work,
    # but the README uses --dataset and explicit year-labelled keys.
    selector.add_argument(
        "--program",
        choices=SUPPORTED_PROGRAMS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force EasyOCR to CPU mode.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after correction.")
    parser.add_argument("--with-index", action="store_true", help="Build the unified runtime DB after correction.")
    parser.add_argument("--only-index", action="store_true", help="Build only the runtime DB from reviewed final artifacts.")
    parser.add_argument("--from", dest="from_stage", choices=FROM_CHOICES, default="ocr")
    parser.add_argument("--keep-intermediates", action="store_true", help="Persist extracted debug artifacts.")
    parser.add_argument("--index-path", default=None, help=argparse.SUPPRESS)
    # Retained only for compatibility with focused metadata tests/advanced
    # internal use. Bundled datasets always provide these values automatically.
    parser.add_argument("--catalog-key", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--academic-year", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    requested = args.dataset or args.program or "it2565"
    try:
        args.dataset = resolve_dataset_key(requested)
    except ValueError as error:
        parser.error(str(error))
    # Some older tests/callers inspect args.program. Keep it as the resolved
    # dataset key rather than the ambiguous unversioned program label.
    args.program = args.dataset
    return args


def _edition_metadata_from_args(args: argparse.Namespace) -> dict[str, str] | None:
    """Validate an optional explicit metadata override (advanced compatibility)."""
    catalog_key = getattr(args, "catalog_key", None)
    academic_year = getattr(args, "academic_year", None)
    if catalog_key is None and academic_year is None:
        return None
    if not isinstance(catalog_key, str) or not catalog_key.strip():
        raise ValueError("catalog_key and academic_year must be supplied together")
    if catalog_key != catalog_key.strip():
        raise ValueError("catalog_key must not have leading or trailing whitespace")
    if isinstance(academic_year, bool) or not isinstance(academic_year, (str, int)):
        raise ValueError("academic_year must be a 4-digit year or explicit year range")
    year_text = str(academic_year)
    if not _ACADEMIC_YEAR_RE.fullmatch(year_text):
        raise ValueError("academic_year must be a 4-digit year or explicit year range")
    return {"catalog_key": catalog_key, "academic_year": year_text}


def _apply_edition_metadata(
    paths: list[str | Path], metadata: dict[str, str] | None
) -> list[Path]:
    """Stamp canonical catalog metadata and preserve edition-safe filenames."""
    if metadata is None:
        return [Path(path) for path in paths]
    if not paths:
        raise ValueError("no curriculum artifacts found for edition metadata")

    token = edition_filename_token(metadata["catalog_key"])
    updates: list[tuple[Path, Path, dict]] = []
    destinations: set[Path] = set()
    for path_value in paths:
        path = Path(path_value)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot apply edition metadata to {path}") from error
        if not isinstance(document, dict):
            raise ValueError(f"curriculum artifact must be a JSON object: {path}")
        catalog = document.get("catalog") or {}
        if not isinstance(catalog, dict):
            raise ValueError(f"curriculum artifact catalog metadata must be an object: {path}")
        existing_key = catalog.get("catalog_key")
        if existing_key is not None and existing_key != metadata["catalog_key"]:
            raise ValueError(
                f"curriculum artifact belongs to catalog {existing_key!r}, "
                f"not {metadata['catalog_key']!r}: {path}"
            )
        document["catalog"] = {**catalog, **metadata}
        destination = path
        marker = f"_{token}_"
        if marker not in f"_{path.stem}_":
            if path.stem.endswith("_corrected"):
                base = path.stem[: -len("_corrected")]
                destination = path.with_name(f"{base}_{token}_corrected{path.suffix}")
            else:
                destination = path.with_name(f"{path.stem}_{token}{path.suffix}")
        resolved_destination = destination.resolve()
        if resolved_destination in destinations:
            raise ValueError(f"duplicate edition artifact identity: {destination}")
        destinations.add(resolved_destination)
        if destination != path and destination.exists():
            raise FileExistsError(f"edition artifact already exists: {destination}")
        if destination != path and path.name.endswith("_corrected.json"):
            old_corrections = path.with_name(path.name.replace("_corrected.json", "_corrections.json"))
            new_corrections = destination.with_name(
                destination.name.replace("_corrected.json", "_corrections.json")
            )
            if old_corrections.exists() and new_corrections.exists():
                raise FileExistsError(f"edition artifact already exists: {new_corrections}")
        updates.append((path, destination, document))

    renamed: list[Path] = []
    for path, destination, document in updates:
        destination.write_text(
            json.dumps(document, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        if destination != path:
            path.unlink()
            corrections_path = path.with_name(path.name.replace("_corrected.json", "_corrections.json"))
            if corrections_path != path and corrections_path.exists():
                tokenized_corrections = destination.with_name(
                    destination.name.replace("_corrected.json", "_corrections.json")
                )
                if tokenized_corrections.exists():
                    raise FileExistsError(f"edition artifact already exists: {tokenized_corrections}")
                corrections_path.rename(tokenized_corrections)
        renamed.append(destination)
    return renamed


def _paths_for_program(paths: list[Path], program_key: str) -> list[Path]:
    wanted = program_key.upper()
    matching: list[Path] = []
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict) and str(document.get("program", "")).upper() == wanted:
            matching.append(path)
    return matching


def _candidate_corrected_paths(llm_dir: Path) -> list[Path]:
    """Read both the new structured layout and legacy flat corrected files."""
    structured = sorted(llm_dir.glob("*/curriculum_*.json"))
    legacy = sorted(llm_dir.glob("*_corrected.json"))
    return structured + legacy


def _corrected_paths_for_program(
    llm_dir: Path,
    program_key: str,
    *,
    catalog_key: str | None = None,
    plan: str | None = None,
) -> list[Path]:
    """Compatibility discovery by JSON identity, never by filename substring."""
    wanted = program_key.upper()
    kept: list[Path] = []
    for path in _candidate_corrected_paths(llm_dir):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or str(doc.get("program", "")).upper() != wanted:
            continue
        if plan is not None and str(doc.get("plan", "")).casefold() != plan.casefold():
            continue
        if catalog_key is not None:
            catalog = doc.get("catalog")
            if not isinstance(catalog, dict) or catalog.get("catalog_key") != catalog_key:
                continue
        kept.append(path)

    if catalog_key is not None:
        if not kept:
            raise FileNotFoundError(
                f"no corrected artifact for program={program_key}, plan={plan}, "
                f"catalog_key={catalog_key!r} under {llm_dir}"
            )
        if len(kept) > 1:
            raise ValueError(
                f"ambiguous corrected artifacts for program={program_key}, plan={plan}, "
                f"catalog_key={catalog_key!r}: {', '.join(map(str, kept))}"
            )
    return kept


def _corrected_paths_for_dataset(llm_dir: Path, dataset_key: str) -> list[Path]:
    paths = sorted((llm_dir / dataset_key).glob("curriculum_*.json"))
    if paths:
        return paths
    config = get_dataset_config(dataset_key)
    return _corrected_paths_for_program(
        llm_dir,
        config.program,
        catalog_key=config.catalog_key,
    )


def _dataset_scoped_prepare(
    dataset_key: str,
    ocr_root: Path,
    extracted_root: Path,
    consolidated_root: Path,
    edition_metadata: dict[str, str] | None = None,
) -> dict:
    from src.pipeline.tools.preparation import tool as _pd

    key = resolve_dataset_key(dataset_key)
    if not _pd.prepare_program(
        key,
        ocr_root / key,
        extracted_root,
        consolidated_root,
        BASE_DIR,
        edition_metadata,
    ):
        raise _pd.PreparationError(
            f"No usable OCR source for dataset '{key}' under {ocr_root}"
        )
    return {"prepared_programs": [key]}


# Backward-compatible helper name used by older focused tests/imports.
def _program_scoped_prepare(
    program_key: str,
    ocr_root: Path,
    extracted_root: Path,
    consolidated_root: Path,
    edition_metadata: dict[str, str] | None = None,
) -> dict:
    return _dataset_scoped_prepare(
        resolve_dataset_key(program_key),
        ocr_root,
        extracted_root,
        consolidated_root,
        edition_metadata,
    )


def _plan_label(value: object) -> str:
    text = str(value).strip() if value not in (None, "") else "no_plan"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "no_plan"


def _working_full_paths(consolidated_work: Path, dataset_key: str) -> list[Path]:
    return sorted((consolidated_work / dataset_key).glob("**/full/*_full.json"))


def _publish_consolidated_artifacts(
    consolidated_work: Path,
    consolidated_final: Path,
    dataset_key: str,
) -> list[Path]:
    """Publish readable pre-LLM files as consolidated/<dataset>/curriculum_<plan>.json."""
    sources = _working_full_paths(consolidated_work, dataset_key)
    if not sources:
        raise FileNotFoundError(
            f"no consolidated full artifacts found for dataset {dataset_key}"
        )
    destination_dir = consolidated_final / dataset_key
    destination_dir.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    seen_labels: set[str] = set()
    for source in sources:
        document = json.loads(source.read_text(encoding="utf-8"))
        label = _plan_label(document.get("plan"))
        if label in seen_labels:
            raise ValueError(
                f"multiple consolidated artifacts resolve to {dataset_key}/{label}"
            )
        seen_labels.add(label)
        destination = destination_dir / f"curriculum_{label}.json"
        shutil.copy2(source, destination)
        published.append(destination)
    return published


def _consolidated_paths_for_dataset(
    consolidated_final: Path,
    dataset_key: str,
) -> list[Path]:
    simple = sorted((consolidated_final / dataset_key).glob("curriculum_*.json"))
    if simple:
        return simple
    # Read-only fallback for artifacts created before the simplified layout.
    return sorted((consolidated_final / dataset_key).glob("**/full/*_full.json"))


def _publish_corrected_artifacts(
    dataset_key: str,
    corrected_temp_paths: list[Path],
    final_root: Path,
) -> list[Path]:
    """Publish readable final/<dataset>/curriculum_<plan>.json + correction logs."""
    destination_dir = final_root / dataset_key
    destination_dir.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    manifest_items: list[dict[str, str]] = []
    seen_labels: set[str] = set()

    for corrected_path in corrected_temp_paths:
        document = json.loads(corrected_path.read_text(encoding="utf-8"))
        label = _plan_label(document.get("plan"))
        if label in seen_labels:
            raise ValueError(f"multiple corrected artifacts resolve to {dataset_key}/{label}")
        seen_labels.add(label)

        corrections_path = corrected_path.with_name(
            corrected_path.name.replace("_corrected.json", "_corrections.json")
        )
        if not corrections_path.is_file():
            raise FileNotFoundError(corrections_path)

        curriculum_destination = destination_dir / f"curriculum_{label}.json"
        corrections_destination = destination_dir / f"corrections_{label}.json"
        shutil.copy2(corrected_path, curriculum_destination)
        shutil.copy2(corrections_path, corrections_destination)
        published.append(curriculum_destination)
        manifest_items.append(
            {
                "plan": label,
                "curriculum": curriculum_destination.name,
                "corrections": corrections_destination.name,
            }
        )

    config = get_dataset_config(dataset_key)
    manifest = {
        "dataset": config.key,
        "program": config.program,
        "academic_year": config.academic_year,
        "catalog_key": config.catalog_key,
        "artifacts": manifest_items,
    }
    (destination_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return published


def _print_dry_run(dataset_key: str) -> None:
    config = get_dataset_config(dataset_key)
    print(f"Dataset: {config.key}")
    print(f"Program: {config.program}")
    print(f"Academic year: {config.academic_year}")
    print(f"Catalog: {config.catalog_key}")
    print(f"OCR pages: {config.required_pages()}")
    for scope in config.scopes:
        print(
            f"Plan {scope.plan or 'no_plan'}: plan pages={scope.pages}; "
            f"description pages={scope.description_pages}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = get_dataset_config(args.dataset)

    try:
        explicit_metadata = _edition_metadata_from_args(args)
    except ValueError as error:
        print(f"Invalid edition metadata: {error}", file=sys.stderr)
        return 2
    edition_metadata = explicit_metadata or config.edition_metadata

    print("Stages: ocr -> extract -> merge -> correct -> (evaluate) -> (build_index)")
    print(
        f"Dataset={config.key} program={config.program} year={config.academic_year} "
        f"from={args.from_stage}"
    )
    if args.dry_run:
        _print_dry_run(config.key)
        return 0

    output_base = BASE_DIR / "data" / "output"
    ocr_root = output_base / "ocr"
    extracted_final = output_base / "extracted"
    consolidated_final = output_base / "consolidated"
    final_root = output_base / "final"

    if args.only_index:
        from src.pipeline.tools.indexing.tool import run_build_index_stage

        paths = None
        if args.index_path is not None:
            paths = _corrected_paths_for_dataset(final_root, config.key)
        print(run_build_index_stage(paths, args.index_path))
        return 0

    from_stage = args.from_stage
    do_ocr = from_stage == "ocr"
    do_extract_merge = from_stage in ("ocr", "extracted")
    do_correct = from_stage in ("ocr", "extracted", "consolidated")

    with tempfile.TemporaryDirectory(prefix="pipeline_") as temporary_directory:
        tmp_base = Path(temporary_directory)
        extracted_root = extracted_final if args.keep_intermediates else tmp_base / "extracted"
        consolidated_work = tmp_base / "consolidated"

        try:
            if do_ocr:
                from src.pipeline.tools.ocr.tool import run_ocr_stage

                input_dir = config.resolve_input_dir(BASE_DIR)
                run_ocr_stage(
                    config.program,
                    config.required_pages(),
                    args.no_gpu,
                    BASE_DIR,
                    input_dir=input_dir,
                    output_dir=output_base,
                    dataset_key=config.key,
                )

            if do_extract_merge:
                if from_stage == "extracted":
                    from src.pipeline.tools.preparation import tool as _pd

                    extracted_dataset = extracted_final / config.key
                    if not extracted_dataset.is_dir():
                        raise FileNotFoundError(
                            f"persisted extracted dataset not found: {extracted_dataset}"
                        )
                    for scope in config.scopes:
                        _pd._run_merge(
                            extracted_dataset,
                            consolidated_work / config.key,
                            config,
                            scope,
                            _pd._combined_pages(scope, True),
                            edition_metadata,
                        )
                else:
                    _dataset_scoped_prepare(
                        config.key,
                        ocr_root,
                        extracted_root,
                        consolidated_work,
                        edition_metadata,
                    )

                working_full = _working_full_paths(consolidated_work, config.key)
                if not working_full:
                    raise FileNotFoundError(
                        f"no merged full artifacts produced for {config.key}"
                    )
                _apply_edition_metadata(working_full, edition_metadata)
                consolidated_inputs = _publish_consolidated_artifacts(
                    consolidated_work,
                    consolidated_final,
                    config.key,
                )
                print(
                    f"Consolidated {len(consolidated_inputs)} plan artifact(s) under "
                    f"{consolidated_final / config.key}"
                )
            else:
                consolidated_inputs = _consolidated_paths_for_dataset(
                    consolidated_final,
                    config.key,
                )

            if do_correct:
                if not consolidated_inputs:
                    raise FileNotFoundError(
                        f"no consolidated artifacts found for {config.key}"
                    )
                from src.pipeline.tools.correction.tool import run_correct_stage

                corrected_temp = tmp_base / "corrected"
                corrected = run_correct_stage(
                    consolidated_final,
                    corrected_temp,
                    consolidated_inputs,
                )
                final_paths = _publish_corrected_artifacts(
                    config.key,
                    corrected,
                    final_root,
                )
                print(
                    f"Corrected {len(final_paths)} plan artifact(s) under "
                    f"{final_root / config.key}"
                )
            else:
                final_paths = _corrected_paths_for_dataset(final_root, config.key)

            if not args.skip_eval:
                from src.pipeline.tools.evaluation.evaluate import discover_llm_evaluation_pairs
                from src.pipeline.tools.evaluation.tool import run_evaluate_stage

                try:
                    all_pairs = discover_llm_evaluation_pairs(final_root)
                except FileNotFoundError:
                    all_pairs = []
                selected_paths = {path.resolve() for path in final_paths}
                pairs = [
                    (prediction, ground_truth)
                    for prediction, ground_truth in all_pairs
                    if Path(prediction).resolve() in selected_paths
                ]
                if pairs:
                    run_evaluate_stage(pairs=pairs)
                else:
                    print(
                        f"No accepted Ground Truth pair for {config.key}; skipping evaluation."
                    )

            if args.with_index:
                from src.pipeline.tools.indexing.tool import run_build_index_stage

                print(run_build_index_stage(None, args.index_path))

        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    print(f"Done. Final reviewed artifacts: {final_root / config.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
