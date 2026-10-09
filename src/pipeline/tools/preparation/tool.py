"""Prepare extracted and consolidated curriculum data from persisted OCR.

This stage starts at ``data/output/ocr``. Dataset/edition semantics are defined
centrally in :mod:`src.pipeline.datasets`; users no longer have to supply plan
or page ranges manually for bundled curriculum datasets.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

from src.pipeline.datasets import (
    DATASET_CONFIG,
    DatasetConfig as ProgramConfig,
    Scope,
    resolve_dataset_key,
)
from src.pipeline.tools.extraction.tool import run_extraction
from src.pipeline.tools.merge.consolidator import merge_consecutive_files, parse_page_range


# Compatibility name retained for tests and older internal imports.
PROGRAM_CONFIG: dict[str, ProgramConfig] = DATASET_CONFIG
SUPPORTED_PROGRAMS = tuple(PROGRAM_CONFIG)

PAGE_RE = re.compile(r"page_(\d+)", re.IGNORECASE)
OCR_SUFFIXES = {".txt", ".json"}


class PreparationError(RuntimeError):
    """Raised when no usable persisted OCR source can be prepared."""


def _ocr_files(ocr_dir: Path, prefix: str | None = None) -> list[Path]:
    files = []
    if not ocr_dir.is_dir():
        return files
    for path in sorted(ocr_dir.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in OCR_SUFFIXES:
            continue
        if path.name.endswith("_extracted.json"):
            continue
        if prefix and not path.name.casefold().startswith(
            f"{prefix.casefold().rstrip('_')}_"
        ):
            continue
        if PAGE_RE.search(path.name) is None:
            continue
        files.append(path)
    return files


def _pages_in_files(files: Iterable[Path]) -> set[int]:
    pages = set()
    for path in files:
        match = PAGE_RE.search(path.name)
        if match:
            pages.add(int(match.group(1)))
    return pages


def _configured_pages(page_spec: str | None) -> set[int] | None:
    if page_spec is None:
        return None
    return parse_page_range(page_spec)


def _scope_is_usable(files: list[Path], page_spec: str | None) -> bool:
    if not files:
        return False
    required_pages = _configured_pages(page_spec)
    if required_pages is None:
        return True
    return required_pages.issubset(_pages_in_files(files))


def _scope_has_any_files(files: list[Path], page_spec: str | None) -> bool:
    if not files:
        return False
    configured_pages = _configured_pages(page_spec)
    if configured_pages is None:
        return True
    return bool(configured_pages & _pages_in_files(files))


def discover_supported_corpora(ocr_root: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """Return supported OCR dataset directories, preferring explicit year keys."""
    if not ocr_root.is_dir():
        return [], []

    chosen: dict[str, Path] = {}
    unknown: list[str] = []
    for child in sorted(ocr_root.iterdir(), key=lambda path: path.name.casefold()):
        if not child.is_dir():
            continue
        raw_key = child.name.casefold()
        try:
            key = resolve_dataset_key(raw_key)
        except ValueError:
            unknown.append(child.name)
            continue
        previous = chosen.get(key)
        if previous is None or raw_key == key:
            chosen[key] = child

    return sorted(chosen.items()), unknown


def _run_extract(
    project_root: Path,
    ocr_dir: Path,
    output_dir: Path,
    config: ProgramConfig,
    scope: Scope,
) -> None:
    run_extraction(
        ocr_dir,
        output_dir,
        program=config.program,
        plan=scope.plan,
        prefix=config.prefix,
        pages=scope.pages,
        source=None,
        dataset_key=config.dataset_key,
    )


def _run_merge(
    extracted_dir: Path,
    consolidated_dir: Path,
    config: ProgramConfig,
    scope: Scope,
    pages: str | None,
    edition_metadata: dict[str, str] | None = None,
) -> None:
    merge_consecutive_files(
        input_dir=str(extracted_dir),
        output_dir=str(consolidated_dir),
        plan_filter=scope.plan,
        pages=pages,
        prefix=config.prefix,
        desc_pages=scope.description_pages,
        catalog_key=(edition_metadata or {}).get("catalog_key"),
    )


def _combined_pages(scope: Scope, include_descriptions: bool) -> str | None:
    if scope.pages is None or not include_descriptions or scope.description_pages is None:
        return scope.pages
    if _configured_pages(scope.description_pages).issubset(
        _configured_pages(scope.pages)
    ):
        return scope.pages
    return f"{scope.pages},{scope.description_pages}"


def prepare_program(
    key: str,
    program_dir: Path,
    extracted_path: Path,
    consolidated_path: Path,
    root: Path,
    edition_metadata: dict[str, str] | None = None,
) -> bool:
    """Prepare one exact dataset corpus using its deterministic configured scopes."""
    key = resolve_dataset_key(key)
    config = PROGRAM_CONFIG[key]
    configured_metadata = config.edition_metadata
    if edition_metadata not in (None, configured_metadata):
        raise ValueError(
            f"dataset {config.dataset_key!r} requires edition metadata "
            f"{configured_metadata!r}"
        )
    effective_metadata = edition_metadata or configured_metadata
    files = _ocr_files(program_dir, config.prefix)
    if not files:
        print(f"Skipping empty OCR directory: {program_dir}")
        return False

    usable_scopes: list[Scope] = []
    for scope in config.scopes:
        if _scope_is_usable(files, scope.pages):
            _run_extract(root, program_dir, extracted_path, config, scope)
            usable_scopes.append(scope)
        else:
            print(
                f"Skipping scope {config.dataset_key}/{scope.plan or 'no_plan'}: "
                "required OCR pages are missing"
            )

    descriptions_available = False
    if config.shared_description is not None:
        shared = config.shared_description
        if _scope_has_any_files(files, shared.pages):
            _run_extract(root, program_dir, extracted_path, config, shared)
            descriptions_available = True
        else:
            print(f"No shared description OCR pages found for {config.dataset_key}")

    dataset_consolidated = consolidated_path / config.dataset_key
    for scope in usable_scopes:
        _run_merge(
            extracted_path / config.dataset_key,
            dataset_consolidated,
            config,
            scope,
            _combined_pages(
                scope,
                descriptions_available or config.shared_description is None,
            ),
            effective_metadata,
        )

    if usable_scopes:
        from src.pipeline.run import _apply_edition_metadata
        from src.pipeline.tools.merge.consolidator import edition_filename_token

        token = edition_filename_token(effective_metadata["catalog_key"])
        full_files = sorted(
            dataset_consolidated.glob(f"**/full/*_{token}_full.json")
        )
        if not full_files:
            raise PreparationError(
                f"no full consolidated artifact found for {config.dataset_key}"
            )
        _apply_edition_metadata(full_files, effective_metadata)

    return bool(usable_scopes)


def prepare_data(
    project_root: str | Path | None = None,
    ocr_root: str | Path = "data/output/ocr",
    extracted_root: str | Path = "data/output/extracted",
    consolidated_root: str | Path = "data/output/consolidated",
    dataset_keys: Iterable[str] | None = None,
) -> dict:
    """Prepare selected supported OCR datasets, or all discovered datasets."""
    root = Path(project_root or Path(__file__).resolve().parents[4]).resolve()
    ocr_path = Path(ocr_root)
    if not ocr_path.is_absolute():
        ocr_path = root / ocr_path
    extracted_path = Path(extracted_root)
    if not extracted_path.is_absolute():
        extracted_path = root / extracted_path
    consolidated_path = Path(consolidated_root)
    if not consolidated_path.is_absolute():
        consolidated_path = root / consolidated_path

    supported, unknown = discover_supported_corpora(ocr_path)
    if dataset_keys is not None:
        selected = tuple(dict.fromkeys(resolve_dataset_key(str(key)) for key in dataset_keys))
        directories = {key: path for key, path in supported}
        missing = [key for key in selected if key not in directories]
        if missing:
            raise PreparationError(
                f"Selected OCR dataset directory not found: {', '.join(missing)}"
            )
        supported = [(key, directories[key]) for key in selected]

    for name in unknown:
        print(f"Ignoring unsupported OCR directory: {name}")

    prepared_programs = []
    skipped_programs = []
    for key, program_dir in supported:
        if prepare_program(key, program_dir, extracted_path, consolidated_path, root):
            prepared_programs.append(key)
        else:
            skipped_programs.append(key)

    if not prepared_programs:
        raise PreparationError(
            f"No usable supported OCR source found under {ocr_path}"
        )

    return {
        "prepared_programs": prepared_programs,
        "skipped_programs": skipped_programs,
        "unknown_directories": unknown,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare persisted OCR datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(PROGRAM_CONFIG),
        help="Prepare only this exact dataset key; may be supplied multiple times.",
    )
    # Backward-compatible spelling for internal/older commands.
    parser.add_argument(
        "--dataset-key",
        action="append",
        dest="legacy_dataset_keys",
        choices=tuple(PROGRAM_CONFIG),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    selected = (args.dataset or []) + (args.legacy_dataset_keys or [])
    try:
        result = prepare_data(dataset_keys=selected or None)
    except (OSError, ValueError, PreparationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Preparation complete for: " + ", ".join(result["prepared_programs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
