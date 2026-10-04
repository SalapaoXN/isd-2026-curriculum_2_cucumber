"""Preparation tool: processed information -> final RAG-ready staging (scope orchestration).

"Prepare extracted and consolidated curriculum data from persisted OCR.

This stage deliberately starts at ``data/output/ocr``.  It does not invoke OCR,
LLM correction, evaluation, or RAG; those remain independent pipeline stages."""
from __future__ import annotations

import re
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.pipeline.tools.extraction.tool import run_extraction
from src.pipeline.tools.merge.consolidator import merge_consecutive_files, parse_page_range


SUPPORTED_PROGRAMS = ("ait", "bit", "dsba", "gened", "it")


@dataclass(frozen=True)
class Scope:
    """One explicit extraction/merge scope for a program."""

    plan: str | None
    pages: str | None
    description_pages: str | None = None


@dataclass(frozen=True)
class ProgramConfig:
    """Explicit program semantics; none of these values are inferred."""

    program: str
    prefix: str
    scopes: tuple[Scope, ...]
    shared_description: Scope | None = None
    dataset_key: str | None = None
    catalog_key: str | None = None
    academic_year: str | None = None


PROGRAM_CONFIG: dict[str, ProgramConfig] = {
    "ait": ProgramConfig(
        program="AIT",
        prefix="ait",
        scopes=(Scope(plan=None, pages=None, description_pages="287-302"),),
    ),
    "bit": ProgramConfig(
        program="BIT",
        prefix="bit",
        scopes=(
            Scope(plan="no_coop", pages="26-30", description_pages="238-257"),
            Scope(plan="coop", pages="31-35", description_pages="238-257"),
        ),
        shared_description=Scope(plan="coop", pages="238-257"),
    ),
    "dsba": ProgramConfig(
        program="DSBA",
        prefix="dsba",
        scopes=(
            Scope(plan="no_coop", pages="26-32", description_pages="317-344"),
            Scope(plan="coop", pages="33-39", description_pages="317-344"),
        ),
        shared_description=Scope(plan="coop", pages="317-344"),
    ),
    "gened": ProgramConfig(
        program="GENED",
        prefix="gened",
        scopes=(
            Scope(plan="gened", pages="16-30,44-117", description_pages="44-117"),
        ),
    ),
    "dsba2560": ProgramConfig(
        program="DSBA",
        prefix="dsba2560",
        scopes=(
            Scope(plan="no_coop", pages="25-29", description_pages="175-207"),
            Scope(plan="coop", pages="30-34", description_pages="175-207"),
        ),
        shared_description=Scope(plan="coop", pages="175-207"),
        dataset_key="dsba2560",
        catalog_key="dsba-2560",
        academic_year="2560",
    ),
    "gened2557": ProgramConfig(
        program="GENED",
        prefix="gened2557",
        scopes=(
            Scope(plan="gened", pages="11-18,47-92", description_pages="47-92"),
        ),
        dataset_key="gened2557",
        catalog_key="gened-2557",
        academic_year="2557",
    ),
    "it2560": ProgramConfig(
        program="IT",
        prefix="it2560",
        scopes=(
            Scope(plan="no_coop", pages="27-33", description_pages="222-269"),
            Scope(plan="coop", pages="34-40", description_pages="222-269"),
        ),
        shared_description=Scope(plan="coop", pages="222-269"),
        dataset_key="it2560",
        catalog_key="it-2560",
        academic_year="2560",
    ),
    "bit2560": ProgramConfig(
        program="BIT",
        prefix="bit2560",
        scopes=(
            Scope(plan="no_coop", pages="23-26", description_pages="170-192"),
            Scope(plan="coop", pages="27-30", description_pages="170-192"),
        ),
        shared_description=Scope(plan="coop", pages="170-192"),
        dataset_key="bit2560",
        catalog_key="bit-2560",
        academic_year="2560",
    ),
    "it": ProgramConfig(
        program="IT",
        prefix="it",
        scopes=(
            Scope(plan="no_coop", pages="32-38", description_pages="328-371"),
            Scope(plan="coop", pages="39-45", description_pages="328-371"),
        ),
        shared_description=Scope(plan="coop", pages="328-371"),
    ),
}

PAGE_RE = re.compile(r"page_(\d+)", re.IGNORECASE)
OCR_SUFFIXES = {".txt", ".json"}


class PreparationError(RuntimeError):
    """Raised when no usable persisted OCR source can be prepared."""


def _ocr_files(ocr_dir: Path, prefix: str | None = None) -> list[Path]:
    files = []
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
    """Return existing supported directories and unknown directory names."""
    if not ocr_root.is_dir():
        return [], []

    supported = []
    unknown = []
    supported_names = set(PROGRAM_CONFIG)
    for child in sorted(ocr_root.iterdir(), key=lambda path: path.name.casefold()):
        if not child.is_dir():
            continue
        key = child.name.casefold()
        if key in supported_names:
            supported.append((key, child))
        else:
            unknown.append(child.name)
    return supported, unknown


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
    """Prepare one program corpus; shared by prepare_data() and pipeline.py."""
    key = key.casefold()
    config = PROGRAM_CONFIG[key]
    configured_metadata = (
        {"catalog_key": config.catalog_key, "academic_year": config.academic_year}
        if config.catalog_key is not None and config.academic_year is not None
        else None
    )
    if config.dataset_key is not None and edition_metadata not in (None, configured_metadata):
        raise ValueError(
            f"dataset {config.dataset_key!r} requires edition metadata "
            f"{configured_metadata!r}"
        )
    effective_metadata = edition_metadata or configured_metadata
    files = _ocr_files(program_dir, config.prefix)
    if not files:
        print(f"Skipping empty OCR directory: {program_dir}")
        return False

    usable_scopes = []
    for scope in config.scopes:
        if _scope_is_usable(files, scope.pages):
            _run_extract(root, program_dir, extracted_path, config, scope)
            usable_scopes.append(scope)
        else:
            print(
                f"Skipping scope {config.program}/{scope.plan or 'no_plan'}: "
                "required OCR pages are missing"
            )

    descriptions_available = False
    if config.shared_description is not None:
        shared = config.shared_description
        if _scope_has_any_files(files, shared.pages):
            _run_extract(root, program_dir, extracted_path, config, shared)
            descriptions_available = True
        else:
            print(f"No shared description OCR pages found for {config.program}")

    for scope in usable_scopes:
        _run_merge(
            extracted_path / (config.dataset_key or config.program.casefold()),
            consolidated_path / config.dataset_key if config.dataset_key else consolidated_path,
            config,
            scope,
            _combined_pages(
                scope,
                descriptions_available or config.shared_description is None,
            ),
            effective_metadata,
        )

    if config.dataset_key is not None and usable_scopes:
        if effective_metadata is None:
            raise PreparationError(
                f"edition metadata is not configured for {config.dataset_key}"
            )
        from src.pipeline.run import _apply_edition_metadata
        from src.pipeline.tools.merge.consolidator import edition_filename_token

        token = edition_filename_token(effective_metadata["catalog_key"])
        full_files = sorted(
            (consolidated_path / config.dataset_key).glob(
                f"**/full/*_{token}_full.json"
            )
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
        selected = tuple(dict.fromkeys(str(key).casefold() for key in dataset_keys))
        unsupported = sorted(set(selected) - set(PROGRAM_CONFIG))
        if unsupported:
            raise ValueError(f"Unsupported dataset key(s): {', '.join(unsupported)}")
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
    parser = argparse.ArgumentParser(description="Prepare selected persisted OCR datasets.")
    parser.add_argument(
        "--dataset-key",
        action="append",
        choices=tuple(PROGRAM_CONFIG),
        help="Prepare only this exact OCR dataset key; may be supplied multiple times.",
    )
    args = parser.parse_args(argv)
    try:
        result = prepare_data(dataset_keys=args.dataset_key)
    except (OSError, ValueError, PreparationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        "Preparation complete for: "
        + ", ".join(result["prepared_programs"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
