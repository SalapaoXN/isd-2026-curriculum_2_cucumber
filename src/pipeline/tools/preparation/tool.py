"""Preparation tool: processed information -> final RAG-ready staging (scope orchestration).

"Prepare extracted and consolidated curriculum data from persisted OCR.

This stage deliberately starts at ``data/output/ocr``.  It does not invoke OCR,
LLM correction, evaluation, or RAG; those remain independent pipeline stages."""
from __future__ import annotations

import re
import sys
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
        if prefix and not path.name.casefold().startswith(prefix.casefold()):
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
    supported_names = set(SUPPORTED_PROGRAMS)
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
    )


def _run_merge(
    extracted_dir: Path,
    consolidated_dir: Path,
    config: ProgramConfig,
    scope: Scope,
    pages: str | None,
) -> None:
    merge_consecutive_files(
        input_dir=str(extracted_dir),
        output_dir=str(consolidated_dir),
        plan_filter=scope.plan,
        pages=pages,
        prefix=config.prefix,
        desc_pages=scope.description_pages,
    )


def _combined_pages(scope: Scope, include_descriptions: bool) -> str | None:
    if scope.pages is None or not include_descriptions or scope.description_pages is None:
        return scope.pages
    return f"{scope.pages},{scope.description_pages}"


def prepare_program(
    key: str,
    program_dir: Path,
    extracted_path: Path,
    consolidated_path: Path,
    root: Path,
) -> bool:
    """Prepare one program corpus; shared by prepare_data() and pipeline.py."""
    config = PROGRAM_CONFIG[key]
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
            extracted_path / key,
            consolidated_path,
            config,
            scope,
            _combined_pages(
                scope,
                descriptions_available or config.shared_description is None,
            ),
        )

    return bool(usable_scopes)


def prepare_data(
    project_root: str | Path | None = None,
    ocr_root: str | Path = "data/output/ocr",
    extracted_root: str | Path = "data/output/extracted",
    consolidated_root: str | Path = "data/output/consolidated",
) -> dict:
    """Prepare every usable supported OCR corpus under ``ocr_root``."""
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


def main() -> int:
    try:
        result = prepare_data()
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
