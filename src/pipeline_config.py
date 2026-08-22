"""Shared command-line resolution for the curriculum OCR pipeline."""

import re
from pathlib import Path
from typing import Dict, List, Optional


SUPPORTED_PROGRAMS = ("DSBA", "IT", "AIT", "GENED", "BIT")
PROGRAM_BY_INPUT_DIR = {
    "dsba": "DSBA",
    "it": "IT",
    "ait": "AIT",
    "gened": "GENED",
    "bit": "BIT",
}
PLAN_VARIANTS = ("coop", "no_coop", "gened")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def normalize_program(program: str) -> str:
    """Return the canonical program name or raise a useful CLI error."""
    normalized = str(program).strip().upper()
    if normalized not in SUPPORTED_PROGRAMS:
        supported = ", ".join(SUPPORTED_PROGRAMS)
        raise ValueError(
            f"Unsupported program '{program}'. Choose one of: {supported}."
        )
    return normalized


def resolve_program(program: Optional[str], input_dir: Path) -> str:
    """Use an explicit program, or derive it from a supported input directory."""
    if program is not None and str(program).strip():
        return normalize_program(program)

    inferred = PROGRAM_BY_INPUT_DIR.get(input_dir.name.casefold())
    if inferred is None:
        supported_dirs = ", ".join(sorted(PROGRAM_BY_INPUT_DIR))
        raise ValueError(
            f"Cannot derive a program from input directory '{input_dir}'. "
            f"Pass --program explicitly (supported directory names: {supported_dirs})."
        )
    return inferred


def normalize_plan(plan: Optional[str]) -> Optional[str]:
    """Normalize a plan value while keeping an omitted plan as ``None``."""
    if plan is None or not str(plan).strip():
        return None

    normalized = str(plan).strip().lower().replace("-", "_")
    if normalized not in PLAN_VARIANTS:
        supported = ", ".join(PLAN_VARIANTS)
        raise ValueError(f"Unsupported plan '{plan}'. Choose one of: {supported}.")
    return normalized


def resolve_plan(plan: Optional[str], program: str) -> Optional[str]:
    """Validate plan semantics without guessing a co-op variant."""
    program = normalize_program(program)
    normalized = normalize_plan(plan)

    if program == "AIT":
        if normalized is not None:
            raise ValueError("AIT has no study-plan variant; omit --plan.")
        return None

    if program == "GENED":
        if normalized != "gened":
            raise ValueError("GENED requires the explicit plan value --plan gened.")
        return normalized

    if normalized not in {"coop", "no_coop"}:
        raise ValueError(
            f"{program} requires an explicit --plan coop or --plan no_coop; "
            "the co-op variant cannot be inferred."
        )
    return normalized


def discover_page_files(input_dir: Path) -> Dict[int, Path]:
    """Find direct-child ``<group>_page_<NNN>.<ext>`` images by page number."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    group = input_dir.name
    if not group:
        raise ValueError(
            f"Cannot derive page filename prefix from input directory '{input_dir}'."
        )

    pattern = re.compile(
        rf"^{re.escape(group)}_page_(\d{{3}})(\.[^.]+)$", re.IGNORECASE
    )
    extension_order = {extension: index for index, extension in enumerate(IMAGE_EXTENSIONS)}
    page_files: Dict[int, Path] = {}

    for candidate in input_dir.iterdir():
        if not candidate.is_file() or candidate.suffix.casefold() not in extension_order:
            continue
        match = pattern.match(candidate.name)
        if not match:
            continue

        page = int(match.group(1))
        current = page_files.get(page)
        if current is None or extension_order[candidate.suffix.casefold()] < extension_order[
            current.suffix.casefold()
        ]:
            page_files[page] = candidate

    return page_files


def discover_pages(input_dir: Path) -> List[int]:
    """Return discovered page numbers in numeric order, or fail clearly."""
    page_files = discover_page_files(input_dir)
    if not page_files:
        expected = f"{input_dir.name}_page_<NNN>.<ext>"
        raise ValueError(
            f"No valid page images found directly inside '{input_dir}'. "
            f"Expected filenames like {expected}."
        )
    return sorted(page_files)


def plan_label(plan: Optional[str]) -> str:
    """Return a safe filename label without changing the JSON plan value."""
    return "no_plan" if plan is None or not str(plan).strip() else str(plan)
