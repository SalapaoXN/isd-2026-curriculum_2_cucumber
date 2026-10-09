"""Canonical curriculum dataset registry for the preprocessing pipeline.

The user-facing pipeline selects one dataset key (for example ``it2560`` or
``it2565``).  Program identity, academic year, plan page ranges, description
page ranges, source folder aliases, and catalog metadata are resolved here.

This keeps OCR orchestration deterministic and removes the need for users to
manually supply plan/page/catalog flags for the bundled project datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scope:
    """One explicit study-plan scope inside a curriculum dataset."""

    plan: str | None
    pages: str | None
    description_pages: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """All deterministic routing metadata for one curriculum edition."""

    key: str
    program: str
    academic_year: str
    catalog_key: str
    input_dir_names: tuple[str, ...]
    scopes: tuple[Scope, ...]
    shared_description: Scope | None = None

    @property
    def dataset_key(self) -> str:
        """Compatibility alias used by existing preparation code."""
        return self.key

    @property
    def prefix(self) -> str:
        """Generated OCR/extraction filename prefix."""
        return self.key

    @property
    def edition_metadata(self) -> dict[str, str]:
        return {
            "catalog_key": self.catalog_key,
            "academic_year": self.academic_year,
        }

    def required_pages(self) -> list[int]:
        """Return every bundled source page needed for the canonical artifact."""
        pages: set[int] = set()
        for scope in self.scopes:
            pages.update(parse_page_spec(scope.pages))
            pages.update(parse_page_spec(scope.description_pages))
        if self.shared_description is not None:
            pages.update(parse_page_spec(self.shared_description.pages))
            pages.update(parse_page_spec(self.shared_description.description_pages))
        return sorted(pages)

    def resolve_input_dir(self, project_root: str | Path) -> Path:
        """Resolve the exact year-labelled source folder for this dataset."""
        root = Path(project_root)
        input_root = root / "data" / "input"
        for name in self.input_dir_names:
            candidate = input_root / name
            if candidate.is_dir():
                return candidate
        expected = ", ".join(str(input_root / name) for name in self.input_dir_names)
        raise FileNotFoundError(
            f"No year-labelled input directory found for dataset '{self.key}'. "
            f"Expected: {expected}"
        )


def parse_page_spec(value: str | None) -> set[int]:
    """Parse a trusted configuration page specification such as ``32-38,328-371``."""
    if value is None:
        return set()
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid configured page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return pages


DATASET_CONFIG: dict[str, DatasetConfig] = {
    "ait2566": DatasetConfig(
        key="ait2566",
        program="AIT",
        academic_year="2566",
        catalog_key="ait-2566",
        input_dir_names=("ait2566",),
        scopes=(Scope(plan=None, pages="23-26", description_pages="287-302"),),
        shared_description=Scope(plan=None, pages="287-302"),
    ),
    "bit2565": DatasetConfig(
        key="bit2565",
        program="BIT",
        academic_year="2565",
        catalog_key="bit-2565",
        input_dir_names=("bit2565",),
        scopes=(
            Scope(plan="no_coop", pages="26-30", description_pages="238-257"),
            Scope(plan="coop", pages="31-35", description_pages="238-257"),
        ),
        shared_description=Scope(plan="coop", pages="238-257"),
    ),
    "bit2560": DatasetConfig(
        key="bit2560",
        program="BIT",
        academic_year="2560",
        catalog_key="bit-2560",
        input_dir_names=("bit2560",),
        scopes=(
            Scope(plan="no_coop", pages="23-26", description_pages="170-192"),
            Scope(plan="coop", pages="27-30", description_pages="170-192"),
        ),
        shared_description=Scope(plan="coop", pages="170-192"),
    ),
    "dsba2565": DatasetConfig(
        key="dsba2565",
        program="DSBA",
        academic_year="2565",
        catalog_key="dsba-2565",
        input_dir_names=("dsba2565",),
        scopes=(
            Scope(plan="no_coop", pages="26-32", description_pages="317-344"),
            Scope(plan="coop", pages="33-39", description_pages="317-344"),
        ),
        shared_description=Scope(plan="coop", pages="317-344"),
    ),
    "dsba2560": DatasetConfig(
        key="dsba2560",
        program="DSBA",
        academic_year="2560",
        catalog_key="dsba-2560",
        input_dir_names=("dsba2560",),
        scopes=(
            Scope(plan="no_coop", pages="25-29", description_pages="175-207"),
            Scope(plan="coop", pages="30-34", description_pages="175-207"),
        ),
        shared_description=Scope(plan="coop", pages="175-207"),
    ),
    "gened2564": DatasetConfig(
        key="gened2564",
        program="GENED",
        academic_year="2564",
        catalog_key="gened-2564",
        input_dir_names=("gened2564",),
        scopes=(Scope(plan="gened", pages="16-30", description_pages="44-117"),),
        shared_description=Scope(plan="gened", pages="44-117"),
    ),
    "gened2557": DatasetConfig(
        key="gened2557",
        program="GENED",
        academic_year="2557",
        catalog_key="gened-2557",
        input_dir_names=("gened2557",),
        scopes=(Scope(plan="gened", pages="11-18", description_pages="47-92"),),
        shared_description=Scope(plan="gened", pages="47-92"),
    ),
    "it2565": DatasetConfig(
        key="it2565",
        program="IT",
        academic_year="2565",
        catalog_key="it-2565",
        input_dir_names=("it2565",),
        scopes=(
            Scope(plan="no_coop", pages="32-38", description_pages="328-371"),
            Scope(plan="coop", pages="39-45", description_pages="328-371"),
        ),
        shared_description=Scope(plan="coop", pages="328-371"),
    ),
    "it2560": DatasetConfig(
        key="it2560",
        program="IT",
        academic_year="2560",
        catalog_key="it-2560",
        input_dir_names=("it2560",),
        scopes=(
            Scope(plan="no_coop", pages="27-33", description_pages="222-269"),
            Scope(plan="coop", pages="34-40", description_pages="222-269"),
        ),
        shared_description=Scope(plan="coop", pages="222-269"),
    ),
}

# Backward-compatible aliases for the old unversioned CLI names.  New README
# examples use the explicit year-labelled dataset keys above.
DATASET_ALIASES = {
    "ait": "ait2566",
    "bit": "bit2565",
    "dsba": "dsba2565",
    "gened": "gened2564",
    "it": "it2565",
}


def resolve_dataset_key(value: str) -> str:
    key = str(value).strip().casefold()
    key = DATASET_ALIASES.get(key, key)
    if key not in DATASET_CONFIG:
        supported = ", ".join(DATASET_CONFIG)
        raise ValueError(f"Unsupported dataset '{value}'. Choose one of: {supported}.")
    return key


def get_dataset_config(value: str) -> DatasetConfig:
    return DATASET_CONFIG[resolve_dataset_key(value)]
