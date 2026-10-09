"""Build the shared unified curriculum database from reviewed final JSON files."""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from dotenv import load_dotenv

from rag.retrieval.index import (
    ARTIFACTS_DIR,
    DEFAULT_INDEX_NAME,
    ensure_index,
    llm_source_paths,
)
from src.pipeline.datasets import DATASET_ALIASES, DATASET_CONFIG


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CLEAN_FINAL_DIR = _PROJECT_ROOT / "data" / "output" / "final"


def _expand_input_paths(values: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        text = str(value)
        matches = sorted(Path(match) for match in glob.glob(text))
        paths.extend(matches or [Path(value)])
    return paths


def _legacy_catalog_from_filename(path: Path) -> str | None:
    """Map old unversioned final filenames to their known bundled edition."""
    name = path.name.casefold()
    for old_key, dataset_key in DATASET_ALIASES.items():
        if name.startswith(f"merged_{old_key}_"):
            return DATASET_CONFIG[dataset_key].catalog_key
    return None


def _artifact_identity(path: Path) -> tuple[str, str, str | None] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    program = str(document.get("program", "")).strip().upper()
    plan_value = document.get("plan")
    plan = "" if plan_value in (None, "") else str(plan_value).strip().casefold()
    catalog = document.get("catalog")
    catalog_key = catalog.get("catalog_key") if isinstance(catalog, dict) else None
    if not isinstance(catalog_key, str) or not catalog_key:
        catalog_key = _legacy_catalog_from_filename(path)
    if not program:
        return None
    return program, plan, catalog_key


def _default_sources() -> list[Path]:
    """Prefer current ``*_final.json`` artifacts while supporting old layouts.

    Current pipeline output is flat and explicit, for example
    ``data/output/final/it2565_coop_final.json``. During migration, an artifact
    from a newer layout replaces only the exact same catalog/program/plan
    identity from an older layout; other editions remain available.
    """
    current = sorted(_CLEAN_FINAL_DIR.glob("*_final.json"))
    structured_legacy = sorted(_CLEAN_FINAL_DIR.glob("*/curriculum_*.json"))
    flat_legacy = sorted(_CLEAN_FINAL_DIR.glob("*_corrected.json"))

    candidates = [current, structured_legacy, flat_legacy]
    selected: list[Path] = []
    selected_identities: set[tuple[str, str, str | None]] = set()
    for group in candidates:
        for path in group:
            identity = _artifact_identity(path)
            if identity is not None and identity in selected_identities:
                continue
            selected.append(path)
            if identity is not None:
                selected_identities.add(identity)

    if selected:
        return sorted(selected, key=lambda path: path.as_posix().casefold())
    return llm_source_paths()


def _default_supplemental_sources() -> dict[str, Path]:
    return {
        "institution_policy": _PROJECT_ROOT
        / "data"
        / "output"
        / "final"
        / "institution_policy.json",
        "program_requirements": _PROJECT_ROOT
        / "data"
        / "output"
        / "final"
        / "program_requirements.json",
    }


def build_index(
    input_json_paths: Iterable[str | Path] | None = None,
    index_path: str | Path | None = None,
) -> Path:
    """Build or reuse the shared curriculum database for the supplied JSON files."""
    using_defaults = input_json_paths is None
    sources = _default_sources() if using_defaults else input_json_paths
    artifact_index = index_path or ARTIFACTS_DIR / DEFAULT_INDEX_NAME
    supplemental = _default_supplemental_sources() if using_defaults else None
    kwargs = {"index_path": artifact_index}
    if supplemental is not None:
        kwargs["supplemental_json_paths"] = supplemental
    return ensure_index(sources, **kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_json_paths",
        nargs="*",
        help="curriculum JSON files (defaults to reviewed data/output/final artifacts)",
    )
    args = parser.parse_args(argv)
    input_paths = _expand_input_paths(args.input_json_paths)
    print(build_index(input_paths or None))


if __name__ == "__main__":
    main()
