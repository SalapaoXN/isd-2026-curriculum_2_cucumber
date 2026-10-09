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


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CLEAN_FINAL_DIR = _PROJECT_ROOT / "data" / "output" / "final"


def _expand_input_paths(values: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        text = str(value)
        matches = sorted(Path(match) for match in glob.glob(text))
        paths.extend(matches or [Path(value)])
    return paths


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
    if not program:
        return None
    return program, plan, catalog_key if isinstance(catalog_key, str) else None


def _default_sources() -> list[Path]:
    """Prefer the readable dataset layout while supporting unmigrated flat artifacts.

    New pipeline output:
        data/output/final/<dataset>/curriculum_<plan>.json

    Older flat ``*_corrected.json`` files are still included for datasets that
    have not been regenerated yet. If a new structured artifact covers the
    same catalog identity (or the same unscoped program/plan), it wins.
    """
    structured = sorted(_CLEAN_FINAL_DIR.glob("*/curriculum_*.json"))
    legacy = sorted(_CLEAN_FINAL_DIR.glob("*_corrected.json"))

    if not structured:
        if legacy:
            return legacy
        return llm_source_paths()

    structured_identities = {
        identity
        for path in structured
        if (identity := _artifact_identity(path)) is not None
    }
    structured_program_plans = {(program, plan) for program, plan, _ in structured_identities}

    kept_legacy: list[Path] = []
    for path in legacy:
        identity = _artifact_identity(path)
        if identity is None:
            kept_legacy.append(path)
            continue
        program, plan, catalog_key = identity
        if identity in structured_identities:
            continue
        # Old current-edition artifacts often predate catalog metadata. Once a
        # year-labelled structured artifact exists for that program/plan, do
        # not load the unscoped duplicate as a second curriculum edition.
        if catalog_key is None and (program, plan) in structured_program_plans:
            continue
        kept_legacy.append(path)

    return sorted(structured + kept_legacy, key=lambda path: path.as_posix().casefold())


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
