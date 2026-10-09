"""Build the shared unified curriculum database from canonical runtime JSON files."""

from __future__ import annotations

import argparse
import glob
from collections.abc import Iterable, Sequence
from pathlib import Path

from dotenv import load_dotenv

from rag.retrieval.index import (
    ARTIFACTS_DIR,
    DEFAULT_INDEX_NAME,
    ensure_index,
)
from src.pipeline.tools.runtime_artifacts import (
    CANONICAL_DIR,
    audit_shared_course_conflicts,
    canonical_runtime_sources,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _expand_input_paths(values: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        text = str(value)
        matches = sorted(Path(match) for match in glob.glob(text))
        paths.extend(matches or [Path(value)])
    return paths


def _default_sources() -> list[Path]:
    """Read only deterministic post-evaluation canonical runtime artifacts."""
    return canonical_runtime_sources(CANONICAL_DIR)


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
    if using_defaults:
        conflicts = audit_shared_course_conflicts(sources)
        if conflicts:
            raise ValueError(
                "refusing to build from canonical runtime artifacts with "
                f"{len(conflicts)} unresolved shared-course conflict(s):\n- "
                + "\n- ".join(conflicts)
            )
    return ensure_index(sources, **kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_json_paths",
        nargs="*",
        help=f"curriculum JSON files (defaults to canonical runtime artifacts in {CANONICAL_DIR})",
    )
    args = parser.parse_args(argv)
    input_paths = _expand_input_paths(args.input_json_paths)
    print(build_index(input_paths or None))


if __name__ == "__main__":
    main()
