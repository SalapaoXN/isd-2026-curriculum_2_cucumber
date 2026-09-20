"""Build the shared unified curriculum database from consolidated JSON files."""

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


def _default_sources() -> list[Path]:
    """Clean-layout final output first, original layout as fallback."""
    preferred = sorted(_CLEAN_FINAL_DIR.glob("*_corrected.json"))
    if preferred:
        return preferred
    return llm_source_paths()


def build_index(
    input_json_paths: Iterable[str | Path] | None = None,
    index_path: str | Path | None = None,
) -> Path:
    """Build or reuse the shared curriculum database for the supplied JSON files."""
    sources = _default_sources() if input_json_paths is None else input_json_paths
    artifact_index = index_path or ARTIFACTS_DIR / DEFAULT_INDEX_NAME
    return ensure_index(sources, index_path=artifact_index)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_json_paths",
        nargs="*",
        help="curriculum JSON files (defaults to reviewed data/output/final corrected files)",
    )
    args = parser.parse_args(argv)
    input_paths = _expand_input_paths(args.input_json_paths)
    print(build_index(input_paths or None))


if __name__ == "__main__":
    main()
