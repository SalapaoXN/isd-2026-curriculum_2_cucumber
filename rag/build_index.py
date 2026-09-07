"""Build the shared persistent semantic index for curriculum JSON files."""

from __future__ import annotations

import argparse
import glob
from collections.abc import Iterable, Sequence
from pathlib import Path

from rag.retrieval.index import ARTIFACTS_DIR, ensure_index


def _expand_input_paths(values: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        text = str(value)
        matches = sorted(Path(match) for match in glob.glob(text))
        paths.extend(matches or [Path(value)])
    return paths


def build_index(
    input_json_paths: Iterable[str | Path],
    index_path: str | Path | None = None,
) -> Path:
    """Build or reuse the shared semantic index for the supplied JSON files."""
    artifact_index = index_path or ARTIFACTS_DIR / "semantic.db"
    return ensure_index(input_json_paths, index_path=artifact_index)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json_paths", nargs="+", help="consolidated curriculum JSON files")
    args = parser.parse_args(argv)
    print(build_index(_expand_input_paths(args.input_json_paths)))


if __name__ == "__main__":
    main()
