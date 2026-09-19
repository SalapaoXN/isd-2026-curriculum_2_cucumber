"""Indexing tool: final RAG-ready data -> curriculum.db (delegates to rag).

"Build-index stage: corrected JSON -> curriculum.db. No argv here."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

from rag.build_index import build_index


def run_build_index_stage(
    input_json_paths: Iterable[str | Path] | None = None,
    index_path: str | Path | None = None,
) -> Path:
    """Build the runtime curriculum DB and return its path."""
    return build_index(input_json_paths, index_path)
