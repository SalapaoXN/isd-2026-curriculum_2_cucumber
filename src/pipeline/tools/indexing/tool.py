"""Indexing tool: final RAG-ready data -> curriculum.db (delegates to rag).

"Build-index stage: corrected JSON -> curriculum.db. No argv here."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable

from rag.build_index import ARTIFACTS_DIR, DEFAULT_INDEX_NAME, _default_sources, build_index


_UNIFIED_PROGRAMS = frozenset({"AIT", "BIT", "DSBA", "GENED", "IT"})


def _targets_shared_runtime(index_path: str | Path | None) -> bool:
    if index_path is None:
        return True
    raw_path = Path(index_path)
    candidate = (
        ARTIFACTS_DIR / raw_path
        if not raw_path.is_absolute() and raw_path.parent == Path(".")
        else raw_path
    )
    return candidate.resolve() == (ARTIFACTS_DIR / DEFAULT_INDEX_NAME).resolve()


def _validate_unified_sources(
    input_json_paths: Iterable[str | Path] | None,
) -> None:
    paths = list(_default_sources() if input_json_paths is None else input_json_paths)
    programs: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot validate unified curriculum source: {path}") from error
        if not isinstance(document, dict) or not document.get("program"):
            raise ValueError(f"curriculum source has no program identity: {path}")
        programs.add(str(document["program"]).strip().upper())

    missing = sorted(_UNIFIED_PROGRAMS - programs)
    unexpected = sorted(programs - _UNIFIED_PROGRAMS)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing programs: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected programs: {', '.join(unexpected)}")
        raise ValueError(
            "refusing to rebuild shared curriculum.db from an incomplete source set ("
            + "; ".join(details)
            + ")"
        )


def run_build_index_stage(
    input_json_paths: Iterable[str | Path] | None = None,
    index_path: str | Path | None = None,
) -> Path:
    """Build the runtime DB, protecting the shared path from scoped inputs."""
    if _targets_shared_runtime(index_path):
        if input_json_paths is None:
            _validate_unified_sources(None)
            return build_index(None, index_path)
        sources = list(input_json_paths)
        _validate_unified_sources(sources)
        return build_index(sources, index_path)
    return build_index(input_json_paths, index_path)
