"""Indexing tool: final RAG-ready data -> curriculum.db (delegates to rag).

"Build-index stage: corrected JSON -> curriculum.db. No argv here."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from rag.build_index import ARTIFACTS_DIR, DEFAULT_INDEX_NAME, _default_sources, build_index


_UNIFIED_PROGRAMS = frozenset({"AIT", "BIT", "DSBA", "GENED", "IT"})
_PLAN_UNSPECIFIED = object()


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
    expected_editions: Sequence[Mapping[str, object]] | None = None,
) -> None:
    paths = list(_default_sources() if input_json_paths is None else input_json_paths)
    if expected_editions is not None:
        _resolve_expected_edition_sources(paths, expected_editions)
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


def _normalized_plan(value: object) -> str | None:
    if isinstance(value, Mapping):
        value = (
            value.get("plan_code")
            or value.get("code")
            or value.get("id")
            or value.get("plan_name")
            or value.get("name")
            or value.get("title")
        )
    if value is None:
        return None
    text = str(value).strip()
    return text.casefold() if text else None


def _resolve_expected_edition_sources(
    input_json_paths: Iterable[str | Path],
    expected_editions: Sequence[Mapping[str, object]],
) -> list[Path]:
    """Resolve each requested edition by corrected JSON identity, never filename."""
    paths = []
    seen_paths: set[Path] = set()
    for value in input_json_paths:
        path = Path(value)
        resolved_path = path.resolve()
        if resolved_path not in seen_paths:
            seen_paths.add(resolved_path)
            paths.append(path)
    if not expected_editions:
        raise ValueError("expected_editions must contain at least one edition")

    normalized_expectations: list[tuple[str, str | None | object, str]] = []
    seen: set[tuple[str, str | None | object, str]] = set()
    for expected in expected_editions:
        if not isinstance(expected, Mapping):
            raise ValueError("each expected edition must be an object")
        program = expected.get("program")
        catalog_key = expected.get("catalog_key")
        if not isinstance(program, str) or not program.strip():
            raise ValueError("each expected edition requires a program")
        if not isinstance(catalog_key, str) or not catalog_key.strip():
            raise ValueError("each expected edition requires a catalog_key")
        if catalog_key != catalog_key.strip():
            raise ValueError("expected catalog_key must not have surrounding whitespace")
        raw_plan = expected.get("plan", _PLAN_UNSPECIFIED)
        if (
            raw_plan is not _PLAN_UNSPECIFIED
            and raw_plan is not None
            and not isinstance(raw_plan, str)
        ):
            raise ValueError("expected edition plan must be a string or null")
        plan = _normalized_plan(raw_plan) if raw_plan is not _PLAN_UNSPECIFIED else raw_plan
        identity = (program.strip().upper(), plan, catalog_key)
        if identity in seen:
            plan_text = "*" if plan is _PLAN_UNSPECIFIED else str(plan)
            raise ValueError(
                f"duplicate expected edition: {identity[0]}/{plan_text}/{catalog_key}"
            )
        seen.add(identity)
        normalized_expectations.append(identity)

    documents: list[tuple[Path, dict]] = []
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot validate corrected curriculum source: {path}") from error
        if not isinstance(document, dict):
            raise ValueError(f"corrected curriculum source must be an object: {path}")
        documents.append((path, document))

    resolved: list[Path] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for program, expected_plan, catalog_key in normalized_expectations:
        matches = []
        for path, document in documents:
            catalog = document.get("catalog")
            if not isinstance(catalog, dict) or catalog.get("catalog_key") != catalog_key:
                continue
            if str(document.get("program", "")).strip().upper() != program:
                continue
            if expected_plan is not _PLAN_UNSPECIFIED:
                if _normalized_plan(document.get("plan", document.get("plan_name"))) != expected_plan:
                    continue
            matches.append(path)
        plan_label = expected_plan if expected_plan is not _PLAN_UNSPECIFIED else "*"
        label = f"{program}/{plan_label}/{catalog_key}"
        if not matches:
            missing.append(label)
        elif len(matches) > 1:
            ambiguous.append(label)
        else:
            resolved.append(matches[0])

    if missing or ambiguous:
        details = []
        if missing:
            details.append("missing edition(s): " + ", ".join(missing))
        if ambiguous:
            details.append("ambiguous edition(s): " + ", ".join(ambiguous))
        raise ValueError(
            "expected curriculum edition validation failed ("
            + "; ".join(details)
            + ")"
        )
    return resolved


def run_build_index_stage(
    input_json_paths: Iterable[str | Path] | None = None,
    index_path: str | Path | None = None,
    expected_editions: Sequence[Mapping[str, object]] | None = None,
) -> Path:
    """Build the index, optionally validating/selecting exact curriculum editions.

    The shared runtime still ingests every supplied/default catalog after exact
    expectations are validated. A separate index can be scoped to the resolved
    expected-edition artifacts.
    """
    if _targets_shared_runtime(index_path):
        if input_json_paths is None and expected_editions is None:
            _validate_unified_sources(None)
            return build_index(None, index_path)
        if input_json_paths is None:
            sources = list(_default_sources())
            _validate_unified_sources(sources, expected_editions)
            return build_index(None, index_path)
        sources = list(input_json_paths)
        _validate_unified_sources(sources, expected_editions)
        return build_index(sources, index_path)
    if expected_editions is not None:
        sources = list(_default_sources() if input_json_paths is None else input_json_paths)
        selected = _resolve_expected_edition_sources(sources, expected_editions)
        return build_index(selected, index_path)
    return build_index(input_json_paths, index_path)
