"""Retrieve ranked evidence for a natural-language query."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .embedder import embed_texts
from .search import search
from .vector_store import score_candidate_vectors


_SEMANTIC_CHUNKS_TABLE = "semantic_chunks"
_EXPLICIT_COURSE_CODE = re.compile(r"(?<![0-9])([0-9]{8})(?![0-9])")
CONSTRAINED_RETRIEVAL_STATES = (
    "empty_structural_candidates",
    "description_missing",
    "vector_missing_or_invalid",
    "scored",
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ConstrainedTopicRetrievalResult:
    """Immutable state for one constrained topic-retrieval execution."""

    status: str
    candidates: tuple[Mapping[str, Any], ...] = ()
    scored_candidates: tuple[Mapping[str, Any], ...] = ()
    missing_description_course_ids: tuple[Any, ...] = ()
    missing_vector_chunk_ids: tuple[str, ...] = ()
    invalid_vector_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in CONSTRAINED_RETRIEVAL_STATES:
            raise ValueError(f"unsupported constrained retrieval status: {self.status!r}")
        object.__setattr__(
            self,
            "candidates",
            tuple(_freeze(candidate) for candidate in self.candidates),
        )
        object.__setattr__(
            self,
            "scored_candidates",
            tuple(_freeze(candidate) for candidate in self.scored_candidates),
        )
        object.__setattr__(
            self,
            "missing_description_course_ids",
            tuple(self.missing_description_course_ids),
        )
        object.__setattr__(
            self,
            "missing_vector_chunk_ids",
            tuple(self.missing_vector_chunk_ids),
        )
        object.__setattr__(
            self,
            "invalid_vector_chunk_ids",
            tuple(self.invalid_vector_chunk_ids),
        )


def _explicit_course_codes(query_text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(match.group(1) for match in _EXPLICIT_COURSE_CODE.finditer(query_text))
    )


def _normalize_course_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().casefold()


def _exact_course_name_codes(
    db_path: str | Path,
    query_text: str,
) -> tuple[str, ...]:
    normalized_query = _normalize_course_name(query_text)
    if not normalized_query:
        return ()
    try:
        with closing(sqlite3.connect(str(db_path))) as connection:
            rows = connection.execute(
                """
                SELECT course_code_normalized, name_th, name_en
                FROM courses
                WHERE name_th IS NOT NULL OR name_en IS NOT NULL
                """
            ).fetchall()
    except sqlite3.Error:
        return ()

    matched_codes: set[str] = set()
    for course_code, name_th, name_en in rows:
        if any(
            normalized_name and normalized_name in normalized_query
            for normalized_name in (
                _normalize_course_name(name_th),
                _normalize_course_name(name_en),
            )
        ):
            matched_codes.add(str(course_code))
    if len(matched_codes) != 1:
        return ()
    return tuple(matched_codes)


def _chunk_course_codes(chunk: dict[str, Any]) -> list[str]:
    values = chunk.get("course_code")
    if isinstance(values, (list, tuple, set)):
        return [str(value) for value in values]
    if values is None:
        return []
    return [str(values)]


def _course_code_candidates(
    db_path: str | Path,
    course_codes: tuple[str, ...],
) -> tuple[set[str], int]:
    try:
        with closing(sqlite3.connect(str(db_path))) as connection:
            rows = connection.execute(
                f"SELECT chunk_id, chunk_json FROM {_SEMANTIC_CHUNKS_TABLE}"
            ).fetchall()
    except sqlite3.Error:
        return set(), 0

    candidate_ids: set[str] = set()
    requested_codes = set(course_codes)
    for chunk_id, chunk_json in rows:
        chunk = json.loads(chunk_json)
        if requested_codes.intersection(_chunk_course_codes(chunk)):
            candidate_ids.add(chunk_id)
    return candidate_ids, len(rows)


def fetch_course_description_evidence(
    db_path: str | Path,
    course_id: int,
) -> list[dict[str, Any]]:
    """Fetch description evidence for one persisted course identity.

    This direct path intentionally does not embed or search.  A missing
    description is represented by an empty list, and no distance is invented
    for evidence that was not produced by nearest-neighbor search.
    """
    try:
        with closing(sqlite3.connect(str(db_path))) as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, chunk_json
                FROM {_SEMANTIC_CHUNKS_TABLE}
                WHERE chunk_json IS NOT NULL
                ORDER BY chunk_id
                """
            ).fetchall()
    except sqlite3.Error:
        return []

    evidence: list[dict[str, Any]] = []
    for chunk_id, chunk_json in rows:
        try:
            chunk = json.loads(chunk_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(chunk, dict):
            continue
        if chunk.get("course_id") != course_id:
            continue
        if chunk.get("chunk_type") != "description":
            continue

        provenance = [
            dict(reference)
            for reference in chunk.get("provenance", [])
            if isinstance(reference, dict)
        ]
        result = dict(chunk)
        result.update(
            {
                "chunk_id": chunk_id,
                "text": chunk.get("text", ""),
                "distance": None,
                "source_page": [
                    reference.get("source_page")
                    for reference in provenance
                    if reference.get("source_page") is not None
                ],
                "provenance": provenance,
            }
        )
        evidence.append(result)
    return evidence


def map_course_candidates_to_description_evidence(
    db_path: str | Path,
    candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Attach direct description evidence to already-constrained candidates.

    Candidate records are copied without deduplication so the same logical
    course can remain present in multiple structural partitions.  A candidate
    with no description receives an explicit empty ``description_evidence``
    tuple; an empty input therefore remains distinguishable from missing
    description evidence for a non-empty candidate set.
    """
    mapped: list[dict[str, Any]] = []
    for candidate in candidates:
        if "course_id" not in candidate:
            raise ValueError("each candidate must contain course_id")
        mapped_candidate = dict(candidate)
        mapped_candidate["description_evidence"] = tuple(
            fetch_course_description_evidence(db_path, candidate["course_id"])
        )
        mapped.append(mapped_candidate)
    return tuple(mapped)


def enrich_candidate_description_scores(
    mapped_candidates: Iterable[Mapping[str, Any]],
    candidate_scores: Mapping[str, Any],
) -> dict[str, Any]:
    """Join candidate-only scores to mapped description evidence.

    Only description evidence already attached to a supplied candidate is
    enriched.  Score rows for unrelated chunks are ignored, and the same
    description score may therefore be copied into separate partition records
    without merging those records.
    """
    score_by_chunk_id = {
        score["chunk_id"]: score["distance"]
        for score in candidate_scores.get("scores", [])
    }
    enriched_candidates: list[dict[str, Any]] = []
    for candidate in mapped_candidates:
        enriched_candidate = dict(candidate)
        enriched_evidence: list[dict[str, Any]] = []
        for evidence in candidate.get("description_evidence", ()):
            enriched = dict(evidence)
            chunk_id = enriched.get("chunk_id")
            if chunk_id in score_by_chunk_id:
                enriched["distance"] = float(score_by_chunk_id[chunk_id])
            else:
                enriched["distance"] = None
            enriched_evidence.append(enriched)
        enriched_candidate["description_evidence"] = tuple(enriched_evidence)
        enriched_candidates.append(enriched_candidate)

    return {
        "candidates": tuple(enriched_candidates),
        "missing_chunk_ids": list(candidate_scores.get("missing_chunk_ids", [])),
        "invalid_chunk_ids": list(candidate_scores.get("invalid_chunk_ids", [])),
    }


def make_constrained_topic_retrieval_result(
    mapped_candidates: Iterable[Mapping[str, Any]],
    candidate_scores: Mapping[str, Any],
) -> ConstrainedTopicRetrievalResult:
    """Build immutable constrained-retrieval state from mapped candidates/scores."""
    candidates = tuple(mapped_candidates)
    missing_description_course_ids = tuple(
        candidate.get("course_id")
        for candidate in candidates
        if not candidate.get("description_evidence")
    )
    scored_candidates = tuple(
        candidate
        for candidate in candidates
        if any(
            evidence.get("distance") is not None
            for evidence in candidate.get("description_evidence", ())
        )
    )
    missing_vector_chunk_ids = tuple(candidate_scores.get("missing_chunk_ids", ()))
    invalid_vector_chunk_ids = tuple(candidate_scores.get("invalid_chunk_ids", ()))

    if not candidates:
        status = "empty_structural_candidates"
    elif scored_candidates:
        status = "scored"
    elif missing_description_course_ids and all(
        not candidate.get("description_evidence") for candidate in candidates
    ):
        status = "description_missing"
    else:
        status = "vector_missing_or_invalid"

    return ConstrainedTopicRetrievalResult(
        status=status,
        candidates=candidates,
        scored_candidates=scored_candidates,
        missing_description_course_ids=missing_description_course_ids,
        missing_vector_chunk_ids=missing_vector_chunk_ids,
        invalid_vector_chunk_ids=invalid_vector_chunk_ids,
    )


def retrieve_constrained_topic_evidence(
    db_path: str | Path,
    topic: str,
    candidate_courses: Iterable[Mapping[str, Any]],
) -> ConstrainedTopicRetrievalResult:
    """Retrieve topic evidence only within already-constrained courses."""
    mapped_candidates = map_course_candidates_to_description_evidence(
        db_path, tuple(candidate_courses)
    )
    empty_scores = {
        "scores": [],
        "missing_chunk_ids": [],
        "invalid_chunk_ids": [],
    }
    if not mapped_candidates:
        return make_constrained_topic_retrieval_result(mapped_candidates, empty_scores)

    description_chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    for candidate in mapped_candidates:
        for evidence in candidate.get("description_evidence", ()):
            chunk_id = evidence.get("chunk_id")
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                description_chunk_ids.append(chunk_id)
    if not description_chunk_ids:
        return make_constrained_topic_retrieval_result(mapped_candidates, empty_scores)

    query_embeddings = embed_texts([topic])
    if len(query_embeddings) != 1:
        raise ValueError("the query embedder must return one embedding")
    candidate_scores = score_candidate_vectors(
        db_path, query_embeddings[0], description_chunk_ids
    )
    enriched = enrich_candidate_description_scores(
        mapped_candidates, candidate_scores
    )
    return make_constrained_topic_retrieval_result(
        enriched["candidates"], candidate_scores
    )


def retrieve(
    db_path: str | Path,
    query_text: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Embed one query and return ranked evidence with source references."""
    course_codes = _explicit_course_codes(query_text)
    if not course_codes:
        course_codes = _exact_course_name_codes(db_path, query_text)
    candidate_ids: set[str] | None = None
    search_k = k
    if course_codes:
        candidate_ids, semantic_chunk_count = _course_code_candidates(
            db_path, course_codes
        )
        if not candidate_ids:
            return []
        search_k = max(k, semantic_chunk_count)

    query_embeddings = embed_texts([query_text])
    if len(query_embeddings) != 1:
        raise ValueError("the query embedder must return one embedding")

    hits = search(db_path, query_embeddings[0], k=search_k)
    if candidate_ids is not None:
        hits = [hit for hit in hits if hit["chunk_id"] in candidate_ids]
    evidence: list[dict[str, Any]] = []
    for hit in hits:
        provenance = [dict(reference) for reference in hit.get("provenance", [])]
        evidence.append(
            {
                "chunk_id": hit["chunk_id"],
                "text": hit["text"],
                "distance": float(hit["distance"]),
                "source_page": [
                    reference.get("source_page")
                    for reference in provenance
                    if reference.get("source_page") is not None
                ],
                "provenance": provenance,
            }
        )

    evidence.sort(key=lambda item: (item["distance"], item["chunk_id"]))
    return evidence[:k]


__all__ = [
    "CONSTRAINED_RETRIEVAL_STATES",
    "ConstrainedTopicRetrievalResult",
    "fetch_course_description_evidence",
    "enrich_candidate_description_scores",
    "make_constrained_topic_retrieval_result",
    "map_course_candidates_to_description_evidence",
    "retrieve_constrained_topic_evidence",
    "retrieve",
]
