"""Retrieve ranked evidence for a natural-language query."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from collections.abc import Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .embedder import embed_texts
from .search import search
from .threshold import is_topic_match
from .vector_store import compare_stored_vectors, score_candidate_vectors


_SEMANTIC_CHUNKS_TABLE = "semantic_chunks"
_EXPLICIT_COURSE_CODE = re.compile(r"(?<![0-9])([0-9]{8})(?![0-9])")
CONSTRAINED_TOPIC_DISTANCE_THRESHOLD = 0.4428954516935646
CONSTRAINED_RETRIEVAL_STATES = (
    "empty_structural_candidates",
    "description_missing",
    "vector_missing_or_invalid",
    "no_threshold_matches",
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


SIMILARITY_STATES = ("complete", "valid_empty", "insufficient_evidence")


def _stable_partition_key(partition: Mapping[str, Any]) -> str:
    return json.dumps(dict(partition), ensure_ascii=False, sort_keys=True, default=str)


def _description_partition(evidence: Mapping[str, Any]) -> dict[str, Any]:
    partition = evidence.get("partition")
    if isinstance(partition, Mapping):
        return dict(partition)
    if evidence.get("plan") is not None:
        return {"plan": evidence.get("plan")}
    return {}


def _fetch_description_chunk(
    db_path: str | Path,
    chunk_id: str,
) -> dict[str, Any] | None:
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("chunk_id must be a non-empty string")
    try:
        with closing(sqlite3.connect(str(db_path))) as connection:
            row = connection.execute(
                f"SELECT chunk_id, chunk_json FROM {_SEMANTIC_CHUNKS_TABLE} "
                "WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        chunk = json.loads(row[1])
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(chunk, dict) or chunk.get("chunk_type") != "description":
        return None
    provenance = [
        dict(reference)
        for reference in chunk.get("provenance", ())
        if isinstance(reference, Mapping)
    ]
    result = dict(chunk)
    result.update(
        {
            "chunk_id": row[0],
            "text": chunk.get("text", ""),
            "distance": None,
            "provenance": provenance,
            "source_page": [
                reference.get("source_page")
                for reference in provenance
                if reference.get("source_page") is not None
            ],
        }
    )
    return result


def _description_evidence_validation_error(
    evidence: Any,
    chunk_id: str,
) -> str | None:
    if not isinstance(evidence, Mapping):
        return "invalid_supplied_evidence"
    if (
        not isinstance(evidence.get("chunk_id"), str)
        or not evidence["chunk_id"].strip()
        or evidence.get("chunk_id") != chunk_id
    ):
        return "inconsistent_evidence"
    if evidence.get("chunk_type") != "description":
        return "invalid_supplied_evidence"
    if (
        not isinstance(evidence.get("program"), str)
        or not evidence["program"].strip()
        or not isinstance(evidence.get("course_code"), str)
        or not evidence["course_code"].strip()
    ):
        return "invalid_supplied_evidence"
    if not _description_partition(evidence):
        return "invalid_supplied_evidence"
    text = evidence.get("text", evidence.get("description"))
    if not isinstance(text, str) or not text.strip():
        return "invalid_supplied_evidence"
    provenance = evidence.get("provenance")
    if (
        not isinstance(provenance, (list, tuple))
        or not provenance
        or any(not isinstance(reference, Mapping) for reference in provenance)
    ):
        return "invalid_supplied_evidence"
    return None


def _evidence_matches_persisted_description(
    supplied: Mapping[str, Any],
    persisted: Mapping[str, Any],
) -> bool:
    for field in ("chunk_id", "chunk_type", "program", "course_code"):
        if supplied.get(field) != persisted.get(field):
            return False
    if (
        "course_id" in supplied
        and "course_id" in persisted
        and supplied["course_id"] != persisted["course_id"]
    ):
        return False
    supplied_text = supplied.get("text", supplied.get("description"))
    if supplied_text != persisted.get("text", persisted.get("description")):
        return False
    if tuple(_freeze(reference) for reference in supplied["provenance"]) != tuple(
        _freeze(reference) for reference in persisted.get("provenance", ())
    ):
        return False
    supplied_partition = _description_partition(supplied)
    persisted_partition = _description_partition(persisted)
    return all(
        key not in persisted_partition
        or persisted_partition[key] == value
        for key, value in supplied_partition.items()
    )


def _similarity_evidence(
    db_path: str | Path,
    chunk_id: str,
    supplied: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if supplied is not None:
        invalid_reason = _description_evidence_validation_error(supplied, chunk_id)
        if invalid_reason is not None:
            return None, invalid_reason
        persisted = _fetch_description_chunk(db_path, chunk_id)
        if persisted is None:
            return None, "description_missing"
        if not _evidence_matches_persisted_description(supplied, persisted):
            return None, "inconsistent_evidence"
        return dict(supplied), None
    fetched = _fetch_description_chunk(db_path, chunk_id)
    if fetched is None:
        return None, "description_missing"
    invalid_reason = _description_evidence_validation_error(fetched, chunk_id)
    if invalid_reason is not None:
        return None, invalid_reason
    return fetched, None


@dataclass(frozen=True, slots=True)
class SimilarityPair:
    """One partition-local raw cosine comparison of two descriptions."""

    status: str
    partition: Mapping[str, Any]
    left: Mapping[str, Any]
    right: Mapping[str, Any]
    cosine_distance: float | None = None
    cosine_similarity: float | None = None
    reason: str | None = None
    missing_chunk_ids: tuple[str, ...] = ()
    invalid_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"complete", "insufficient_evidence"}:
            raise ValueError(f"unsupported similarity pair status: {self.status!r}")
        if not isinstance(self.partition, Mapping):
            raise TypeError("partition must be a mapping")
        if not isinstance(self.left, Mapping) or not isinstance(self.right, Mapping):
            raise TypeError("similarity sides must be mappings")
        if self.status == "complete":
            if self.cosine_distance is None or self.cosine_similarity is None:
                raise ValueError("complete similarity pair needs cosine values")
            if not math.isfinite(self.cosine_distance) or not math.isfinite(self.cosine_similarity):
                raise ValueError("cosine values must be finite")
        elif self.cosine_distance is not None or self.cosine_similarity is not None:
            raise ValueError("insufficient similarity cannot claim cosine values")
        object.__setattr__(self, "partition", _freeze(self.partition))
        object.__setattr__(self, "left", _freeze(self.left))
        object.__setattr__(self, "right", _freeze(self.right))
        object.__setattr__(self, "missing_chunk_ids", tuple(self.missing_chunk_ids))
        object.__setattr__(self, "invalid_chunk_ids", tuple(self.invalid_chunk_ids))


@dataclass(frozen=True, slots=True)
class SimilarityEvidence:
    """Per-partition exact-course similarity evidence and raw summaries."""

    status: str
    pairs: tuple[SimilarityPair, ...] = ()
    unmatched_partitions: tuple[Mapping[str, Any], ...] = ()
    mean_distance: float | None = None
    min_distance: float | None = None
    max_distance: float | None = None

    def __post_init__(self) -> None:
        if self.status not in SIMILARITY_STATES:
            raise ValueError(f"unsupported similarity status: {self.status!r}")
        pairs = tuple(self.pairs)
        if any(not isinstance(pair, SimilarityPair) for pair in pairs):
            raise TypeError("pairs must contain SimilarityPair values")
        unmatched = tuple(_freeze(partition) for partition in self.unmatched_partitions)
        if any(not isinstance(partition, Mapping) for partition in unmatched):
            raise TypeError("unmatched_partitions must contain mappings")
        complete_distances = [
            pair.cosine_distance for pair in pairs if pair.status == "complete"
        ]
        if self.status == "valid_empty" and pairs:
            raise ValueError("valid_empty similarity cannot contain pairs")
        if self.status == "complete":
            if not pairs or len(complete_distances) != len(pairs):
                raise ValueError("complete similarity needs complete pairs")
            expected = (
                sum(complete_distances) / len(complete_distances),
                min(complete_distances),
                max(complete_distances),
            )
            if (self.mean_distance, self.min_distance, self.max_distance) != expected:
                raise ValueError("similarity summaries do not match pair distances")
        elif any(
            value is not None
            for value in (self.mean_distance, self.min_distance, self.max_distance)
        ):
            raise ValueError("non-complete similarity cannot claim summaries")
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "unmatched_partitions", unmatched)


def compare_exact_description_vectors(
    db_path: str | Path,
    left_chunk_id: str,
    right_chunk_id: str,
    *,
    left_evidence: Mapping[str, Any] | None = None,
    right_evidence: Mapping[str, Any] | None = None,
    partition: Mapping[str, Any] | None = None,
) -> SimilarityPair:
    """Compare two exact persisted description chunks without global search."""
    if any(
        not isinstance(chunk_id, str) or not chunk_id.strip()
        for chunk_id in (left_chunk_id, right_chunk_id)
    ):
        return SimilarityPair(
            "insufficient_evidence",
            {},
            {},
            {},
            reason="invalid_supplied_evidence",
        )
    if partition is not None and (
        not isinstance(partition, Mapping) or not partition
    ):
        return SimilarityPair(
            "insufficient_evidence",
            {},
            {},
            {},
            reason="invalid_supplied_evidence",
        )
    left, left_reason = _similarity_evidence(db_path, left_chunk_id, left_evidence)
    right, right_reason = _similarity_evidence(db_path, right_chunk_id, right_evidence)
    left_value = left or {}
    right_value = right or {}
    left_partition = _description_partition(left_value)
    right_partition = _description_partition(right_value)
    selected_partition = dict(partition or left_partition)
    if left_reason is not None or right_reason is not None:
        missing_chunk_ids = tuple(
            chunk_id
            for chunk_id, reason in (
                (left_chunk_id, left_reason),
                (right_chunk_id, right_reason),
            )
            if reason == "description_missing"
        )
        return SimilarityPair(
            "insufficient_evidence",
            selected_partition,
            left_value,
            right_value,
            reason=left_reason or right_reason,
            missing_chunk_ids=missing_chunk_ids,
        )
    if (
        partition is not None
        and (left_partition != selected_partition or right_partition != selected_partition)
    ):
        return SimilarityPair(
            "insufficient_evidence",
            selected_partition,
            left_value,
            right_value,
            reason="partition_mismatch",
        )
    if left is not None and right is not None and left_partition != right_partition:
        return SimilarityPair(
            "insufficient_evidence",
            selected_partition,
            left_value,
            right_value,
            reason="partition_mismatch",
        )
    if left is None or right is None:
        return SimilarityPair(
            "insufficient_evidence",
            selected_partition,
            left_value,
            right_value,
            reason="description_missing",
        )

    vector_result = compare_stored_vectors(db_path, left_chunk_id, right_chunk_id)
    if vector_result["status"] != "complete":
        return SimilarityPair(
            "insufficient_evidence",
            selected_partition,
            left,
            right,
            reason="vector_missing_or_invalid",
            missing_chunk_ids=tuple(vector_result["missing_chunk_ids"]),
            invalid_chunk_ids=tuple(vector_result["invalid_chunk_ids"]),
        )
    return SimilarityPair(
        "complete",
        selected_partition,
        left,
        right,
        cosine_distance=vector_result["cosine_distance"],
        cosine_similarity=vector_result["cosine_similarity"],
    )


def _course_records(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (str, bytes)):
        raise TypeError("course input must be a mapping or iterable of mappings")
    records = tuple(value)
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError("course input must contain mappings")
    return records


def _partitioned_descriptions(
    value: Any,
) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    partitions: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for course in _course_records(value):
        raw_partition = course.get("partition", {})
        if not isinstance(raw_partition, Mapping):
            raise ValueError("course partition must be a mapping")
        partition = dict(raw_partition)
        key = _stable_partition_key(partition)
        if key not in partitions:
            partitions[key] = (partition, [])
        descriptions = course.get("description_evidence", ()) or ()
        for description in descriptions:
            if not isinstance(description, Mapping):
                raise ValueError("description evidence must contain mappings")
            if not isinstance(description.get("chunk_id"), str) or not description["chunk_id"]:
                raise ValueError("description evidence needs a chunk_id")
            enriched = dict(description)
            enriched.setdefault("program", course.get("program"))
            enriched.setdefault("course_code", course.get("course_code"))
            enriched.setdefault("course_id", course.get("course_id"))
            enriched["partition"] = dict(partition)
            partitions[key][1].append(enriched)
    return partitions


def _unique_descriptions(
    descriptions: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...] | None:
    unique: dict[str, dict[str, Any]] = {}
    for description in descriptions:
        text = description.get("text", "")
        if not isinstance(text, str):
            return None
        unique.setdefault(text, dict(description))
    return tuple(unique.values())


def aggregate_exact_course_similarity(
    db_path: str | Path,
    left_course: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    right_course: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    selected_plan: str | None = None,
) -> SimilarityEvidence:
    """Compare exact-course descriptions only in matching preserved partitions."""
    if selected_plan is not None and (
        not isinstance(selected_plan, str) or not selected_plan.strip()
    ):
        return SimilarityEvidence(status="insufficient_evidence")
    try:
        left_partitions = _partitioned_descriptions(left_course)
        right_partitions = _partitioned_descriptions(right_course)
    except (TypeError, ValueError):
        return SimilarityEvidence(status="insufficient_evidence")
    if selected_plan is not None:
        left_partitions = {
            key: value
            for key, value in left_partitions.items()
            if value[0].get("plan") == selected_plan
        }
        right_partitions = {
            key: value
            for key, value in right_partitions.items()
            if value[0].get("plan") == selected_plan
        }

    common_keys = sorted(set(left_partitions) & set(right_partitions))
    unmatched = [
        left_partitions[key][0] for key in sorted(set(left_partitions) - set(common_keys))
    ] + [
        right_partitions[key][0] for key in sorted(set(right_partitions) - set(common_keys))
    ]
    if not common_keys:
        return SimilarityEvidence(
            status="valid_empty",
            unmatched_partitions=tuple(unmatched),
        )

    pairs: list[SimilarityPair] = []
    for key in common_keys:
        left_partition, left_descriptions = left_partitions[key]
        right_partition, right_descriptions = right_partitions[key]
        left_unique = _unique_descriptions(left_descriptions)
        right_unique = _unique_descriptions(right_descriptions)
        if left_unique is None or right_unique is None or len(left_unique) != 1 or len(right_unique) != 1:
            pairs.append(
                SimilarityPair(
                    "insufficient_evidence",
                    left_partition,
                    left_unique[0] if left_unique else {},
                    right_unique[0] if right_unique else {},
                    reason="multiple_non_identical_descriptions"
                    if (left_unique and len(left_unique) > 1)
                    or (right_unique and len(right_unique) > 1)
                    else "description_missing",
                )
            )
            continue
        pairs.append(
            compare_exact_description_vectors(
                db_path,
                left_unique[0]["chunk_id"],
                right_unique[0]["chunk_id"],
                left_evidence=left_unique[0],
                right_evidence=right_unique[0],
                partition=left_partition,
            )
        )

    if any(pair.status != "complete" for pair in pairs):
        return SimilarityEvidence(
            status="insufficient_evidence",
            pairs=tuple(pairs),
            unmatched_partitions=tuple(unmatched),
        )
    distances = [pair.cosine_distance for pair in pairs]
    return SimilarityEvidence(
        status="complete",
        pairs=tuple(pairs),
        unmatched_partitions=tuple(unmatched),
        mean_distance=sum(distances) / len(distances),
        min_distance=min(distances),
        max_distance=max(distances),
    )


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


def _normalize_lexical_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character if character.isalnum() else " " for character in value)


def _ascii_phrase_match(topic: str, text: str) -> bool:
    topic_tokens = topic.split()
    if not topic_tokens:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(
        re.escape(token) for token in topic_tokens
    ) + r"(?!\w)"
    return re.search(pattern, text) is not None


def lexical_topic_match(
    topic: str,
    candidate: Mapping[str, Any],
) -> bool:
    """Return whether a candidate has conservative lexical topic evidence.

    Course names and description text are checked without semantic expansion.
    ASCII topics use whole-token phrase matching; non-ASCII topics use the
    complete normalized phrase.  The sole alias pair is AI/artificial
    intelligence.
    """
    normalized_topic = _normalize_lexical_text(topic)
    if not normalized_topic:
        return False

    topic_variants = (normalized_topic,)
    if normalized_topic in {"ai", "artificial intelligence"}:
        topic_variants = ("ai", "artificial intelligence")

    searchable_texts: list[str] = []
    for field in ("course_name", "name_th", "name_en"):
        normalized_value = _normalize_lexical_text(candidate.get(field))
        if normalized_value:
            searchable_texts.append(normalized_value)
    for evidence in candidate.get("description_evidence", ()):
        if not isinstance(evidence, Mapping):
            continue
        normalized_value = _normalize_lexical_text(
            evidence.get("text", evidence.get("description"))
        )
        if normalized_value:
            searchable_texts.append(normalized_value)

    for text in searchable_texts:
        for variant in topic_variants:
            if variant.isascii() and _ascii_phrase_match(variant, text):
                return True
            if not variant.isascii() and variant in text:
                return True
    return False


def _candidate_topic_matches(
    topic: str,
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    matches: list[Mapping[str, Any]] = []
    for candidate in candidates:
        evidence = candidate.get("description_evidence", ())
        if not evidence:
            continue
        lexical_match = lexical_topic_match(topic, candidate)
        semantic_match = any(
            evidence_item.get("distance") is not None
            and is_topic_match(
                evidence_item["distance"], CONSTRAINED_TOPIC_DISTANCE_THRESHOLD
            )
            for evidence_item in evidence
            if isinstance(evidence_item, Mapping)
        )
        if lexical_match or semantic_match:
            matches.append(candidate)
    return tuple(matches)


def make_constrained_topic_retrieval_result(
    mapped_candidates: Iterable[Mapping[str, Any]],
    candidate_scores: Mapping[str, Any],
    topic_matches: Iterable[Mapping[str, Any]] | None = None,
) -> ConstrainedTopicRetrievalResult:
    """Build immutable constrained-retrieval state from mapped candidates/scores."""
    candidates = tuple(mapped_candidates)
    missing_description_course_ids = tuple(
        candidate.get("course_id")
        for candidate in candidates
        if not candidate.get("description_evidence")
    )
    if topic_matches is None:
        scored_candidates = tuple(
            candidate
            for candidate in candidates
            if any(
                evidence.get("distance") is not None
                and is_topic_match(
                    evidence["distance"], CONSTRAINED_TOPIC_DISTANCE_THRESHOLD
                )
                for evidence in candidate.get("description_evidence", ())
                if isinstance(evidence, Mapping)
            )
        )
    else:
        scored_candidates = tuple(topic_matches)
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
    elif candidate_scores.get("scores"):
        status = "no_threshold_matches"
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
    topic_matches = _candidate_topic_matches(topic, enriched["candidates"])
    return make_constrained_topic_retrieval_result(
        enriched["candidates"], candidate_scores, topic_matches
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
    "CONSTRAINED_TOPIC_DISTANCE_THRESHOLD",
    "ConstrainedTopicRetrievalResult",
    "SIMILARITY_STATES",
    "SimilarityEvidence",
    "SimilarityPair",
    "aggregate_exact_course_similarity",
    "compare_exact_description_vectors",
    "fetch_course_description_evidence",
    "enrich_candidate_description_scores",
    "make_constrained_topic_retrieval_result",
    "map_course_candidates_to_description_evidence",
    "lexical_topic_match",
    "retrieve_constrained_topic_evidence",
    "retrieve",
]
