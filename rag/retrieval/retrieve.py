"""Retrieve ranked evidence for a natural-language query."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .embedder import embed_texts
from .search import search


_SEMANTIC_CHUNKS_TABLE = "semantic_chunks"
_EXPLICIT_COURSE_CODE = re.compile(r"(?<![0-9])([0-9]{8})(?![0-9])")


def _explicit_course_code(query_text: str) -> str | None:
    match = _EXPLICIT_COURSE_CODE.search(query_text)
    return match.group(1) if match is not None else None


def _chunk_course_codes(chunk: dict[str, Any]) -> list[str]:
    values = chunk.get("course_code")
    if isinstance(values, (list, tuple, set)):
        return [str(value) for value in values]
    if values is None:
        return []
    return [str(values)]


def _course_code_candidates(
    db_path: str | Path,
    course_code: str,
) -> tuple[set[str], int]:
    try:
        with closing(sqlite3.connect(str(db_path))) as connection:
            rows = connection.execute(
                f"SELECT chunk_id, chunk_json FROM {_SEMANTIC_CHUNKS_TABLE}"
            ).fetchall()
    except sqlite3.Error:
        return set(), 0

    candidate_ids: set[str] = set()
    for chunk_id, chunk_json in rows:
        chunk = json.loads(chunk_json)
        if course_code in _chunk_course_codes(chunk):
            candidate_ids.add(chunk_id)
    return candidate_ids, len(rows)


def retrieve(
    db_path: str | Path,
    query_text: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Embed one query and return ranked evidence with source references."""
    course_code = _explicit_course_code(query_text)
    candidate_ids: set[str] | None = None
    search_k = k
    if course_code is not None:
        candidate_ids, semantic_chunk_count = _course_code_candidates(
            db_path, course_code
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


__all__ = ["retrieve"]
