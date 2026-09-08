"""Join vector hits to structured retrieval chunks and provenance."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .chunks import build_chunks
from .vector_store import nearest_neighbor_search


_PERSISTED_CHUNKS_TABLE = "semantic_chunks"


def _persisted_chunks(
    database_path: str | Path,
    chunk_ids: list[str],
) -> dict[str, dict[str, Any]] | None:
    if not chunk_ids:
        return {}
    placeholders = ", ".join("?" for _ in chunk_ids)
    try:
        with closing(sqlite3.connect(str(database_path))) as connection:
            rows = connection.execute(
                f"SELECT chunk_id, chunk_json FROM {_PERSISTED_CHUNKS_TABLE} "
                f"WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
    except sqlite3.Error:
        return None
    return {row[0]: json.loads(row[1]) for row in rows}


def search(
    db_path: str | Path,
    query_embedding: Any,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return deterministic vector hits enriched with chunk text and provenance."""
    hits = nearest_neighbor_search(db_path, query_embedding, limit=k)
    chunks_by_id = _persisted_chunks(
        db_path, [hit["chunk_id"] for hit in hits]
    )
    if chunks_by_id is None:
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in build_chunks(db_path)}

    results: list[dict[str, Any]] = []
    for hit in hits:
        chunk_id = hit["chunk_id"]
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise KeyError(f"vector hit has no persisted chunk: {chunk_id}")
        results.append(
            {
                "chunk_id": chunk_id,
                "distance": float(hit["distance"]),
                "text": chunk["text"],
                "provenance": [dict(reference) for reference in chunk["provenance"]],
            }
        )

    results.sort(key=lambda result: (result["distance"], result["chunk_id"]))
    return results


__all__ = ["search"]
