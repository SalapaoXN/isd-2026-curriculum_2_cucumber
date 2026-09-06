"""Join vector hits to structured retrieval chunks and provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunks import build_chunks
from .vector_store import nearest_neighbor_search


def search(
    db_path: str | Path,
    query_embedding: Any,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return deterministic vector hits enriched with chunk text and provenance."""
    chunks_by_id = {
        chunk["chunk_id"]: chunk for chunk in build_chunks(db_path)
    }
    hits = nearest_neighbor_search(db_path, query_embedding, limit=k)

    results: list[dict[str, Any]] = []
    for hit in hits:
        chunk_id = hit["chunk_id"]
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise KeyError(f"vector hit has no structured chunk: {chunk_id}")
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
