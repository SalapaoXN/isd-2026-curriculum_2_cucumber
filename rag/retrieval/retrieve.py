"""Retrieve ranked evidence for a natural-language query."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .embedder import embed_texts
from .search import search


def retrieve(
    db_path: str | Path,
    query_text: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Embed one query and return ranked evidence with source references."""
    query_embeddings = embed_texts([query_text])
    if len(query_embeddings) != 1:
        raise ValueError("the query embedder must return one embedding")

    hits = search(db_path, query_embeddings[0], k=k)
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
    return evidence


__all__ = ["retrieve"]
