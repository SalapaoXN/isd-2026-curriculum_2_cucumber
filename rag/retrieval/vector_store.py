"""SQLite-vec storage and nearest-neighbor search for retrieval chunks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np


VECTOR_TABLE = "vector"
VECTOR_COLUMN = "embed_byte"
EMBEDDING_DIMENSION = 384


def _load_sqlite_vec(connection: sqlite3.Connection) -> None:
    """Load sqlite-vec without requiring a machine-specific extension path."""
    import sqlite_vec

    connection.enable_load_extension(True)
    try:
        sqlite_vec.load(connection)
    finally:
        connection.enable_load_extension(False)


def _create_vector_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {VECTOR_TABLE} USING vec0(
            {VECTOR_COLUMN} float[{EMBEDDING_DIMENSION}],
            +chunk_id TEXT
        )
        """
    )


def create_vector_table(database_path: str | Path) -> None:
    """Create the sqlite-vec table in an existing RAG SQLite database."""
    with closing(sqlite3.connect(str(database_path))) as connection:
        _load_sqlite_vec(connection)
        _create_vector_table(connection)
        connection.commit()


def _chunk_ids(chunks: Iterable[str | Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") if isinstance(chunk, Mapping) else chunk
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("each vector must have a non-empty chunk_id")
        if chunk_id in ids:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        ids.append(chunk_id)
    return ids


def _embedding_matrix(embeddings: Any, count: int) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1 and count == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape != (count, EMBEDDING_DIMENSION):
        raise ValueError(
            f"embeddings must have shape ({count}, {EMBEDDING_DIMENSION})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def insert_embeddings(
    database_path: str | Path,
    chunks: Iterable[str | Mapping[str, Any]],
    embeddings: Any,
) -> None:
    """Insert float32 embeddings in chunk order, keyed by each chunk_id."""
    ids = _chunk_ids(chunks)
    matrix = _embedding_matrix(embeddings, len(ids))
    rows = [
        (matrix[index].tobytes(), chunk_id) for index, chunk_id in enumerate(ids)
    ]

    with closing(sqlite3.connect(str(database_path))) as connection:
        _load_sqlite_vec(connection)
        _create_vector_table(connection)
        if rows:
            connection.executemany(
                f"INSERT INTO {VECTOR_TABLE} ({VECTOR_COLUMN}, chunk_id) "
                "VALUES (?, ?)",
                rows,
            )
        connection.commit()


def _query_embedding(embedding: Any) -> bytes:
    matrix = _embedding_matrix(embedding, 1)
    return matrix[0].tobytes()


def nearest_neighbor_search(
    database_path: str | Path,
    embedding: Any,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return nearest chunk IDs and sqlite-vec distances."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    query_vector = _query_embedding(embedding)
    with closing(sqlite3.connect(str(database_path))) as connection:
        _load_sqlite_vec(connection)
        rows = connection.execute(
            f"""
            SELECT chunk_id, distance
            FROM {VECTOR_TABLE}
            WHERE {VECTOR_COLUMN} MATCH ? AND k = ?
            ORDER BY distance
            """,
            (query_vector, limit),
        ).fetchall()
    return [
        {"chunk_id": row[0], "distance": float(row[1])}
        for row in rows
    ]


def score_candidate_vectors(
    database_path: str | Path,
    query_embedding: Any,
    candidate_chunk_ids: Iterable[str],
) -> dict[str, Any]:
    """Score only the supplied stored vectors, without k-nearest search.

    The returned mapping keeps valid scores separate from missing and invalid
    IDs.  Invalid vectors are never assigned a distance, and result ordering
    is deterministic regardless of database row order.
    """
    candidate_ids: list[str] = []
    seen: set[str] = set()
    for chunk_id in candidate_chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("each candidate chunk_id must be a non-empty string")
        if chunk_id not in seen:
            seen.add(chunk_id)
            candidate_ids.append(chunk_id)
    if not candidate_ids:
        return {
            "scores": [],
            "missing_chunk_ids": [],
            "invalid_chunk_ids": [],
        }

    query_matrix = _embedding_matrix(query_embedding, 1)
    query_vector = query_matrix[0]
    query_norm = float(np.linalg.norm(query_vector))
    if not np.isfinite(query_norm) or query_norm == 0.0:
        raise ValueError("query embedding must have a finite non-zero norm")

    placeholders = ", ".join("?" for _ in candidate_ids)
    with closing(sqlite3.connect(str(database_path))) as connection:
        rows = connection.execute(
            f"SELECT chunk_id, {VECTOR_COLUMN} FROM {VECTOR_TABLE} "
            f"WHERE chunk_id IN ({placeholders})",
            candidate_ids,
        ).fetchall()

    rows_by_id = {row[0]: row[1] for row in rows}
    missing_ids = [chunk_id for chunk_id in candidate_ids if chunk_id not in rows_by_id]
    invalid_ids: list[str] = []
    scores: list[dict[str, Any]] = []
    for chunk_id in candidate_ids:
        if chunk_id not in rows_by_id:
            continue
        try:
            vector = np.frombuffer(rows_by_id[chunk_id], dtype=np.float32)
            vector = _embedding_matrix(vector, 1)[0]
            vector_norm = float(np.linalg.norm(vector))
            if not np.isfinite(vector_norm) or vector_norm == 0.0:
                raise ValueError("candidate vector must have a finite non-zero norm")
            similarity = float(
                np.dot(query_vector, vector) / (query_norm * vector_norm)
            )
        except (TypeError, ValueError):
            invalid_ids.append(chunk_id)
            continue
        scores.append(
            {
                "chunk_id": chunk_id,
                "distance": float(1.0 - similarity),
            }
        )

    scores.sort(key=lambda result: (result["distance"], result["chunk_id"]))
    return {
        "scores": scores,
        "missing_chunk_ids": missing_ids,
        "invalid_chunk_ids": invalid_ids,
    }


__all__ = [
    "EMBEDDING_DIMENSION",
    "VECTOR_COLUMN",
    "VECTOR_TABLE",
    "create_vector_table",
    "insert_embeddings",
    "nearest_neighbor_search",
    "score_candidate_vectors",
]
