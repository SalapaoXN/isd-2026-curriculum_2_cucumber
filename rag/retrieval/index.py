"""Persistent semantic index construction and querying."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from rag.structured.loader import load_json_to_sqlite

from .chunks import build_chunks
from .embedder import EMBEDDING_DIMENSION, MODEL_NAME, embed_texts
from .vector_store import insert_embeddings, nearest_neighbor_search


ARTIFACTS_DIR = Path("rag_artifacts")
_METADATA_TABLE = "semantic_index_metadata"
_CHUNKS_TABLE = "semantic_chunks"


def _effective_settings(
    embedding_model_identity: str | None,
    vector_dimension: int | None,
) -> tuple[str, int]:
    model_identity = embedding_model_identity or MODEL_NAME
    dimension = EMBEDDING_DIMENSION if vector_dimension is None else vector_dimension
    if not isinstance(model_identity, str) or not model_identity.strip():
        raise ValueError("embedding_model_identity must be a non-empty string")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("vector_dimension must be a positive integer")
    return model_identity, dimension


def _embedder(
    embed_texts_callable: Callable[[Iterable[str]], Any] | None,
) -> Callable[[Iterable[str]], Any]:
    if embed_texts_callable is not None:
        if not callable(embed_texts_callable):
            raise ValueError("embed_texts_callable must be callable")
        return embed_texts_callable
    return embed_texts


def _source_fingerprint(input_path: Path) -> str:
    return hashlib.sha256(input_path.read_bytes()).hexdigest()


def index_path_for_source(
    input_json_path: str | Path,
    artifact_dir: str | Path = ARTIFACTS_DIR,
) -> Path:
    return Path(artifact_dir) / f"{Path(input_json_path).stem}.db"


def _artifact_path(
    input_json_path: Path,
    index_path: str | Path | None,
    artifact_dir: str | Path,
) -> Path:
    artifact_root = Path(artifact_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    candidate = (
        Path(index_path)
        if index_path is not None
        else index_path_for_source(input_json_path, artifact_root)
    ).resolve()
    try:
        candidate.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("semantic index database must be stored under rag_artifacts") from error
    if candidate == input_json_path.resolve():
        raise ValueError("input JSON and semantic index database paths must differ")
    return candidate


def _remove_database(database_path: Path) -> None:
    for candidate in (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
        Path(f"{database_path}-journal"),
    ):
        if not candidate.exists():
            continue
        if not candidate.is_file():
            raise IsADirectoryError(candidate)
        candidate.unlink()


def _create_index_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_METADATA_TABLE} (
            metadata_id INTEGER PRIMARY KEY CHECK (metadata_id = 1),
            source_json_fingerprint TEXT NOT NULL,
            embedding_model_identity TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_CHUNKS_TABLE} (
            chunk_id TEXT PRIMARY KEY,
            chunk_json TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )


def _chunks_json(chunks: Iterable[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("each semantic chunk must have a non-empty chunk_id")
        text = str(chunk.get("text", ""))
        rows.append(
            (
                chunk_id,
                json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                text,
            )
        )
    return rows


def _write_metadata(
    database_path: Path,
    source_fingerprint: str,
    model_identity: str,
    vector_dimension: int,
    chunks: list[Mapping[str, Any]],
) -> None:
    chunk_rows = _chunks_json(chunks)
    with closing(sqlite3.connect(str(database_path))) as connection:
        _create_index_tables(connection)
        connection.execute(f"DELETE FROM {_CHUNKS_TABLE}")
        connection.executemany(
            f"INSERT INTO {_CHUNKS_TABLE} (chunk_id, chunk_json, text) VALUES (?, ?, ?)",
            chunk_rows,
        )
        connection.execute(f"DELETE FROM {_METADATA_TABLE}")
        connection.execute(
            f"""
            INSERT INTO {_METADATA_TABLE} (
                metadata_id, source_json_fingerprint, embedding_model_identity,
                vector_dimension, chunk_count
            ) VALUES (1, ?, ?, ?, ?)
            """,
            (source_fingerprint, model_identity, vector_dimension, len(chunks)),
        )
        connection.commit()


def _is_valid_index(
    database_path: Path,
    source_fingerprint: str,
    model_identity: str,
    vector_dimension: int,
) -> bool:
    if not database_path.is_file():
        return False
    try:
        with closing(sqlite3.connect(str(database_path))) as connection:
            row = connection.execute(
                f"""
                SELECT source_json_fingerprint, embedding_model_identity,
                       vector_dimension, chunk_count
                FROM {_METADATA_TABLE}
                WHERE metadata_id = 1
                """
            ).fetchone()
            if row is None:
                return False
            chunk_count = connection.execute(
                f"SELECT COUNT(*) FROM {_CHUNKS_TABLE}"
            ).fetchone()[0]
    except sqlite3.Error:
        return False
    return row == (source_fingerprint, model_identity, vector_dimension, chunk_count)


def ensure_index(
    input_json_path: str | Path,
    index_path: str | Path | None = None,
    *,
    artifact_dir: str | Path = ARTIFACTS_DIR,
    embed_texts_callable: Callable[[Iterable[str]], Any] | None = None,
    embedding_model_identity: str | None = None,
    vector_dimension: int | None = None,
) -> Path:
    """Build or reuse the persistent semantic index for one source JSON file."""
    input_path = Path(input_json_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    database_path = _artifact_path(input_path, index_path, artifact_dir)
    model_identity, dimension = _effective_settings(
        embedding_model_identity, vector_dimension
    )
    source_fingerprint = _source_fingerprint(input_path)

    if _is_valid_index(database_path, source_fingerprint, model_identity, dimension):
        return database_path

    _remove_database(database_path)
    load_json_to_sqlite(input_path, database_path)
    chunks = list(build_chunks(database_path))
    embed = _embedder(embed_texts_callable)
    embeddings = embed(chunk["text"] for chunk in chunks)
    insert_embeddings(database_path, chunks, embeddings)
    _write_metadata(
        database_path,
        source_fingerprint,
        model_identity,
        dimension,
        chunks,
    )
    return database_path


def _source_pages(chunk: Mapping[str, Any]) -> list[Any]:
    pages = chunk.get("source_page")
    if pages is not None:
        if isinstance(pages, (list, tuple, set)):
            return [page for page in pages if page is not None]
        return [pages]
    provenance = chunk.get("provenance")
    if isinstance(provenance, list):
        return [
            reference.get("source_page")
            for reference in provenance
            if isinstance(reference, Mapping) and reference.get("source_page") is not None
        ]
    return []


def search_index(
    index_database_path: str | Path,
    query_text: str,
    top_k: int = 5,
    *,
    embed_texts_callable: Callable[[Iterable[str]], Any] | None = None,
    vector_dimension: int | None = None,
) -> list[dict[str, Any]]:
    """Search a valid persistent index, embedding only the supplied question."""
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")
    _, dimension = _effective_settings(None, vector_dimension)
    embed = _embedder(embed_texts_callable)
    query_embedding = embed([query_text])
    hits = nearest_neighbor_search(index_database_path, query_embedding, limit=top_k)
    hit_ids = [hit["chunk_id"] for hit in hits]
    if not hit_ids:
        return []

    placeholders = ", ".join("?" for _ in hit_ids)
    with closing(sqlite3.connect(str(index_database_path))) as connection:
        rows = connection.execute(
            f"SELECT chunk_id, chunk_json FROM {_CHUNKS_TABLE} "
            f"WHERE chunk_id IN ({placeholders})",
            hit_ids,
        ).fetchall()
    chunks = {row[0]: json.loads(row[1]) for row in rows}

    results: list[dict[str, Any]] = []
    for hit in hits:
        chunk_id = hit["chunk_id"]
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise KeyError(f"vector hit has no persisted semantic chunk: {chunk_id}")
        results.append(
            {
                "chunk_id": chunk_id,
                "distance": float(hit["distance"]),
                "text": chunk.get("text", ""),
                "source_page": _source_pages(chunk),
                "provenance": [dict(reference) for reference in chunk.get("provenance", [])],
            }
        )
    results.sort(key=lambda result: (result["distance"], result["chunk_id"]))
    return results


def query_index(
    input_json_path: str | Path,
    query_text: str,
    top_k: int = 5,
    *,
    index_path: str | Path | None = None,
    artifact_dir: str | Path = ARTIFACTS_DIR,
    embed_texts_callable: Callable[[Iterable[str]], Any] | None = None,
    embedding_model_identity: str | None = None,
    vector_dimension: int | None = None,
) -> list[dict[str, Any]]:
    """Ensure a source index is current, then search it with one query embedding."""
    database_path = ensure_index(
        input_json_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        embed_texts_callable=embed_texts_callable,
        embedding_model_identity=embedding_model_identity,
        vector_dimension=vector_dimension,
    )
    return search_index(
        database_path,
        query_text,
        top_k=top_k,
        embed_texts_callable=embed_texts_callable,
        vector_dimension=vector_dimension,
    )


__all__ = [
    "ARTIFACTS_DIR",
    "ensure_index",
    "index_path_for_source",
    "query_index",
    "search_index",
]
