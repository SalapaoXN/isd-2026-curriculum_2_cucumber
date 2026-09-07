"""Persistent unified curriculum database construction and vector querying."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from rag.structured.loader import load_jsons_to_sqlite

from .chunks import build_chunks
from .embedder import EMBEDDING_DIMENSION, MODEL_NAME, embed_texts
from .vector_store import insert_embeddings, nearest_neighbor_search


ARTIFACTS_DIR = Path("rag_artifacts")
DEFAULT_INDEX_NAME = "curriculum.db"
_METADATA_TABLE = "semantic_index_metadata"
_SOURCES_TABLE = "semantic_index_sources"
_CHUNKS_TABLE = "semantic_chunks"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONSOLIDATED_OUTPUTS_DIR = _PROJECT_ROOT / "consolidated_outputs"
_SOURCE_IDENTITY_FIELDS = (
    "source_document_key",
    "document_key",
    "document_id",
    "source_filename",
    "document_filename",
    "source_uri",
    "source_locator",
    "source",
)
_RELATIONAL_TABLES = {
    "catalogs",
    "programs",
    "courses",
    "curriculum_plans",
    "plan_placements",
    "prerequisites",
    "provenance",
}
_RELATIONAL_VIEWS = {
    "v_plan_courses",
    "v_semester_credits",
    "v_prerequisite_edges",
}


def canonical_source_paths() -> list[Path]:
    """Return every canonical consolidated curriculum document in the repository."""
    paths = sorted(_CONSOLIDATED_OUTPUTS_DIR.glob("merged_*_full.json"))
    if not paths:
        raise FileNotFoundError(
            "no canonical consolidated curriculum JSON files found in "
            f"{_CONSOLIDATED_OUTPUTS_DIR}"
        )
    return paths


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _source_paths(input_json_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(input_json_paths, (str, Path)):
        values = [input_json_paths]
    else:
        values = list(input_json_paths)
    if not values:
        raise ValueError("at least one consolidated JSON input is required")

    paths: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    paths.sort(key=str)
    return paths


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


def _source_document_key(entry: Mapping[str, Any]) -> str | None:
    for field in _SOURCE_IDENTITY_FIELDS:
        value = _as_text(entry.get(field))
        if value is not None and value.strip():
            return value.strip()
    return None


def _document_metadata(input_path: Path) -> tuple[str, str, str | None]:
    with input_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, Mapping):
        raise ValueError("consolidated JSON must contain an object at the top level")

    program = _as_text(document.get("program")) or "UNKNOWN"
    plan_value = document.get("plan")
    if isinstance(plan_value, Mapping):
        plan = _as_text(
            plan_value.get("plan_code")
            or plan_value.get("code")
            or plan_value.get("id")
            or plan_value.get("plan_name")
            or plan_value.get("name")
            or plan_value.get("title")
        ) or "default"
    else:
        plan = _as_text(plan_value) or _as_text(document.get("plan_name")) or "default"
    return program, plan, _source_document_key(document)


def index_path_for_source(
    input_json_paths: str | Path | Iterable[str | Path],
    artifact_dir: str | Path = ARTIFACTS_DIR,
) -> Path:
    """Return the shared curriculum database path for one or more source files."""
    return Path(artifact_dir) / DEFAULT_INDEX_NAME


def _artifact_path(
    input_json_path: Path,
    index_path: str | Path | None,
    artifact_dir: str | Path,
) -> Path:
    artifact_root = Path(artifact_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    if index_path is None:
        candidate = artifact_root / DEFAULT_INDEX_NAME
    else:
        raw_path = Path(index_path)
        candidate = (
            artifact_root / raw_path
            if not raw_path.is_absolute() and raw_path.parent == Path(".")
            else raw_path
        ).resolve()
    try:
        candidate.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("curriculum database must be stored under rag_artifacts") from error
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
        CREATE TABLE IF NOT EXISTS {_SOURCES_TABLE} (
            source_file_identity TEXT PRIMARY KEY,
            source_filename TEXT NOT NULL,
            source_json_fingerprint TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_CHUNKS_TABLE} (
            chunk_id TEXT PRIMARY KEY,
            chunk_json TEXT NOT NULL,
            text TEXT NOT NULL,
            source_file_identity TEXT NOT NULL
        )
        """
    )


def _source_metadata_rows(
    source_rows: list[tuple[str, str, str]],
) -> str:
    return json.dumps(
        [
            {
                "source_file_identity": identity,
                "source_filename": filename,
                "source_json_fingerprint": fingerprint,
            }
            for identity, filename, fingerprint in source_rows
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _chunks_json(chunks: Iterable[Mapping[str, Any]]) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        source_file_identity = chunk.get("source_file_identity")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("each semantic chunk must have a non-empty chunk_id")
        if not isinstance(source_file_identity, str) or not source_file_identity:
            raise ValueError("each semantic chunk must have a source_file_identity")
        rows.append(
            (
                chunk_id,
                json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                str(chunk.get("text", "")),
                source_file_identity,
            )
        )
    return rows


def _write_index_contents(
    database_path: Path,
    source_rows: list[tuple[str, str, str]],
    model_identity: str,
    vector_dimension: int,
    chunks: list[Mapping[str, Any]],
) -> None:
    chunk_rows = _chunks_json(chunks)
    with closing(sqlite3.connect(str(database_path))) as connection:
        _create_index_tables(connection)
        connection.execute(f"DELETE FROM {_CHUNKS_TABLE}")
        connection.executemany(
            f"""
            INSERT INTO {_CHUNKS_TABLE}
                (chunk_id, chunk_json, text, source_file_identity)
            VALUES (?, ?, ?, ?)
            """,
            chunk_rows,
        )
        connection.execute(f"DELETE FROM {_SOURCES_TABLE}")
        connection.executemany(
            f"""
            INSERT INTO {_SOURCES_TABLE}
                (source_file_identity, source_filename, source_json_fingerprint)
            VALUES (?, ?, ?)
            """,
            source_rows,
        )
        connection.execute(f"DELETE FROM {_METADATA_TABLE}")
        connection.execute(
            f"""
            INSERT INTO {_METADATA_TABLE} (
                metadata_id, source_json_fingerprint, embedding_model_identity,
                vector_dimension, chunk_count
            ) VALUES (1, ?, ?, ?, ?)
            """,
            (
                _source_metadata_rows(source_rows),
                model_identity,
                vector_dimension,
                len(chunks),
            ),
        )
        connection.commit()


def _has_unified_schema(connection: sqlite3.Connection) -> bool:
    rows = connection.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')"
    ).fetchall()
    tables = {name for name, kind in rows if kind == "table"}
    views = {name for name, kind in rows if kind == "view"}
    return _RELATIONAL_TABLES <= tables and _RELATIONAL_VIEWS <= views


def _is_valid_index(
    database_path: Path,
    source_rows: list[tuple[str, str, str]],
    model_identity: str,
    vector_dimension: int,
) -> bool:
    if not database_path.is_file():
        return False
    try:
        with closing(sqlite3.connect(str(database_path))) as connection:
            if not _has_unified_schema(connection):
                return False
            metadata = connection.execute(
                f"""
                SELECT source_json_fingerprint, embedding_model_identity,
                       vector_dimension, chunk_count
                FROM {_METADATA_TABLE}
                WHERE metadata_id = 1
                """
            ).fetchone()
            if metadata is None:
                return False
            stored_sources = connection.execute(
                f"""
                SELECT source_file_identity, source_filename, source_json_fingerprint
                FROM {_SOURCES_TABLE}
                ORDER BY source_file_identity
                """
            ).fetchall()
            chunk_count = connection.execute(
                f"SELECT COUNT(*) FROM {_CHUNKS_TABLE}"
            ).fetchone()[0]
    except sqlite3.Error:
        return False
    return (
        metadata
        == (
            _source_metadata_rows(source_rows),
            model_identity,
            vector_dimension,
            chunk_count,
        )
        and stored_sources == source_rows
    )


def _unique_values(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value is not None and value not in result:
            result.append(value)
    return result


def _one_or_many(values: Iterable[Any]) -> Any:
    unique = _unique_values(values)
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else unique


def _enrich_chunks(
    database_path: Path,
    chunks: Iterable[Mapping[str, Any]],
    source_by_catalog_id: Mapping[int, tuple[Path, str, str, str | None]],
) -> list[dict[str, Any]]:
    course_codes: dict[int, str] = {}
    course_catalog_ids: dict[int, int] = {}
    placement_metadata: dict[
        int, tuple[Any, Any, int | None, int | None, int, list[str]]
    ] = {}
    placements_by_course: defaultdict[int, list[tuple[Any, Any]]] = defaultdict(list)
    group_codes: defaultdict[int, list[str]] = defaultdict(list)
    group_catalog_ids: dict[int, int] = {}
    provenance_keys: dict[int, str] = {}
    try:
        with closing(sqlite3.connect(str(database_path))) as connection:
            for row in connection.execute(
                "SELECT course_id, catalog_id, course_code FROM courses"
            ):
                course_id = int(row[0])
                course_catalog_ids[course_id] = int(row[1])
                course_codes[course_id] = row[2]
            for row in connection.execute(
                """
                SELECT placements.placement_id,
                       programs.program_code,
                       COALESCE(plans.plan_code, plans.plan_key),
                       placements.course_id,
                       placements.alternative_group_id,
                       plans.catalog_id
                FROM plan_placements AS placements
                JOIN curriculum_plans AS plans
                    ON plans.plan_id = placements.plan_id
                JOIN programs
                    ON programs.program_id = plans.program_id
                """
            ):
                placement_id = int(row[0])
                course_id = int(row[3]) if row[3] is not None else None
                group_id = int(row[4]) if row[4] is not None else None
                codes = []
                if course_id is not None and course_id in course_codes:
                    codes.append(course_codes[course_id])
                    placements_by_course[course_id].append((row[1], row[2]))
                placement_metadata[placement_id] = (
                    row[1],
                    row[2],
                    course_id,
                    group_id,
                    int(row[5]),
                    codes,
                )
            for row in connection.execute(
                """
                SELECT members.alternative_group_id, courses.course_code
                FROM alternative_course_group_members AS members
                JOIN courses ON courses.course_id = members.course_id
                ORDER BY members.alternative_group_id, members.member_order,
                         members.alternative_group_member_id
                """
            ):
                group_codes[int(row[0])].append(row[1])
            group_catalog_ids = {
                int(row[0]): int(row[1])
                for row in connection.execute(
                    "SELECT alternative_group_id, catalog_id "
                    "FROM alternative_course_groups"
                )
            }
            provenance_keys = {
                int(row[0]): row[1]
                for row in connection.execute(
                    "SELECT provenance_id, source_document_key FROM provenance"
                )
                if row[1] is not None and str(row[1]).strip()
            }
    except sqlite3.Error:
        pass

    def source_context(chunk: Mapping[str, Any]) -> tuple[Path, str, str, str | None]:
        catalog_id: int | None = None
        placement_id = chunk.get("placement_id")
        course_id = chunk.get("course_id")
        group_id = chunk.get("alternative_group_id")
        if placement_id is not None:
            placement = placement_metadata.get(int(placement_id))
            if placement is not None:
                catalog_id = placement[4]
        if catalog_id is None and course_id is not None:
            catalog_id = course_catalog_ids.get(int(course_id))
        if catalog_id is None and group_id is not None:
            catalog_id = group_catalog_ids.get(int(group_id))
        if catalog_id is not None and catalog_id in source_by_catalog_id:
            return source_by_catalog_id[catalog_id]
        if len(source_by_catalog_id) == 1:
            return next(iter(source_by_catalog_id.values()))
        raise ValueError("cannot determine a source document for a retrieval chunk")

    enriched: list[dict[str, Any]] = []
    for original_chunk in chunks:
        chunk = dict(original_chunk)
        input_path, default_program, default_plan, default_source_document_key = (
            source_context(chunk)
        )
        references = [
            dict(reference)
            for reference in chunk.get("provenance", [])
            if isinstance(reference, Mapping)
        ]
        source_document_keys: list[str] = []
        source_pages: list[Any] = []
        for reference in references:
            provenance_id = reference.get("provenance_id")
            key = reference.get("source_document_key")
            if key is None and provenance_id is not None:
                key = provenance_keys.get(int(provenance_id))
            if key is None:
                key = default_source_document_key
            if key is not None and str(key).strip():
                key_text = str(key).strip()
                reference["source_document_key"] = key_text
                source_document_keys.append(key_text)
            page = reference.get("source_page")
            if page is not None:
                source_pages.append(page)
        chunk["provenance"] = references

        program_values: list[Any] = [default_program]
        plan_values: list[Any] = [default_plan]
        course_code_values: list[Any] = []
        placement_id = chunk.get("placement_id")
        course_id = chunk.get("course_id")
        group_id = chunk.get("alternative_group_id")
        if placement_id is not None and int(placement_id) in placement_metadata:
            placement = placement_metadata[int(placement_id)]
            program_values.append(placement[0])
            plan_values.append(placement[1])
            course_code_values.extend(placement[5])
            if placement[3] is not None:
                course_code_values.extend(group_codes[int(placement[3])])
        elif course_id is not None:
            course_code = course_codes.get(int(course_id))
            if course_code is not None:
                course_code_values.append(course_code)
            for program, plan in placements_by_course.get(int(course_id), []):
                program_values.append(program)
                plan_values.append(plan)
        elif group_id is not None:
            course_code_values.extend(group_codes[int(group_id)])

        if chunk.get("course_code") is not None:
            existing_codes = chunk["course_code"]
            if isinstance(existing_codes, list):
                course_code_values.extend(existing_codes)
            else:
                course_code_values.append(existing_codes)

        if chunk.get("source_page") is not None:
            existing_pages = chunk["source_page"]
            if isinstance(existing_pages, (list, tuple, set)):
                source_pages.extend(existing_pages)
            else:
                source_pages.append(existing_pages)

        chunk["original_chunk_id"] = chunk.get("chunk_id")
        chunk["source_file_identity"] = str(input_path)
        chunk["source_filename"] = input_path.name
        chunk["program"] = _one_or_many(program_values)
        chunk["plan"] = _one_or_many(plan_values)
        chunk["course_code"] = _one_or_many(course_code_values)
        chunk["source_page"] = _unique_values(source_pages)
        chunk["source_document_key"] = _one_or_many(source_document_keys)
        enriched.append(chunk)
    return enriched


def _make_unique_chunk_ids(chunks: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        original_id = chunk["chunk_id"]
        candidate = original_id
        if candidate in seen:
            prefix = hashlib.sha256(
                str(chunk["source_file_identity"]).encode("utf-8")
            ).hexdigest()[:12]
            candidate = f"{prefix}:{original_id}"
            suffix = 2
            while candidate in seen:
                candidate = f"{prefix}:{suffix}:{original_id}"
                suffix += 1
        chunk["chunk_id"] = candidate
        seen.add(candidate)


def ensure_index(
    input_json_paths: str | Path | Iterable[str | Path],
    index_path: str | Path | None = None,
    *,
    artifact_dir: str | Path = ARTIFACTS_DIR,
    embed_texts_callable: Callable[[Iterable[str]], Any] | None = None,
    embedding_model_identity: str | None = None,
    vector_dimension: int | None = None,
) -> Path:
    """Build or reuse the unified curriculum database for source JSON files."""
    source_paths = _source_paths(input_json_paths)
    input_for_path = source_paths[0]
    database_path = _artifact_path(input_for_path, index_path, artifact_dir)
    model_identity, dimension = _effective_settings(
        embedding_model_identity, vector_dimension
    )
    source_rows = [
        (str(path), path.name, _source_fingerprint(path)) for path in source_paths
    ]

    if _is_valid_index(database_path, source_rows, model_identity, dimension):
        return database_path

    _remove_database(database_path)
    catalog_ids = load_jsons_to_sqlite(source_paths, database_path)
    source_by_catalog_id = {
        catalog_id: (source_path, *_document_metadata(source_path))
        for source_path, catalog_id in zip(source_paths, catalog_ids, strict=True)
    }
    all_chunks = _enrich_chunks(
        database_path,
        build_chunks(database_path),
        source_by_catalog_id,
    )
    _make_unique_chunk_ids(all_chunks)

    embed = _embedder(embed_texts_callable)
    embeddings = embed(chunk["text"] for chunk in all_chunks)
    insert_embeddings(database_path, all_chunks, embeddings)
    _write_index_contents(
        database_path,
        source_rows,
        model_identity,
        dimension,
        all_chunks,
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
    """Search a persistent index, embedding only the supplied question."""
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")
    _effective_settings(None, vector_dimension)
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
    input_json_paths: str | Path | Iterable[str | Path],
    query_text: str,
    top_k: int = 5,
    *,
    index_path: str | Path | None = None,
    artifact_dir: str | Path = ARTIFACTS_DIR,
    embed_texts_callable: Callable[[Iterable[str]], Any] | None = None,
    embedding_model_identity: str | None = None,
    vector_dimension: int | None = None,
) -> list[dict[str, Any]]:
    """Ensure the curriculum database is current, then search it by embedding."""
    database_path = ensure_index(
        input_json_paths,
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
    "DEFAULT_INDEX_NAME",
    "canonical_source_paths",
    "ensure_index",
    "index_path_for_source",
    "query_index",
    "search_index",
]
