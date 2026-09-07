"""End-to-end structured curriculum retrieval demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.retrieval.embedder import embed_texts
from rag.retrieval.retrieve import retrieve
from rag.retrieval.vector_store import insert_embeddings
from rag.retrieval.chunks import build_chunks
from rag.structured.loader import load_json_to_sqlite


def _reset_database(input_json_path: str | Path, database_path: str | Path) -> None:
    input_path = Path(input_json_path).resolve()
    output_path = Path(database_path)
    if output_path.resolve() == input_path:
        raise ValueError("input JSON and output database paths must differ")

    for candidate in (
        output_path,
        Path(f"{output_path}-wal"),
        Path(f"{output_path}-shm"),
        Path(f"{output_path}-journal"),
    ):
        if not candidate.exists():
            continue
        if not candidate.is_file():
            raise IsADirectoryError(candidate)
        candidate.unlink()


def run_demo(
    input_json_path: str | Path,
    database_path: str | Path,
    query_text: str,
    top_k: int = 5,
) -> None:
    """Build a retrieval database and print ranked evidence for one query."""
    _reset_database(input_json_path, database_path)
    load_json_to_sqlite(input_json_path, database_path)
    chunks = build_chunks(database_path)
    embeddings = embed_texts(chunk["text"] for chunk in chunks)
    insert_embeddings(database_path, chunks, embeddings)

    evidence = retrieve(database_path, query_text, k=top_k)
    for rank, result in enumerate(evidence, start=1):
        pages = ", ".join(str(page) for page in result["source_page"])
        print(f"{rank}. [{result['distance']:.6f}] {result['chunk_id']}")
        print(result["text"])
        print(f"source_page: {pages or 'unknown'}")
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json_path", type=Path)
    parser.add_argument("database_path", type=Path)
    parser.add_argument("query_text", nargs="?", help="one Thai query")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    query_text = args.query_text or input("Thai query: ")
    run_demo(args.input_json_path, args.database_path, query_text, args.top_k)


if __name__ == "__main__":
    main()
