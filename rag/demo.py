"""End-to-end structured curriculum retrieval demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.retrieval.index import query_index


def run_demo(
    input_json_path: str | Path,
    query_text: str,
    top_k: int = 5,
) -> None:
    """Reuse or build an artifact-scoped retrieval index and print evidence."""
    evidence = query_index(input_json_path, query_text, top_k=top_k)
    for rank, result in enumerate(evidence, start=1):
        pages = ", ".join(str(page) for page in result["source_page"])
        print(f"{rank}. [{result['distance']:.6f}] {result['chunk_id']}")
        print(result["text"])
        print(f"source_page: {pages or 'unknown'}")
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json_path", type=Path)
    parser.add_argument("query_text", nargs="?", help="one Thai query")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    query_text = args.query_text or input("Thai query: ")
    run_demo(args.input_json_path, query_text, args.top_k)


if __name__ == "__main__":
    main()
