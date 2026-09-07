"""Semantic evidence demo over the unified curriculum database."""

from __future__ import annotations

import argparse

from rag.retrieval.index import canonical_source_paths, query_index


def run_demo(
    query_text: str,
    top_k: int = 5,
) -> None:
    """Reuse the canonical unified database and print semantic evidence."""
    evidence = query_index(canonical_source_paths(), query_text, top_k=top_k)
    for rank, result in enumerate(evidence, start=1):
        pages = ", ".join(str(page) for page in result["source_page"])
        print(f"{rank}. [{result['distance']:.6f}] {result['chunk_id']}")
        print(result["text"])
        print(f"source_page: {pages or 'unknown'}")
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_text", nargs="?", help="one Thai query")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    query_text = args.query_text or input("Thai query: ")
    run_demo(query_text, args.top_k)


if __name__ == "__main__":
    main()
