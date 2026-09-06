"""Command-line demo for routed curriculum QA."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rag.providers.gemini import make_gemini_callable
from rag.qa import ask


def _print_source_pages(pages: Any) -> None:
    if pages is None:
        return
    if isinstance(pages, (list, tuple, set)):
        pages = [page for page in pages if page is not None]
        if not pages:
            return
        value = ", ".join(str(page) for page in pages)
    else:
        value = str(pages)
    print(f"source_page: {value}")


def _print_structured_result(result: dict[str, Any]) -> None:
    print(f"sql: {result['sql']}")
    columns = result["columns"]
    rows = result["rows"]
    print(f"columns: {columns}")
    print(f"rows: {rows}")

    if "source_page" in columns:
        source_page_index = columns.index("source_page")
        _print_source_pages(
            [row[source_page_index] for row in rows if row[source_page_index] is not None]
        )


def _print_semantic_results(results: list[dict[str, Any]]) -> None:
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. [{result['distance']:.6f}] {result['chunk_id']}")
        print(result["text"])
        _print_source_pages(result.get("source_page"))
        print()


def run_hybrid_demo(
    db_path: str | Path,
    question: str,
    structured_model_callable: Callable[[str], str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run routed QA and print the selected route and raw result."""
    response = ask(
        db_path,
        question,
        structured_model_callable=structured_model_callable,
        top_k=top_k,
    )
    print(f"selected route: {response['route']}")
    if response["route"] == "structured":
        _print_structured_result(response["result"])
    else:
        _print_semantic_results(response["result"])
    return response


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--structured-provider",
        choices=("gemini",),
        help="provider for structured SQL generation",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    structured_model_callable: Callable[[str], str] | None = None,
) -> None:
    args = _parse_args(argv)
    if structured_model_callable is None and args.structured_provider == "gemini":
        structured_model_callable = make_gemini_callable()
    run_hybrid_demo(
        args.db_path,
        args.question,
        structured_model_callable=structured_model_callable,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
