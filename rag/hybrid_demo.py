"""Command-line demo for routed curriculum QA."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag.answer import answer_question
from rag.providers.gemini import make_gemini_callable
from rag.retrieval.index import query_index
from rag.qa import ask
from rag.router import route_question


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
    answer_model_callable: Callable[[str], str] | None = None,
    source_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run routed QA and print the grounded final answer."""
    route = route_question(question)
    if route == "structured":
        response = ask(
            db_path,
            question,
            structured_model_callable=structured_model_callable,
            top_k=top_k,
        )
    else:
        semantic_source_path = source_json_path
        if semantic_source_path is None and Path(db_path).suffix.casefold() == ".json":
            semantic_source_path = db_path
        if semantic_source_path is None:
            raise ValueError("source_json_path is required for semantic route")
        response = {
            "route": route,
            "result": query_index(semantic_source_path, question, top_k=top_k),
        }
    if response["route"] == "structured":
        final_answer = answer_question(
            question,
            response["route"],
            structured_result=response["result"],
            answer_model_callable=answer_model_callable,
        )
    else:
        final_answer = answer_question(
            question,
            response["route"],
            semantic_chunks=response["result"],
            answer_model_callable=answer_model_callable,
        )
    response["final_answer"] = final_answer
    print(f"Question: {question}")
    print(f"Route: {response['route']}")
    print(f"Final Answer: {final_answer}")
    return response


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("question")
    parser.add_argument(
        "--source-json",
        dest="source_json_path",
        type=Path,
        help="consolidated JSON source for semantic indexing",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--structured-provider",
        choices=("gemini",),
        help="provider for structured SQL generation and final answers",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    structured_model_callable: Callable[[str], str] | None = None,
    answer_model_callable: Callable[[str], str] | None = None,
) -> None:
    load_dotenv()
    args = _parse_args(argv)
    if args.structured_provider == "gemini":
        if structured_model_callable is None or answer_model_callable is None:
            gemini_callable = make_gemini_callable()
            if structured_model_callable is None:
                structured_model_callable = gemini_callable
            if answer_model_callable is None:
                answer_model_callable = gemini_callable
    run_kwargs: dict[str, Any] = {
        "structured_model_callable": structured_model_callable,
        "top_k": args.top_k,
        "answer_model_callable": answer_model_callable,
    }
    if args.source_json_path is not None:
        run_kwargs["source_json_path"] = args.source_json_path
    run_hybrid_demo(
        args.db_path,
        args.question,
        **run_kwargs,
    )


if __name__ == "__main__":
    main()
