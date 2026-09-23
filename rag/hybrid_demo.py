"""Command-line demo for unified curriculum QA."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag.grounded_answer import GroundedAnswerResult
from rag.providers.gemini import make_gemini_callable
from rag.retrieval.index import (
    ARTIFACTS_DIR,
    DEFAULT_INDEX_NAME,
    ensure_index,
)
from rag.qa import ask
from rag.resolution import QueryContext

DEFAULT_CURRICULUM_DB_PATH = ARTIFACTS_DIR / DEFAULT_INDEX_NAME

# H17: public JSON contract for caller-owned structural context.
# Only fields supported by the actual QueryContext may appear.
_ALLOWED_CONTEXT_KEYS = frozenset(
    {
        "program",
        "plan",
        "plans",
        "year",
        "years",
        "semester",
        "semesters",
        "category",
        "course_code",
        "operations",
    }
)

# Explicitly rejected factual-injection / trace fields (subset; any
# unknown key is also rejected, but these get a clear error).
_FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "answer",
        "answer_text",
        "final_answer",
        "count",
        "credits",
        "credit_units",
        "credits_raw",
        "existence",
        "exists",
        "prerequisite",
        "prerequisites",
        "prerequisite_text",
        "description",
        "descriptions",
        "claim",
        "claims",
        "evidence",
        "provenance",
        "sql",
        "rows",
        "columns",
        "model_trace",
        "trace",
        "status",
        "action",
        "route",
        "topic",
        "judgement",
    }
)


def conversation_context_to_dict(context: QueryContext | None) -> dict[str, Any] | None:
    """Serialize structural context only; never factual answer values."""
    if context is None:
        return None
    if not isinstance(context, QueryContext):
        raise TypeError("conversation_context must be a QueryContext or dict")
    payload: dict[str, Any] = {}
    if context.program is not None:
        payload["program"] = context.program
    if context.plan is not None:
        payload["plan"] = context.plan
    if context.years:
        payload["years"] = list(context.years)
    if context.semesters:
        payload["semesters"] = list(context.semesters)
    if context.category is not None:
        payload["category"] = context.category
    if context.course_code is not None:
        payload["course_code"] = context.course_code
    if context.operations:
        payload["operations"] = list(context.operations)
    return payload


def parse_conversation_context(
    value: QueryContext | dict[str, Any] | None,
) -> QueryContext | None:
    """Validate untrusted client JSON into a structural QueryContext.

    Fail-closed: unknown keys, factual-injection keys, wrong types, or
    multi-course targets raise ValueError/TypeError.  No DB access, no
    model calls, no factual inference.
    """
    if value is None:
        return None
    if isinstance(value, QueryContext):
        return value
    if not isinstance(value, dict):
        raise TypeError("conversation_context must be an object")
    unknown = set(value) - _ALLOWED_CONTEXT_KEYS
    if unknown:
        forbidden = sorted(k for k in unknown if k in _FORBIDDEN_CONTEXT_KEYS)
        detail = (
            f"forbidden factual-injection fields: {forbidden}"
            if forbidden
            else f"unknown conversation_context fields: {sorted(unknown)}"
        )
        raise ValueError(detail)

    def _opt_str(key: str) -> str | None:
        raw = value.get(key)
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{key} must be a non-empty string")
        return raw.strip()

    program = _opt_str("program")
    category = _opt_str("category")
    course_code = _opt_str("course_code")

    plan = _opt_str("plan")
    if "plans" in value and value["plans"] is not None:
        plans_raw = value["plans"]
        if isinstance(plans_raw, str):
            plans_raw = [plans_raw]
        if (
            not isinstance(plans_raw, (list, tuple))
            or not plans_raw
            or len(plans_raw) != 1
        ):
            raise ValueError("plans must be a single-element list")
        single = plans_raw[0]
        if not isinstance(single, str) or not single.strip():
            raise ValueError("plans must contain a non-empty string")
        single = single.strip()
        if plan is not None and plan.casefold() != single.casefold():
            raise ValueError("conflicting plan/plans values")
        plan = single

    def _opt_int_list(*keys: str) -> tuple[int, ...]:
        for key in keys:
            if key in value and value[key] is not None:
                raw = value[key]
                if isinstance(raw, int) and not isinstance(raw, bool):
                    raw = [raw]
                if not isinstance(raw, (list, tuple)):
                    raise ValueError(f"{key} must be a list of integers")
                out: list[int] = []
                for item in raw:
                    if not isinstance(item, int) or isinstance(item, bool):
                        raise ValueError(f"{key} must contain integers")
                    out.append(item)
                return tuple(out)
        return ()

    years = _opt_int_list("years", "year")
    semesters = _opt_int_list("semesters", "semester")

    operations: tuple[str, ...] = ()
    if "operations" in value and value["operations"] is not None:
        raw_ops = value["operations"]
        if not isinstance(raw_ops, (list, tuple)):
            raise ValueError("operations must be a list of strings")
        ops: list[str] = []
        for item in raw_ops:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("operations must contain non-empty strings")
            ops.append(item.strip())
        operations = tuple(ops)

    try:
        return QueryContext(
            program=program,
            plan=plan,
            years=years,
            semesters=semesters,
            category=category,
            course_code=course_code,
            operations=operations,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid conversation_context: {exc}") from exc


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
    source_json_path: str | Path | Iterable[str | Path] | None = None,
    intent_model_callable: Callable[[str], str] | None = None,
    synthesize_answer: bool = False,
) -> dict[str, Any]:
    """Run unified curriculum QA and print its grounded final answer."""
    response = answer_question_once(
        db_path,
        question,
        structured_model_callable=structured_model_callable,
        top_k=top_k,
        answer_model_callable=answer_model_callable,
        source_json_path=source_json_path,
        intent_model_callable=intent_model_callable,
        synthesize_answer=synthesize_answer,
    )
    print(f"Question: {question}")
    result = response.get("result")
    if isinstance(result, GroundedAnswerResult):
        print(f"Final Answer: {result.final_answer}")
        return response
    if response["route"] is None:
        print(f"Resolution: {response['result']}")
        return response
    if "final_answer" in response:
        print(f"Final Answer: {response['final_answer']}")
    else:
        print(f"Resolution: {response['result']}")
    return response


def answer_question_once(
    db_path: str | Path,
    question: str,
    structured_model_callable: Callable[[str], str] | None = None,
    top_k: int = 5,
    answer_model_callable: Callable[[str], str] | None = None,
    source_json_path: str | Path | Iterable[str | Path] | None = None,
    intent_model_callable: Callable[[str], str] | None = None,
    synthesize_answer: bool = False,
    conversation_context: QueryContext | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one QA request without printing or selecting a retrieval route."""
    if source_json_path is not None:
        db_path = ensure_index(source_json_path, index_path=db_path)

    parsed_context = parse_conversation_context(conversation_context)
    ask_kwargs: dict[str, Any] = {
        "structured_model_callable": structured_model_callable,
        "top_k": top_k,
        "answer_model_callable": answer_model_callable,
        "intent_model_callable": intent_model_callable,
    }
    if parsed_context is not None:
        ask_kwargs["conversation_context"] = parsed_context
    if synthesize_answer:
        ask_kwargs["synthesize_answer"] = True
    response = ask(db_path, question, **ask_kwargs)
    # Expose validated structural next_context separately as JSON-safe dict.
    # Never merge it into GroundedAnswerResult; canonical evidence stays authoritative.
    next_context = response.get("next_context")
    if isinstance(next_context, QueryContext):
        response = dict(response)
        serialized = conversation_context_to_dict(next_context)
        if serialized is not None:
            response["next_context"] = serialized
        else:
            response.pop("next_context", None)
    result = response.get("result")
    if response["route"] is None:
        if isinstance(result, GroundedAnswerResult):
            projected = dict(response)
            projected.update(
                final_answer=result.final_answer,
                provenance=result.provenance,
                status=result.status,
            )
            return projected
        return response
    return response


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_argument")
    parser.add_argument("second_argument", nargs="?")
    parser.add_argument(
        "--source-json",
        dest="source_json_paths",
        type=Path,
        action="append",
        help="consolidated JSON source for the unified database",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--structured-provider",
        choices=("gemini",),
        help="provider for structured SQL generation and final answers",
    )
    args = parser.parse_args(argv)
    if args.second_argument is None:
        args.db_path = DEFAULT_CURRICULUM_DB_PATH
        args.question = args.first_argument
        args.uses_default_database = True
    else:
        args.db_path = Path(args.first_argument)
        args.question = args.second_argument
        args.uses_default_database = False
    return args


def main(
    argv: Sequence[str] | None = None,
    structured_model_callable: Callable[[str], str] | None = None,
    answer_model_callable: Callable[[str], str] | None = None,
    intent_model_callable: Callable[[str], str] | None = None,
) -> None:
    load_dotenv()
    args = _parse_args(argv)
    if args.source_json_paths:
        args.db_path = ensure_index(args.source_json_paths, index_path=args.db_path)
    if structured_model_callable is None or answer_model_callable is None:
        gemini_callable = make_gemini_callable()
        if structured_model_callable is None:
            structured_model_callable = gemini_callable
        if answer_model_callable is None:
            answer_model_callable = gemini_callable
        if intent_model_callable is None:
            intent_model_callable = gemini_callable
    run_kwargs: dict[str, Any] = {
        "structured_model_callable": structured_model_callable,
        "top_k": args.top_k,
        "answer_model_callable": answer_model_callable,
        "intent_model_callable": intent_model_callable,
    }
    run_hybrid_demo(
        args.db_path,
        args.question,
        **run_kwargs,
    )


if __name__ == "__main__":
    main()
