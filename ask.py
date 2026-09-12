"""User-facing command line interface for curriculum QA."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag.hybrid_demo import DEFAULT_CURRICULUM_DB_PATH, answer_question_once
from rag.providers.gemini import make_gemini_callable


EXIT_COMMANDS = {"exit", "quit"}
_INTERNAL_PROVENANCE_SYNTAX = re.compile(
    r"[\[\{\"'`*_\-]*(?:source_page|source_file|source_filename|provenance)"
    r"[\]\}\"'`*_\-]*\s*[:=]",
    re.IGNORECASE,
)


def _as_sequence(value: Any) -> list[Any]:
    if value is None or isinstance(value, (str, bytes)):
        return [] if value is None else [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _provenance_entries(value: Any) -> list[Mapping[str, Any]]:
    """Collect only provenance-bearing values from an existing QA result."""
    entries: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if any(
            key in value
            for key in (
                "source_page",
                "source_filename",
                "document_filename",
                "source",
                "page",
            )
        ):
            entries.append(value)
        for key in ("provenance", "source_provenance"):
            for nested in _as_sequence(value.get(key)):
                entries.extend(_provenance_entries(nested))
        columns = value.get("columns")
        rows = value.get("rows")
        if isinstance(columns, Sequence) and not isinstance(columns, (str, bytes)):
            for row in _as_sequence(rows):
                if isinstance(row, Mapping):
                    entries.extend(_provenance_entries(row))
                elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                    entries.extend(
                        _provenance_entries(dict(zip(columns, row)))
                    )
        for key in ("rows", "sql_rows", "retrieved_chunks", "semantic"):
            if key not in value:
                continue
            for nested in _as_sequence(value.get(key)):
                entries.extend(_provenance_entries(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            entries.extend(_provenance_entries(nested))
    return entries


def _format_sources(response: Mapping[str, Any]) -> str:
    candidates: list[tuple[str, str, str, str]] = []
    for entry in _provenance_entries(response.get("result")):
        program = entry.get("program")
        filename = entry.get("source_filename") or entry.get("document_filename")
        if filename is None:
            filename = entry.get("source")
        pages = entry.get("source_page")
        if pages is None:
            pages = entry.get("page")
        page_values = _as_sequence(pages)
        if not page_values:
            page_values = [None]
        for page in page_values:
            parts = []
            if program:
                parts.append(str(program))
            if filename:
                parts.append(str(filename))
            if page is not None:
                parts.append(f"หน้า {page}")
            if not parts:
                continue
            candidates.append(
                (
                    str(program or ""),
                    str(filename or ""),
                    str(page or ""),
                    " / ".join(parts),
                )
            )

    references: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    rich_pages = {
        page
        for _program, filename, page, _reference in candidates
        if filename and page
    }
    for program, filename, page, reference in candidates:
        identity = (program, filename, page)
        if identity in seen:
            continue
        if not filename and page in rich_pages:
            continue
        seen.add(identity)
        references.append(reference)
    return ", ".join(references) if references else "ไม่พบ provenance ในผลลัพธ์"


def _clean_answer_for_display(answer: Any) -> str:
    """Remove lines containing internal provenance/debug syntax for CLI display."""
    text = answer if isinstance(answer, str) else str(answer)
    return "\n".join(
        line
        for line in text.splitlines()
        if not _INTERNAL_PROVENANCE_SYNTAX.search(line)
    ).strip()


def _print_result(question: str, response: Mapping[str, Any], *, show_question: bool) -> None:
    if show_question:
        print(f"ถาม: {question}")
    print(f"ตอบ: {_clean_answer_for_display(response.get('final_answer', ''))}")
    print(f"แหล่งข้อมูล: {_format_sources(response)}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="one curriculum question")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    db_path = Path(DEFAULT_CURRICULUM_DB_PATH)
    if not db_path.is_file():
        print(
            "ไม่พบฐานข้อมูลหลักสูตรที่สร้างไว้แล้ว "
            "โปรดรัน: python -m rag.build_index",
            file=sys.stderr,
        )
        return 1

    try:
        provider = make_gemini_callable()
        if args.question is not None:
            question = args.question.strip()
            if not question:
                print("คำถามต้องไม่เป็นค่าว่าง", file=sys.stderr)
                return 1
            response = answer_question_once(
                db_path,
                question,
                structured_model_callable=provider,
                top_k=10,
                answer_model_callable=provider,
            )
            _print_result(question, response, show_question=True)
            return 0

        while True:
            try:
                question = input("ถาม: ").strip()
            except EOFError:
                print()
                return 0
            if question.casefold() in EXIT_COMMANDS:
                return 0
            if not question:
                continue
            response = answer_question_once(
                db_path,
                question,
                structured_model_callable=provider,
                top_k=10,
                answer_model_callable=provider,
            )
            _print_result(question, response, show_question=False)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
