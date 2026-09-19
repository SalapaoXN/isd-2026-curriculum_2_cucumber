"""User-facing command line interface for curriculum QA."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from rag.hybrid_demo import DEFAULT_CURRICULUM_DB_PATH, answer_question_once
from rag.grounded_answer import GroundedAnswerResult
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


def _source_page_sort_key(page: str) -> tuple[int, Any]:
    try:
        return (0, int(page))
    except (TypeError, ValueError):
        return (1, page)


def _source_pages_by_book(source: Any) -> list[tuple[str, list[str]]]:
    """Group displayable source pages by book, sorted ascending, deduplicated."""
    pages_by_book: dict[str, dict[str, None]] = {}
    order: list[str] = []
    for entry in _provenance_entries(source):
        program = entry.get("program")
        book = str(program).strip() if program not in (None, "") else ""
        if not book:
            continue
        if book not in pages_by_book:
            pages_by_book[book] = {}
            order.append(book)
        pages = entry.get("source_page")
        if pages is None:
            pages = entry.get("page")
        for page in _as_sequence(pages):
            if page is None or isinstance(page, bool):
                continue
            text = str(page).strip()
            if not text:
                continue
            pages_by_book[book].setdefault(text, None)
    return [
        (book, sorted(pages_by_book[book], key=_source_page_sort_key))
        for book in order
    ]


def _format_sources(response: Mapping[str, Any]) -> str:
    result = response.get("result")
    source = result.provenance if isinstance(result, GroundedAnswerResult) else result
    grouped = _source_pages_by_book(source)
    with_pages = [(book, pages) for book, pages in grouped if pages]
    if with_pages:
        return "\n\n".join(
            f"เล่มหลักสูตร: {book}\nหน้า: {', '.join(pages)}"
            for book, pages in with_pages
        )
    if grouped:
        return "\n\n".join(
            f"เล่มหลักสูตร: {book}\nหน้า: ไม่ระบุ" for book, _ in grouped
        )
    return "ไม่พบ provenance ในผลลัพธ์"


def _clean_answer_for_display(answer: Any) -> str:
    """Remove lines containing internal provenance/debug syntax for CLI display."""
    text = answer if isinstance(answer, str) else str(answer)
    return "\n".join(
        line
        for line in text.splitlines()
        if not _INTERNAL_PROVENANCE_SYNTAX.search(line)
    ).strip()


_BLOCKED_STATUSES = frozenset(
    {
        "insufficient_evidence",
        "unsupported",
        "clarify_program",
        "context_conflict",
        "valid_empty",
    }
)


def _blocked_status_message(status: Any) -> str | None:
    """Render one blocked status as concise Thai, or None when answerable."""
    if not isinstance(status, str) or status not in _BLOCKED_STATUSES:
        return None
    if status == "insufficient_evidence":
        return "ไม่พบหลักฐานเพียงพอในเล่มหลักสูตรสำหรับคำถามนี้ โปรดระบุรหัสวิชา หลักสูตร หรือช่วงปีและเทอมให้ชัดเจนขึ้น"
    if status == "unsupported":
        return "คำถามนี้อยู่นอกเหนือขอบเขตข้อมูลหลักสูตรที่ระบบรองรับ"
    if status == "clarify_program":
        return "คำถามนี้ยังระบุหลักสูตรไม่ชัดเจน โปรดระบุหลักสูตรที่ต้องการถาม เช่น IT, DSBA, BIT หรือ AIT"
    if status == "context_conflict":
        return "เงื่อนไขในคำถามขัดแย้งกัน โปรดทบทวนคำถามแล้วถามใหม่อีกครั้ง"
    if status == "valid_empty":
        return "ไม่พบข้อมูลตามเงื่อนไขที่ถามในเล่มหลักสูตร"
    return "ระบบยังไม่สามารถตอบคำถามนี้ได้ โปรดระบุรหัสวิชา หลักสูตร และแผนการเรียนให้ชัดเจน"


def _blocked_ambiguity_kinds(value: Any) -> tuple[str, ...]:
    """Return normalized ambiguity markers without exposing internal values."""
    if isinstance(value, (str, bytes)):
        items: list[Any] = [value]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        items = _as_sequence(value)
    kinds: list[str] = []
    for item in items:
        text = str(item).casefold()
        if "program" in text and "program" not in kinds:
            kinds.append("program")
        elif "plan" in text and "plan" not in kinds:
            kinds.append("plan")
    return tuple(kinds)


def _blocked_clarification(result: Mapping[str, Any]) -> str | None:
    """Render a blocked mapping as concise Thai, or None when not blocked."""
    status = result.get("status")
    action = result.get("action")
    key = action if isinstance(action, str) and action.strip() else status
    if not isinstance(key, str) or key not in _BLOCKED_STATUSES:
        return None
    ambiguity = _blocked_ambiguity_kinds(result.get("blocking_ambiguity"))
    needs_program = key == "clarify_program" or "program" in ambiguity
    needs_plan = "plan" in ambiguity
    if needs_program and needs_plan:
        return "คำถามนี้ยังระบุหลักสูตรและแผนการเรียนไม่ชัดเจน โปรดระบุหลักสูตรและแผนสหกิจหรือไม่สหกิจที่ต้องการถาม"
    if needs_program:
        return "คำถามนี้ยังระบุหลักสูตรไม่ชัดเจน โปรดระบุหลักสูตรที่ต้องการถาม เช่น IT, DSBA, BIT หรือ AIT"
    if needs_plan:
        return "คำถามนี้ยังระบุแผนการเรียนไม่ชัดเจน โปรดระบุแผนสหกิจหรือไม่สหกิจที่ต้องการถาม"
    if key == "context_conflict" or _as_sequence(result.get("context_conflicts")):
        return "เงื่อนไขในคำถามขัดแย้งกัน โปรดทบทวนคำถามแล้วถามใหม่อีกครั้ง"
    if key == "insufficient_evidence":
        return "ไม่พบหลักฐานเพียงพอในเล่มหลักสูตรสำหรับคำถามนี้ โปรดระบุรหัสวิชา หลักสูตร หรือช่วงปีและเทอมให้ชัดเจนขึ้น"
    if key == "unsupported":
        return "คำถามนี้อยู่นอกเหนือขอบเขตข้อมูลหลักสูตรที่ระบบรองรับ"
    if key == "valid_empty":
        return "ไม่พบข้อมูลตามเงื่อนไขที่ถามในเล่มหลักสูตร"
    return "ระบบยังไม่สามารถตอบคำถามนี้ได้ โปรดระบุรหัสวิชา หลักสูตร และแผนการเรียนให้ชัดเจน"


def _print_result(question: str, response: Mapping[str, Any], *, show_question: bool) -> None:
    if show_question:
        print(f"ถาม: {question}")
    result = response.get("result")
    if isinstance(result, GroundedAnswerResult):
        answer = result.final_answer
        if not answer.strip():
            blocked_answer = _blocked_status_message(result.status)
            if blocked_answer is not None:
                answer = blocked_answer
    elif isinstance(result, Mapping) and (
        result.get("status") == "no_data" or result.get("action") == "no_data"
    ):
        answer = "ไม่พบข้อมูลนี้ในเล่มหลักสูตร"
    else:
        blocked_answer = (
            _blocked_clarification(result) if isinstance(result, Mapping) else None
        )
        if blocked_answer is not None:
            answer = blocked_answer
        else:
            answer = _clean_answer_for_display(response.get("final_answer", ""))
    print(f"ตอบ: {answer}")
    print("แหล่งข้อมูล:")
    print(_format_sources(response))


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
