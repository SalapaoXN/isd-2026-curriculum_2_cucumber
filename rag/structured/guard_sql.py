"""Small safety guard for deterministic, read-only SQLite queries."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final


_WORD_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_INTEGER_RE: Final = re.compile(r"\d+$")
_FORBIDDEN: Final = {
    "ALTER",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "UPDATE",
    "VACUUM",
}
_STATEMENT_WORDS: Final = {
    "ALTER",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "SELECT",
    "UPDATE",
    "VACUUM",
}


@dataclass(frozen=True)
class _Token:
    value: str
    kind: str
    start: int
    end: int
    depth: int


def _skip_quoted(sql: str, start: int, quote: str) -> int:
    index = start + 1
    while index < len(sql):
        if sql[index] != quote:
            index += 1
            continue
        if index + 1 < len(sql) and sql[index + 1] == quote:
            index += 2
            continue
        return index + 1
    raise ValueError("unterminated quoted SQL value")


def _skip_bracket_identifier(sql: str, start: int) -> int:
    end = sql.find("]", start + 1)
    if end < 0:
        raise ValueError("unterminated bracketed SQL identifier")
    return end + 1


def _tokenize(sql: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    depth = 0
    while index < len(sql):
        character = sql[index]
        if character.isspace():
            index += 1
            continue

        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = len(sql) if newline < 0 else newline + 1
            continue
        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated SQL comment")
            index = end + 2
            continue

        if character in "'\"`":
            index = _skip_quoted(sql, index, character)
            continue
        if character == "[":
            index = _skip_bracket_identifier(sql, index)
            continue

        if character == "(":
            tokens.append(_Token(character, "punctuation", index, index + 1, depth))
            depth += 1
            index += 1
            continue
        if character == ")":
            if depth == 0:
                raise ValueError("unbalanced SQL parentheses")
            depth -= 1
            tokens.append(_Token(character, "punctuation", index, index + 1, depth))
            index += 1
            continue

        word = _WORD_RE.match(sql, index)
        if word is not None:
            tokens.append(
                _Token(word.group(0), "word", word.start(), word.end(), depth)
            )
            index = word.end()
            continue

        if character.isdigit():
            end = index + 1
            while end < len(sql) and sql[end].isdigit():
                end += 1
            tokens.append(_Token(sql[index:end], "number", index, end, depth))
            index = end
            continue

        tokens.append(_Token(character, "punctuation", index, index + 1, depth))
        index += 1

    if depth != 0:
        raise ValueError("unbalanced SQL parentheses")
    return tokens


def _validate_statement(tokens: list[_Token]) -> None:
    if not tokens:
        raise ValueError("SQL must contain one SELECT or WITH statement")

    semicolons = [index for index, token in enumerate(tokens) if token.value == ";"]
    if len(semicolons) > 1:
        raise ValueError("multiple SQL statements are not allowed")
    if semicolons:
        semicolon_index = semicolons[0]
        if tokens[semicolon_index].depth != 0 or semicolon_index != len(tokens) - 1:
            raise ValueError("multiple SQL statements are not allowed")

    for token in tokens:
        if token.kind == "word" and token.value.upper() in _FORBIDDEN:
            raise ValueError(f"SQL keyword is not allowed: {token.value}")

    first_word = next(
        (token.value.upper() for token in tokens if token.kind == "word"), None
    )
    if first_word not in {"SELECT", "WITH"}:
        raise ValueError("only SELECT and WITH queries are allowed")

    if first_word == "WITH":
        main_statement = next(
            (
                token.value.upper()
                for token in tokens[1:]
                if token.kind == "word"
                and token.depth == 0
                and token.value.upper() in _STATEMENT_WORDS
            ),
            None,
        )
        if main_statement != "SELECT":
            raise ValueError("WITH must contain a SELECT statement")


def _next_token(tokens: list[_Token], index: int) -> tuple[int, _Token] | None:
    next_index = index + 1
    if next_index >= len(tokens):
        return None
    return next_index, tokens[next_index]


def _signed_integer(
    tokens: list[_Token], index: int
) -> tuple[int, int, int] | None:
    token = tokens[index]
    sign = 1
    start = token.start
    if token.value in {"+", "-"}:
        sign = -1 if token.value == "-" else 1
        next_value = _next_token(tokens, index)
        if next_value is None:
            return None
        index, token = next_value
        if token.kind != "number":
            return None
    elif token.kind != "number":
        return None

    if not _INTEGER_RE.fullmatch(token.value):
        return None
    return sign * int(token.value), start, token.end


def _limit_replacement(
    sql: str, tokens: list[_Token], limit_index: int, max_limit: int
) -> tuple[int, int] | None:
    value_index = limit_index + 1
    if value_index >= len(tokens):
        raise ValueError("LIMIT must have a literal integer")

    parsed = _signed_integer(tokens, value_index)
    if parsed is None:
        raise ValueError("LIMIT must have a literal integer")
    value, start, end = parsed

    following_index = value_index + 1
    if tokens[value_index].value in {"+", "-"}:
        following_index += 1
    if following_index < len(tokens):
        following = tokens[following_index]
        if following.depth != 0:
            raise ValueError("LIMIT expression is not supported")
        if following.value == ",":
            count = _signed_integer(tokens, following_index + 1)
            if count is None:
                raise ValueError("LIMIT must have a literal integer")
            value, start, end = count
        elif following.value not in {"OFFSET", ";"}:
            raise ValueError("LIMIT expression is not supported")

    if value <= max_limit:
        return None
    return start, end


def guard_sql(sql: str, max_limit: int = 100) -> str:
    """Validate a read-only SQL query and apply a bounded result limit."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a non-empty string")
    if isinstance(max_limit, bool) or not isinstance(max_limit, int) or max_limit < 1:
        raise ValueError("max_limit must be a positive integer")

    tokens = _tokenize(sql)
    _validate_statement(tokens)

    limits = [
        index
        for index, token in enumerate(tokens)
        if token.kind == "word"
        and token.value.upper() == "LIMIT"
        and token.depth == 0
    ]
    if len(limits) > 1:
        raise ValueError("multiple top-level LIMIT clauses are not allowed")

    if limits:
        replacement = _limit_replacement(sql, tokens, limits[0], max_limit)
        if replacement is None:
            return sql
        start, end = replacement
        return sql[:start] + str(max_limit) + sql[end:]

    last_token = tokens[-1]
    insert_at = last_token.start if last_token.value == ";" else last_token.end
    prefix = sql[:insert_at].rstrip()
    suffix = sql[insert_at:]
    return f"{prefix} LIMIT {max_limit}{suffix}"


__all__ = ["guard_sql"]
