"""Small safety guard for deterministic, read-only SQLite queries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
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
        _validate_with_structure(tokens)


_RELATION_TERMINATORS: Final = frozenset(
    {"WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "UNION", "JOIN", "ON"}
)


def _matching_parenthesis(tokens: list[_Token], opening_index: int) -> int:
    if (
        opening_index >= len(tokens)
        or tokens[opening_index].value != "("
    ):
        raise ValueError("ambiguous SQL parenthesized relation")
    opening_depth = tokens[opening_index].depth
    for index in range(opening_index + 1, len(tokens)):
        token = tokens[index]
        if token.value == ")" and token.depth == opening_depth:
            return index
    raise ValueError("ambiguous SQL parenthesized relation")


def _normalized_tokens(tokens: list[_Token]) -> list[_Token]:
    if not tokens:
        return []
    base_depth = tokens[0].depth
    return [replace(token, depth=token.depth - base_depth) for token in tokens]


def _validate_cte_column_list(
    tokens: list[_Token], opening_index: int, closing_index: int
) -> None:
    expected_name = True
    expected_depth = tokens[opening_index].depth + 1
    for token in tokens[opening_index + 1 : closing_index]:
        if expected_name:
            if token.kind != "word" or token.depth != expected_depth:
                raise ValueError("malformed CTE column list")
            expected_name = False
        else:
            if token.value != "," or token.depth != expected_depth:
                raise ValueError("malformed CTE column list")
            expected_name = True
    if expected_name:
        raise ValueError("malformed CTE column list")


def _validate_with_structure(tokens: list[_Token]) -> None:
    """Validate the supported WITH/CTE grammar before relation extraction."""
    if not tokens or tokens[0].value.upper() != "WITH":
        raise ValueError("WITH must contain a valid CTE list")

    index = 1
    if (
        index < len(tokens)
        and tokens[index].kind == "word"
        and tokens[index].value.upper() == "RECURSIVE"
    ):
        index += 1

    while True:
        if index >= len(tokens) or tokens[index].kind != "word":
            raise ValueError("WITH must contain a valid CTE name")
        cte_name_depth = tokens[index].depth
        if cte_name_depth != 0:
            raise ValueError("WITH CTE name must be top-level")
        index += 1

        if index < len(tokens) and tokens[index].value == "(":
            column_list_end = _matching_parenthesis(tokens, index)
            _validate_cte_column_list(tokens, index, column_list_end)
            index = column_list_end + 1

        if (
            index >= len(tokens)
            or tokens[index].kind != "word"
            or tokens[index].value.upper() != "AS"
        ):
            raise ValueError("CTE must contain AS")
        index += 1

        if index >= len(tokens) or tokens[index].value != "(":
            raise ValueError("CTE body must be parenthesized")
        body_end = _matching_parenthesis(tokens, index)
        body = _normalized_tokens(tokens[index + 1 : body_end])
        if not body:
            raise ValueError("CTE body must contain SELECT or WITH")
        _validate_statement(body)
        index = body_end + 1

        if index < len(tokens) and tokens[index].value == ",":
            index += 1
            continue
        break

    if (
        index >= len(tokens)
        or tokens[index].kind != "word"
        or tokens[index].depth != 0
        or tokens[index].value.upper() != "SELECT"
    ):
        raise ValueError("WITH must be followed by a main SELECT")


def _cte_names(tokens: list[_Token]) -> set[str]:
    names: set[str] = set()
    for index, token in enumerate(tokens[:-2]):
        if token.kind != "word":
            continue
        as_index = index + 1
        if tokens[as_index].value == "(":
            column_list_end = _matching_parenthesis(tokens, as_index)
            as_index = column_list_end + 1
        if as_index + 1 >= len(tokens):
            continue
        as_token = tokens[as_index]
        opening = tokens[as_index + 1]
        if (
            as_token.kind == "word"
            and as_token.value.upper() == "AS"
            and opening.value == "("
            and opening.depth == token.depth
        ):
            names.add(token.value.casefold())
    return names


def _validate_relation_name(
    tokens: list[_Token],
    index: int,
    allowed: set[str],
    cte_names: set[str],
) -> int:
    if index >= len(tokens):
        raise ValueError("ambiguous SQL relation reference")
    if tokens[index].value == "(":
        closing_index = _matching_parenthesis(tokens, index)
        nested_tokens = _normalized_tokens(tokens[index + 1 : closing_index])
        if not nested_tokens:
            raise ValueError("ambiguous SQL parenthesized relation")
        _validate_statement(nested_tokens)
        _validate_relations(nested_tokens, allowed)
        return closing_index + 1
    if tokens[index].kind != "word":
        raise ValueError("ambiguous SQL relation reference")
    relation = tokens[index].value.casefold()
    if index + 1 < len(tokens) and tokens[index + 1].value == ".":
        raise ValueError("schema-qualified SQL relations are not allowed")
    if relation not in allowed and relation not in cte_names:
        raise ValueError(f"SQL relation is not allowed: {tokens[index].value}")
    return index + 1


def _validate_relations(tokens: list[_Token], allowed_relations: Iterable[str]) -> None:
    allowed: set[str] = set()
    for relation in allowed_relations:
        if (
            not isinstance(relation, str)
            or _WORD_RE.fullmatch(relation) is None
        ):
            raise ValueError("allowed SQL relations must be simple identifiers")
        allowed.add(relation.casefold())

    cte_names = _cte_names(tokens)
    for index, token in enumerate(tokens):
        if token.kind != "word" or token.value.upper() not in {"FROM", "JOIN"}:
            continue
        relation_index = _validate_relation_name(
            tokens, index + 1, allowed, cte_names
        )
        if token.value.upper() != "FROM":
            continue

        # Also reject comma-separated FROM sources.  Commas nested in an
        # expression or subquery are at a deeper token depth and are ignored.
        source_depth = token.depth
        scan_index = relation_index
        while scan_index < len(tokens):
            scanned = tokens[scan_index]
            if scanned.depth < source_depth:
                break
            if scanned.depth == source_depth:
                if (
                    scanned.kind == "word"
                    and scanned.value.upper() in _RELATION_TERMINATORS
                ):
                    break
                if scanned.value == ",":
                    relation_index = _validate_relation_name(
                        tokens, scan_index + 1, allowed, cte_names
                    )
                    scan_index = relation_index
                    continue
            scan_index += 1


def _validate_relation_allowlist(
    tokens: list[_Token], allowed_relations: Iterable[str] | None
) -> None:
    if allowed_relations is not None:
        _validate_relations(tokens, allowed_relations)


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


def guard_sql(
    sql: str,
    max_limit: int = 100,
    *,
    allowed_relations: Iterable[str] | None = None,
) -> str:
    """Validate a read-only SQL query and apply a bounded result limit."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a non-empty string")
    if isinstance(max_limit, bool) or not isinstance(max_limit, int) or max_limit < 1:
        raise ValueError("max_limit must be a positive integer")

    tokens = _tokenize(sql)
    _validate_statement(tokens)
    _validate_relation_allowlist(tokens, allowed_relations)

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
