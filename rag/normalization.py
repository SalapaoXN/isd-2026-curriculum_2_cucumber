"""Pure surface normalization for Natural QA questions."""

from __future__ import annotations

import re


_SURFACE_REPLACEMENTS = (
    ("ตอนปีสามเทอมสอง", "ปี 3 เทอม 2"),
    ("ตอนปีสามเทอมปลาย", "ปี 3 เทอม 2"),
    ("ปีสองเทอมสอง", "ปี 2 เทอม 2"),
    ("ปีสองเทอมปลาย", "ปี 2 เทอม 2"),
    ("ปีสามเทอมสอง", "ปี 3 เทอม 2"),
    ("ปีสามเทอมปลาย", "ปี 3 เทอม 2"),
    ("ตอนปีสาม", "ปี 3"),
    ("ปีสอง", "ปี 2"),
    ("ปีสาม", "ปี 3"),
    ("เทอมสอง", "เทอม 2"),
    ("เทอมปลาย", "เทอม 2"),
)
_COMPOUND_SURFACES = frozenset(
    source for source, _ in _SURFACE_REPLACEMENTS[:6]
)
_SURFACE_PATTERN = re.compile(
    "|".join(re.escape(source) for source, _ in _SURFACE_REPLACEMENTS)
)


def _replace_surface(match: re.Match[str], question: str) -> str:
    value = match.group(0)
    for source, replacement in _SURFACE_REPLACEMENTS:
        if value == source:
            if (
                source in _COMPOUND_SURFACES
                and match.end() < len(question)
                and not question[match.end()].isspace()
                and question[match.end()].isalnum()
            ):
                return f"{replacement} "
            return replacement
    raise AssertionError(f"unhandled surface token: {value}")


def normalize_thai_surface(question: str) -> str:
    """Normalize supported Thai year/semester surface forms only."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    return _SURFACE_PATTERN.sub(
        lambda match: _replace_surface(match, question), question
    )


__all__ = ["normalize_thai_surface"]
