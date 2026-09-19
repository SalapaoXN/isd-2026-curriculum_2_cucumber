"""Data contracts for the pipeline (mirrors original JSON schemas).

These models document the input/output contracts; pipeline code passes plain
dicts with exactly these shapes so behavior stays identical to the original.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Provenance:
    program: str | None = None
    source_filename: str | None = None
    source_page: int | None = None
    document_page: int | None = None
    document_category: str | None = None


@dataclass(frozen=True)
class Course:
    code: str
    name_th: str = ""
    name_en: str = ""
    credits: str = ""
    category: str = ""
    type: str = ""
    prerequisite: str = ""
    year: Any = None
    semester: Any = None
    flexible_year_semester: Any = None
    note: str = ""
    desc_th: str = ""
    desc_en: str = ""
    source_provenance: tuple = ()


@dataclass
class CurriculumDocument:
    """One merged program/plan document (``merged_*_full.json`` shape)."""

    source: str = ""
    description: str = ""
    program: str = ""
    plan: str | None = None
    audit_course_codes: list = field(default_factory=list)
    courses: list = field(default_factory=list)
    total_courses: int = 0

    def to_dict(self) -> dict:
        d: dict = {
            "source": self.source,
            "description": self.description,
            "program": self.program,
            "plan": self.plan,
            "courses": self.courses,
        }
        if self.audit_course_codes:
            d["audit_course_codes"] = self.audit_course_codes
        if self.total_courses:
            d["total_courses"] = self.total_courses
        return d


# Final RAG-ready artifact: ``*_corrected.json`` has the same schema as
# CurriculumDocument; only ``name_th``/``name_en`` may differ (LLM spelling fix).
# Companion ``*_corrections.json`` is a list of
# ``{course_code, field, before, after}`` records.

# Fail-closed rules preserved everywhere:
# - never infer program from a course-code prefix
# - never fuzzy-match courses; ambiguous corrections raise
# - never overwrite non-empty text with empty text
# - terminal numeric suffixes (1/2) are guarded
# - "not specified" markers are never "corrected"
