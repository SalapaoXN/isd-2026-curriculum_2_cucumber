"""HTTP request and response schemas for the CUCUMBER FastAPI app."""

from typing import Any

from pydantic import BaseModel, Field, StrictStr


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    conversation_context: dict[str, Any] | None = Field(default=None)
    home_program: StrictStr | None = Field(default=None)


class AskResponse(BaseModel):
    question: str
    answer: str
    status: str
    action: str | None = None
    route: str | None = None
    hard_task_type: str | None = None
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    next_context: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    plan_results: list[dict[str, Any]] | None = None


class PlanInfo(BaseModel):
    plan_key: str
    plan_name: str | None = None


class EditionInfo(BaseModel):
    catalog_key: str | None = None
    academic_year: str | None = None
    plans: list[PlanInfo] = Field(default_factory=list)


class ProgramInfo(BaseModel):
    program_code: str
    plans: list[PlanInfo] = Field(default_factory=list)
    editions: list[EditionInfo] = Field(default_factory=list)


class ProgramsResponse(BaseModel):
    programs: list[ProgramInfo] = Field(default_factory=list)


class CurriculumItem(BaseModel):
    catalog_key: str | None = None
    academic_year: str | None = None
    program: str | None = None
    plan_key: str | None = None
    year: int | None = None
    semester: int | None = None
    placement_category: str | None = None
    course_code: str | None = None
    name_en: str | None = None
    name_th: str | None = None
    credits: str | None = None
    credit_units: int | None = None
    category: str | None = None
    course_type: str | None = None
    prerequisite_text: str | None = None


class CurriculumResponse(BaseModel):
    items: list[CurriculumItem] = Field(default_factory=list)
    total: int = 0
    limit: int = 200
    offset: int = 0


class CourseDetailResponse(BaseModel):
    course: dict[str, Any]
    placements: list[dict[str, Any]] = Field(default_factory=list)
    prerequisites: list[dict[str, Any]] = Field(default_factory=list)
