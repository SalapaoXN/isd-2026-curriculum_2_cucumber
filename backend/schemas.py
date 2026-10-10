"""HTTP contracts for the curriculum assistant."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=3000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=3000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()


class CurriculumScope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    program_code: str | None = None
    academic_year: str | None = None
    study_year: int | None = None
    semester: int | None = None
    plan: str | None = None


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    intent: Literal["course_question", "general_conversation", "clarification"]
    use_rag: bool
    standalone_query: str = Field(min_length=1, max_length=3000)
    needs_clarification: bool
    clarification_question: str | None
    clarification_options: list[str] = Field(default_factory=list, max_length=12)
    exhaustive_course_list: bool = False
    program_code: str | None = None
    academic_year: str | None = None
    study_year: int | None = None
    semester: int | None = None
    plan: str | None = None
    course_detail: bool = False
    prerequisite_question: bool = False
    course_code: str | None = None
    course_name: str | None = None
    institution_rule: bool = False
    rule_query: str | None = None
    compare_curricula: bool = False
    left_scope: CurriculumScope | None = None
    right_scope: CurriculumScope | None = None
    comparison_operation: Literal["intersection", "left_only", "right_only", "all"] = "intersection"


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sufficient: bool
    answer: str = Field(min_length=1, max_length=5000)
    source_ids: list[str] = Field(max_length=8)


class GeneralDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1, max_length=3000)
