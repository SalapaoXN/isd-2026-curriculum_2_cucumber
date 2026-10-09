"""HTTP request and response schemas for the CUCUMBER FastAPI app."""

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

ShortStr = Annotated[StrictStr, Field(min_length=1, max_length=80)]
CatalogStr = Annotated[StrictStr, Field(min_length=1, max_length=128)]
StudyYear = Annotated[StrictInt, Field(ge=1, le=5)]
StudySemester = Annotated[StrictInt, Field(ge=1, le=2)]


class CourseRef(BaseModel):
    """One bounded course identity (focus or retained result).

    Keys mirror the downstream focus/result validators exactly
    (course_code, course_name, program, catalog_key); length caps match
    the strictest downstream bound so abuse fails at the request
    boundary with 422 instead of mid-pipeline.
    """

    model_config = ConfigDict(extra="forbid")
    course_code: StrictStr = Field(min_length=1, max_length=64)
    course_name: StrictStr | None = Field(
        default=None, min_length=1, max_length=160
    )
    program: ShortStr | None = None
    catalog_key: CatalogStr | None = None


class LastAnswerCourseRef(BaseModel):
    """Bounded previous-answer reference to a course turn."""

    model_config = ConfigDict(extra="forbid")
    route: Literal["course"]
    course_code: ShortStr
    operations: list[
        Literal["sum_credits", "placement", "describe", "prerequisite", "existence"]
    ] = Field(min_length=1, max_length=8)
    program: ShortStr | None = None
    catalog_key: CatalogStr | None = None


class LastAnswerPolicyRef(BaseModel):
    """Bounded previous-answer reference to a policy turn."""

    model_config = ConfigDict(extra="forbid")
    route: Literal["policy"]
    policy_kind: ShortStr
    program: ShortStr | None = None
    catalog_key: CatalogStr | None = None
    plan: ShortStr | None = None
    amount: StrictInt | None = Field(default=None, ge=0)
    evidence_ids: list[ShortStr] | None = Field(default=None, max_length=50)


class LastNormalOperation(BaseModel):
    """Closed shape of the last successful ordinary list operation."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["list_courses"]
    program: ShortStr
    catalog_key: CatalogStr | None = None
    plan: ShortStr | None = None
    years: list[StudyYear] = Field(default_factory=list, max_length=6)
    semesters: list[StudySemester] = Field(default_factory=list, max_length=3)


class StudyPlanContext(BaseModel):
    """Bounded legacy study-plan context (exact supported shape only)."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["seven_term_plan"]
    program: ShortStr
    catalog_key: CatalogStr
    plan: ShortStr


class ConversationContext(BaseModel):
    """Bounded client conversation state for every QA mode.

    Only fields actually emitted by the API or consumed by the legacy
    and semantic pipelines are accepted; anything else (operand state,
    facts, trace data, giant nested objects) is rejected with 422
    BEFORE expensive semantic/provider work. Comparison operands and
    clarification progress travel exclusively through the dedicated
    clarification transport fields, never through this context.

    The retained-course bound (50) matches the largest context the
    backends themselves emit; semantic retention still keeps at most
    20 for new answers.
    """

    model_config = ConfigDict(extra="forbid")
    program: ShortStr | None = None
    catalog_key: CatalogStr | None = None
    plan: ShortStr | None = None
    plans: list[ShortStr] | None = Field(default=None, max_length=1)
    year: StudyYear | None = None
    years: list[StudyYear] = Field(default_factory=list, max_length=6)
    semester: StudySemester | None = None
    semesters: list[StudySemester] = Field(default_factory=list, max_length=3)
    category: ShortStr | None = None
    course_code: ShortStr | None = None
    operations: list[ShortStr] | None = Field(default=None, max_length=8)
    focus_course: CourseRef | None = None
    focus_catalog_key: CatalogStr | None = None
    result_courses: list[CourseRef] = Field(default_factory=list, max_length=50)
    result_scope_program: ShortStr | None = None
    result_set_empty: StrictBool | None = None
    last_answer: LastAnswerCourseRef | LastAnswerPolicyRef | None = None
    last_normal_operation: LastNormalOperation | None = None
    semantic_topic: ShortStr | None = None
    study_plan_context: StudyPlanContext | None = None
    pending_catalog_selection: StrictBool | None = None

    @model_validator(mode="before")
    @classmethod
    def _wrap_bare_scalars(cls, data: Any) -> Any:
        # Legacy clients may send a bare string/int where a singleton
        # list is canonical. Normalize before validation so the same
        # downstream code sees one shape; bounds still apply to the list.
        if not isinstance(data, dict):
            return data
        wrapped = dict(data)
        if isinstance(wrapped.get("plans"), str):
            wrapped["plans"] = [wrapped["plans"]]
        for key in ("years", "semesters"):
            value = wrapped.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                wrapped[key] = [value]
        return wrapped


class ClarificationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimension: Literal["plan", "catalog"]
    program: StrictStr = Field(min_length=1, max_length=80)
    operand: Literal["left", "right"] | None

    @model_validator(mode="after")
    def normal_target_is_catalog_only(self) -> "ClarificationTarget":
        if self.operand is None and self.dimension != "catalog":
            raise ValueError("normal clarification target is supported only for catalog")
        return self


class ClarificationResolution(ClarificationTarget):
    operand: Literal["left", "right"]
    value: StrictStr = Field(min_length=1, max_length=80)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    conversation_context: ConversationContext | None = Field(default=None)
    home_program: StrictStr | None = Field(default=None)
    clarification_resolution: ClarificationResolution | None = None
    clarification_resolutions: list[ClarificationResolution] | None = Field(default=None, max_length=4)

    @model_validator(mode="after")
    def unambiguous_clarification_transport(self) -> "AskRequest":
        if self.clarification_resolution is not None and self.clarification_resolutions is not None:
            raise ValueError("use singular or plural clarification resolutions, not both")
        return self


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
    clarification_target: ClarificationTarget | None = None


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
