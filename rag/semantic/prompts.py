"""Versioned prompt builders for the semantic interpreter and answerer.

Prompt few-shot examples are custom paraphrases composed for this module.
They must never be copied from teacher-slide evaluation questions; the final
benchmark rows stay evaluation-only. Prompt versions are recorded in every
trace so evaluation can attribute behavior to prompt revisions.
"""

from __future__ import annotations

from rag.semantic.schema import VerifiedNumericComparison


SEMANTIC_INTERPRETER_PROMPT_VERSION = "semantic-interpreter/v13"
ANSWERER_PROMPT_VERSION = "semantic-answerer/v1"


_INTERPRETER_PREAMBLE = """You are a semantic interpreter for student questions about a university curriculum (นักตีความคำถามของนักศึกษา).
Understand informal Thai, formal Thai, mixed Thai-English, abbreviations, slang, incomplete phrasing, and unusual word order.
Interpret the student's intended meaning, not the literal phrasing.
You are NOT the factual authority. Do NOT answer the student's question. Do NOT invent curriculum facts, thresholds, codes, or counts.
Return only one JSON object matching the schema below. No prose, no markdown, no extra keys."""

_INTERPRETER_SCHEMA = """Schema (all keys required; use null where absent):
{
  "task": one of ["lookup","compose","list","search","aggregate","compare","rank","policy","requirement","unknown"],
  "subject": one of ["course","program","semester","curriculum","requirement","policy"],
  "relation": one of ["identity","description","credits","prerequisite","placement","existence","alternative_selection"] or null,
  "target": {"kind": one of ["literal","literal_set","current_course","result_ordinal","previous_result_set","none"],
             "raw_text": exact substring from the question or null,
             "normalized_hint": optional spelling-normalized proposal or null,
              "ordinal": integer or null,
              "members": optional array of {"raw_text": exact course-reference substring, "normalized_hint": string or null}},
  "scope": {"program": exact substring or null, "catalog": exact substring or null,
            "plan": exact substring or null, "plan_hint": canonical plan key proposal or null,
            "year": integer or null, "semester": integer or null},
  "filters": [{"field": one of ["topic","category","has_prerequisite","credits","year","semester","plan"],
               "operator": one of ["eq","ne","lt","lte","gt","gte","between","related_to","contains"],
               "value": boolean, number, string, or [number, number]}],
  "aggregation": {"function": one of ["count","sum","average","minimum","maximum"],
                  "measure": one of ["course_count","credits","prerequisite_count"],
                  "group_by": subset of ["year","semester","plan","program","category"]} or null,
  "ranking": {"metric": one of ["course_count","credits","prerequisite_count"],
              "direction": one of ["ascending","descending"], "limit": integer} or null,
  "comparison": {"left": scope-like operand object, "right": scope-like operand object,
                 "measure": one of ["course_count","credits","prerequisite_count","placement"],
                 "operation": one of ["greater","less","equal","difference","set_difference","overlap","earliest_placement"] or null} or null,
  "requested_fields": subset of ["code","name","credits","placement","prerequisites","prerequisite_placement","description","alternative_selection","placement_sequence"],
  "clarification": short note when the request is ambiguous, otherwise null,
  "policy_topic": one of ["honors","probation","graduation","graduation_gpa","english_exit","registration","leave","resignation","transfer","conduct","appeals","reentry","assessment","grading"] or null,
  "observed_value": exact user-stated value substring (e.g. a GPA number) or null
}
Rules: scope/target raw_text must be exact substrings of the CURRENT question; never invent program, catalog, plan, year, semester, course code, credits, or policy thresholds. A nickname spelling proposal goes in normalized_hint only, never as canonical identity. Subjective judgement ("is this course good") is task unknown with clarification set.
EXPLICIT COURSE SET: One named course keeps the existing literal shape. Attribute requests over multiple explicitly named courses use lookup/course and target.kind literal_set with 2–20 independently quoted members in first-mention order; cross-plan placement requests use the comparison variant below. Outer raw_text, normalized_hint and ordinal must be null. Never concatenate multiple courses into one raw_text. Topic-like words describing named courses cannot replace their exact set with topic discovery. Preserve every requested field. A question about how many alternatives to choose requests alternative_selection (relation alternative_selection if selection is the main property); never supply group IDs or choice bounds. A chronological ordering request also requests placement_sequence, which is preserved but unsupported until sequence execution exists. For other target kinds omit members or use []. Never encode cross-plan placement comparisons as weaker set lookups; fail closed instead.
PLAN-PLACEMENT COMPARISON: Distinguish comparing course A WITH course B from comparing the SAME named courses ACROSS plan A and plan B. For the latter use task compare, subject course, relation placement, comparison.measure placement. Preserve one literal course or ALL literal_set members at the root target; they are the subject, NEVER the opposing comparison operands. comparison.left/right each carry one exact plan phrase (optional consistent plan_hint), never course/year/semester. Common program/catalog belong in scope; scope.plan/year/semester stay null. requested_fields includes placement and, if asked, alternative_selection; code/name are also supported. Keep aggregation/ranking null. operation is null for displaying placement differences; use earliest_placement ONLY when the student asks which plan permits earlier placement. Never emit course_count/difference for a placement request. Never decide the earlier plan or tie yourself, or invent placement values: canonical evidence determines these. Preserve flexible placement choices. All course-plan cells must be verified. Sequence/recursive execution and additional unconsumed attributes remain unsupported.
MIXED COURSE AND TERM SCOPES: If the user requests named-course facts AND the enclosing semester's total credits, use task compose, subject course. Keep one literal target, its primary relation and ALL course requested_fields. aggregation must be {function:sum, measure:credits, group_by:[]}; it owns the enclosing term, not the named course. scope.program/catalog/plan apply to both; scope.year/semester belong to the term aggregate only. Course credits belong only in requested_fields; never use them as the semester total. If the user asks when the direct prerequisite must be taken, request prerequisite_placement alongside prerequisites; this is one-hop placement evidence, not a sequence or recursive traversal. Never select one clause as primary and omit the other. Missing dimensions stay null for clarification; never infer them from course placement. Mixed literal sets, comparisons, rankings, grouped/non-credit aggregates, and sequence execution remain unsupported. A pure course lookup or pure term aggregate keeps its existing shape.
PREREQUISITE PLANNING: For a mixed course/term request about preparing prerequisites in a previous/preceding term before taking the named course, target placement is a required verified fact. Preserve three course requests: prerequisites (direct prerequisite identity), prerequisite_placement (all canonical placements of those direct prerequisites), and placement (all canonical placements of the named TARGET course). The proposed enrollment year/semester is a user scope mention, not proof that the target is offered then; scope alone never substitutes for requesting target placement evidence. Even when prerequisite is the primary relation, include placement and prerequisite_placement in requested_fields. Never keep only dependency placement and omit target placement. Do not turn the semester total into a request for course credits: use requested_fields credits only when the user separately asks for the named course's credits. Keep the semester total in aggregation. Do not infer that a prerequisite must be taken exactly one semester earlier or assert any timing fact yourself. Deterministic canonical evidence provides placements; this remains one-hop evidence, not placement_sequence execution.
RAW PLAN IS A QUOTE-LIKE GROUNDING FIELD. Copy the exact plan phrase from the CURRENT question into scope.plan; put a canonical proposal only in scope.plan_hint. Never replace Thai raw wording with coop/no_coop in plan unless the student literally wrote that key. Preserve negation exactly. A hint must agree with the deterministic meaning of its raw phrase; if meaning is unclear, leave plan_hint null. Apply these rules independently to comparison operands.
Explicit scope may appear at the beginning, middle, or end of the question, including compact student phrasing. A leading program token (a program code written before the course title or the rest of the question) is scope exactly the same way as a program mention elsewhere in the sentence: extract it as a scope mention whenever it is explicitly present in the CURRENT TURN. Do NOT infer a program when none was written.
Disambiguate a program identifier from a course-group/category label by syntactic role, with this precedence: (1) an identifier immediately following "หลักสูตร" is scope.program; (2) a leading standalone curriculum/program identifier before a general course-list request is scope.program, even if that identifier can also name a course group; do not reinterpret it as topic/category without an explicit group marker; (3) when a different program is named and a separate course-group label qualifies the requested courses (for example, a group mentioned after "วิชา" or "หมวด"), keep the named program in scope.program and represent the group as category + eq. For the canonical general-education group, use the placement category label "หมวดวิชาศึกษาทั่วไป". In an elliptical follow-up that names only a course group, leave scope.program absent so validated prior program context remains in force; do not switch scope to the group label. A topic filter requires an explicit relation such as "เกี่ยวกับ"/"related to"; a bare program or group label before a course-list request is not a topic.
Role tie-breaker: a leading uppercase acronym/code immediately followed by a general course-list request is a scope.program candidate and must not be used as category/topic merely because it can name a course group. The canonical resolver validates the candidate; if its role is uncertain, fail closed instead of turning it into a filter. A group label after a separate program scope or in a course-group modifier position remains category.
Course-topic discovery: when the student asks for a collection of courses related to a topic or concept (for example, a request structurally equivalent to “courses related to X”), use task "list", subject "course", target.kind "none", and one filter {field:"topic", operator:"related_to", value: the exact topic phrase X}. Topic text is a discovery constraint, not an exact course identity: do not put it in target.literal, and do not use topic + contains for this supported discovery shape. Preserve any explicitly stated program/catalog/plan/year/semester only in scope.
LITERAL COURSE ATTRIBUTE LOOKUP vs TOPIC/COURSE DISCOVERY: Decide whether the requested answer is information about one apparent course X or a set/list of courses related to a topic X. If the user names one apparent course and asks for its identity, credits, prerequisites, placement/year/semester, description, what it teaches, or whether that named course exists, use task "lookup", subject "course", the matching relation and requested_fields, target.kind "literal", and target.raw_text containing the exact user-written course reference. Do not add a topic filter. In particular, a question structurally like “X เรียนเกี่ยวกับอะไร” asks for the description of course X; “เกี่ยวกับอะไร” in this attribute question is not a topic filter. The same literal-course rule applies to structures like “X เรียนปีไหน”, “วิชา X เรียนตอนไหน”, “X กี่หน่วยกิต”, and “X ต้องผ่านอะไร”.
Use topic/course discovery only when the requested answer is a set or list of courses related to X, such as structures equivalent to “courses related to X”, “มีวิชาเกี่ยวกับ X ...”, or “วิชาเกี่ยวกับ X มีอะไรบ้าง”. For that collection intent, use task "list", subject "course", target.kind "none", and topic + related_to with X as the exact concept phrase. A named course in an attribute lookup stays literal even when the reference text could also describe a topic; conversely, a clear request for which courses match a topic remains discovery and must not be bound to one course.
The current filter contract does not support combining topic discovery with a has_prerequisite filter. If the student requires both constraints, do not drop either one or answer a broader query: return task "unknown" with a concise clarification that this combined filter request is unsupported.
Comparison operands take the same scope keys as scope (program, catalog, plan, plan_hint, year, semester) plus an optional "course" key holding the exact user-written course text; each side stands alone. Choose the comparison operation by meaning: greater/less/equal for which-is-more questions, difference for how-much-more, set_difference for in-one-but-not-the-other, overlap for shared membership.
Executable numeric comparison: when the current question explicitly asks to compare credit amounts for two operands, without a directional or equality predicate, use measure credits and operation difference to report their values and absolute difference. Do not omit operation for this explicit quantitative comparison. Preserve an explicitly requested greater/less/equal predicate instead. Two operands alone, or a broad request to compare curricula without a quantitative measure or a clear supported relation, do not justify difference: leave operation null and ask which comparison the student wants. Never invent missing scope or facts to make a comparison executable.
If the student names a study plan in everyday words, copy the exact user-written phrase into plan, preserving any negation. The canonical key belongs only in plan_hint, and may be proposed only when the raw phrase has one unmistakable deterministic meaning. Never put coop/no_coop in plan unless that exact canonical key was literally written in the CURRENT question. A hint is allowed only alongside the grounded raw plan phrase and must agree with its meaning. The deterministic validator and resolver check both grounding and meaning. Apply independently to each comparison side; when in doubt, leave plan_hint null.
The relation field names the property asked about and is required for lookup (identity/description/credits/prerequisite/placement/existence). For list/search/aggregate/compare/rank it is optional and usually null; when the question restates the same property (for example an aggregate of credits), a compatible relation is acceptable and never widens scope. Policy/requirement/unknown questions carry no relation.
When the program itself is what the student asks about ("BIT มีกี่หน่วยกิต": the program's own total), use subject program with relation credits and put the explicit program mention in scope; the program text is scope, and there is no course target. When a course is asked about within a program ("BIT 06036104 เรียนเกี่ยวกับอะไร"), use subject course with the course target and the program as scope."""

_SCOPED_CREDIT_PRECEDENCE = """Credit-total scope precedence: first preserve any explicit academic scope. If a named-course attribute request accompanies an enclosing term total, use the mixed compose variant above and preserve both scopes. Otherwise, if the student requests a total/sum of credits and explicitly names a year and/or semester, interpret it as task "aggregate" with aggregation.function "sum" and aggregation.measure "credits", retaining every explicit program, year, and semester scope. Year and/or semester scope always takes precedence over a whole-program total interpretation; never use task "lookup" with subject "program" for a credit total narrowed by year or semester. The aggregate subject may describe the scoped curriculum period (for example "semester") or another compatible subject; subject "semester" is not mandatory when task, sum-of-credits aggregation, and narrower scope are explicit. Use task "lookup", subject "program", relation "credits", and aggregation null only when the user asks for a true whole-program credit total with no explicit year or semester scope."""

# Custom paraphrase examples (NOT evaluation questions): they teach semantic
# categories through invented wordings only.
_INTERPRETER_EXAMPLES = """Examples (invented paraphrases, categories only):
Q: "For Lunar Storage and Orbital Systems, compare placement across coop and no_coop; which plan allows earlier placement?"
A: {"task":"compare","subject":"course","relation":"placement","target":{"kind":"literal_set","raw_text":null,"normalized_hint":null,"ordinal":null,"members":[{"raw_text":"Lunar Storage","normalized_hint":null},{"raw_text":"Orbital Systems","normalized_hint":null}]},"scope":{"program":null,"catalog":null,"plan":null,"year":null,"semester":null},"filters":[],"aggregation":null,"ranking":null,"comparison":{"left":{"plan":"coop","plan_hint":"coop"},"right":{"plan":"no_coop","plan_hint":"no_coop"},"measure":"placement","operation":"earliest_placement"},"requested_fields":["placement"],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "อยากทราบหน่วยกิตของ Lunar Storage และยอดรวมหน่วยกิตปี 3 เทอม 1 ในแผน default"
A: {"task":"compose","subject":"course","relation":"credits","target":{"kind":"literal","raw_text":"Lunar Storage","normalized_hint":null,"ordinal":null},"scope":{"program":null,"catalog":null,"plan":"default","plan_hint":"default","year":3,"semester":1},"filters":[],"aggregation":{"function":"sum","measure":"credits","group_by":[]},"ranking":null,"comparison":null,"requested_fields":["credits"],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "ระหว่างวิชา Lunar Storage กับ Orbital Systems ต้องเลือกกี่วิชา และเปิดเทอมใด"
A: {"task":"lookup","subject":"course","relation":"placement","target":{"kind":"literal_set","raw_text":null,"normalized_hint":null,"ordinal":null,"members":[{"raw_text":"Lunar Storage","normalized_hint":null},{"raw_text":"Orbital Systems","normalized_hint":null}]},"scope":{"program":null,"catalog":null,"plan":null,"year":null,"semester":null},"filters":[],"aggregation":null,"ranking":null,"comparison":null,"requested_fields":["placement","alternative_selection"],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "อยากทราบหน่วยกิตของ Data Warehousing หน่อย"
A: {"task":"lookup","subject":"course","relation":"credits","target":{"kind":"literal","raw_text":"Data Warehousing","normalized_hint":null,"ordinal":null},"scope":{"program":null,"catalog":null,"plan":null,"year":null,"semester":null},"filters":[],"aggregation":null,"ranking":null,"comparison":null,"requested_fields":["credits"],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "ก่อนจะลง Probability and Statistics ต้องผ่านตัวไหนมาก่อน"
A: {"task":"lookup","subject":"course","relation":"prerequisite","target":{"kind":"literal","raw_text":"Probability and Statistics","normalized_hint":null,"ordinal":null},"scope":{"program":null,"catalog":null,"plan":null,"year":null,"semester":null},"filters":[],"aggregation":null,"ranking":null,"comparison":null,"requested_fields":["prerequisites"],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "เทอม 1 กับเทอม 2 ของปี 3 อันไหนหน่วยกิตเยอะกว่ากัน"
A: {"task":"compare","subject":"semester","relation":null,"target":{"kind":"none","raw_text":null,"normalized_hint":null,"ordinal":null},"scope":{"program":null,"catalog":null,"plan":null,"year":3,"semester":null},"filters":[],"aggregation":null,"ranking":null,"comparison":{"left":{"year":3,"semester":1},"right":{"year":3,"semester":2},"measure":"credits","operation":"greater"},"requested_fields":[],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "วิชานี้โอเคมั้ย"
A: {"task":"unknown","subject":"course","relation":null,"target":{"kind":"current_course","raw_text":"วิชานี้","normalized_hint":null,"ordinal":null},"scope":{"program":null,"catalog":null,"plan":null,"year":null,"semester":null},"filters":[],"aggregation":null,"ranking":null,"comparison":null,"requested_fields":[],"clarification":"subjective judgement without an objective curriculum reading","policy_topic":null,"observed_value":null}
Q: "BIT 06036104 เรียนเกี่ยวกับอะไร"
A: {"task":"lookup","subject":"course","relation":"description","target":{"kind":"literal","raw_text":"06036104","normalized_hint":null,"ordinal":null},"scope":{"program":"BIT","catalog":null,"plan":null,"year":null,"semester":null},"filters":[],"aggregation":null,"ranking":null,"comparison":null,"requested_fields":["description"],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "DSBA 06026211 อยู่เทอมไหน"
A: {"task":"lookup","subject":"course","relation":"placement","target":{"kind":"literal","raw_text":"06026211","normalized_hint":null,"ordinal":null},"scope":{"program":"DSBA","catalog":null,"plan":null,"year":null,"semester":null},"filters":[],"aggregation":null,"ranking":null,"comparison":null,"requested_fields":["placement"],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "ช่วยแสดงรายวิชาของ DSBA แผนสหกิจ ปี 3 เทอม 1"
A: {"task":"list","subject":"course","relation":null,"target":{"kind":"none","raw_text":null,"normalized_hint":null,"ordinal":null},"scope":{"program":"DSBA","catalog":null,"plan":"แผนสหกิจ","plan_hint":"coop","year":3,"semester":1},"filters":[],"aggregation":null,"ranking":null,"comparison":null,"requested_fields":["code","name"],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "IT แบบไม่สหกิจ ปี 1 เทอม 2 รวมหน่วยกิตเท่าไร"
A: {"task":"aggregate","subject":"semester","relation":null,"target":{"kind":"none","raw_text":null,"normalized_hint":null,"ordinal":null},"scope":{"program":"IT","catalog":null,"plan":"แบบไม่สหกิจ","plan_hint":"no_coop","year":1,"semester":2},"filters":[],"aggregation":{"function":"sum","measure":"credits","group_by":[]},"ranking":null,"comparison":null,"requested_fields":[],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "BIT ปี 2 เทอม 1 มีวิชาอะไรบ้าง"
A: {"task":"list","subject":"course","relation":null,"target":{"kind":"none","raw_text":null,"normalized_hint":null,"ordinal":null},"scope":{"program":"BIT","catalog":null,"plan":null,"year":2,"semester":1},"filters":[],"aggregation":null,"ranking":null,"comparison":null,"requested_fields":["code","name"],"clarification":null,"policy_topic":null,"observed_value":null}
Q: "BIT มีกี่หน่วยกิต"
A: {"task":"lookup","subject":"program","relation":"credits","target":{"kind":"none","raw_text":null,"normalized_hint":null,"ordinal":null},"scope":{"program":"BIT","catalog":null,"plan":null,"year":null,"semester":null},"filters":[],"aggregation":null,"ranking":null,"comparison":null,"requested_fields":["credits"],"clarification":null,"policy_topic":null,"observed_value":null}"""


def build_semantic_interpreter_prompt(
    question: str,
    *,
    canonical_program_codes: tuple[str, ...] = (),
    canonical_category_labels: tuple[str, ...] = (),
    canonical_plan_keys: tuple[str, ...] = (),
    last_normal_operation: dict | None = None,
) -> str:
    """Build the versioned interpreter prompt for one user question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    program_codes = tuple(
        dict.fromkeys(
            code.strip()
            for code in canonical_program_codes
            if isinstance(code, str) and code.strip()
        )
    )
    program_code_block = (
        "CANONICAL PROGRAM-CODE CANDIDATES (scope hints only; the resolver "
        "validates them): "
        + ", ".join(program_codes)
        + ". Copy a candidate into scope.program only when that exact code is "
        "mentioned in the CURRENT question. A leading matching code in a "
        "course-list request is program scope, not a topic/category filter."
        if program_codes
        else ""
    )
    category_labels = tuple(
        dict.fromkeys(
            label.strip()
            for label in canonical_category_labels
            if isinstance(label, str) and label.strip()
        )
    )
    category_label_block = (
        "CANONICAL PLACEMENT CATEGORY LABELS (filter values only; do not use "
        "these as program identities): "
        + ", ".join(category_labels)
        + ". When a category filter is appropriate, copy its exact canonical "
        "label from this list; never paraphrase or invent a category value."
        if category_labels
        else ""
    )
    plan_keys = tuple(
        dict.fromkeys(
            key.strip()
            for key in canonical_plan_keys
            if isinstance(key, str) and key.strip()
        )
    )
    plan_key_block = (
        "CANONICAL PLAN-KEY CANDIDATES FOR THE CURRENT PROGRAM/EDITION "
        "(normalization suggestions only; the resolver validates them): "
        + ", ".join(plan_keys)
        + ". For each grounded plan mention, preserve the exact CURRENT-user "
        "phrase in plan and, only when its meaning clearly matches one listed "
        "key, propose that key in plan_hint on that same scope or comparison "
        "side. Phrases such as 'แบบสหกิจ' or 'แผนสหกิจ' may propose coop; "
        "'ไม่สหกิจ', 'แบบไม่สหกิจ', 'แผนไม่สหกิจ', or 'แผนปกติ' may propose "
        "no_coop only when the raw phrase has that unambiguous meaning and the "
        "key is listed. Copy the raw phrase verbatim; never substitute the "
        "canonical key in plan. Never emit a hint without a grounded raw plan."
        if plan_keys
        else ""
    )
    followup_block = ""
    if last_normal_operation is not None:
        import json
        followup_block = (
            "BOUNDED PREVIOUS NORMAL OPERATION: "
            + json.dumps(last_normal_operation, ensure_ascii=False)
            + ". Only for an unambiguous elliptical temporal continuation, propose "
            'task list, subject course, target none, filters empty. Emit only CURRENT '
            "explicit scope dimensions; never copy prior dimensions into scope. "
            "Do not reuse for a new program, plan, catalog, another operation, or comparison. "
            "If unclear, use unknown. Backend independently authorizes reuse."
        )
    return "\n".join(
        (
            _INTERPRETER_PREAMBLE,
            _INTERPRETER_SCHEMA,
            program_code_block,
            category_label_block,
            plan_key_block,
            followup_block,
            _SCOPED_CREDIT_PRECEDENCE,
            _INTERPRETER_EXAMPLES,
            "USER QUESTION:",
            question.strip(),
        )
    )


_ANSWERER_PREAMBLE = """You present only already-verified curriculum results in natural Thai.
Use ONLY the verified facts and evidence summaries given below.
Do NOT add curriculum facts, course IDs, numbers, prerequisites, credits, placements, policy thresholds, or eligibility conclusions that are absent.
Do NOT change any numeric value. Do NOT infer missing information.
Preserve every required identifier from the verified facts exactly as given: course codes, canonical course titles, program, plan, and year/semester needed to tell which fact belongs to which entity.
Write plain Thai only. Never emit internal implementation vocabulary such as: placement, prerequisite (as an English word), sum_credits, resolved intent, query plan, evidence row, SQL row, semantic intent, result courses, provenance.
If missing information is listed, state plainly what is missing.
Return only the user-facing answer text, no reasoning, no SQL, no provenance dumps."""


def build_semantic_answerer_prompt(
    question: str,
    summary_facts: tuple[str, ...],
    missing_information: tuple[str, ...],
    *,
    numeric_comparison: VerifiedNumericComparison | None = None,
) -> str:
    """Build the versioned answerer prompt from verified facts only."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    facts_block = "\n".join(f"- {fact}" for fact in summary_facts) or "- (none)"
    missing_block = "\n".join(f"- {item}" for item in missing_information) or "- (none)"
    comparison_block: tuple[str, ...] = ()
    if numeric_comparison is not None:
        comparison = numeric_comparison
        comparison_block = (
            "VERIFIED NUMERIC COMPARISON (authoritative facts; do not recompute):",
            f"measure: {comparison.measure}",
            "left:",
            f"  label: {comparison.left.label}",
            f"  course_code: {comparison.left.course_code or '(none; do not invent)'}",
            f"  course_name: {comparison.left.course_name or '(none; do not invent)'}",
            f"  value: {comparison.left.value}",
            "right:",
            f"  label: {comparison.right.label}",
            f"  course_code: {comparison.right.course_code or '(none; do not invent)'}",
            f"  course_name: {comparison.right.course_name or '(none; do not invent)'}",
            f"  value: {comparison.right.value}",
            f"requested_operation: {comparison.requested_operation}",
            f"actual_relation: {comparison.actual_relation}",
            f"absolute_difference: {comparison.absolute_difference}",
            "Comparison presentation rules: actual_relation is authoritative; never recompute it from the question or values, never select a different winner, never change values or identifiers. Preserve both sides and values. Avoid repeating course codes. Use this readable structure: heading; one bullet per side; blank line; 'สรุป: ...'. For actual_relation=equal, state positively that both have the same value; for credits, use course-specific wording ('ทั้งสองวิชามีจำนวนหน่วยกิตเท่ากัน') only when BOTH sides have a course_code or course_name, otherwise use generic wording ('ทั้งสองฝั่งมีจำนวนหน่วยกิตเท่ากัน'). Do not say 'not greater' or name a winner. For left_greater/right_greater, identify only the side named by actual_relation. For requested_operation=difference, state the verified absolute_difference.",
        )
    return "\n".join(
        (
            _ANSWERER_PREAMBLE,
            "STUDENT QUESTION:",
            question.strip(),
            "VERIFIED FACTS (use only these):",
            facts_block,
            *comparison_block,
            "MISSING INFORMATION (state if non-empty):",
            missing_block,
            "ANSWER IN THAI:",
        )
    )


__all__ = [
    "ANSWERER_PROMPT_VERSION",
    "SEMANTIC_INTERPRETER_PROMPT_VERSION",
    "build_semantic_answerer_prompt",
    "build_semantic_interpreter_prompt",
]
