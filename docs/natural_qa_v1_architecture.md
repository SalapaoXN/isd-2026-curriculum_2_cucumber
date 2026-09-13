# Natural QA v1 — Phase 3A QuerySpec Contract

## Purpose

Phase 3A defines a normalized semantic representation of one standalone
Natural QA v1 question. The representation is called `QuerySpec`. It is a
contract between surface-question understanding and later database
resolution, retrieval planning, policy checks, and answer composition.

`QuerySpec` describes what the student asked. It does not decide whether the
request is allowed, whether an entity exists, which database IDs match, which
retrieval route to use, or how the answer should be written.

## QuerySpec fields

The conceptual fields are:

| Field | Type | Meaning |
|---|---|---|
| `original_question` | `str` | The exact user question, unchanged. |
| `normalized_question` | `str` | The same question after surface normalization only. |
| `program` | `str \| None` | Explicitly stated program, or absent. |
| `plans` | `tuple[str, ...]` | Explicitly stated plan keys, in user order. |
| `years` | `tuple[int, ...]` | Explicitly stated curriculum years, in user order. |
| `semesters` | `tuple[int, ...]` | Explicitly stated semesters, in user order. |
| `course_codes` | `tuple[str, ...]` | Explicit 8-digit course codes, in user order. |
| `course_name` | `str \| None` | Course-name text as stated; it is not resolved here. |
| `category` | `str \| None` | Explicit course category, separate from topic. |
| `topic` | `str \| None` | Explicit topic or subject concept, without semantic rewriting. |
| `operations` | `tuple[str, ...]` | Requested curriculum operations, in question order. |
| `group_by` | `tuple[str, ...]` | Explicit grouping or comparison axes, in question order. |
| `judgement` | `str` | Requested judgement mode. |

Absent scalar information remains `None`; absent collections are empty
tuples. Multiple years and course codes are preserved in the order used by
the student. No implicit plan is selected when the question does not name
one.

## Allowed operation and judgement values

The operation vocabulary is exactly:

`list`, `describe`, `count`, `sum_credits`, `existence`, `compare`,
`earliest`, `placement`, `prerequisite`, `similarity`

The judgement vocabulary is exactly:

`none`, `quantity`, `workload`, `preference`, `unsupported`

These are deliberately small semantic vocabularies. They describe requested
work without encoding a separate enum for every wording or question pattern.

## Separation from later stages

`QuerySpec` MUST NOT contain:

- `action`
- `blocking_ambiguity`
- `capabilities`
- resolved database IDs
- a retrieval route

Those belong to later resolution and planning stages. In particular,
`structured`, `semantic`, and `hybrid` remain diagnostic route labels rather
than QuerySpec intent categories.

Phase 3A does not:

- query the database
- resolve course or program identity
- make ambiguity, no-data, or unsupported decisions
- retrieve semantic chunks
- generate SQL
- synthesize an answer

## Surface normalization

Normalization is limited to surface form. It makes equivalent explicit
number/semester forms easier for later stages to consume, but it does not
change the student’s meaning.

Examples:

- `ปีสอง` → `ปี 2`
- `ปีสาม` → `ปี 3`
- `ตอนปีสาม` → `ปี 3`
- `เทอมสอง` → `เทอม 2`
- `เทอมปลาย` → `เทอม 2`

Normalization must not perform semantic rewrites such as:

- `คอม` → `programming`
- `data` → `database`
- `หนัก` → `ยาก`

The original question is retained alongside the normalized form so later
stages can preserve wording and inspect what was actually stated.

## Entity rules

- Do not infer a program from a course-code prefix.
- Preserve multiple years and course codes in user order.
- Keep `course_name` as unresolved text.
- Keep `category` and `topic` as separate concepts.
- Represent absent information as `None` or an empty tuple.
- Do not select a plan implicitly.

Resolution of a course name or code, including cross-program ambiguity, is a
later stage. QuerySpec records the surface entity; it does not claim that the
entity is unique or present in the curriculum database.

## Multi-operation examples

The following cases demonstrate that one question can request more than one
operation without requiring a new intent enum for the combination:

- `nq_005`: `sum_credits` + `compare`
- `nq_010`: `placement` + `earliest` + `compare`
- `nq_023`: `count` + `compare`
- `nq_030`: `describe` + `placement`
- `nq_036`: `count` + `sum_credits`

Operations remain an ordered tuple. Later planning can determine how the
requested operations share evidence and how their results are composed.

## Worked QuerySpec examples

### nq_017

Question: `IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง`

```text
original_question: "IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง"
normalized_question: "IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง"
program: "IT"
plans: ()
years: (2,)
semesters: ()
course_codes: ()
course_name: None
category: None
topic: "database"
operations: ("list",)
group_by: ()
judgement: "none"
```

### nq_023

Question: `IT ปี 2 กับปี 3 ปีไหนมีวิชา programming เยอะกว่า`

```text
original_question: "IT ปี 2 กับปี 3 ปีไหนมีวิชา programming เยอะกว่า"
normalized_question: "IT ปี 2 กับปี 3 ปีไหนมีวิชา programming เยอะกว่า"
program: "IT"
plans: ()
years: (2, 3)
semesters: ()
course_codes: ()
course_name: None
category: None
topic: "programming"
operations: ("count", "compare")
group_by: ("year",)
judgement: "none"
```

### nq_036

Question: `IT ปีสามเทอมปลายเรียนหนักไหม`

```text
original_question: "IT ปีสามเทอมปลายเรียนหนักไหม"
normalized_question: "IT ปี 3 เทอม 2 เรียนหนักไหม"
program: "IT"
plans: ()
years: (3,)
semesters: (2,)
course_codes: ()
course_name: None
category: None
topic: None
operations: ("count", "sum_credits")
group_by: ()
judgement: "workload"
```

The `หนัก` wording remains a workload judgement. It is not rewritten to the
unsupported difficulty concept `ยาก`.

## Why this addresses the baseline

The Phase 2 baseline showed that Thai surface variants were missed, structural
constraints were not consistently available to semantic retrieval, ambiguous
requests proceeded too far, and multi-operation evidence was difficult to
compose. A compact QuerySpec creates one neutral place to preserve normalized
entities and requested operations before those later decisions occur.

It does not create intent-specific enums: lists, comparisons, descriptions,
placements, and workload questions are represented by reusable fields and
operation tuples. This keeps normalization, resolution, policy, retrieval,
aggregation, and answer synthesis separate while allowing each later stage to
consume the same explicit request representation.

## Comparison and grouping dimension

`group_by` preserves a grouping or comparison axis explicitly requested by the
student so later evidence planning does not need to re-parse either question
string. Its allowed values are exactly:

`plan`, `year`, `semester`, `course`

`group_by` records question understanding only. It contains no database IDs or
resolved values, is not database resolution, and is not a retrieval route. It
may be non-empty even when the corresponding entity collection is empty.

Examples:

- `nq_005`: `semesters = ()`, `operations = ("sum_credits", "compare")`,
  `group_by = ("semester",)`
- `nq_009`: `group_by = ("plan",)`
- `nq_010`: `plans = ()`, `operations = ("placement", "earliest", "compare")`,
  `group_by = ("plan",)`
- `nq_023`: `years = (2, 3)`, `operations = ("count", "compare")`,
  `group_by = ("year",)`
- `nq_024`: `semesters = ()`, `operations = ("count", "compare")`,
  `group_by = ("semester",)`
- `nq_031`: `group_by = ("course",)`
- `nq_032`: `group_by = ("course",)`

Empty `semesters` with `group_by = ("semester",)` means compare or group
across applicable semesters; it does not mean that a semester was explicitly
selected. Likewise, empty `plans` with `group_by = ("plan",)` means compare
or group across applicable plans; it does not select a plan implicitly.

The Evidence Planner must consume `group_by` rather than re-parsing
`original_question` or `normalized_question` to rediscover a comparison axis.
Existing QuerySpec entity and operation semantics remain unchanged.

The frozen fixture remains the Phase 1 behavioral contract. It is not required
to serialize every internal QuerySpec field one-to-one, so its schema is not
changed here. Focused QuerySpec/`group_by` regression tests will be specified
later in Phase 3F and implemented during Phase 4B.

Phase 3A is documentation only. It does not define concrete implementation
files, classes, APIs, database queries, or the Phase 3B–3F design.

## Phase 3B — Resolution and Guard Order

Phase 3B defines the precedence of deterministic resolution and policy guards
after a QuerySpec exists. It separates exact entity identity and scope from
later evidence planning. The result of this stage is a conceptual resolution
outcome containing only:

- `action`
- `blocking_ambiguity`
- resolved program scope
- resolved course candidates or identities

The outcome is not a database result, retrieval result, or answer. QuerySpec
itself remains unchanged and immutable throughout resolution.

### Resolution and guard precedence

The stages run in this order:

1. **Unsupported guard**

   If `QuerySpec.judgement` is `unsupported`, set `action` to
   `unsupported` and stop before retrieval.

2. **Deterministic exact-entity resolution**

   Resolve every explicit exact course reference independently using relational
   curriculum data only.

   Course-code resolution:
   - uses exact normalized 8-digit course-code equality only
   - does not use prefix inference, partial matching, fuzzy matching,
     embeddings, or vector search

   Course-name resolution:
   - searches canonical `name_th` and `name_en`
   - is deterministic lexical identity matching only
   - may normalize case and collapse surrounding/repeated whitespace
   - may use an exact normalized full-name match
   - may use whole-token or contiguous whole-token phrase containment
   - must not use arbitrary substring matching
   - must not use embeddings, semantic similarity, or fuzzy meaning expansion

   Example:
   `NOSQL` may lexically identify `NOSQL DATABASE SYSTEMS`
   because `NOSQL` is a complete lexical token.

   QuerySpec controls whether text is a course name or a topic.
   Phase 3B MUST NOT promote `QuerySpec.topic` into a course-name lookup.
   Therefore topical concepts such as `data`, `database`, `AI`, `web`,
   `network`, and `programming` remain topics when QuerySpec classified
   them as topics.

   If QuerySpec.program is explicit, every exact-entity lookup is restricted
   to that program. Candidates from other programs must not be used as fallback.

   If QuerySpec.program is absent, exact-entity lookup may inspect all loaded
   programs only to establish identity and ambiguity.

   Matches that differ only by curriculum plan are not separate course
   identities. Collapse them conceptually by:

   `(program, course_code)`

   before testing no-data or ambiguity.

3. **Exact-entity no-data guard**

   Evaluate every explicit exact entity reference separately.

   If any required explicit course-code reference or explicit course-name
   reference has zero logical course candidates in the requested scope:

   - action = `no_data`
   - blocking_ambiguity = ()
   - stop before retrieval

   This guard occurs before generic missing-program clarification.

   Examples:

   `06019999 เรียนอะไร`
   -> the explicit code has zero candidates
   -> `no_data`

   A hypothetical comparison containing one existing course code and one
   nonexistent course code must also stop as `no_data`; the system must not
   continue with only the surviving course.

   If an explicit program was supplied, absence inside that program is
   authoritative for this request. The resolver must not silently search
   another program and substitute that result.

4. **Exact-entity/program ambiguity guard**

   After plan-duplicate collapse, inspect every successfully resolved exact
   entity reference.

   If QuerySpec.program is absent and an explicit course code or explicit
   course-name reference resolves to logical course identities in more than
   one program:

   - action = `clarify_program`
   - blocking_ambiguity = ("program",)
   - stop before retrieval

   `nq_012` is the canonical Natural QA v1 example:

   `วิชา NOSQL เรียนเรื่องอะไรบ้าง`

   Lexical resolution finds:
   - IT / 06016414 / NOSQL DATABASE SYSTEMS
   - DSBA / 06026207 / NOSQL DATABASE SYSTEMS

   Therefore:
   - action = `clarify_program`
   - blocking_ambiguity = ("program",)

   For a single-course request, one uniquely resolved exact course identity may
   supply the resolved program scope.

   For a multi-course request with no explicit `QuerySpec.program`, resolve
   every required exact course reference independently. All references must
   exist, and after plan-duplicate collapse each reference must be uniquely
   resolved. The resulting logical identities must all belong to the same
   program before they may jointly supply the resolved program scope.

   If every reference is individually unique but the combined resolved
   identities span more than one program:

   - action = `clarify_program`
   - blocking_ambiguity = ("program",)
   - stop before retrieval or evidence planning

   Example:

   course A -> uniquely IT
   course B -> uniquely DSBA

   The individual identities are not ambiguous, but the request has no single
   resolved program scope. It must not silently compose a cross-program answer.

   If all required exact references resolve uniquely to the same program, that
   shared program may supply the resolved program scope. This is DB resolution,
   not parser inference.

5. **Generic missing-program guard**

   Apply this guard only after exact-entity resolution, exact-entity no-data,
   and exact-entity ambiguity have been handled.

   A Natural QA v1 request requires program scope when it operates over a
   curriculum collection or over topic/category/plan/year/semester scope,
   unless a uniquely resolved exact course identity has already supplied
   one program scope.

   If:
   - QuerySpec.program is absent, AND
   - no uniquely resolved exact course identity supplies program scope, AND
   - the request requires program scope,

   then:

   - action = `clarify_program`
   - blocking_ambiguity = ("program",)
   - stop before semantic retrieval or evidence planning

   This rule covers the generic Natural QA v1 clarify cases:
   `nq_016`, `nq_020`, `nq_025`, `nq_027`, `nq_028`,
   `nq_029`, `nq_033`, and `nq_035`.

   Missing plan is NOT a blocking ambiguity.

6. **Continue to evidence planning**

   Otherwise set `action` to `answer`. This means that the request may proceed
   to the Evidence Planner; it does not synthesize an answer at this stage.

### Resolution semantics

An exact unique course resolution may supply a resolved program scope even
when the student did not explicitly state a program. That is a result of
deterministic database resolution, not parser inference. An absent plan is
not a blocking ambiguity in v1: the later planning stage must preserve all
applicable plans and must never silently select `coop` or `no_coop`.

Course-name identity resolution is deterministic lexical matching.

It may use:
- case/whitespace normalization
- exact normalized full-name match
- exact normalized token/phrase containment

It must NOT use:
- embeddings
- semantic similarity
- fuzzy meaning expansion

Example:
"NOSQL" may lexically match "NOSQL DATABASE SYSTEMS".

Generic topic words such as "data" remain QuerySpec.topic and must not
be promoted to course_name identity resolution.

The only v1 blocking ambiguity defined here is `program`. No provenance,
course identity, or program scope may be fabricated. If a required exact
entity is unavailable or ambiguous, the corresponding guard must stop the
request rather than silently widening the scope.

### Precedence examples

- `nq_038` → `unsupported`; an unsupported salary claim must not become a
  program clarification.
- `nq_039` → `no_data`; the missing exact course must not become a program
  clarification.
- `nq_012` → `clarify_program`; `NOSQL` has exact-name candidates in IT and
  DSBA.
- `nq_016` → `clarify_program`; the general topic question has no program.
- `nq_006` → continue after exact resolution; the explicit course code can
  provide the course identity/program scope when uniquely resolved.
- `nq_011` → continue after exact resolution; the explicit course code can
  provide the course identity/program scope before description retrieval.
- `nq_031` and `nq_032` → resolve both explicit course codes independently;
  a multi-course request must not continue with only partial exact evidence.

### Phase 3B boundary

Phase 3B performs deterministic relational identity resolution and the ordered
guards above. It does not perform:

- topic or vector retrieval
- aggregation
- comparison
- plan-aware composition
- answer synthesis

It does not define concrete Python classes, filenames, APIs, or SQL, and it
does not design Phase 3C–3F.

## Phase 3C — Evidence Planner Matrix

Phase 3C defines how the Evidence Planner converts a `QuerySpec` and the
Phase 3B Resolution Outcome into deterministic evidence requirements. The
planner decides structural scope partitions, target course candidate sets,
required evidence primitives, dependencies between evidence requests, and
how `QuerySpec.group_by` partitions evidence.

The planner is deterministic from those two inputs. It does not re-parse
`original_question` or `normalized_question`.

### Evidence primitives

The planner uses exactly these conceptual evidence primitives:

1. `course_set`
   - logical course identities inside a structural scope
   - relational source
2. `placement_facts`
   - plan, year, semester, and flexible-placement facts
   - relational source
3. `credit_facts`
   - credit information needed by later deterministic aggregation
   - relational source
4. `prerequisite_facts`
   - prerequisite relations for exact resolved courses
   - relational source
5. `description_evidence`
   - description/content evidence for exact resolved courses
   - semantic corpus constrained to the resolved identity
6. `topic_matches`
   - semantic topic relevance over an already constrained candidate course set
   - must have a parent `course_set`
   - must not be a global vector-search result

No separate evidence type is introduced for an individual Natural QA case.

### Global planning order

The dependency order is:

```text
structural scope
→ relational candidate course_set
→ semantic narrowing if QuerySpec.topic exists
→ remaining evidence requests over that narrowed target set
```

Structural filters therefore occur before topic matching. `topic_matches`
receives course identities from `course_set`, and later list, count,
existence, and credit operations consume the same topic-matched target set.
The planner never performs global semantic retrieval and filters by
program, plan, year, semester, or category afterward.

For example, `IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง` is planned as:

```text
IT + applicable plans + year 2
→ course_set
→ topic_matches("database", candidates=course_set)
→ later list composition
```

### Operation-to-evidence matrix

| Operation | Evidence requirements |
|---|---|
| `list` | `course_set`; if `topic` exists, `course_set → topic_matches`; list construction is later. |
| `describe` | `description_evidence` for every resolved exact course identity; never global topic retrieval. |
| `count` | Final target course identities; if `topic` exists, consume `topic_matches`; counting is later. |
| `sum_credits` | `credit_facts` for final target identities; if `topic` exists, request credits only for the topic-matched set; arithmetic is later. |
| `existence` | Final target course identities; if `topic` exists, consume `topic_matches`; the boolean decision is later. |
| `compare` | Derived operation; request evidence for its base operations and preserve partitions from `group_by`. It creates no retrieval source by itself. |
| `earliest` | Derived from `placement_facts`; it creates no separate retriever. Ordering/minimum calculation is later. |
| `placement` | `placement_facts` from the relational source. |
| `prerequisite` | `prerequisite_facts` from the relational source only; do not use semantic/vector retrieval. |
| `similarity` | Separate `description_evidence` for every resolved exact course; similarity/comparison is later. |

### Grouping and partition semantics

The Evidence Planner must consume `QuerySpec.group_by` and must not rediscover
comparison axes by parsing either question string.

- `group_by = ("plan",)` maintains separate plan partitions.
- `group_by = ("year",)` maintains separate year partitions.
- `group_by = ("semester",)` maintains separate semester partitions.
- `group_by = ("course",)` maintains separate exact-course partitions.

Grouping occurs before later aggregation or comparison. Partitions must not be
merged early. If the corresponding entity collection is empty, `group_by`
means enumerate applicable values within the resolved structural scope; it does
not imply that the student explicitly selected one value.

Structural plan partitions compose with every requested `group_by` dimension.
For example, when applicable plans are `coop` and `no_coop`, `years = (2, 3)`,
and `group_by = ("year",)`, preserve all four partitions:

- `coop` / year 2
- `coop` / year 3
- `no_coop` / year 2
- `no_coop` / year 3

Do not merge plans merely because plan is not the comparison axis. Apply the
same rule to semester grouping, including `nq_005`: preserve each applicable
plan's semester partitions before later comparison or aggregation.

### Plan-aware evidence

When the program is resolved and `QuerySpec.plans` is empty, preserve every
applicable plan as a separate planning scope. Never silently choose `coop` or
`no_coop`, and do not merge plan evidence in Phase 3C. Phase 3E may later
decide whether identical results can be summarized or differing results must be
reported separately. Phase 3C only preserves the evidence partitions.

### Bare comparison

For `nq_009`, `IT สหกิจกับไม่สหกิจต่างกันยังไง`, the conceptual request is:

```text
operations = ("compare",)
group_by = ("plan",)
```

The planner requests `course_set` separately for each requested plan partition
and uses `placement_facts` only when placement differences are needed. These
are existing relational primitives; no additional evidence type is introduced.
It does not invoke semantic topic retrieval because no topic/content request
exists. Deterministic plan diff/composition is deferred to Phase 3E.

### Exact-course rules

For exact-course `describe`:

```text
resolved exact course
→ description_evidence scoped to that identity
```

The planner does not plan a global semantic search of the question. For exact
course `similarity`, each resolved course receives separate description
evidence before later comparison. Exact-course `placement` and
`prerequisite` use relational `placement_facts` and `prerequisite_facts`.

### Multi-operation target sets

All operations share the same structural and topic-narrowing criteria.
`group_by` partitions that same target relation, and downstream operations
consume the corresponding partitions. Grouping does not create independent
membership logic per operation. For `nq_022`, `ปีสองของ IT เรียนคอมหนักมั้ย`,
the plan is:

```text
IT / applicable plans / year 2
→ course_set
→ topic_matches("คอม", candidates=course_set)
→ the same matched set feeds count
→ credit_facts(courses=the same matched set) → later sum_credits
```

The planner must not count only topic-matched courses while summing the whole
year. In contrast, `nq_036`, `IT ปีสามเทอมปลายเรียนหนักไหม`, has no topic:

```text
IT / applicable plans / year 3 / semester 2
→ course_set
→ the same full structural set feeds count and credit_facts → later sum_credits
```

### Worked evidence-plan examples

- `nq_005`: `sum_credits + compare`, grouped by semester; request course and
  credit evidence partitioned by semester.
- `nq_009`: bare plan comparison; request `course_set` separately for each
  plan partition and add `placement_facts` only when placement differences are
  needed.
- `nq_010`: exact resolved course with `placement + earliest + compare`,
  grouped by plan; request `placement_facts` partitioned by plan, then defer
  earliest and comparison.
- `nq_017`: constrain IT/year 2 through `course_set` before requesting
  `topic_matches("database")`.
- `nq_022`: use one topic-narrowed target set for both count and credits.
- `nq_023`: preserve year partitions and perform constrained programming topic
  matching independently within each year.
- `nq_030`: request both `description_evidence` and `placement_facts` for the
  exact resolved course; composition is later.
- `nq_031`: request separate description evidence for both exact courses before
  later similarity comparison.
- `nq_036`: use the full structural course set for count and credits; there is
  no semantic topic step.

### Conceptual EvidencePlan

A conceptual EvidencePlan contains only scope partitions, candidate or target
course identities, evidence requests, dependencies between requests, and
`group_by`. Every evidence primitive must preserve source provenance sufficient
for later grounded answer and citation rendering. Citation rendering itself
remains outside Phase 3C; this invariant does not add a seventh evidence
primitive. For example:

```text
R1 = relational course_set(scope)
R2 = topic_matches(topic, candidates=R1)
R3 = credit_facts(courses=R2)

count consumes R2
sum_credits consumes R3
```

This is a deterministic dependency graph, not an agent workflow. Phase 3C
does not define Python classes, production filenames, SQL statements, vector
query implementations, or APIs.

### Phase 3C boundary

The planner does not answer the student, perform aggregation, arithmetic,
comparison, or similarity judgement, execute vector retrieval, generate
free-form SQL, or synthesize answer wording. It only describes the evidence
needed for later stages. In particular, it defers:

- Phase 3D: vector/index filtering, candidate-constrained nearest-neighbor
  strategy, semantic thresholds/ranking, and topic-match mechanics.
- Phase 3E: deduplication, count/credit arithmetic, comparison, earliest
  calculations, similarity judgement, plan merge/diff semantics, workload or
  preference judgement, and composition.
- The later answer layer: natural-language synthesis and final
  provenance/citation rendering.

Do not use `structured`, `semantic`, or `hybrid` as the primary planner intent
taxonomy.

## Phase 3D — Constrained Semantic Retrieval

Phase 3D defines how semantic evidence is retrieved after Phase 3C has
provided structural scope and candidate identities. Natural QA v1 has two
semantic retrieval paths. Both are constrained by the evidence plan and both
preserve structural partitions and provenance.

### Exact-course content path

For a resolved exact course identity:

```text
resolved exact course identity
→ direct fetch of its `description` chunk
```

This path is used by `describe` and `similarity`. It does not perform global
nearest-neighbor search. If the description chunk is missing, the result is
empty semantic evidence for that identity; metadata must not be silently
substituted as description evidence.

For similarity, each exact course receives its own description evidence. The
courses remain separate for the later similarity judgement.

### Constrained topic path

Topic retrieval starts from the relational `course_set` produced by Phase 3C.
Semantic ranking operates only over description chunks belonging to those
candidate logical courses. Placement and other metadata chunks are not topic
scoring units. The system must not perform a global search first and then
filter the results by program, plan, year, semester, category, or course set.

The query embedding is produced from `QuerySpec.topic`, not from the full user
question. Phase 3D does not semantically rewrite topic text; for example,
`data`, `database`, `AI`, `web`, `network`, and `programming` remain the topic
values understood by the earlier stage.

### Scoring unit and partitions

There is one semantic topic score per logical course. A course's description
chunk is the scoring unit. Structural membership remains separately attached
to every applicable plan/year/semester partition. The same logical course may
reuse one topic score in multiple structural partitions, but scoring never
merges those partitions.

All structurally eligible candidate descriptions are scored. Topic membership
must not be defined by top-k membership. A top-k limit may be applied later for
presentation, but never for count, existence, comparison, or any other
membership decision.

### Distance convention

Natural QA v1 uses one explicit cosine-distance convention for semantic
relevance:

```text
cosine_distance = 1 - cosine_similarity
lower distance = closer semantic match
```

The convention is shared by exact-course and topic semantic scoring where a
distance is reported. Existing stored embeddings and vectors are reused. Phase
3D introduces no vector database and does not require a corpus rebuild solely
to document this convention.

### Relevance threshold policy

Topic membership uses one fixed threshold for the frozen embedding model. No
numeric threshold is invented or frozen in Phase 3D. The threshold must be
calibrated later on a small manually reviewed development set before final
end-to-end evaluation. It must not be tuned from final held-out evaluation
results, and there is no per-query dynamic threshold heuristic.

### Empty and missing evidence

The planner and retrieval result distinguish these cases:

1. The structural candidate set is empty.
2. Structural candidates exist, but some or all lack description evidence.
3. Descriptions exist, but none pass the fixed relevance threshold.

None of these cases fabricates a topic match. Missing descriptions do not fall
back silently to metadata, and a zero topic-match result is not converted into
an unrelated global search result.

### Topic-match result contract

Each topic match preserves:

- logical course identity
- semantic distance
- source description chunk/evidence
- source provenance
- structural partition membership

Plan, year, semester, and other scope partitions therefore remain available
to later aggregation, comparison, and composition without re-parsing the
question or reconstructing membership from the semantic score.

### Runtime/index relationship

The existing `semantic_chunks` table and sqlite-vec vectors remain in the same
SQLite runtime database. Existing vectors can be reused; Natural QA v1 does
not require automatic rebuilding during retrieval. The existing global search
path may remain for compatibility, but constrained topic retrieval must not
depend on global top-k results followed by structural filtering.

### Worked examples

- `nq_011`, `06016414 เรียนเกี่ยวกับอะไร`: resolve the exact course, directly
  fetch its `description` chunk, and return empty description evidence if that
  chunk is absent; do not search globally or substitute metadata.
- `nq_017`, `IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง`: create the IT/year-2
  relational `course_set`, embed only topic `database`, and score only the
  eligible description chunks.
- `nq_023`, `IT ปี 2 กับปี 3 ปีไหนมีวิชา programming เยอะกว่า`: preserve the
  `coop`/`no_coop` × year-2/year-3 partitions, then score topic `programming`
  within each partition's candidate course set without using top-k as
  membership.
- `nq_031`, `06016414 กับ 06016419 เนื้อหาคล้ายกันไหม`: directly fetch one
  description chunk for each resolved course and keep the two evidence sets
  separate for later comparison.
- Empty topic-match example: when a non-empty structural `course_set` has
  descriptions but none pass the fixed threshold, return an explicit empty
  `topic_matches` result with the candidate scope and available provenance;
  do not fabricate matches or fall back to metadata/global results.

### Phase 3D boundary

Phase 3D does not define the numeric threshold calibration experiment or its
implementation details. It defers to Phase 3E deduplication, count and credit
arithmetic, earliest calculations, comparison, similarity judgement, workload
and preference judgement, plan merge/diff semantics, and composition.

Concrete production APIs, classes, and focused tests are deferred to Phase 3F
and Phase 4. Final answer synthesis and citation rendering remain in the later
answer layer.

## Phase 3E — Aggregation, Comparison, and Composition

Phase 3E consumes the partitioned evidence planned by Phase 3C and retrieved
under Phase 3D. It derives deterministic values from grounded evidence and
composes those values without re-parsing the question or issuing new evidence
requests.

### Logical-course identity and deduplication

The logical course identity is:

```text
(program, course_code)
```

Within one structural partition, duplicate chunks or records for the same
logical course are counted once. Deduplication is local to that partition. It
must never collapse evidence across preserved plan, year, semester, or exact
course partitions. A course appearing in two plan or year partitions remains
present in both partitions even when its logical identity is the same.

Alternative-course groups remain group records during composition. Their
existing deterministic choice and credit semantics are reused; group members
must not be converted into separately required courses.

### Deterministic aggregation

Aggregation uses the final target relation and its corresponding evidence.
`count` has two distinct semantics, depending on what the question counts:

- Required-load count: for structural questions asking how many courses a
  student must take, such as `nq_003`, each ordinary required course
  contributes 1. An alternative group contributes its deterministic required
  choice count, using existing group semantics such as `minimum_choices`.
  Its members are not counted as separately required courses.
- Topic/category option count: for questions counting available or matching
  courses, such as `nq_018` or elective-topic counts, count unique logical
  course identities in the final matched target set. Alternative-group
  members may remain distinct course options, but the result must not imply
  that every option is required.

The evidence path selects the count semantics; Phase 3E does not re-parse
`original_question` or `normalized_question` to choose a mode:

- a count over a structural curriculum-obligation relation uses
  `required_load`;
- a count over `topic_matches` or another concrete filtered course-option
  relation uses `option_count`.

If the evidence path does not determine a safe count semantic, the Phase 3E
composition stage retains an internal insufficient-evidence or
indeterminate-count-semantics state rather than guessing. This state is not a
new public action or QuerySpec operation.

`existence` is true exactly when the relevant final target count is greater
than zero. `sum_credits` consumes `credit_facts` for the final target set.

Credit totals reuse the existing deterministic credit and alternative-group
semantics. Required-load aggregation accounts for an alternative group
according to its established choice rule rather than once per member, while
option counts retain distinct matching member identities where they are
options. No LLM arithmetic is used for count, existence, credits, or any
intermediate total.

Every aggregate retains the exact target components that produced it. The two
lineage shapes are explicit:

```text
option_count = 4
→ four matched logical course identities and their supporting evidence

required_load = 4
→ three ordinary required course identities
→ one alternative-group record with required-choice contribution = 1
→ do not fabricate a selected group member
```

For `required_load`, the alternative-group record and its numeric contribution
are preserved even when the curriculum states only a choice requirement. For
`option_count`, the lineage identifies the exact matched logical course
identities. Neither representation is reduced to a misleading list of
independently required members.

### Earliest placement

`earliest` is derived from `placement_facts`. Only valid normalized
`(year, semester)` values participate in lexicographic comparison:

```text
(year_a, semester_a) < (year_b, semester_b)
```

Flexible placement choices are normalized into their deterministic set of
valid choices and included in the comparison. Ties are preserved as all tied
plans, courses, or placements; no arbitrary winner is selected. If timing is
missing or no valid timing can be compared, the result does not guess an
earliest value and retains the missing/insufficient-evidence state.

### Comparison

`compare` is derived only from already computed base values and respects
`QuerySpec.group_by`. It does not create a new retrieval request.

- `count` comparisons compare numeric course counts.
- credit comparisons compare computed credit totals.
- `earliest` comparisons compare valid `(year, semester)` values.
- Every tie is preserved; a unique winner is exposed only when the computed
  base values prove it.

The compared values remain in their original structural partitions. A
comparison never silently merges plans, years, semesters, or courses merely to
produce one scalar.

### Plan-aware composition

When multiple applicable plans are in scope, compute each plan independently
first, including its target set, aggregates, placements, comparisons, and
provenance. Equivalent results may later be summarized together. Differing
results remain separate, and differing plan results must never be unioned into
one factual result.

A presentation-level collapse must not discard the plan partitions, component
evidence, or provenance lineage. The evidence remains available even when the
later answer layer chooses a concise shared presentation.

### Deterministic plan-result equivalence

Before any presentation-level plan collapse, compare normalized derived
values. Two plan results are factually equivalent only when the relevant
normalized values are equal under the same operation and count semantics:

- course sets: the same canonical logical-course set
- counts: the same numeric result under the same required-load or option-count
  semantics
- credits: the same numeric total
- placement: the same normalized placement set
- earliest: the same normalized earliest value or set
- composed deterministic facts: corresponding normalized components are equal

Different provenance does not make factual results unequal. However, a
presentation collapse must preserve provenance from every contributing plan,
along with its partition and component evidence. If any normalized factual
value differs, retain separate plan results.

For the bare plan comparison in `nq_009`, compute deterministically:

- common courses
- `coop`-only courses
- `no_coop`-only courses
- placement differences for common courses when placement facts are available

Each set is derived from the independently computed plan course sets. Missing
placement information is reported as unavailable, not inferred.

### Multi-operation composition

Multi-operation composition reuses the same Phase 3C/3D structural and topic
target relation. It does not re-query the database, re-run semantic retrieval,
or re-parse either question string to obtain a different target set.

`nq_030` combines `description_evidence` and `placement_facts` for the same
resolved course identity, then composes the description and placement facts.
Every component retains its own supporting provenance. Grouped operations use
the corresponding preserved partition of that same relation.

### Judgement semantics

Judgement is applied only to grounded derived values and evidence:

- `quantity` is grounded in an actual count or proportion. It must not become
  an unsupported subjective claim.
- `workload` uses course count and credits as an explicit proxy. Conclusions
  must be qualified as workload by count/credits and must never infer course
  difficulty.
- `preference` may state that grounded course content matches the stated
  interest, but must not claim that a course or plan is objectively “best”.
- An insufficient-evidence or indeterminate-count-semantics state is distinct
  from `unsupported`; Phase 3E must not create or assign the `unsupported`
  judgement for this condition.
- `unsupported` should already have stopped in the Phase 3B unsupported guard;
  Phase 3E must not manufacture a basis for it.

### Similarity and nq_031

Similarity compares only the two grounded exact-course descriptions retained
by Phase 3D. A bounded grounded synthesis may identify supported similarities
and differences, but it may not introduce external facts or infer
prerequisite/placement claims from descriptive similarity.

If either description is insufficient, the result reports a limited comparison
rather than filling the gap with unrelated evidence. Provenance for both
courses remains attached to the comparison components.

### Empty-result semantics

Phase 3E distinguishes a valid empty result from insufficient or missing
evidence:

- A valid empty result means the grounded target relation was evaluated and
  contains no matching members.
- Insufficient/missing evidence means the required relation, fact, or
  description was unavailable or could not be evaluated safely.

These states must not be converted into each other. The system never fabricates
data to turn missing evidence into a valid empty result or to turn a valid
empty result into a positive match.

### Provenance lineage

Every derived value remains traceable to the evidence that supports it. This
lineage includes the partition, logical course identity, source fact, and
provenance references for each component. Count lineage follows the explicit
`option_count` and `required_load` shapes above; a generic count must not be
interpreted as a list of independently required courses.

No transformation may discard the provenance needed for a later grounded
answer or citation. Citation rendering itself remains outside Phase 3E.

### Worked examples

- `nq_003`: use `required_load` over the structural curriculum-obligation
  relation. Ordinary required courses contribute 1; each alternative group
  contributes its deterministic required choice count, without counting every
  group member as required.
- `nq_004`: derive semester/year credit totals from `credit_facts`; arithmetic
  is deterministic and retains the contributing courses.
- `nq_005`: compute credit totals independently for each semester partition,
  then compare those totals while preserving plan × semester evidence.
- `nq_009`: derive common, `coop`-only, and `no_coop`-only course sets, plus
  available placement differences for common courses.
- `nq_010`: compare valid placement timings for the exact course across plan
  partitions, preserve ties, and expose an earliest plan only when unique.
- `nq_018`: use IT/year 3 and applicable plans to form one constrained AI
  target relation, count distinct matching course options within each
  preserved plan, and compose the plan-aware result. It has no `compare`
  operation and no multi-year partition.
- `nq_021`: apply the quantity judgement to the grounded topic-course count;
  do not replace it with an unsupported subjective conclusion.
- `nq_022`: use the same topic-narrowed course relation for both count and
  credit facts, then qualify workload using those count/credit proxies.
- `nq_023`: compare programming counts separately for year 2 and year 3, with
  plan partitions preserved underneath each year.
- `nq_024`: count database-topic matches separately by semester and compare
  those counts without treating an empty semester collection as one selected
  semester.
- `nq_030`: compose the exact course description and placement components from
  their shared identity, retaining provenance for each component.
- `nq_031`: compare only the two grounded descriptions and report a limited
  result if either description is insufficient.
- `nq_032`: compute each course's valid placement timing independently, then
  compare the resulting timings and preserve ties or missing timing.
- `nq_036`: derive count and credit totals from the same full structural set,
  then describe workload explicitly as a count/credit proxy.

### Phase 3E boundary

Phase 3E does not parse Thai, resolve entities, perform semantic retrieval or
vector scoring, calibrate a semantic threshold, generate free-form SQL, or
render the final answer or citations. Those responsibilities remain in the
earlier resolution/retrieval stages or the later answer layer as specified.

## Phase 3F — Implementation Mapping and Regression Strategy

Phase 3F maps the frozen Phase 3A–3E contracts into independently reviewable
Phase 4 micro-tasks. It is an implementation map, not a production change and
not an approval to begin implementation. Each checkpoint has one narrow
contract, a small set of likely existing touch points, focused regressions, and
a gate that includes the frozen Phase 1 specification tests plus
`git diff --check`.

### Dependency order

```text
4A → 4B → 4C → 4D → 4E → 4F → 4G → 4H → 4I → Phase 5
```

The order follows data dependencies: surface forms become a neutral
`QuerySpec`, exact entities and guards establish safe scope, the Evidence
Planner defines partitions, constrained retrieval supplies topic evidence,
threshold calibration freezes membership, and only then do deterministic
aggregation, judgement, and end-to-end integration proceed.

### 4A — Thai surface normalization

1. Frozen contract implemented: perform only the Phase 3A surface
   normalizations (`ปีสอง`/`ปีสาม`/`ตอนปีสาม`, `เทอมสอง`/`เทอมปลาย`) needed by
   QuerySpec. Do not rewrite meanings such as `คอม`, `data`, or `หนัก`.
2. Likely existing files: the normalization/QA entry path in `rag/qa.py` and
   only the compatibility portion of `rag/router.py` if route detection needs
   the same normalized surface. `route_question()` remains diagnostic.
3. New file: none is required by the architecture mapping; introduce a small
   dedicated normalization module only if the existing entry path cannot own
   this pure operation without coupling it to retrieval.
4. Out of scope: DB access, entity resolution, routing redesign, semantic
   rewriting, retrieval, judgement, answer synthesis, and all later stages.
5. Focused tests: normalization cases for `ปีสอง`, `ปีสาม`, `ตอนปีสาม`,
   `เทอมสอง`, and `เทอมปลาย`, plus negative assertions for `คอม`, `data`, and
   `หนัก`.
6. Completion gate: normalized surfaces are deterministic and lossless for
   QuerySpec, frozen Natural QA v1 tests pass, and the focused normalization
   tests plus `git diff --check` pass.

### 4B — QuerySpec understanding

1. Frozen contract implemented: deterministically extract program, plans,
   years, semesters, course codes, course name, category, topic, operations,
   `group_by`, and judgement while retaining original and normalized question
   text. This stage does not resolve identities or retrieve evidence.
2. Likely existing files: `rag/qa.py` as the single QA orchestration entry
   point, with route detection in `rag/router.py` kept as a diagnostic
   compatibility concern rather than the intent taxonomy.
3. New file: none required by the contract. A small QuerySpec-owned module is
   justified only if it is needed to keep parsing independent from DB and
   retrieval code; its filename and API are implementation decisions for this
   checkpoint, not part of Phase 3F.
4. Out of scope: relational lookups, embeddings, vector search, ambiguity or
   no-data decisions, SQL generation, aggregation, and answer synthesis.
5. Focused tests: all scalar/collection fields, the exact operation and
   judgement enums, `group_by` for `nq_005`, `nq_010`, `nq_023`, `nq_024`,
   `nq_031`, and `nq_032`, and multi-operation cases such as `nq_005`,
   `nq_010`, `nq_023`, `nq_030`, and `nq_036`.
6. Completion gate: QuerySpec values are deterministic, preserve user order,
   do not infer plans or programs, and do not perform DB/retrieval work;
   focused tests, frozen fixture/spec tests, and `git diff --check` pass.

### 4C — Exact resolution and ordered guards

1. Frozen contract implemented: exact normalized code/name resolution,
   program scoping, logical plan collapse, per-reference no-data, exact-name
   program ambiguity, and generic missing-program guards in this order:
   unsupported → exact resolution/no-data → ambiguity → generic missing
   program.
2. Likely existing files: `rag/structured/qa.py` and its existing relational
   query/helper boundary; `rag/qa.py` only for passing the resolution outcome
   through the QA path.
3. New file: none required unless the existing structured boundary cannot
   isolate resolution from evidence execution.
4. Out of scope: embeddings, vector identity resolution, topic retrieval,
   aggregation, comparison, composition, and answer synthesis.
5. Focused tests: `nq_012`, `nq_038`, `nq_039`, uniquely resolved exact-course
   cases such as `nq_006` and `nq_011`, and multi-course same-program versus
   cross-program resolution. Assert no vector/model call occurs before a
   guard completes.
6. Completion gate: every explicit reference is resolved independently,
   missing references do not yield partial evidence, and the reviewed guard
   precedence is observable in focused tests, frozen spec tests, and
   `git diff --check`.

### 4D — Evidence Planner and structural partitions

1. Frozen contract implemented: map QuerySpec plus the resolution outcome to
   the six Phase 3C evidence primitives, preserving applicable plan ×
   `group_by` partitions and dependencies.
2. Likely existing files: `rag/qa.py` for orchestration and
   `rag/structured/qa.py` for deterministic structural evidence; existing
   query helpers remain the source of relational facts.
3. New file: none required if planning remains a narrow composition layer;
   a dedicated planner module is justified only to prevent plan construction
   from being duplicated in route-specific code.
4. Out of scope: vector execution, topic scoring, threshold selection,
   aggregation/arithmetic, comparison, judgement, and answer rendering.
5. Focused tests: `nq_005`, `nq_009`, `nq_017`, `nq_022`, `nq_023`, and
   `nq_030`, including preserved plan/year/semester partitions and provenance
   on every evidence primitive.
6. Completion gate: the same target relation drives all requested operations,
   partitions are not merged early, no vector work or aggregation occurs, and
   focused, frozen spec, and diff-check gates pass.

### 4E — Constrained semantic retrieval

1. Frozen contract implemented: direct exact-course description fetch and
   candidate-constrained topic scoring over description chunks only; embed
   `QuerySpec.topic`, score all eligible candidates, preserve partitions and
   provenance, and keep the explicit cosine-distance convention.
2. Likely existing files: `rag/retrieval/retrieve.py` for the Natural QA
   constrained path, `rag/retrieval/search.py` for persisted hit enrichment,
   and `rag/retrieval/chunks.py`/`rag/retrieval/index.py` only where current
   chunk metadata and stored-vector access require a compatible extension.
3. New file: none required; no new vector database. A small calibration/test
   fixture is deferred to 4F only if needed.
4. Out of scope: QuerySpec parsing, entity resolution, program guards,
   aggregation, comparison, answer synthesis, embedding-model changes, and
   global-search-then-filter behavior for Natural QA v1.
5. Focused tests: candidate leakage, exact-course direct retrieval, missing
   descriptions, empty topic matches, no global top-k membership truncation,
   metadata chunks excluded from topic scoring, partition retention, and
   provenance. Use `tests/test_rag_retrieve.py` and focused QA integration
   tests with injected local embeddings.
6. Completion gate: no eligible course outside the Phase 3C course set can
   match, membership is not defined by `k`, and existing vectors are reused;
   focused retrieval tests, frozen spec tests, and `git diff --check` pass.

### 4F — Semantic threshold calibration

1. Frozen contract implemented: calibrate one fixed cosine-distance threshold
   for the frozen embedding model on a small manually reviewed development set
   and freeze the resulting configuration before final evaluation.
2. Likely existing files: the retrieval configuration boundary in
   `rag/retrieval/retrieve.py` or `rag/retrieval/index.py`, plus a focused
   calibration test/configuration location selected during implementation.
3. New file: a small checked-in development fixture or calibration artifact
   may be genuinely necessary so the threshold decision is reproducible; it
   must be separate from the held-out Natural QA evaluation artifact.
4. Out of scope: tuning from final evaluation results, per-query dynamic
   thresholds, embedding/reranker changes, corpus rebuilds solely for design,
   and aggregation or answer wording.
5. Focused tests: threshold boundary behavior, frozen model/dimension
   compatibility, reproducible calibration input, and distinction between
   empty structural candidates, missing descriptions, and no threshold
   matches.
6. Completion gate: one fixed threshold is reproducibly loaded and tested,
   its development evidence is reviewable, no held-out score influenced it,
   and focused, frozen spec, and diff-check gates pass.

### 4G — Deterministic aggregation, comparison, and composition

1. Frozen contract implemented: local logical-course deduplication,
   `required_load` versus `option_count`, alternative-group accounting,
   count/existence/credits/earliest, ties, normalized plan-result equivalence,
   plan diff, and multi-operation composition without LLM arithmetic.
2. Likely existing files: `rag/structured/qa.py` for deterministic facts and
   `rag/qa.py` for combining evidence results. The existing structured query
   helpers remain authoritative for credits, placement, prerequisites, and
   groups.
3. New file: none required unless a small pure composition helper is needed
   to keep derived-value logic out of SQL execution.
4. Out of scope: Thai parsing, entity resolution, semantic scoring or
   threshold calibration, free-form SQL, judgement or similarity synthesis,
   and final answer/citation rendering.
5. Focused tests: `nq_003`, `nq_004`, `nq_005`, `nq_009`, `nq_010`, `nq_018`,
   `nq_021`, `nq_022`, `nq_023`, `nq_024`, `nq_030`, `nq_032`, and `nq_036`,
   including alternative groups, ties, plan partitions, and provenance
   lineage. For `nq_021` and `nq_022`, 4G covers only the grounded count/
   proportion and count/credit inputs; their quantity/workload conclusions
   belong to 4H.
6. Completion gate: derived values are deterministic, plan results are
   computed independently before any presentation collapse, and every value
   remains traceable; focused composition tests, frozen spec tests, and
   `git diff --check` pass.

### 4H — Grounded judgement and similarity

1. Frozen contract implemented: quantity, workload, preference, grounded
   similarity, and insufficient-evidence handling from Phase 3E. Workload
   remains a count/credit proxy, and similarity uses only the two exact
   grounded descriptions. The Phase 3B/4C `unsupported` guard must stop an
   unsupported request before it reaches 4H; 4H does not implement that
   judgement.
2. Likely existing files: `rag/answer.py` for bounded grounded synthesis and
   `rag/qa.py` only if judgement metadata must pass through the existing result
   contract.
3. New file: none required; a small pure policy helper is justified only if
   it keeps subjective claims out of free-form generation.
4. Out of scope: retrieval changes, new facts, external knowledge, generic
   answer grading, deciding `unsupported`, and final citation formatting
   beyond preserving the evidence needed by the later answer layer.
5. Focused tests: workload versus difficulty, preference versus “best”,
   bounded two-description similarity for `nq_031`, insufficient evidence
   versus valid empty result, and a regression assertion that an upstream
   `unsupported` request never reaches judgement or similarity synthesis.
6. Completion gate: every judgement is grounded or safely qualified, no
   external facts are introduced, and focused judgement/answer tests, frozen
   spec tests, and `git diff --check` pass.

### 4I — Pipeline integration and provenance regression

1. Frozen contract implemented: wire the completed stages through the
   existing QA path with the smallest compatible changes, preserve the exact
   no-data fallback `ไม่พบข้อมูลนี้ในเล่มหลักสูตร`, keep route labels
   diagnostic, and ensure `ask.py` never silently rebuilds the runtime index.
2. Likely existing files: `rag/qa.py`, `rag/hybrid_demo.py`, `ask.py`, and
   `rag/answer.py`; `rag/router.py` is touched only if an integration blocker
   is demonstrated, not to make route hints the correctness criterion.
3. New file: no new architecture layer is required. A focused integration
   test module is the only expected addition if existing tests cannot express
   the end-to-end contract.
4. Out of scope: OCR and preparation pipeline changes, new models, new vector
   stores, Gold/evaluator changes, DB schema/data changes, and broad router or
   prompt redesign.
5. Focused tests: representative structural-only evidence, constrained-topic
   semantic evidence, mixed-evidence composition, no-data, provenance,
   exact-name, comparison, alternative-group, and fallback cases;
   `tests/test_rag_qa.py`, `tests/test_rag_hybrid_demo.py`,
   `tests/test_rag_answer.py`, and `tests/test_natural_qa_spec.py` are the
   existing regression boundaries. Route labels may be observed
   diagnostically, but a route-label match is not an acceptance criterion.
6. Completion gate: all focused regressions and frozen Natural QA v1 tests
   pass, provenance reaches grounded answers, no silent rebuild or external
   call is introduced by tests, and `git diff --check` passes. Phase 5 alone
   owns the final frozen 40-question end-to-end evaluation.

### Cross-checks and Phase 3F boundaries

Every Phase 3A–3E responsibility has one primary checkpoint above: surface
normalization (4A), neutral understanding (4B), ordered resolution/guards
(4C), evidence planning (4D), constrained semantic retrieval (4E), threshold
calibration (4F), deterministic derivation (4G), grounded judgement (4H),
and compatible integration/provenance regression (4I). The boundaries are
deliberately explicit so no checkpoint both parses and resolves, both retrieves
and aggregates, or both derives facts and rewrites the final answer.

The frozen `tests/test_natural_qa_spec.py` remains a Phase 1 contract test; it
is not weakened or repurposed as a production-behavior test. Every checkpoint
is independently reviewable with focused tests, the frozen spec tests, and
`git diff --check`. Expensive Gemini calls and the complete 40-case E2E run
are not required for each checkpoint; the final held-out evaluation belongs to
Phase 5.
