# Final automated hardening handoff — 2026-10-08

## Verdict

**READY FOR USER UAT WITH DOCUMENTED LIMITATIONS.**

Automated gates passed for the tested scope. Live interpreter coverage,
especially Cluster C, remains unverified and is a required manual gate.
This is not a final submission freeze or a claim that all semantics are safe.

## Results

| Gate | Result |
|---|---|
| Semantic focused | 173 passed |
| Backend contracts | 39 passed (context bounds 21, P1 operational 10, transport 5, startup 3) |
| SQL resource + guard | 33 passed (resource 9) |
| P0 integrity/fallback/count-policy | 24 + 12 + 8 passed, included above |
| Held-out mocked red team | 66 PASS / 44 SAFE_FAIL / 0 FAIL, 110 cases |
| Frontend unit | 53 passed |
| Frontend build | Passed; 40 modules, Vite 5.4.21 |
| Full Python | 2,552 run; 2,549 passed, 3 skipped; 0 failures/errors; 159.787s |

Full-suite skips: two tests depend on absent authoritative
`ground_truth/rag/unseen_factual_v1.json`; one is the opt-in live QP3 provider
seam. They are environment/optional-provider omissions, not scored passes.

Before this continuation, two premature full runs exposed schema
compatibility regressions. The final run includes the repaired bounded
`last_answer` objects, `course_name` fields, and corrected stale test fixtures.
The added backend test-package marker ensures those contracts are now
included in root discovery.

## P2.1 context transport

Explicit Pydantic models reject unknown fields, giant nesting, oversized
strings/lists and bool-as-int before semantic routing. An actual 2MB HTTP
payload returns 422 and a route spy proves no semantic/provider call occurred.
Only supported normal/legacy reference fields are admitted; clarification
operands travel separately. Canonical identity is still re-grounded downstream.

Maximum input retention is 50 courses because legacy producers can emit 50;
new semantic answers retain 20. Study years/semesters are strict bounded
integers. Frontend normalization drops invalid stored state and preserves
legitimate reference objects and empty-result ownership.

Transport JSON decoding itself is not bounded by this Pydantic model.
Request-byte limits/rate limiting belong to a future deployment boundary.

## P2.2 SQL resources

- Read-only URI, statement/relation guard, scope installation and LIMIT preserved.
- SQL length capped at 20,000 characters.
- Allocation-amplifier functions randomblob/zeroblob/repeat rejected, including
  double-quoted, bracketed and backtick function names.
- Model-query execution gets a 1,000,000-VM-step budget checked every 1,000 steps,
  retained through fetchall and removed in finally.
- Where Python supports it, SQLite's value/row length limit is 1,000,000 bytes.
- Cross joins and large sorts abort; recursive CTEs are rejected; normal nested
  queries and representative curriculum queries remain valid.
- Resource abort maps to sqlite_error and cannot invoke answer generation.

These are SQLite-level controls, not a process-level hard CPU/memory quota.
Older Python without `Connection.setlimit` retains guard/VM controls only.

## Dependency audit

No package upgrades were performed. `pip-audit` was not installed; an OSV
batch query checked all 86 installed Python distributions instead.
The following 9 unique issues produced 18 records due to aliases:

| Package | Advisory | Severity | Classification |
|---|---|---|---|
| pip 25.0.1 | GHSA-4xh5-x5gv-qwph — fallback tar symlinks | Moderate | DEFER_DEPLOYMENT/toolchain |
| pip 25.0.1 | GHSA-58qw-9mgm-455v — tar/ZIP interpretation | Moderate | DEFER_DEPLOYMENT/toolchain |
| pip 25.0.1 | GHSA-6vgw-5pg2-w6jp — path traversal | Low | DEFER_DEPLOYMENT/toolchain |
| pip 25.0.1 | GHSA-jp4c-xjxw-mgf9 — untrusted functionality | Moderate | DEFER_DEPLOYMENT/toolchain |
| pip 25.0.1 | GHSA-qwm4-qh6w-59xr — double-encoded index URL | Moderate | DEFER_DEPLOYMENT/toolchain |
| pip 25.0.1 | GHSA-wf93-45jw-7689 — entry-point path traversal | Moderate | DEFER_DEPLOYMENT/toolchain |
| urllib3 2.7.0 | GHSA-8988-9cw3-xx77 — HTTPS proxy TLS settings | High | DEFER_DEPLOYMENT; no student-selected proxy |
| urllib3 2.7.0 | GHSA-gh4c-6fx4-qh6g — deflate streaming loop | Moderate | DEFER_DEPLOYMENT; hostile upstream response required |
| urllib3 2.7.0 | GHSA-vxq7-64xx-v4gw — chunk-size memory growth | High | DEFER_DEPLOYMENT; hostile upstream response required |

urllib3 advisories list 2.8.0 as fixed. pip fixes vary from 25.3 through
26.2.0. Reassess/update toolchain and network dependencies before public
deployment; these do not originate from an exposed student input path.

`npm audit`: five affected package nodes, **0 critical / 2 high / 3 moderate**.

| Package group | Finding | Classification in built localhost demo |
|---|---|---|
| Vite | GHSA-4w7w-66w2-5vf9, GHSA-v6wh-96g9-6wx3, GHSA-fx2h-pf6j-xcff: dev-server path/UNC/Windows-deny bypass | DEFER_DEPLOYMENT; Vite dev server not run in submission path |
| esbuild | GHSA-67mh-4wv8-2f99: dev-server cross-origin read | DEFER_DEPLOYMENT; no esbuild dev server |
| source-map-js | GHSA-68fv-2mgg-jv7q: indexed source-map offset DoS | DEFER_DEPLOYMENT/build tooling; no user source-map upload |
| React Router / DOM | GHSA-wrjc-x8rr-h8h6: redirect; GHSA-337j-9hxr-rhxg: SSR hydration constructor injection | NOT_APPLICABLE to current fixed-route client rendering; revisit user navigation/SSR |

No CRITICAL_FIX_NOW/HIGH_FIX_NOW advisory was established for this built
localhost question-answering path. Major upgrades were deliberately avoided.

## Secret and web checks

Redacted tracked/untracked source scan: 582 files, zero credential-pattern
findings. No tracked .env/private-key files. `.env` and built frontend artifacts
are ignored. The local wireframe backup was excluded and untouched.

Source checks found no unsafe HTML rendering, eval/Function, backend
user-controlled subprocess, upload/write endpoint, or client API key. Search
SQL parameters are bound. Operational errors use fixed messages. Generated
SQL is not included in the public ask-response schema.

Static serving is rooted in frontend/assets rather than repository root.
The legacy `/static` mount exposes frontend files, including source/dependency
files present there; restrict it to the built directory before public deployment.
No authentication, rate limiting, or cross-origin policy was introduced; run
only on 127.0.0.1 for the accepted demo scope.

## User manual gates

Cluster C is **not closed**. Probe live interpretation of:

- ปี 2 หรือปี 3 มีวิชาอะไรบ้าง
- แผนสหกิจและไม่สหกิจต่างกันยังไง
- Mixed policy + curriculum two-clause questions
- Topic A และ topic B; topic A หรือ topic B
- นอกจาก X มีอะไรอีก
- Negative/universal/quantified wording

Accept only complete semantics or explicit clarification/unsupported;
silently answering one subset is a failure and blocks freeze.

Also verify real-provider failure/retry, ordinary follow-ups, edition/plan
popup chains, session isolation and provenance. Friend UAT, live evaluation,
browser smoke and latency measurements remain user-run gates.

For browser smoke, rebuild AND restart are required (not executed by agent):

```powershell
Ctrl+C
cd frontend
npm run build
cd ..
$env:CUCUMBER_QA_MODE = "semantic"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/chat`, press `Ctrl+F5`, and verify
`/api/health`: status=ok, database_ready=true, qa_mode=semantic.
