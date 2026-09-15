# CUCUMBER RAG — Current Work

## Baseline

- Branch: `heart`
- Frozen pre-unseen baseline: `f854cc7`
- U01–U30 first unseen run completed.
- U01–U30 is now a regression/development set, not final unseen.

## Completed

- Exact-course credit execution scope fixed.
- Split-line credit parsing fixed and globally dry-run verified.
- Source-backed missing-credit reconciliation implemented.
- Source-page provenance audited end-to-end; no page-loss defect exists.
- Evaluator provenance fixes completed:
  - enclosing-plan inheritance
  - validated OCR/image source-key equivalence
  - validated page-range vs individual-page equivalence
- No full regression evaluation has been rerun after these fixes.
- Phase 6A.0 extraction audit baseline established with deterministic OCR/extracted manifest and hashes.
- Phase 6A OCR→Extracted audit found a generic description-credit parser defect; OCR/source noise is separated from parser defects.
- Phase 6A.1 removed fabricated description-credit fallback; parenthetical OCR credits are now preserved without guessing a leading unit.
- Phase 6A.2 extraction regeneration exposed five description records requiring source review after fabricated-credit removal.
- Phase 6A.2a audit: four records are parser-recoverable; BIT 06036135 needs separate source-backed review.
- Phase 6A.2b added safe description-credit syntax normalization for numerically complete OCR fragments without numeric inference.
- Phase 6A.2c found unanimous source credits; four candidates are safe with a conflict guard, while BIT 06036135 remains review.
- Phase 6A.2d verification blocked: 3/4 target credits normalized, IT 06016454 remained empty, and unrelated extraction drift appeared.
- Phase 6A.2d-a comparison artifact: stable path/code/occurrence comparison found only the three intended 6A.2b credit changes; no parser side effect or nondeterminism was proven.
- Phase 6A.2e audit: IT 06016454 has one interleaved credit-fragment candidate, but persisted OCR JSON has no coordinates; same-field ownership is unproven and the parser input is line-only.
- Phase 6A.2e-a OCR metadata audit: page 354 exists; EasyOCR can emit geometry in memory, but the active pipeline requests text-only output and serializes no bounding boxes; geometry is useful for credit fields and printed document pages.
- Phase 6A.2e-b design audit: optional raw OCR detections with quadrilateral bbox/text/confidence and image dimensions preserve `text_lines` compatibility while supporting spatial credit and printed-page detection; legacy JSON remains readable.
- Phase 6A.2e-c added backward-compatible OCR geometry preservation while retaining the existing `text_lines` interface.
- Phase 6A.2e-d page-354 dry-run: target geometry was recovered, but detail=1 produced material text differences that block compatibility approval.
- Phase 6A.2e-e audit: 10 detail=0 pages showed mixed propagation (6), boundary ambiguity (2), and tail-window misses (2); geometry/detail=1 work is deferred.
- Phase 6A document-page propagation restored existing detail=0 detector results without offset inference or detector broadening.
- Phase 6A.2e-g audit: 209 source pages classified; 119 direct, 87 edge-verified, and 3 safe bounded-offset candidates; no unresolved pages.
- Phase 6A.2e-i design: canonical 209-entry source-verified page map validated; 119 OCR-detected, 87 source-verified, and 3 bounded-offset entries.
- Phase 6A.2e-j implementation: deterministic resolver reproduces 209 audited mappings via OCR, bounded GENED logic, and audited source-verified overrides; the map remains verification-only.
- Phase 6A.2e-k extraction regeneration: 209/209 source-scoped pages propagated with no non-document-page semantic drift; raw OCR unchanged.
- Phase 6A.2f extraction: source-verified IT 06016454 and BIT 06036135 credits closed; 2,065 records and document_page provenance preserved.
- Phase 6A.3 foundation gate: OCR→Extracted inventory, credit, provenance, coverage, and determinism checks passed.
- Phase 6B.1 merge audit: consolidation rules are ready; current outputs are stale for document_page metadata and require controlled regeneration.
- Phase 6B.2a audit: merge-time raw description re-extraction bypasses complete Extracted credits and causes 34 serialized credit degradations.
- Phase 6B.2b merge fix: persisted Extracted complete credits are preserved through page-range and full consolidation with focused regressions passing.
- Phase 6B.3 foundation gate: Extracted→Consolidated traceability, factual integrity, credit, isolation, provenance, loss, and determinism checks passed.
- Phase 6C.1 audit: Consolidated→Corrected differences are stale/manual state; correction output lacks a non-empty text guard.
- Phase 6C.2 regeneration: blocked before AIT output by unavailable Gemini network access; 16-file OS-temp snapshot preserved.
- Phase 6C.2a audit: offline Corrected rebase is safe; 839 source-aware matches plus deterministic IT duplicate recovery cover all 841 records.
- Phase 6C.2b rebase: all eight Corrected scopes deterministically rebased from current Consolidated data; 841/841 records preserved.
- Phase 6C.2c audit: real-Gemini Corrected artifacts pass 841/841 identity, factual, multiplicity, name-safety, and log checks.
- Phase 6C.2d routing: restored default LLM output destination to `outputs/llm/`; verified with focused tests.
- Phase 6C.2e promotion: Real-Gemini Corrected artifacts promoted byte-for-byte to canonical `outputs/llm/`; 841/841 verified.
- Phase 6C.3a cleanup: removed 16 verified misrouted temporary Corrected/log artifacts; canonical `outputs/llm/` unchanged.
- Phase 6C.3 audit: canonical Corrected foundation and downstream handoff pass; offline-rebase scaffolding requires separate cleanup.
- Phase 6C.3b cleanup: removed offline-rebase scaffolding; canonical 841-record Corrected baseline remains intact.
- Phase 6C.3c gate: final Consolidated→Corrected integrity, routing, and focused-test checks passed.
- Phase 6A-R.2: dual Consolidated/LLM vs GT evaluator implemented and baseline reports generated.
- Phase 6A-R.3a: prerequisite alternative normalization fixed; dual reports regenerated.
- Phase 6A-R.4: source-backed Ground Truth correction batch applied; dual reports regenerated.

## Current Task
Phase 6A-R.4 — Audit Remaining Evaluation Mismatches
Status: AUDIT ONLY

Goal:
Audit the remaining evaluation mismatches after the source-backed Ground
Truth correction batch.

## Next

- Pass OCR → Extracted foundation gate.
- Audit Extracted → Consolidated.
- Audit Consolidated → Corrected.
- Audit Corrected → SQLite.
- Unfreeze RAG only after all data-foundation gates pass.

## Guardrails

- Preserve unrelated dirty working-tree changes.
- Use targeted `rg` and small code windows only.
- One primary problem per micro-task.
- No question-specific hardcoding.
- No DB patch to hide upstream defects.
- No runtime use of unseen/evaluation expected answers.
- Preserve program / plan / course / provenance identities.
- Prefer fail-closed behavior.
- Do not rerun full evaluation unless explicitly requested.
- No git add / commit / push unless explicitly requested.
