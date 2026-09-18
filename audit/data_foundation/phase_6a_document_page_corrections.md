# Phase 6A document_page regeneration

- Regenerated the extracted layer using the deterministic resolver; raw OCR was not rerun.
- Canonical map: `audit/data_foundation/phase_6a_document_page_map.json`
- Source-scoped evidence: 119 OCR-detected, 87 source-verified, 3 bounded-offset.
- 209/209 canonical physical pages now carry the expected `document_page`.
- `source_page`, source identity, and extracted semantics were unchanged; only document-page propagation changed.
- Thirty legacy OCR files without source-page identity were not map-resolved or rewritten.
- IT 06016454 and BIT 06036135 remain in their existing pending credit state.
