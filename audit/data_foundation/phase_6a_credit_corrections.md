# Phase 6A.2f Source-Verified Credit Corrections

## IT 06016454

- Program/plan: IT / coop description scope
- Source filename: `it_page_354.png`
- Source page: 354
- Previous extracted credit: empty
- Corrected credit: `3(3-0-6)`
- Evidence: authoritative IT academic-plan records for both coop and no-coop
  plans agree on `3(3-0-6)`; the source page shows the same course and credit
  structure.
- The OCR credit is split across non-adjacent lines (`3(3-0-` and `6)`), so
  the parser intentionally leaves it unresolved rather than crossing intervening
  course text.

## BIT 06036135

- Program/plan: BIT / coop description scope
- Source filename: `bit_page_252.png` (OCR artifact: `bit_page_252_ocr.json`)
- Source page: 252
- Previous extracted credit: empty
- Corrected credit: `3(3-0-6)`
- Evidence: authoritative BIT coop and no-coop academic-plan records agree on
  `3(3-0-6)`; the source page identifies course 06036135.
- The OCR credit is incomplete (`3(2-2`), so the parser cannot safely infer
  the missing structure from the tuple fragment.

Corrections are applied only for the exact program, plan, course code, source
filename, and source page identities above, and only when authoritative source
data supplies the full credit string. Raw OCR remains unchanged.
