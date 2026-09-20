# data/

Canonical pipeline data layout (clean rebuild).

- `input/` — place source page images here, mirroring the original `inputs/`:
  `input/<program>/<program>_page_NNN.png` (ait, bit, dsba, gened, it, rule).
  Original `inputs/` were not vendored (large binaries); copy them in to run OCR.
- `output/` — pipeline working tree. Intermediates are in-memory/temp by default
  (`--keep-intermediates` persists `extracted/` + `consolidated/` for debug).
- `output/final/` — **canonical final RAG-ready output** (`*_corrected.json` +
  `*_corrections.json`). This replaces the original `outputs/llm/`.
  RAG (`rag.build_index`) consumes ONLY this directory.

Pre-seeded reference copies of the original final outputs are under
`submission/` and `reports/evaluation_reference/` for behavior comparison.
