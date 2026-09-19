# scripts/

Maintenance, evaluation, validation, report-generation, and developer tools.
Production pipeline stages and the user QA entrypoint remain at the
repository root or under `rag/`/`src/` by design; see
`docs/project_structure.md`.

## Maintained evaluation

- `evaluate_gold_questions.py` — maintained RAG evaluation tooling.
  Executes Gold Questions through the project RAG pipeline and captures
  results. Imported by `tests/test_evaluate_gold_questions.py`.

## Developer/evaluation utility

- `generate_semantic_threshold_dev.py` — developer/evaluation tooling.
  Generates the local semantic-threshold development review set from the
  runtime database.

## Historical/submission tooling

- `build_conversion_report.py` — historical/submission tooling.
  Builds the Lab 8B curriculum conversion report from existing artifacts;
  reads historical `work/` artifacts and references `submission/`.
- `verify_curriculum.py` — historical/submission validation tooling.
  Validates the Lab 8B IT curriculum submission under `submission/`.

The two submission-related scripts above are classified as historical
because their inputs, outputs, and defaults are centered on the frozen
`submission/` package. That classification does not mean they are approved
for deletion.
