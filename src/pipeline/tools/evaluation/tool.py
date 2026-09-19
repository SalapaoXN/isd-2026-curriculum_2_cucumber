"""Evaluation tool: verifies corrected output vs ground truth (reports/).

"Evaluate stage: corrected JSON vs ground truth -> reports. No argv here."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.tools.evaluation.evaluate import discover_llm_evaluation_pairs, evaluate_pair, write_evaluation_reports


def run_evaluate_stage(
    llm_dir: str | Path | None = None,
    ground_truth_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    pairs: list[tuple[str | Path, str | Path]] | None = None,
) -> dict:
    """Evaluate corrected artifacts and return the report payload."""
    if pairs is None:
        resolved = discover_llm_evaluation_pairs(llm_dir, ground_truth_dir)
    else:
        resolved = [(Path(p), Path(g)) for p, g in pairs]
    cases = [evaluate_pair(gt, pred) for pred, gt in resolved]
    return write_evaluation_reports(cases, reports_dir) if reports_dir is not None else write_evaluation_reports(cases)
