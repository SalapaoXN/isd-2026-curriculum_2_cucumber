"""Merge tool: corrected information -> merged information.

"Merge stage: extracted JSON -> consolidated page_ranges + full files. No argv here."""
from __future__ import annotations

from src.pipeline.tools.merge.consolidator import merge_consecutive_files


def run_merge_stage(
    input_dir: str | Path,
    output_dir: str | Path,
    plan_filter: str | None = None,
    pages: str | None = None,
    prefix: str | None = None,
    desc_pages: str | None = None,
) -> None:
    """Merge one scope with identical logic to merge_consecutive.py."""
    merge_consecutive_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        plan_filter=plan_filter,
        pages=pages,
        prefix=prefix,
        desc_pages=desc_pages,
    )
