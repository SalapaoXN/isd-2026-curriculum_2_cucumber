"""Correction tool: extracted information -> corrected information (names only).

"Correct stage: consolidated full JSON -> llm corrected JSON. No argv here."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.tools.correction.corrector import correct_json_files, discover_consolidated_inputs


def run_correct_stage(
    consolidated_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    input_paths: list[str | Path] | None = None,
) -> list[Path]:
    """Correct full consolidated files and return corrected artifact paths."""
    from pathlib import Path as _Path

    paths = [ _Path(p) for p in input_paths ] if input_paths is not None else discover_consolidated_inputs(
        consolidated_dir if consolidated_dir is not None else "data/output/consolidated"
    )
    return correct_json_files(paths, output_dir=output_dir)
