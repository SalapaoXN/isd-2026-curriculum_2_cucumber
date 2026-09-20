"""OCR tool: images -> persisted OCR artifacts (in-memory lines + persisted txt/json).

"OCR stage: images -> persisted OCR artifacts. No argv here."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.config import normalize_program
from src.pipeline.tools.ocr.pipeline_runner import parse_pages, run_ocr


def run_ocr_stage(
    program: str,
    pages: str | list[int] | None = None,
    no_gpu: bool = False,
    project_root: str | Path | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Run OCR for one program and return the OCR output directory."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[4]
    normalized = normalize_program(program)
    key = normalized.casefold()
    resolved_input = Path(input_dir) if input_dir is not None else root / "data" / "input" / key
    resolved_output = Path(output_dir) if output_dir is not None else root / "data" / "output"
    if isinstance(pages, str):
        page_list: list[int] | None = parse_pages(pages)
    else:
        page_list = list(pages) if pages is not None else None
    return run_ocr(
        input_dir=resolved_input,
        output_dir=resolved_output,
        program=normalized,
        pages=page_list,
        no_gpu=no_gpu,
    )
