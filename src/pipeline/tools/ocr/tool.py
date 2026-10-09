"""OCR tool: images -> persisted OCR artifacts."""
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
    dataset_key: str | None = None,
) -> Path:
    """Run OCR for one configured dataset and return its OCR output directory.

    Page selection is an internal orchestration detail. The public end-to-end
    CLI resolves the required page list from ``src.pipeline.datasets``.
    """
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[4]
    )
    normalized = normalize_program(program)
    key = dataset_key or normalized.casefold()
    resolved_input = (
        Path(input_dir)
        if input_dir is not None
        else root / "data" / "input" / key
    )
    resolved_output = (
        Path(output_dir)
        if output_dir is not None
        else root / "data" / "output"
    )
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
        dataset_key=dataset_key,
    )
