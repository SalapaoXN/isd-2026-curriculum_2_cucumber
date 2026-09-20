import json
from pathlib import Path
from typing import Any, List, Sequence

from src.pipeline.utils.page_metadata import document_page_from_lines, parse_document_page


def save_ocr_results(
    text_lines: List[str],
    output_dir: Path,
    base_name: str,
    source_filename: str | Path | None = None,
    source_page: int | None = None,
    program: str | None = None,
    document_page: int | None = None,
    detections: Sequence[dict[str, Any]] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    """Save extracted text to both plain text (.txt) and metadata (.json) format."""
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_file = output_dir / f"{base_name}_ocr.txt"
    json_file = output_dir / f"{base_name}_ocr.json"

    # 1. Save Plain Text File
    extracted_text = "\n".join(text_lines)
    txt_file.write_text(extracted_text, encoding="utf-8")
    print(f" Text output saved to: {txt_file}")

    # 2. Save Structured JSON File
    if document_page is None:
        document_page = document_page_from_lines(text_lines)
    else:
        document_page = parse_document_page(document_page)

    json_data = {
        "filename": base_name,
        "line_count": len(text_lines),
        "text_lines": text_lines,
        "document_page": document_page,
    }
    if source_filename is not None:
        json_data["source_filename"] = Path(source_filename).name
    if source_page is not None:
        json_data["source_page"] = source_page
    if program is not None:
        json_data["program"] = program
    if detections is not None:
        json_data["detections"] = _json_safe(detections)
    if image_width is not None:
        json_data["image_width"] = _json_safe(image_width)
    if image_height is not None:
        json_data["image_height"] = _json_safe(image_height)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f" JSON metadata saved to: {json_file}")


def _json_safe(value: Any) -> Any:
    """Convert numpy-like values from OCR into JSON-serializable values."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value
