import json
from pathlib import Path
from typing import List


def save_ocr_results(text_lines: List[str], output_dir: Path, base_name: str) -> None:
    """Save extracted text to both plain text (.txt) and metadata (.json) format."""
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_file = output_dir / f"{base_name}_ocr.txt"
    json_file = output_dir / f"{base_name}_ocr.json"

    # 1. Save Plain Text File
    extracted_text = "\n".join(text_lines)
    txt_file.write_text(extracted_text, encoding="utf-8")
    print(f"✓ Text output saved to: {txt_file}")

    # 2. Save Structured JSON File
    json_data = {
        "filename": base_name,
        "line_count": len(text_lines),
        "text_lines": text_lines,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        
    print(f"✓ JSON metadata saved to: {json_file}")