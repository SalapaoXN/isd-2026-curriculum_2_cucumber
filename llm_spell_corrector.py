"""Spell corrector using Gemini (gemini-3.5-flash-lite) for curriculum JSON files."""

import sys
import json
from pathlib import Path
from google import genai
from config import API_KEY


def correct_json_file(file_path: str | Path) -> Path:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = genai.Client(api_key=API_KEY)
    
    prompt = (
        "You are a proofreader and spell checker for university curriculum JSON data. "
        "Correct any spelling errors or OCR typos in text fields (such as course names in Thai and English, "
        "prerequisites, descriptions, notes) while keeping all course codes, credits, years, semesters, "
        "categories, types, structure, and source provenance exactly the same. "
        "Return ONLY the corrected JSON object with valid formatting and no markdown code blocks or extra text."
    )

    contents = [
        prompt,
        json.dumps(data, ensure_ascii=False, indent=2)
    ]

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=contents,
    )

    response_text = response.text.strip()
    if response_text.startswith("```"):
        lines = response_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        response_text = "\n".join(lines).strip()

    corrected_data = json.loads(response_text)

    output_path = path.with_name(f"{path.stem}_corrected.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrected_data, f, ensure_ascii=False, indent=4)

    print(f"Corrected JSON saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python typhoon_spell_corrector.py <path_to_json>")
        sys.exit(1)
    
    correct_json_file(sys.argv[1])
