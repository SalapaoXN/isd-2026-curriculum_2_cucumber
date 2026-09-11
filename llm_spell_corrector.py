"""Safely correct curriculum text with Gemini in bounded record batches."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from google import genai

from config import API_KEY


MODEL = "gemini-3.5-flash-lite"
BATCH_SIZE = 10
TEXT_FIELDS_ORDER = ("name_th", "name_en", "desc_th", "desc_en", "note")
TEXT_FIELDS = frozenset(TEXT_FIELDS_ORDER)
IDENTITY_FIELDS = ("course_code", "code", "course_id", "id")

CORRECTION_PROMPT = (
    "You are a proofreader for university curriculum JSON course records. "
    "Correct only spelling errors and OCR typos in these five text fields: "
    "name_th, name_en, desc_th, desc_en, and note. "
    "Only the values of those five fields may change. Everything else is immutable, "
    "including course identity and code, credits, year, semester, flexible timing, "
    "category, type, prerequisites, program and plan metadata, provenance, all keys, "
    "record count, and record order. Do not add, remove, or reorder records or fields. "
    "Return ONLY a valid JSON array containing exactly the same records in the same order; "
    "do not use markdown code fences or add commentary."
)


def _records_from_document(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        records = document
    elif isinstance(document, dict) and isinstance(document.get("courses"), list):
        records = document["courses"]
    else:
        raise ValueError(
            "Unsupported curriculum JSON structure; expected a record list or an object with courses"
        )

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Curriculum records must all be JSON objects")
    return records


def _record_identity(record: dict[str, Any], index: int) -> tuple[str, Any]:
    for field in IDENTITY_FIELDS:
        if field in record:
            return field, record[field]
    return "index", index


def _record_course_code(record: dict[str, Any]) -> Any:
    if "course_code" in record:
        return record["course_code"]
    return record.get("code")


def _same_json(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(before, after) for before, after in zip(left, right)
        )
    return left == right


def _decode_response(response: Any, batch_number: int) -> list[Any]:
    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError(f"Gemini batch {batch_number} returned no JSON text")

    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines).strip()

    try:
        decoded = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Gemini batch {batch_number} returned invalid JSON") from error

    if not isinstance(decoded, list):
        raise ValueError(f"Gemini batch {batch_number} must return a JSON array")
    return decoded


def _validate_batch(
    before_batch: list[dict[str, Any]],
    after_batch: list[Any],
    batch_number: int,
    start_index: int,
) -> list[dict[str, Any]]:
    if len(after_batch) != len(before_batch):
        raise ValueError(
            f"Gemini batch {batch_number} changed record count: "
            f"expected {len(before_batch)}, got {len(after_batch)}"
        )

    validated: list[dict[str, Any]] = []
    for offset, (before, after) in enumerate(zip(before_batch, after_batch)):
        index = start_index + offset
        if not isinstance(after, dict):
            raise ValueError(f"Gemini batch {batch_number} record {index} is not a JSON object")
        if before.keys() != after.keys():
            raise ValueError(f"Gemini batch {batch_number} record {index} changed JSON structure")
        if _record_identity(before, index) != _record_identity(after, index):
            raise ValueError(f"Gemini batch {batch_number} record {index} changed record identity")

        for field in before:
            if field in TEXT_FIELDS:
                if after[field] is not None and not isinstance(after[field], str):
                    raise ValueError(
                        f"Gemini batch {batch_number} record {index} changed text field {field} to a non-text value"
                    )
            elif not _same_json(before[field], after[field]):
                raise ValueError(
                    f"Gemini batch {batch_number} record {index} changed immutable field {field}"
                )
        validated.append(after)
    return validated


def _write_json_pair(
    corrected_path: Path,
    corrections_path: Path,
    corrected_data: Any,
    corrections: list[dict[str, Any]],
) -> None:
    targets = ((corrected_path, corrected_data), (corrections_path, corrections))
    temporary_paths: list[Path] = []
    try:
        for target, payload in targets:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                dir=str(target.parent),
                text=True,
            )
            temporary_path = Path(temporary_name)
            temporary_paths.append(temporary_path)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
                json.dump(payload, output, ensure_ascii=False, indent=4)
                output.write("\n")

        os.replace(temporary_paths[0], corrected_path)
        os.replace(temporary_paths[1], corrections_path)
        temporary_paths.clear()
    finally:
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def correct_json_file(file_path: str | Path, output_dir: str | Path | None = None) -> Path:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Input file is not valid JSON: {path}") from error

    before_records = _records_from_document(document)
    corrected_records = copy.deepcopy(before_records)
    corrections: list[dict[str, Any]] = []

    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as error:
        raise RuntimeError("Gemini client initialization failed") from error

    for start_index in range(0, len(before_records), BATCH_SIZE):
        before_batch = before_records[start_index : start_index + BATCH_SIZE]
        batch_number = (start_index // BATCH_SIZE) + 1
        contents = [
            CORRECTION_PROMPT,
            json.dumps(before_batch, ensure_ascii=False, indent=2),
        ]
        try:
            response = client.models.generate_content(model=MODEL, contents=contents)
        except Exception as error:
            raise RuntimeError(f"Gemini batch {batch_number} failed") from error

        after_batch = _decode_response(response, batch_number)
        validated_batch = _validate_batch(
            before_batch,
            after_batch,
            batch_number,
            start_index,
        )

        for offset, (before, after) in enumerate(zip(before_batch, validated_batch)):
            corrected_record = corrected_records[start_index + offset]
            for field in TEXT_FIELDS_ORDER:
                if field not in before or before[field] == after[field]:
                    continue
                corrections.append(
                    {
                        "course_code": _record_course_code(before),
                        "field": field,
                        "before": before[field],
                        "after": after[field],
                    }
                )
                corrected_record[field] = after[field]

    if isinstance(document, list):
        corrected_data = corrected_records
    else:
        corrected_data = copy.deepcopy(document)
        corrected_data["courses"] = corrected_records

    output_directory = path.parent if output_dir is None else Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    corrected_path = output_directory / f"{path.stem}_corrected.json"
    corrections_path = output_directory / f"{path.stem}_corrections.json"
    _write_json_pair(corrected_path, corrections_path, corrected_data, corrections)

    print(f"Corrected JSON saved to: {corrected_path}")
    print(f"Corrections log saved to: {corrections_path}")
    return corrected_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="python llm_spell_corrector.py <path_to_json> [--output-dir PATH]"
    )
    parser.add_argument("path_to_json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    correct_json_file(args.path_to_json, output_dir=args.output_dir)
