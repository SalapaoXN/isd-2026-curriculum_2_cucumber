"""Safely correct curriculum text with Gemini in bounded unique-text batches."""

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
TEXT_FIELDS_ORDER = ("name_th", "name_en")
TEXT_FIELDS = frozenset(TEXT_FIELDS_ORDER)
UNIT_RESPONSE_FIELDS = frozenset(("unit_index", "field", "text"))

CORRECTION_PROMPT = (
    "You are a careful proofreader for university curriculum text. "
    "Each input item is one correction unit with unit_index, field, and text. "
    "Only name_th and name_en are editable. Make the smallest possible spelling or OCR correction only. Do not rewrite style, "
    "paraphrase, or normalize wording unnecessarily. If the text is already correct, "
    "return it byte-for-byte unchanged. Never invent missing factual content. "
    "Return exactly one item for every input unit_index, each exactly containing only "
    "unit_index, field, and text. Preserve every unit_index exactly once, with no missing "
    "or added indices, and preserve each field value. Return ONLY a valid JSON array with "
    "no markdown code fences or extra commentary."
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


def _record_course_code(record: dict[str, Any]) -> Any:
    if "course_code" in record:
        return record["course_code"]
    return record.get("code")


def _build_correction_units(
    documents: list[Any],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    units: list[dict[str, Any]] = []
    unit_keys: list[tuple[str, str]] = []
    unit_index_by_key: dict[tuple[str, str], int] = {}

    for document in documents:
        for record in _records_from_document(document):
            for field in TEXT_FIELDS_ORDER:
                before = record.get(field)
                if not isinstance(before, str) or before == "":
                    continue
                key = (field, before)
                if key in unit_index_by_key:
                    continue
                unit_index = len(units)
                unit_index_by_key[key] = unit_index
                unit_keys.append(key)
                units.append(
                    {
                        "unit_index": unit_index,
                        "field": field,
                        "text": before,
                    }
                )
    return units, unit_keys


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
) -> dict[int, dict[str, Any]]:
    if len(after_batch) != len(before_batch):
        raise ValueError(
            f"Gemini batch {batch_number} changed unit count: "
            f"expected {len(before_batch)}, got {len(after_batch)}"
        )

    expected_indices = set(range(start_index, start_index + len(before_batch)))
    validated: dict[int, dict[str, Any]] = {}
    for after in after_batch:
        if not isinstance(after, dict):
            raise ValueError(f"Gemini batch {batch_number} item is not a JSON object")
        if after.keys() != UNIT_RESPONSE_FIELDS:
            missing_fields = sorted(UNIT_RESPONSE_FIELDS.difference(after.keys()))
            extra_fields = sorted(set(after.keys()).difference(UNIT_RESPONSE_FIELDS))
            raise ValueError(
                f"Gemini batch {batch_number} response fields are invalid; "
                f"missing={missing_fields}, extra={extra_fields}"
            )

        unit_index = after["unit_index"]
        if type(unit_index) is not int:
            raise ValueError(f"Gemini batch {batch_number} unit_index must be an integer")
        if unit_index in validated:
            raise ValueError(
                f"Gemini batch {batch_number} has duplicate unit_index {unit_index}"
            )
        if unit_index not in expected_indices:
            raise ValueError(
                f"Gemini batch {batch_number} has unknown unit_index {unit_index}"
            )

        expected_unit = before_batch[unit_index - start_index]
        if after["field"] != expected_unit["field"]:
            raise ValueError(
                f"Gemini batch {batch_number} changed field for unit_index {unit_index}"
            )
        if not isinstance(after["text"], str):
            raise ValueError(
                f"Gemini batch {batch_number} text for unit_index {unit_index} is not a string"
            )
        validated[unit_index] = after

    missing_indices = sorted(expected_indices.difference(validated))
    if missing_indices:
        raise ValueError(
            f"Gemini batch {batch_number} is missing unit_index values {missing_indices}"
        )
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


def _output_paths(path: Path, output_dir: str | Path | None) -> tuple[Path, Path]:
    directory = path.parent if output_dir is None else Path(output_dir)
    return (
        directory / f"{path.stem}_corrected.json",
        directory / f"{path.stem}_corrections.json",
    )


def _correct_units(
    units: list[dict[str, Any]],
    unit_keys: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    if not units:
        return {}

    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as error:
        raise RuntimeError("Gemini client initialization failed") from error

    corrected_by_key: dict[tuple[str, str], str] = {}
    for start_index in range(0, len(units), BATCH_SIZE):
        before_batch = units[start_index : start_index + BATCH_SIZE]
        batch_number = (start_index // BATCH_SIZE) + 1
        contents = [
            CORRECTION_PROMPT,
            json.dumps(before_batch, ensure_ascii=False, indent=2),
        ]
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config={"temperature": 0},
            )
        except Exception as error:
            raise RuntimeError(f"Gemini batch {batch_number} failed") from error

        after_batch = _decode_response(response, batch_number)
        validated_batch = _validate_batch(
            before_batch,
            after_batch,
            batch_number,
            start_index,
        )
        for unit_index, after in validated_batch.items():
            corrected_by_key[unit_keys[unit_index]] = after["text"]
    return corrected_by_key


def _reconstruct_document(
    document: Any,
    corrected_by_key: dict[tuple[str, str], str],
) -> tuple[Any, list[dict[str, Any]]]:
    corrected_document = copy.deepcopy(document)
    original_records = _records_from_document(document)
    corrected_records = _records_from_document(corrected_document)
    corrections: list[dict[str, Any]] = []

    for original_record, corrected_record in zip(original_records, corrected_records):
        for field in TEXT_FIELDS_ORDER:
            before = original_record.get(field)
            if not isinstance(before, str) or before == "":
                continue
            key = (field, before)
            if key not in corrected_by_key:
                continue
            after = corrected_by_key[key]
            if before == after:
                continue
            corrected_record[field] = after
            corrections.append(
                {
                    "course_code": _record_course_code(original_record),
                    "field": field,
                    "before": before,
                    "after": after,
                }
            )
    return corrected_document, corrections


def correct_json_files(
    file_paths: list[str | Path],
    output_dir: str | Path | None = None,
) -> list[Path]:
    if not file_paths:
        raise ValueError("At least one input JSON file is required")

    paths = [Path(file_path) for file_path in file_paths]
    documents: list[Any] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as error:
            raise ValueError(f"Input file is not valid JSON: {path}") from error
        _records_from_document(documents[-1])

    output_specs = [_output_paths(path, output_dir) for path in paths]
    output_targets = [target for pair in output_specs for target in pair]
    resolved_targets = [target.resolve() for target in output_targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("Input files would produce colliding output artifact paths")

    units, unit_keys = _build_correction_units(documents)
    corrected_by_key = _correct_units(units, unit_keys)
    reconstructed = [
        _reconstruct_document(document, corrected_by_key)
        for document in documents
    ]

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    for corrected_path, corrections_path in output_specs:
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
    for (corrected_path, corrections_path), (corrected_data, corrections) in zip(
        output_specs,
        reconstructed,
    ):
        _write_json_pair(corrected_path, corrections_path, corrected_data, corrections)
        print(f"Corrected JSON saved to: {corrected_path}")
        print(f"Corrections log saved to: {corrections_path}")
    return [corrected_path for corrected_path, _ in output_specs]


def correct_json_file(
    file_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    return correct_json_files([file_path], output_dir=output_dir)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="python llm_spell_corrector.py <path_to_json> [<path_to_json> ...] [--output-dir PATH]"
    )
    parser.add_argument("path_to_json", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    correct_json_files(args.path_to_json, output_dir=args.output_dir)
