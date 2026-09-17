"""Safely correct curriculum text with Gemini in bounded unique-text batches."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from google import genai

from config import API_KEY


MODEL = "gemini-3.5-flash-lite"
BATCH_SIZE = 50
TEXT_FIELDS_ORDER = ("name_th", "name_en")
TEXT_FIELDS = frozenset(TEXT_FIELDS_ORDER)
CORRECTION_FIELDS = frozenset(("name_th", "name_en", "desc_th", "desc_en"))
UNIT_RESPONSE_FIELDS = frozenset(("unit_index", "field", "text"))
CONSOLIDATED_DIR = Path("outputs/consolidated")
LLM_OUTPUT_DIR = Path("outputs/llm")

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

CANONICAL_NAME_CORRECTIONS = {
    ("GENED", "90642067", "name_th", "ซอฟบอลและเบสบอล"): "ซอฟต์บอลและเบสบอล",
    (
        "GENED",
        "90642118",
        "name_en",
        "APPLICATION SOFTWARE FOR BUSSINESS",
    ): "APPLICATION SOFTWARE FOR BUSINESS",
    (
        "AIT",
        "06046413",
        "name_en",
        "ARTIFICIAL INTELLIGIENCE AND INTERNET OF THING",
    ): "ARTIFICIAL INTELLIGENCE AND INTERNET OF THINGS",
    (
        "AIT",
        "06046422",
        "name_en",
        "ARTIFICIAL INTELLIGIENCE ETHICS",
    ): "ARTIFICIAL INTELLIGENCE ETHICS",
    (
        "IT",
        "06016412",
        "name_en",
        "COMPUTER ORCANIZATON AND OPERATING SSTEM",
    ): "COMPUTER ORGANIZATION AND OPERATING SYSTEM",
    (
        "IT",
        "06016466",
        "name_en",
        "NETWORK AND SYSTEM TROUBLE SHOOTNG",
    ): "NETWORK AND SYSTEM TROUBLE SHOOTING",
    (
        "AIT",
        "06046413",
        "name_th",
        "ปัญญา ประดิษฐ์และอินเทอร์เน็ตประสานสรรพสิง",
    ): "ปัญญาประดิษฐ์และอินเทอร์เน็ตประสานสรรพสิ่ง",
    (
        "DSBA",
        "06026260",
        "name_en",
        "OVERSEA COOPERATIVE EDUCATION IN DATA SCIENCE AND BUSIESS ANALYTICS",
    ): "OVERSEA COOPERATIVE EDUCATION IN DATA SCIENCE AND BUSINESS ANALYTICS",
    (
        "GENED",
        "90642056",
        "name_en",
        "ST EPLDEMICS IN THE 21 CENTURV",
    ): "EPIDEMICS IN THE 21ST CENTURY",
    ("GENED", "90642045", "name_en", "BE MV BEV."): "BEVERAGE",
    (
        "IT",
        "06016418",
        "name_th",
        "การพัฒนาเว็บฝังเซิร์ฟเวอร์",
    ): "การพัฒนาเว็บฝั่งเซิร์ฟเวอร์",
    (
        "IT",
        "06016442",
        "name_th",
        "การออกแบบฮาร์ดแวร์สำหรับอินเทอร์เน็ตแห่งสรรพสิง",
    ): "การออกแบบฮาร์ดแวร์สำหรับอินเทอร์เน็ตแห่งสรรพสิ่ง",
    (
        "IT",
        "06016443",
        "name_th",
        "การวิเคราะห์ข้อมูลและแอปพลิเคชันสำหรับอินเทอร์เน็ตแห่งสรรพสิง",
    ): "การวิเคราะห์ข้อมูลและแอปพลิเคชันสำหรับอินเทอร์เน็ตแห่งสรรพสิ่ง",
    (
        "IT",
        "90643021",
        "name_th",
        "ผู้ ประกอบการสมัยใหม่",
    ): "ผู้ประกอบการสมัยใหม่",
    (
        "GENED",
        "90642134",
        "name_en",
        "KING MONGKUTS REIGN STUDV",
    ): "KING MONGKUTS REIGN STUDY",
}
LITERAL_PRESERVE_VALUES = {
    ("GENED", "90642122", "name_th", "การใช้แอปพลิเคชัน ไมโครคอมพิวเตอร์"):
        "การใช้แอปพลิเคชัน ไมโครคอมพิวเตอร์",
    ("AIT", "06046404", "name_en", "FUNDAMENTAL OF EMBEDDED SYSTEM"):
        "FUNDAMENTAL OF EMBEDDED SYSTEM",
    ("AIT", "06046425", "name_en", "GENERATIVE MODEL"): "GENERATIVE MODEL",
    ("GENED", "90642126", "name_en", "SURVIVORS"): "SURVIVORS",
    ("GENED", "90642154", "name_en", "FALL ABLE"): "FALL ABLE",
}
NON_CORRECTABLE_NAME_VALUES = frozenset(("ไม่ระบุ", "N/A"))


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


def _guard_correction_value(
    record: dict[str, Any],
    field: str,
    before: Any,
    candidate: str,
    *,
    program: Any = None,
) -> str:
    if field in TEXT_FIELDS and before in NON_CORRECTABLE_NAME_VALUES:
        return before
    record_program = record.get("program")
    if record_program is None:
        record_program = program
    identity = (record_program, _record_course_code(record), field, before)
    canonical_after = CANONICAL_NAME_CORRECTIONS.get(identity)
    if canonical_after is not None:
        return canonical_after
    preserved_value = LITERAL_PRESERVE_VALUES.get(identity)
    if preserved_value is not None:
        return preserved_value
    return candidate


def _reject_empty_text_replacement(
    original: Any,
    corrected: str,
    context: str,
) -> None:
    if isinstance(original, str) and original.strip() and not corrected.strip():
        raise ValueError(f"{context} cannot replace non-empty text with empty text")


def _terminal_numeric_suffix(text: Any) -> str | None:
    """Return only a standalone final digit suffix from a name-like value."""
    if not isinstance(text, str) or not text:
        return None
    if len(text) == 1 and text in "123456789":
        return text
    if text[-1] in "123456789" and len(text) > 1 and text[-2].isspace():
        return text[-1]
    return None


def apply_corrections(
    document: Any,
    correction_records: list[dict[str, Any]],
    *,
    approved_document: Any | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Apply reviewed text corrections without rebuilding curriculum records."""
    corrected_document = copy.deepcopy(document)
    original_records = _records_from_document(document)
    corrected_records = _records_from_document(corrected_document)
    document_program = (
        document.get("program") if isinstance(document, dict) else None
    )
    if len(original_records) != len(corrected_records):
        raise ValueError("Correction reconstruction changed curriculum record count")
    approved_records = (
        _records_from_document(approved_document)
        if approved_document is not None
        else None
    )

    applied: list[dict[str, Any]] = []
    for correction_index, correction in enumerate(correction_records):
        if not isinstance(correction, dict):
            raise ValueError(
                f"Correction {correction_index} must be a JSON object"
            )
        required_fields = {"course_code", "field", "before", "after"}
        missing_fields = sorted(required_fields.difference(correction))
        if missing_fields:
            raise ValueError(
                f"Correction {correction_index} is missing fields: {missing_fields}"
            )

        course_code = correction["course_code"]
        field = correction["field"]
        before = correction["before"]
        after = correction["after"]
        if field not in CORRECTION_FIELDS:
            raise ValueError(
                f"Correction {correction_index} has unsupported field: {field!r}"
            )
        if not isinstance(course_code, str) or not course_code:
            raise ValueError(
                f"Correction {correction_index} course_code must be a non-empty string"
            )
        if not isinstance(before, str) or not isinstance(after, str):
            raise ValueError(
                f"Correction {correction_index} before/after must be strings"
            )

        matching_indexes = [
            index
            for index, record in enumerate(corrected_records)
            if _record_course_code(record) == course_code
            and record.get(field) == before
        ]
        if len(matching_indexes) > 1:
            raise ValueError(
                f"Correction {correction_index} is ambiguous for "
                f"{course_code}/{field}: matched {len(matching_indexes)} records"
            )
        if matching_indexes:
            target_record = corrected_records[matching_indexes[0]]
            candidate_after = _guard_correction_value(
                target_record,
                field,
                before,
                after,
                program=document_program,
            )
            _reject_empty_text_replacement(
                target_record.get(field),
                candidate_after,
                f"Correction {correction_index}",
            )
            target_record[field] = candidate_after
            if before != candidate_after:
                applied.append(
                    {
                        "course_code": _record_course_code(
                            original_records[matching_indexes[0]]
                        ),
                        "field": field,
                        "before": before,
                        "after": candidate_after,
                    }
                )
            continue

        already_applied = any(
            _record_course_code(record) == course_code
            and record.get(field) == after
            for record in corrected_records
        )
        if already_applied:
            continue

        if approved_records is not None:
            approved_value = any(
                _record_course_code(record) == course_code
                and record.get(field) == after
                for record in approved_records
            )
            same_code_indexes = [
                index
                for index, record in enumerate(corrected_records)
                if _record_course_code(record) == course_code
            ]
            if approved_value and len(same_code_indexes) == 1:
                target_index = same_code_indexes[0]
                candidate_after = _guard_correction_value(
                    corrected_records[target_index],
                    field,
                    original_records[target_index].get(field),
                    after,
                    program=document_program,
                )
                _reject_empty_text_replacement(
                    original_records[target_index].get(field),
                    candidate_after,
                    f"Correction {correction_index}",
                )
                corrected_records[target_index][field] = candidate_after
                if before != candidate_after:
                    applied.append(
                        {
                            "course_code": _record_course_code(
                                original_records[target_index]
                            ),
                            "field": field,
                            "before": original_records[target_index].get(field),
                            "after": candidate_after,
                        }
                    )
                continue

        raise ValueError(
            f"Correction {correction_index} did not match "
            f"{course_code}/{field} before value"
        )

    return corrected_document, applied


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
        _reject_empty_text_replacement(
            expected_unit["text"],
            after["text"],
            f"Gemini batch {batch_number} unit_index {unit_index}",
        )
        if (
            expected_unit["field"] in TEXT_FIELDS
            and _terminal_numeric_suffix(expected_unit["text"])
            != _terminal_numeric_suffix(after["text"])
        ):
            after = dict(after)
            after["text"] = expected_unit["text"]
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
    directory = LLM_OUTPUT_DIR if output_dir is None else Path(output_dir)
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
    total_batches = (len(units) + BATCH_SIZE - 1) // BATCH_SIZE
    for start_index in range(0, len(units), BATCH_SIZE):
        before_batch = units[start_index : start_index + BATCH_SIZE]
        batch_number = (start_index // BATCH_SIZE) + 1
        print(f"[{batch_number}/{total_batches}] correcting {len(before_batch)} unique units...")
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
        print(f"[{batch_number}/{total_batches}] done")
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
            candidate_after = _guard_correction_value(
                original_record,
                field,
                before,
                after,
                program=document.get("program") if isinstance(document, dict) else None,
            )
            if before == candidate_after:
                continue
            corrected_record[field] = candidate_after
            corrections.append(
                {
                    "course_code": _record_course_code(original_record),
                    "field": field,
                    "before": before,
                    "after": candidate_after,
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


def discover_consolidated_inputs(
    consolidated_dir: str | Path = CONSOLIDATED_DIR,
) -> list[Path]:
    """Discover only full consolidated curriculum files in stable order."""
    root = Path(consolidated_dir)
    paths = sorted(
        (
            path
            for path in root.glob("**/full/merged_*_full.json")
            if path.is_file()
        ),
        key=lambda path: path.as_posix().casefold(),
    )
    if not paths:
        raise FileNotFoundError(
            f"No full consolidated curriculum files found under {root}"
        )

    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Consolidated input is not valid JSON: {path}") from error
        _records_from_document(document)
    return paths


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage="python llm_spell_corrector.py [<path_to_json> ...] [--output-dir PATH]"
    )
    parser.add_argument(
        "path_to_json",
        nargs="*",
        help=(
            "Explicit curriculum JSON path(s); when omitted, discover full "
            "consolidated files under outputs/consolidated/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LLM_OUTPUT_DIR,
        help="Directory for *_corrected.json and *_corrections.json (default: outputs/llm)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_arguments(argv)
        paths = args.path_to_json or discover_consolidated_inputs()
        correct_json_files(paths, output_dir=args.output_dir)
    except Exception as error:
        print(f" Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
