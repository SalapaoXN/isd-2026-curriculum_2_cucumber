"""Reviewed runtime artifacts and plan-scoped course preflight."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.structured.loader import (
    _course_codes,
    _course_identity_discriminator,
    _course_variant_discriminator,
    _member_value as _loader_member_value,
    _normalized_course_code,
    _normalized_course_name,
    _provenance_entries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_DIR = PROJECT_ROOT / "data" / "output" / "final"
CANONICAL_DIR = PROJECT_ROOT / "data" / "output" / "canonical"
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"
LEGACY_CORRECTIONS_PATH = (
    PROJECT_ROOT / "data" / "corrections" / "legacy_2560_source_verified_corrections.json"
)
PREREQUISITE_CORRECTIONS_PATH = (
    PROJECT_ROOT / "data" / "corrections" / "source_verified_prerequisite_corrections.json"
)

STRUCTURED_FIELDS = (
    "code",
    "name_th",
    "name_en",
    "credits",
    "year",
    "semester",
    "category",
    "type",
    "prerequisite",
    "flexible_year_semester",
    "note",
)

LEGACY_2560_CATALOGS = frozenset({"bit-2560", "dsba-2560", "it-2560"})


@dataclass(frozen=True)
class CanonicalizationResult:
    paths: tuple[Path, ...]
    plan_gt_fields_applied: int
    shared_general_education_fields_applied: int
    legacy_corrections_applied: int
    prerequisite_corrections_applied: int = 0


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _document_identity(document: Mapping[str, Any], path: Path) -> tuple[str, str, str | None]:
    catalog = document.get("catalog")
    catalog_key = catalog.get("catalog_key") if isinstance(catalog, Mapping) else None
    program = document.get("program")
    if not isinstance(catalog_key, str) or not catalog_key.strip():
        raise ValueError(f"curriculum artifact has no catalog_key: {path}")
    if not isinstance(program, str) or not program.strip():
        raise ValueError(f"curriculum artifact has no program: {path}")
    plan_value = document.get("plan")
    if isinstance(plan_value, Mapping):
        plan_value = plan_value.get("plan_code", plan_value.get("name"))
    plan = str(plan_value).strip().casefold() if plan_value not in (None, "") else None
    return catalog_key.strip(), program.strip().upper(), plan


def _course_member_value(value: Any, index: int, count: int, *, split_lines: bool = False) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == count:
        return value[index]
    return _loader_member_value(value, index, count, split_lines)


def _normalized_placement(fields: Mapping[str, Any]) -> tuple[str | None, str | None]:
    result = []
    for name in ("year", "semester"):
        value = fields.get(name)
        result.append(None if value is None else str(value).strip())
    return result[0], result[1]


def _flatten_courses(courses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row_index, course in enumerate(courses):
        codes = _course_codes(course)
        for member_index, code in enumerate(codes):
            fields: dict[str, Any] = {"code": code}
            for field in STRUCTURED_FIELDS:
                if field == "code" or field not in course:
                    continue
                fields[field] = _course_member_value(
                    course[field],
                    member_index,
                    len(codes),
                    split_lines=field in {"name_th", "name_en"},
                )
            flattened.append(
                {
                    "row_index": row_index,
                    "member_index": member_index,
                    "code": code,
                    "fields": fields,
                    "placement": _normalized_placement(fields),
                }
            )
    return flattened


def _same_name(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for field in ("name_en", "name_th"):
        left_value = left.get(field)
        right_value = right.get(field)
        if left_value is None or right_value is None:
            continue
        if _normalized_course_name(
            left_value, casefold=field == "name_en"
        ) == _normalized_course_name(right_value, casefold=field == "name_en"):
            return True
    return False


def _flatten_ground_truth(
    courses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    flattened = _flatten_courses(courses)
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in flattened:
        by_code[_normalized_course_code(entry["code"])].append(entry)
    return flattened, by_code


def _match_ground_truth(
    raw_courses: Sequence[Mapping[str, Any]],
    ground_truth_courses: Sequence[Mapping[str, Any]],
    *,
    identity: tuple[str, str, str | None],
    require_all_ground_truth_codes: bool,
) -> dict[tuple[int, int], dict[str, Any]]:
    raw_entries = _flatten_courses(raw_courses)
    gt_entries, gt_by_code = _flatten_ground_truth(ground_truth_courses)
    raw_by_code_placement: dict[
        tuple[str, tuple[str | None, str | None]], list[dict[str, Any]]
    ] = defaultdict(list)
    raw_codes = set()
    for entry in raw_entries:
        code_key = _normalized_course_code(entry["code"])
        raw_codes.add(code_key)
        raw_by_code_placement[(code_key, entry["placement"])].append(entry)

    if require_all_ground_truth_codes:
        gt_codes = {_normalized_course_code(entry["code"]) for entry in gt_entries}
        missing = sorted(gt_codes - raw_codes)
        if missing:
            raise ValueError(
                "plan Ground Truth course codes have no matching reviewed-final record "
                f"for {identity[1]}/{identity[2] or '-'} ({identity[0]}): "
                + ", ".join(missing)
            )

    assignments: dict[tuple[int, int], dict[str, Any]] = {}
    for (code_key, placement), raw_group in raw_by_code_placement.items():
        candidates = gt_by_code.get(code_key, [])
        if not candidates:
            continue

        same_placement = [
            candidate
            for candidate in candidates
            if candidate["placement"] == placement
            and candidate["placement"] != (None, None)
        ]
        if same_placement:
            scoped = same_placement
        elif all(candidate["placement"] == (None, None) for candidate in candidates):
            scoped = candidates
        else:
            scoped = candidates

        if len(scoped) == 1:
            for raw_entry in raw_group:
                assignments[(raw_entry["row_index"], raw_entry["member_index"])] = scoped[0]
            continue

        available = list(scoped)
        unresolved = []
        for raw_entry in raw_group:
            exact = [
                candidate
                for candidate in available
                if _same_name(raw_entry["fields"], candidate["fields"])
            ]
            if len(exact) == 1:
                chosen = exact[0]
                assignments[(raw_entry["row_index"], raw_entry["member_index"])] = chosen
                available.remove(chosen)
            else:
                unresolved.append(raw_entry)

        if not unresolved:
            continue
        if len(available) == 1:
            chosen = available[0]
            for raw_entry in unresolved:
                assignments[(raw_entry["row_index"], raw_entry["member_index"])] = chosen
            continue
        if len(unresolved) == len(available):
            # Repeated placeholder codes can be distinct elective slots at the
            # same placement. If their labels are damaged, source row order is
            # the deterministic identity; no fuzzy text match is used.
            for raw_entry, chosen in zip(unresolved, available, strict=True):
                assignments[(raw_entry["row_index"], raw_entry["member_index"])] = chosen
            continue

        labels = [
            f"{entry['fields'].get('name_en')!r}/{entry['fields'].get('name_th')!r}"
            for entry in unresolved
        ]
        raise ValueError(
            "ambiguous Ground Truth course identity for "
            f"{identity[1]}/{identity[2] or '-'} {identity[0]}:{unresolved[0]['code']} "
            f"at placement {placement}: {labels}"
        )
    return assignments


def _compact_member_values(values: list[Any], *, field: str) -> Any:
    if not values:
        return None
    if all(value == values[0] for value in values[1:]):
        return values[0]
    if field == "code":
        return values
    if field in {"name_th", "name_en"}:
        if any(not isinstance(value, str) for value in values):
            raise ValueError(
                f"multi-course row has an unrepresentable {field} member"
            )
        # The loader's established multi-member representation is one name
        # per line, rather than a JSON list.
        return "\n".join(values)
    if field == "credits":
        return " or ".join(str(value) for value in values)
    raise ValueError(
        f"multi-course row has differing {field} values; split it into separate rows"
    )


def _apply_member_overrides(
    course: dict[str, Any],
    assignments: Mapping[int, Mapping[str, Any]],
    *,
    fields: Sequence[str],
) -> int:
    codes = _course_codes(course)
    applied = 0
    for field in fields:
        if field not in STRUCTURED_FIELDS:
            raise ValueError(f"unsupported structured canonical field: {field}")
        selected_values = []
        field_was_present = False
        for member_index, code in enumerate(codes):
            entry = assignments.get(member_index)
            if entry is not None and field in entry["fields"]:
                selected_values.append(entry["fields"][field])
                field_was_present = True
            elif field == "code":
                selected_values.append(code)
            else:
                selected_values.append(
                    _course_member_value(
                        course.get(field),
                        member_index,
                        len(codes),
                        split_lines=field in {"name_th", "name_en"},
                    )
                )
        if field_was_present or field == "code":
            course[field] = _compact_member_values(selected_values, field=field)
            if field_was_present:
                applied += sum(
                    1
                    for member_index in range(len(codes))
                    if member_index in assignments
                    and field in assignments[member_index]["fields"]
                )
    return applied


def _load_legacy_corrections(path: Path) -> list[dict[str, Any]]:
    document = _read_json_object(path, label="legacy correction artifact")
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported legacy correction schema in {path}")
    corrections = document.get("corrections")
    if not isinstance(corrections, list):
        raise ValueError(f"legacy correction artifact must contain corrections: {path}")
    allowed_fields = set(STRUCTURED_FIELDS) - {"code"}
    seen_identities: set[tuple[str, str, str, str]] = set()
    for index, correction in enumerate(corrections):
        if not isinstance(correction, dict):
            raise ValueError(f"legacy correction {index} must be an object")
        catalog_key = correction.get("catalog_key")
        if catalog_key not in LEGACY_2560_CATALOGS:
            raise ValueError(f"legacy correction {index} is not scoped to a 2560 catalog")
        if not isinstance(correction.get("program"), str) or not correction["program"].strip():
            raise ValueError(f"legacy correction {index} has no program")
        if not isinstance(correction.get("plan"), str) or not correction["plan"].strip():
            raise ValueError(f"legacy correction {index} has no plan")
        if not isinstance(correction.get("course_code"), str) or not correction["course_code"].strip():
            raise ValueError(f"legacy correction {index} has no course_code")
        correction_identity = (
            str(catalog_key),
            str(correction["program"]).strip().upper(),
            str(correction["plan"]).strip().casefold(),
            _normalized_course_code(correction["course_code"]),
        )
        if correction_identity in seen_identities:
            raise ValueError(
                f"duplicate legacy correction identity in {path}: "
                f"{correction_identity}"
            )
        seen_identities.add(correction_identity)
        fields = correction.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise ValueError(f"legacy correction {index} has no fields")
        if set(fields) - allowed_fields:
            raise ValueError(f"legacy correction {index} contains unsupported fields")
        sources = correction.get("source_provenance")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"legacy correction {index} requires source provenance")
        catalog_year = str(catalog_key).rsplit("-", 1)[-1]
        source_directory = (
            PROJECT_ROOT / "data" / "input" / f"{correction['program'].casefold()}{catalog_year}"
        )
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError(
                    f"legacy correction {index} source provenance must contain objects"
                )
            source_filename = source.get("source_filename")
            if (
                not isinstance(source_filename, str)
                or not source_filename
                or Path(source_filename).name != source_filename
                or source.get("document_category") != "plan"
                or isinstance(source.get("source_page"), bool)
                or not isinstance(source.get("source_page"), int)
            ):
                raise ValueError(
                    f"legacy correction {index} requires a plan source filename"
                )
            if not (source_directory / source_filename).is_file():
                raise FileNotFoundError(
                    f"legacy correction source document not found: "
                    f"{source_directory / source_filename}"
                )
    return corrections


def _apply_legacy_corrections(
    document: dict[str, Any],
    identity: tuple[str, str, str | None],
    corrections: Sequence[Mapping[str, Any]],
) -> tuple[int, set[int]]:
    if identity[0] not in LEGACY_2560_CATALOGS:
        return 0, set()
    used: set[int] = set()
    applied = 0
    courses = document.get("courses")
    if not isinstance(courses, list):
        raise ValueError("curriculum artifact must contain a courses list")
    for correction_index, correction in enumerate(corrections):
        correction_identity = (
            correction["catalog_key"],
            str(correction["program"]).strip().upper(),
            str(correction["plan"]).strip().casefold(),
        )
        if correction_identity != identity:
            continue
        if correction_index in used:
            raise ValueError(f"legacy correction applied more than once: {correction_index}")
        code_key = _normalized_course_code(correction["course_code"])
        matching = []
        for row_index, course in enumerate(courses):
            if not isinstance(course, dict):
                continue
            codes = _course_codes(course)
            for member_index, code in enumerate(codes):
                if _normalized_course_code(code) == code_key:
                    matching.append((row_index, member_index, len(codes)))
        if len(matching) != 1:
            raise ValueError(
                "legacy correction must match exactly one course identity: "
                f"{identity[1]}/{identity[2]} {identity[0]}:{correction['course_code']} "
                f"(matches={len(matching)})"
            )
        row_index, member_index, member_count = matching[0]
        assignments = {member_index: {"fields": correction["fields"]}}
        applied += _apply_member_overrides(
            courses[row_index],
            assignments,
            fields=tuple(correction["fields"]),
        )
        provenance_value = courses[row_index].get("source_provenance")
        if isinstance(provenance_value, Mapping):
            provenance = [dict(provenance_value)]
        elif isinstance(provenance_value, list):
            provenance = [dict(entry) for entry in provenance_value if isinstance(entry, Mapping)]
        else:
            provenance = []
        for source in correction["source_provenance"]:
            provenance_entry = {
                key: source[key]
                for key in (
                    "program",
                    "source_filename",
                    "source_page",
                    "document_page",
                    "document_category",
                )
                if key in source
            }
            provenance_entry["excerpt"] = source.get("evidence")
            if provenance_entry not in provenance:
                provenance.append(provenance_entry)
        courses[row_index]["source_provenance"] = provenance
        used.add(correction_index)
    return applied, used


def _load_prerequisite_corrections(path: Path) -> list[dict[str, Any]]:
    document = _read_json_object(path, label="source-verified prerequisite correction artifact")
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported prerequisite correction schema in {path}")
    corrections = document.get("corrections")
    if not isinstance(corrections, list):
        raise ValueError(f"prerequisite correction artifact must contain corrections: {path}")

    seen: set[tuple[str, str, str, str]] = set()
    for index, correction in enumerate(corrections):
        if not isinstance(correction, dict):
            raise ValueError(f"prerequisite correction {index} must be an object")
        required_text = ("catalog_key", "program", "plan", "course_code", "observed_value", "corrected_value")
        if any(not isinstance(correction.get(key), str) or not correction[key].strip()
               for key in required_text):
            raise ValueError(f"prerequisite correction {index} has missing identity or values")
        identity = (
            correction["catalog_key"].strip(),
            correction["program"].strip().upper(),
            correction["plan"].strip().casefold(),
            _normalized_course_code(correction["course_code"]),
        )
        if identity in seen:
            raise ValueError(f"duplicate prerequisite correction identity in {path}: {identity}")
        seen.add(identity)

        sources = correction.get("source_provenance")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"prerequisite correction {index} requires source provenance")
        catalog_year = identity[0].rsplit("-", 1)[-1]
        source_directory = PROJECT_ROOT / "data" / "input" / f"{identity[1].casefold()}{catalog_year}"
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError(f"prerequisite correction {index} source must be an object")
            filename = source.get("source_filename")
            page = source.get("source_page")
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
                or source.get("document_category") not in {"plan", "description"}
                or not isinstance(source.get("evidence"), str)
                or not source["evidence"].strip()
            ):
                raise ValueError(f"prerequisite correction {index} has invalid source provenance")
            if not (source_directory / filename).is_file():
                raise FileNotFoundError(
                    f"prerequisite correction source document not found: {source_directory / filename}"
                )
    return corrections


def _apply_prerequisite_corrections(
    document: dict[str, Any],
    identity: tuple[str, str, str | None],
    corrections: Sequence[Mapping[str, Any]],
) -> tuple[int, set[int]]:
    courses = document.get("courses")
    if not isinstance(courses, list):
        raise ValueError("curriculum artifact must contain a courses list")

    used: set[int] = set()
    applied = 0
    for correction_index, correction in enumerate(corrections):
        correction_identity = (
            correction["catalog_key"].strip(),
            correction["program"].strip().upper(),
            correction["plan"].strip().casefold(),
        )
        if correction_identity != identity:
            continue
        code_key = _normalized_course_code(correction["course_code"])
        matches: list[tuple[int, int, int]] = []
        for row_index, course in enumerate(courses):
            if not isinstance(course, dict):
                continue
            codes = _course_codes(course)
            for member_index, code in enumerate(codes):
                if _normalized_course_code(code) == code_key:
                    matches.append((row_index, member_index, len(codes)))
        if len(matches) != 1:
            raise ValueError(
                "prerequisite correction must match exactly one course identity: "
                f"{identity[1]}/{identity[2]} {identity[0]}:{correction['course_code']} "
                f"(matches={len(matches)})"
            )

        row_index, member_index, member_count = matches[0]
        course = courses[row_index]
        observed = _loader_member_value(
            course.get("prerequisite"), member_index, member_count
        )
        if observed not in {correction["observed_value"], correction["corrected_value"]}:
            raise ValueError(
                "source-verified prerequisite correction no longer matches extracted text: "
                f"{identity[1]}/{identity[2]} {correction['course_code']} "
                f"(expected={correction['observed_value']!r} or "
                f"{correction['corrected_value']!r}, found={observed!r})"
            )

        if observed != correction["corrected_value"]:
            assignments = {
                member_index: {"fields": {"prerequisite": correction["corrected_value"]}}
            }
            applied += _apply_member_overrides(
                course, assignments, fields=("prerequisite",)
            )
        prior_observation = course.get("prerequisite_observed")
        if prior_observation not in (None, observed):
            raise ValueError(
                f"conflicting prerequisite_observed value for {correction['course_code']}"
            )
        course["prerequisite_observed"] = observed

        provenance_value = course.get("source_provenance")
        if isinstance(provenance_value, Mapping):
            provenance = [dict(provenance_value)]
        elif isinstance(provenance_value, list):
            provenance = [dict(entry) for entry in provenance_value if isinstance(entry, Mapping)]
        else:
            provenance = []
        for source in correction["source_provenance"]:
            entry = {
                key: source[key]
                for key in ("program", "source_filename", "source_page", "document_page", "document_category")
                if key in source
            }
            entry["excerpt"] = source["evidence"]
            if entry not in provenance:
                provenance.append(entry)
        course["source_provenance"] = provenance
        used.add(correction_index)
    return applied, used


def _canonicalize_document(
    source_path: Path,
    document: dict[str, Any],
    *,
    corrections: Sequence[Mapping[str, Any]],
    prerequisite_corrections: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int, set[int], int, set[int]]:
    identity = _document_identity(document, source_path)
    if not isinstance(document.get("courses"), list):
        raise ValueError(f"curriculum artifact has no courses list: {source_path}")
    legacy_count, used_corrections = _apply_legacy_corrections(document, identity, corrections)
    prerequisite_count, used_prerequisite_corrections = _apply_prerequisite_corrections(
        document, identity, prerequisite_corrections
    )
    return (
        document,
        legacy_count,
        used_corrections,
        prerequisite_count,
        used_prerequisite_corrections,
    )


def _write_json_atomically(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def canonicalize_runtime_artifacts(
    final_dir: str | Path = FINAL_DIR,
    output_dir: str | Path = CANONICAL_DIR,
    ground_truth_dir: str | Path = GROUND_TRUTH_DIR,
    corrections_path: str | Path = LEGACY_CORRECTIONS_PATH,
    prerequisite_corrections_path: str | Path = PREREQUISITE_CORRECTIONS_PATH,
) -> CanonicalizationResult:
    """Copy reviewed finals and apply source-verified curriculum corrections.

    ``ground_truth_dir`` remains accepted for compatibility with existing callers;
    it is intentionally never read when creating runtime artifacts.
    """
    source_root = Path(final_dir)
    destination_root = Path(output_dir)
    del ground_truth_dir
    corrections_file = Path(corrections_path)
    source_paths = sorted(source_root.glob("*_final.json"))
    if not source_paths:
        raise FileNotFoundError(f"no reviewed curriculum final artifacts in {source_root}")

    corrections = _load_legacy_corrections(corrections_file)
    prerequisite_corrections = _load_prerequisite_corrections(
        Path(prerequisite_corrections_path)
    )
    canonical_documents = []
    legacy_count = 0
    prerequisite_count = 0
    used_corrections: set[int] = set()
    used_prerequisite_corrections: set[int] = set()
    for source_path in source_paths:
        source_document = _read_json_object(source_path, label="reviewed final")
        # Keep the in-memory source untouched; canonicalization only writes to
        # the separate runtime artifact directory.
        canonical_document = json.loads(json.dumps(source_document, ensure_ascii=False))
        result = _canonicalize_document(
            source_path,
            canonical_document,
            corrections=corrections,
            prerequisite_corrections=prerequisite_corrections,
        )
        (
            canonical_document,
            legacy_applied,
            used,
            prerequisite_applied,
            used_prerequisites,
        ) = result
        legacy_count += legacy_applied
        prerequisite_count += prerequisite_applied
        used_corrections.update(used)
        used_prerequisite_corrections.update(used_prerequisites)
        canonical_documents.append((source_path.name, canonical_document))

    unused = sorted(set(range(len(corrections))) - used_corrections)
    if unused:
        labels = [
            f"{corrections[index]['catalog_key']}:{corrections[index]['course_code']}"
            for index in unused
        ]
        raise ValueError("unused source-verified legacy corrections: " + ", ".join(labels))

    unused_prerequisite_corrections = sorted(
        set(range(len(prerequisite_corrections))) - used_prerequisite_corrections
    )
    if unused_prerequisite_corrections:
        labels = [
            f"{prerequisite_corrections[index]['catalog_key']}:"
            f"{prerequisite_corrections[index]['course_code']}"
            for index in unused_prerequisite_corrections
        ]
        raise ValueError(
            "unused source-verified prerequisite corrections: " + ", ".join(labels)
        )

    expected_names = {name for name, _ in canonical_documents}
    if destination_root.exists():
        existing_final_names = {path.name for path in destination_root.glob("*_final.json")}
        unexpected = sorted(existing_final_names - expected_names)
        if unexpected:
            raise ValueError(
                "canonical runtime directory contains stale/unrecognized final artifacts: "
                + ", ".join(unexpected)
            )
    for name, document in canonical_documents:
        _write_json_atomically(destination_root / name, document)

    return CanonicalizationResult(
        paths=tuple(destination_root / name for name, _ in canonical_documents),
        plan_gt_fields_applied=0,
        shared_general_education_fields_applied=0,
        legacy_corrections_applied=legacy_count,
        prerequisite_corrections_applied=prerequisite_count,
    )


def _is_plan_backed(course: Mapping[str, Any]) -> bool:
    provenance_value = course.get("source_provenance", course.get("provenance"))
    entries = _provenance_entries(provenance_value)
    if not entries:
        return True
    return any(str(entry.get("document_category", "")).strip().casefold() == "plan" for entry in entries)


def audit_shared_course_conflicts(
    paths: Sequence[str | Path],
) -> list[str]:
    """Return unresolved name/credit conflicts that the loader cannot represent.

    Different observations in separate plans are valid runtime variants and are
    retained as separate course identities by the loader. Differing facts within
    one plan receive the same deterministic variant identities during loading.
    """
    observations: dict[
        tuple[str, str, str | None, str, str],
        dict[str, list[tuple[Any, Any, str]]],
    ] = defaultdict(lambda: defaultdict(list))
    for path_value in paths:
        path = Path(path_value)
        document = _read_json_object(path, label="canonical runtime curriculum")
        catalog_key, program, plan = _document_identity(document, path)
        courses = document.get("courses")
        if not isinstance(courses, list):
            raise ValueError(f"canonical runtime artifact has no courses list: {path}")
        for course in courses:
            if not isinstance(course, dict):
                raise ValueError(f"course row must be an object in {path}")
            codes = _course_codes(course)
            plan_backed = _is_plan_backed(course)
            if not plan_backed:
                continue
            source = f"{path.name} [{program}/{plan or '-'}]"
            for member_index, code in enumerate(codes):
                name_th = _course_member_value(
                    course.get("name_th"), member_index, len(codes), split_lines=True
                )
                name_en = _course_member_value(
                    course.get("name_en"), member_index, len(codes), split_lines=True
                )
                credits = _course_member_value(
                    course.get("credits"), member_index, len(codes)
                )
                discriminator = _course_identity_discriminator(code, name_en, name_th)
                variant_discriminator = _course_variant_discriminator(
                    discriminator,
                    {
                        "name_th": name_th,
                        "name_en": name_en,
                        "credits": credits,
                    },
                )
                identity = (
                    catalog_key,
                    program,
                    plan or "",
                    _normalized_course_code(code),
                    variant_discriminator,
                )
                if name_th is not None:
                    normalized = _normalized_course_name(name_th, casefold=False)
                    observations[identity]["name_th"].append((normalized, name_th, source))
                if name_en is not None:
                    normalized = _normalized_course_name(name_en, casefold=True)
                    observations[identity]["name_en"].append((normalized, name_en, source))
                if credits is not None:
                    observations[identity]["credits"].append((credits, credits, source))

    conflicts = []
    for identity, fields in sorted(observations.items()):
        catalog_key, program, plan, code, discriminator = identity
        for field, values in fields.items():
            normalized_values = {value[0] for value in values}
            if len(normalized_values) <= 1:
                continue
            rendered = "; ".join(
                f"{raw!r} @ {source}" for _, raw, source in values
            )
            label = f"{catalog_key}:{code}"
            if discriminator:
                label += f" ({discriminator})"
            conflicts.append(
                f"{label} {field}: {rendered} [program={program}, plan={plan or '-'}]"
            )
    return conflicts


def canonical_runtime_sources(
    canonical_dir: str | Path = CANONICAL_DIR,
) -> list[Path]:
    root = Path(canonical_dir)
    sources = sorted(root.glob("*_final.json"))
    if not sources:
        raise FileNotFoundError(
            f"no canonical runtime artifacts in {root}; run evaluation and "
            "python -m src.pipeline.tools.canonicalize_runtime first"
        )
    return sources


def preflight_runtime_artifacts(
    canonical_dir: str | Path = CANONICAL_DIR,
) -> list[str]:
    return audit_shared_course_conflicts(canonical_runtime_sources(canonical_dir))


def main_canonicalize() -> int:
    result = canonicalize_runtime_artifacts()
    print(
        f"Canonicalized {len(result.paths)} curriculum artifact(s) in {CANONICAL_DIR}; "
        "Ground Truth field applications=0, "
        f"legacy source corrections={result.legacy_corrections_applied}, "
        "source-verified prerequisite corrections="
        f"{result.prerequisite_corrections_applied}."
    )
    return 0


def main_preflight() -> int:
    sources = canonical_runtime_sources()
    conflicts = audit_shared_course_conflicts(sources)
    if conflicts:
        print(
            f"Preflight failed: {len(conflicts)} unresolved shared-course conflict(s):"
        )
        for conflict in conflicts:
            print(f"- {conflict}")
        return 1
    print(
        f"Preflight passed: {len(sources)} canonical curriculum artifact(s), "
        "0 loader-blocking course conflicts. Differing facts are retained as variants."
    )
    return 0
