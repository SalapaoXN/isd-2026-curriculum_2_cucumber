"""Post-evaluation canonical curriculum artifacts and shared-course preflight."""

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

# The program-specific plan GT owns all structured plan fields.
PLAN_GROUND_TRUTH = {
    ("ait-2566", "AIT", None): Path("AIT/AIT_academic_plan.json"),
    ("bit-2565", "BIT", "coop"): Path("BIT/BIT_academic_plan_coop.json"),
    ("bit-2565", "BIT", "no_coop"): Path("BIT/BIT_academic_plan_no_coop.json"),
    ("dsba-2565", "DSBA", "coop"): Path("DSBA/DSBA_academic_plan_coop.json"),
    ("dsba-2565", "DSBA", "no_coop"): Path("DSBA/DSBA_academic_plan_no_coop.json"),
    ("gened-2564", "GENED", "gened"): Path("general_education_ground_truth.json"),
    ("it-2565", "IT", "coop"): Path("IT/IT_academic_plan_coop.json"),
    ("it-2565", "IT", "no_coop"): Path("IT/IT_academic_plan_no_coop.json"),
}

# Shared general-education GT supplies course facts. Course type and placement
# stay plan-specific, so those fields remain from the program plan GT above.
SHARED_GENERAL_EDUCATION_FIELDS = frozenset(
    {
        "code",
        "name_th",
        "name_en",
        "credits",
        "category",
        "prerequisite",
        "flexible_year_semester",
        "note",
    }
)
LEGACY_2560_CATALOGS = frozenset({"bit-2560", "dsba-2560", "it-2560"})


@dataclass(frozen=True)
class CanonicalizationResult:
    paths: tuple[Path, ...]
    plan_gt_fields_applied: int
    shared_general_education_fields_applied: int
    legacy_corrections_applied: int


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


def _apply_ground_truth(
    document: dict[str, Any],
    ground_truth: Mapping[str, Any],
    *,
    identity: tuple[str, str, str | None],
    fields: Sequence[str],
    require_all_ground_truth_codes: bool,
) -> int:
    courses = document.get("courses")
    gt_courses = ground_truth.get("courses")
    if not isinstance(courses, list) or not isinstance(gt_courses, list):
        raise ValueError("curriculum and Ground Truth artifacts must contain courses lists")
    assignments = _match_ground_truth(
        courses,
        gt_courses,
        identity=identity,
        require_all_ground_truth_codes=require_all_ground_truth_codes,
    )
    applied = 0
    for row_index, course in enumerate(courses):
        if not isinstance(course, dict):
            raise ValueError(f"course row {row_index} must be an object")
        member_assignments = {
            member_index: assignments[(row_index, member_index)]
            for member_index in range(len(_course_codes(course)))
            if (row_index, member_index) in assignments
        }
        applied += _apply_member_overrides(course, member_assignments, fields=fields)
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


def _canonicalize_document(
    source_path: Path,
    document: dict[str, Any],
    *,
    ground_truth_dir: Path,
    corrections: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int, int, int, set[int]]:
    identity = _document_identity(document, source_path)
    if not isinstance(document.get("courses"), list):
        raise ValueError(f"curriculum artifact has no courses list: {source_path}")
    plan_count = 0
    shared_count = 0
    used_corrections: set[int] = set()

    plan_gt_relative = PLAN_GROUND_TRUTH.get(identity)
    if plan_gt_relative is not None:
        plan_gt_path = ground_truth_dir / plan_gt_relative
        plan_gt = _read_json_object(plan_gt_path, label="plan Ground Truth")
        plan_count += _apply_ground_truth(
            document,
            plan_gt,
            identity=identity,
            fields=STRUCTURED_FIELDS,
            require_all_ground_truth_codes=True,
        )

    if identity[0] not in LEGACY_2560_CATALOGS:
        shared_path = ground_truth_dir / "general_education_ground_truth.json"
        if shared_path.is_file():
            shared_gt = _read_json_object(shared_path, label="shared general-education Ground Truth")
            shared_count += _apply_ground_truth(
                document,
                shared_gt,
                identity=identity,
                fields=tuple(SHARED_GENERAL_EDUCATION_FIELDS),
                require_all_ground_truth_codes=False,
            )

    legacy_count, used_corrections = _apply_legacy_corrections(
        document, identity, corrections
    )
    return document, plan_count, shared_count, legacy_count, used_corrections


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
) -> CanonicalizationResult:
    """Copy reviewed finals, apply post-evaluation GT and 2560 source corrections."""
    source_root = Path(final_dir)
    destination_root = Path(output_dir)
    gt_root = Path(ground_truth_dir)
    corrections_file = Path(corrections_path)
    source_paths = sorted(source_root.glob("*_final.json"))
    if not source_paths:
        raise FileNotFoundError(f"no reviewed curriculum final artifacts in {source_root}")

    corrections = _load_legacy_corrections(corrections_file)
    canonical_documents = []
    plan_count = 0
    shared_count = 0
    legacy_count = 0
    used_corrections: set[int] = set()
    for source_path in source_paths:
        source_document = _read_json_object(source_path, label="reviewed final")
        # Keep the in-memory source untouched; canonicalization only writes to
        # the separate runtime artifact directory.
        canonical_document = json.loads(json.dumps(source_document, ensure_ascii=False))
        result = _canonicalize_document(
            source_path,
            canonical_document,
            ground_truth_dir=gt_root,
            corrections=corrections,
        )
        canonical_document, plan_applied, shared_applied, legacy_applied, used = result
        plan_count += plan_applied
        legacy_count += legacy_applied
        shared_count += shared_applied
        used_corrections.update(used)
        canonical_documents.append((source_path.name, canonical_document))

    unused = sorted(set(range(len(corrections))) - used_corrections)
    if unused:
        labels = [
            f"{corrections[index]['catalog_key']}:{corrections[index]['course_code']}"
            for index in unused
        ]
        raise ValueError("unused source-verified legacy corrections: " + ", ".join(labels))

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
        plan_gt_fields_applied=plan_count,
        shared_general_education_fields_applied=shared_count,
        legacy_corrections_applied=legacy_count,
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
    """Return all loader-blocking name/credit conflicts across plan-backed facts."""
    observations: dict[
        tuple[str, str, str, str],
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
                identity = (
                    catalog_key,
                    program,
                    _normalized_course_code(code),
                    discriminator,
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
        catalog_key, program, code, discriminator = identity
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
                f"{label} {field}: {rendered} [program={program}]"
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
        f"plan-GT field applications={result.plan_gt_fields_applied}, "
        f"shared-GE field applications={result.shared_general_education_fields_applied}, "
        f"legacy source corrections={result.legacy_corrections_applied}."
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
        "0 unresolved shared-course conflicts."
    )
    return 0
