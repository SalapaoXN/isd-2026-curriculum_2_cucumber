"""Evaluate Consolidated and LLM curriculum layers against shared GT data.

This module intentionally keeps the legacy ``evaluate.py`` CLI unchanged.  It
uses the same curriculum Ground Truth population for both input layers and
keeps record alignment separate from field scoring so placement errors remain
visible rather than becoming identity errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evaluate import calculate_cer, calculate_wer


FIELDS = ("code", "name_th", "name_en", "credits", "year", "semester", "prerequisite")
TEXT_FIELDS = {"name_th", "name_en"}
PLACEMENT_FIELDS = {"year", "semester"}

SCOPE_FILES = {
    ("AIT", None): "ait_no_plan",
    ("BIT", "coop"): "bit_coop",
    ("BIT", "no_coop"): "bit_no_coop",
    ("DSBA", "coop"): "dsba_coop",
    ("DSBA", "no_coop"): "dsba_no_coop",
    ("GENED", "gened"): "gened_gened",
    ("IT", "coop"): "it_coop",
    ("IT", "no_coop"): "it_no_coop",
}

GT_FILES = {
    ("AIT", None): Path("AIT/AIT_academic_plan.json"),
    ("BIT", "coop"): Path("BIT/BIT_academic_plan_coop.json"),
    ("BIT", "no_coop"): Path("BIT/BIT_academic_plan_no_coop.json"),
    ("DSBA", "coop"): Path("DSBA/DSBA_academic_plan_coop.json"),
    ("DSBA", "no_coop"): Path("DSBA/DSBA_academic_plan_no_coop.json"),
    ("GENED", "gened"): Path("general_education_ground_truth.json"),
    ("IT", "coop"): Path("IT/IT_academic_plan_coop.json"),
    ("IT", "no_coop"): Path("IT/IT_academic_plan_no_coop.json"),
}


@dataclass(frozen=True)
class Scope:
    program: str
    plan: str | None

    def label(self) -> str:
        return f"{self.program}/{self.plan or 'none'}"


@dataclass(frozen=True)
class ScopedRecord:
    scope: Scope
    item: dict[str, Any]
    code_expression: str
    occurrence: int

    @property
    def key(self) -> tuple[str, str | None, str, int]:
        return (
            self.scope.program,
            self.scope.plan,
            self.code_expression,
            self.occurrence,
        )

    @property
    def identity(self) -> str:
        return json.dumps(
            {
                "program": self.scope.program,
                "plan": self.scope.plan,
                "code": self.code_expression,
                "occurrence": self.occurrence,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def _raw(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", _raw(value)).strip().lower()


def normalize_code_expression(value: Any) -> str:
    """Normalize only whitespace/case; never expand wildcard expressions."""

    return _collapse_ws(value)


def _recordize(items: Iterable[dict[str, Any]], scope: Scope) -> list[ScopedRecord]:
    counts: Counter[str] = Counter()
    result = []
    for item in items:
        code_expression = normalize_code_expression(item.get("code"))
        counts[code_expression] += 1
        result.append(
            ScopedRecord(scope, item, code_expression, counts[code_expression])
        )
    return result


def _typed_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


_CREDIT_TOKEN = re.compile(
    r"^\s*([0-9]+)\s*\(\s*([0-9xX]+)\s*-\s*([0-9xX]+)\s*-\s*([0-9xX]+)\s*\)\s*$"
)
_CREDIT_SEPARATOR = re.compile(r"\s*(?:หรือ|\bor\b)\s*", re.IGNORECASE)


def _credit_token(value: str) -> tuple[str, str, str, str] | None:
    match = _CREDIT_TOKEN.fullmatch(value)
    if not match:
        return None
    return tuple(part.lower() for part in match.groups())  # type: ignore[return-value]


def _credit_shape(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    text = _raw(value)
    if not text:
        return ("empty",)

    parts = _CREDIT_SEPARATOR.split(text)
    tokens = [_credit_token(part) for part in parts]
    if all(token is not None for token in tokens):
        normalized = tuple(tokens)  # type: ignore[arg-type]
        if len(normalized) > 1:
            return ("alternatives", tuple(sorted(normalized)))
        return ("single", normalized[0])

    # A line break can represent an explicit alternative in extracted data,
    # but only accept it when every resulting piece is a complete credit token.
    line_parts = [part for part in re.split(r"\r?\n", text) if part.strip()]
    line_tokens = [_credit_token(part) for part in line_parts]
    if len(line_tokens) > 1 and all(token is not None for token in line_tokens):
        return ("alternatives", tuple(sorted(line_tokens)))  # type: ignore[arg-type]

    # Malformed and wildcard values remain distinguishable from missing data.
    return ("raw", _collapse_ws(text))


_PREREQUISITE_ALTERNATIVE_RE = re.compile(
    r"^\s*(\d{8})(?:(?:\s*(?:,|หรือ|\bor\b)\s*)(\d{8}))+\s*$",
    re.IGNORECASE,
)


def _normalize_prerequisite(value: Any) -> Any:
    normalized = _collapse_ws(value)
    if normalized in {"", "none", "ไม่มี"}:
        return "<no_prerequisite>"
    alternative_match = _PREREQUISITE_ALTERNATIVE_RE.fullmatch(normalized)
    if alternative_match:
        codes = tuple(re.findall(r"\d{8}", alternative_match.group(0)))
        return ("or", tuple(sorted(codes)))
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    return normalized


def _normalized_value(field: str, value: Any) -> Any:
    if field == "credits":
        return _credit_shape(value)
    if field in {"year", "semester"}:
        return _typed_value(value)
    if field == "prerequisite":
        return _normalize_prerequisite(value)
    if field == "code":
        return normalize_code_expression(value)
    return _collapse_ws(value)


def field_matches(field: str, gt_value: Any, output_value: Any) -> tuple[bool, bool]:
    """Return strict and conservative normalized equality for one field."""

    strict = _raw(gt_value) == _raw(output_value)
    if field in {"year", "semester"}:
        strict = _typed_value(gt_value) == _typed_value(output_value)
    return strict, _normalized_value(field, gt_value) == _normalized_value(
        field, output_value
    )


def _present(item: dict[str, Any], field: str) -> bool:
    # A missing GT key means that field has no denominator.  Explicit None is
    # still a meaningful prerequisite absence, but not a placement value.
    if field not in item:
        return False
    if field in PLACEMENT_FIELDS and item.get(field) is None:
        return False
    return True


def align_records(
    gt_items: list[dict[str, Any]],
    output_items: list[dict[str, Any]],
    program: str,
    plan: str | None,
) -> dict[str, Any]:
    scope = Scope(program, plan)
    gt_records = _recordize(gt_items, scope)
    output_records = _recordize(output_items, scope)
    output_by_key: dict[tuple[str, str | None, str, int], list[ScopedRecord]] = defaultdict(list)
    for record in output_records:
        output_by_key[record.key].append(record)

    pairs: list[tuple[ScopedRecord, ScopedRecord]] = []
    missing: list[ScopedRecord] = []
    ambiguous: list[ScopedRecord] = []
    used_output: set[int] = set()
    for gt_record in gt_records:
        candidates = output_by_key.get(gt_record.key, [])
        if len(candidates) == 1:
            pair = candidates[0]
            pairs.append((gt_record, pair))
            used_output.add(id(pair))
        elif len(candidates) > 1:
            ambiguous.append(gt_record)
        else:
            missing.append(gt_record)

    extras = [record for record in output_records if id(record) not in used_output]
    return {
        "gt_records": gt_records,
        "output_records": output_records,
        "pairs": pairs,
        "missing": missing,
        "extras": extras,
        "ambiguous": ambiguous,
    }


def _field_metric_template(field: str) -> dict[str, Any]:
    result = {
        "denominator": 0,
        "strict_correct": 0,
        "strict_incorrect": 0,
        "strict_accuracy": 0.0,
        "normalized_correct": 0,
        "normalized_incorrect": 0,
        "normalized_accuracy": 0.0,
    }
    if field in TEXT_FIELDS:
        result.update({"cer": 0.0, "wer": 0.0})
    return result


def _error_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def evaluate_records(
    gt_items: list[dict[str, Any]],
    output_items: list[dict[str, Any]],
    program: str,
    plan: str | None,
    layer: str,
) -> dict[str, Any]:
    alignment = align_records(gt_items, output_items, program, plan)
    metrics = {field: _field_metric_template(field) for field in FIELDS}
    errors: list[dict[str, Any]] = []

    for gt_record in alignment["gt_records"]:
        output_record = next(
            (pred for actual, pred in alignment["pairs"] if actual is gt_record),
            None,
        )
        if output_record is None:
            continue
        for field in FIELDS:
            if not _present(gt_record.item, field):
                continue
            metric = metrics[field]
            metric["denominator"] += 1
            gt_value = gt_record.item.get(field)
            output_value = output_record.item.get(field)
            strict, normalized = field_matches(field, gt_value, output_value)
            metric["strict_correct"] += int(strict)
            metric["strict_incorrect"] += int(not strict)
            metric["normalized_correct"] += int(normalized)
            metric["normalized_incorrect"] += int(not normalized)
            if field in TEXT_FIELDS:
                gt_text = _collapse_ws(gt_value)
                output_text = _collapse_ws(output_value)
                metric["cer"] += calculate_cer(gt_text, output_text)
                metric["wer"] += calculate_wer(gt_text, output_text)
            if not strict or not normalized:
                errors.append(
                    {
                        "layer": layer,
                        "program": program,
                        "plan": plan or "",
                        "occurrence": gt_record.occurrence,
                        "course_identity": gt_record.identity,
                        "field": field,
                        "gt_value": _error_value(gt_value),
                        "output_value": _error_value(output_value),
                        "strict_match": strict,
                        "normalized_match": normalized,
                        "error_category": "FIELD_MISMATCH",
                    }
                )

    for record in alignment["missing"]:
        for field in FIELDS:
            if not _present(record.item, field):
                continue
            metrics[field]["denominator"] += 1
            metrics[field]["strict_incorrect"] += 1
            metrics[field]["normalized_incorrect"] += 1
            if field in TEXT_FIELDS:
                gt_text = _collapse_ws(record.item.get(field))
                metrics[field]["cer"] += calculate_cer(gt_text, "")
                metrics[field]["wer"] += calculate_wer(gt_text, "")
            errors.append(
                {
                    "layer": layer,
                    "program": program,
                    "plan": plan or "",
                    "occurrence": record.occurrence,
                    "course_identity": record.identity,
                    "field": field,
                    "gt_value": _error_value(record.item.get(field)),
                    "output_value": "",
                    "strict_match": False,
                    "normalized_match": False,
                    "error_category": "MISSING_OUTPUT",
                }
            )

    for record in alignment["extras"]:
        errors.append(
            {
                "layer": layer,
                "program": program,
                "plan": plan or "",
                "occurrence": record.occurrence,
                "course_identity": record.identity,
                "field": "__record__",
                "gt_value": "",
                "output_value": _error_value(record.item.get("code")),
                "strict_match": False,
                "normalized_match": False,
                "error_category": "EXTRA_OUTPUT",
            }
        )

    for record in alignment["ambiguous"]:
        errors.append(
            {
                "layer": layer,
                "program": program,
                "plan": plan or "",
                "occurrence": record.occurrence,
                "course_identity": record.identity,
                "field": "__record__",
                "gt_value": _error_value(record.item.get("code")),
                "output_value": "",
                "strict_match": False,
                "normalized_match": False,
                "error_category": "AMBIGUOUS_MATCH",
            }
        )

    for metric in metrics.values():
        denominator = metric["denominator"]
        metric["strict_accuracy"] = round(
            metric["strict_correct"] / denominator, 6
        ) if denominator else 0.0
        metric["normalized_accuracy"] = round(
            metric["normalized_correct"] / denominator, 6
        ) if denominator else 0.0
        if "cer" in metric:
            metric["cer"] = round(metric["cer"] / denominator, 6) if denominator else 0.0
            metric["wer"] = round(metric["wer"] / denominator, 6) if denominator else 0.0

    return {
        "layer": layer,
        "gt_record_count": len(alignment["gt_records"]),
        "output_record_count": len(alignment["output_records"]),
        "matched_gt_rows": len(alignment["pairs"]),
        "missing_output_rows": len(alignment["missing"]),
        "extra_output_rows": len(alignment["extras"]),
        "ambiguous_rows": len(alignment["ambiguous"]),
        "field_metrics": metrics,
        "errors": errors,
        "alignment": alignment,
    }


def _scope_basename(scope: Scope) -> str:
    return SCOPE_FILES[(scope.program, scope.plan)]


def _expected_path_map(root: Path, suffix: str) -> dict[Scope, Path]:
    result = {}
    for (program, plan), basename in SCOPE_FILES.items():
        result[Scope(program, plan)] = root / f"merged_{basename}_full{suffix}.json"
    return result


def _load_courses(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("courses"), list):
        raise ValueError(f"curriculum file must contain a courses list: {path}")
    return document["courses"]


def _discover_consolidated(root: Path) -> dict[Scope, Path]:
    paths = sorted(root.glob("**/full/merged_*_full.json"))
    expected = _expected_path_map(root, "")
    by_basename = {path.name: path for path in paths}
    if set(by_basename) != {path.name for path in expected.values()}:
        raise ValueError("Consolidated inventory does not contain exactly the eight expected FULL files")
    return {scope: by_basename[path.name] for scope, path in expected.items()}


def _discover_llm(root: Path) -> dict[Scope, Path]:
    paths = sorted(root.glob("*_corrected.json"))
    expected = _expected_path_map(root, "_corrected")
    by_basename = {path.name: path for path in paths}
    if set(by_basename) != {path.name for path in expected.values()}:
        raise ValueError("LLM inventory does not contain exactly the eight expected corrected files")
    return {scope: by_basename[path.name] for scope, path in expected.items()}


def _discover_gt(root: Path) -> dict[Scope, Path]:
    result = {Scope(program, plan): root / relative for (program, plan), relative in GT_FILES.items()}
    missing = [path for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing expected Ground Truth files: {missing}")
    return result


def load_layer_records(
    gt_root: Path,
    consolidated_root: Path,
    llm_root: Path,
) -> tuple[dict[Scope, list[dict[str, Any]]], dict[Scope, list[dict[str, Any]]], dict[Scope, list[dict[str, Any]]]]:
    gt_paths = _discover_gt(gt_root)
    consolidated_paths = _discover_consolidated(consolidated_root)
    llm_paths = _discover_llm(llm_root)
    gt = {scope: _load_courses(path) for scope, path in gt_paths.items()}
    consolidated = {scope: _load_courses(path) for scope, path in consolidated_paths.items()}
    llm = {scope: _load_courses(path) for scope, path in llm_paths.items()}
    return gt, consolidated, llm


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_layer_report(result: dict[str, Any], directory: Path) -> None:
    summary = {key: value for key, value in result.items() if key not in {"errors", "alignment"}}
    _write_json(directory / "summary.json", summary)
    field_rows = []
    for field, metrics in result["field_metrics"].items():
        row = {"layer": result["layer"], "field": field, **metrics}
        field_rows.append(row)
    field_columns = ("layer", "field", "denominator", "strict_correct", "strict_incorrect", "strict_accuracy", "normalized_correct", "normalized_incorrect", "normalized_accuracy", "cer", "wer")
    for row in field_rows:
        row.setdefault("cer", "")
        row.setdefault("wer", "")
    _write_csv(directory / "field_metrics.csv", field_rows, field_columns)
    error_columns = ("layer", "program", "plan", "occurrence", "course_identity", "field", "gt_value", "output_value", "strict_match", "normalized_match", "error_category")
    _write_csv(directory / "errors.csv", result["errors"], error_columns)


def _transition_row(field: str, mode: str, consolidated: list[bool], llm: list[bool]) -> dict[str, Any]:
    counts = Counter(("CORRECT" if c else "WRONG") + "_TO_" + ("CORRECT" if l else "WRONG") for c, l in zip(consolidated, llm))
    return {
        "field": field,
        "comparison_mode": mode,
        "correct_to_correct": counts["CORRECT_TO_CORRECT"],
        "wrong_to_correct": counts["WRONG_TO_CORRECT"],
        "correct_to_wrong": counts["CORRECT_TO_WRONG"],
        "wrong_to_wrong": counts["WRONG_TO_WRONG"],
    }


def compare_layers(
    gt_by_scope: dict[Scope, list[dict[str, Any]]],
    consolidated_by_scope: dict[Scope, list[dict[str, Any]]],
    llm_by_scope: dict[Scope, list[dict[str, Any]]],
) -> dict[str, Any]:
    deltas = []
    transitions = []
    non_name_differences = []
    for field in FIELDS:
        normalized_con: list[bool] = []
        normalized_llm: list[bool] = []
        strict_con: list[bool] = []
        strict_llm: list[bool] = []
        for scope in gt_by_scope:
            con_alignment = align_records(gt_by_scope[scope], consolidated_by_scope[scope], scope.program, scope.plan)
            llm_alignment = align_records(gt_by_scope[scope], llm_by_scope[scope], scope.program, scope.plan)
            con_pairs = {gt.key: pred for gt, pred in con_alignment["pairs"]}
            llm_pairs = {gt.key: pred for gt, pred in llm_alignment["pairs"]}
            for gt_record in _recordize(gt_by_scope[scope], scope):
                if not _present(gt_record.item, field):
                    continue
                con_pred = con_pairs.get(gt_record.key)
                llm_pred = llm_pairs.get(gt_record.key)
                con_strict, con_normalized = field_matches(field, gt_record.item.get(field), con_pred.item.get(field) if con_pred else None)
                llm_strict, llm_normalized = field_matches(field, gt_record.item.get(field), llm_pred.item.get(field) if llm_pred else None)
                strict_con.append(con_strict)
                strict_llm.append(llm_strict)
                normalized_con.append(con_normalized)
                normalized_llm.append(llm_normalized)
                if field not in TEXT_FIELDS and con_pred and llm_pred:
                    if _normalized_value(field, con_pred.item.get(field)) != _normalized_value(field, llm_pred.item.get(field)):
                        non_name_differences.append({"scope": scope.label(), "identity": gt_record.identity, "field": field})
        con_accuracy = sum(normalized_con) / len(normalized_con) if normalized_con else 0.0
        llm_accuracy = sum(normalized_llm) / len(normalized_llm) if normalized_llm else 0.0
        improvements = sum(not c and l for c, l in zip(normalized_con, normalized_llm))
        regressions = sum(c and not l for c, l in zip(normalized_con, normalized_llm))
        deltas.append({
            "field": field,
            "consolidated_accuracy": round(con_accuracy, 6),
            "llm_accuracy": round(llm_accuracy, 6),
            "delta_percentage_points": round((llm_accuracy - con_accuracy) * 100, 4),
            "improvements": improvements,
            "regressions": regressions,
            "net_improvement": improvements - regressions,
        })
        transitions.append(_transition_row(field, "normalized", normalized_con, normalized_llm))
        transitions.append(_transition_row(field, "strict", strict_con, strict_llm))
    return {"field_deltas": deltas, "transitions": transitions, "non_name_layer_differences": non_name_differences}


def _validate_real_population(
    gt_by_scope: dict[Scope, list[dict[str, Any]]],
    consolidated_by_scope: dict[Scope, list[dict[str, Any]]],
    llm_by_scope: dict[Scope, list[dict[str, Any]]],
) -> None:
    gt_total = sum(len(items) for items in gt_by_scope.values())
    con_total = sum(len(items) for items in consolidated_by_scope.values())
    llm_total = sum(len(items) for items in llm_by_scope.values())
    if (gt_total, con_total, llm_total) != (839, 841, 841):
        raise ValueError(f"unexpected population counts: GT={gt_total}, consolidated={con_total}, LLM={llm_total}")
    expected_extras = {"IT/coop", "IT/no_coop"}
    for layer, data in (("consolidated", consolidated_by_scope), ("llm", llm_by_scope)):
        actual_extras = set()
        for scope in gt_by_scope:
            alignment = align_records(gt_by_scope[scope], data[scope], scope.program, scope.plan)
            if alignment["missing"] or alignment["ambiguous"]:
                raise ValueError(f"unexpected {layer} alignment for {scope.label()}")
            actual_extras.update(
                f"{record.scope.program}/{record.scope.plan or 'none'}"
                for record in alignment["extras"]
            )
        if actual_extras != expected_extras:
            raise ValueError(f"unexpected {layer} extra scopes: {sorted(actual_extras)}")


def run_dual_evaluation(
    gt_root: Path = Path("ground_truth"),
    consolidated_root: Path = Path("outputs/consolidated"),
    llm_root: Path = Path("outputs/llm"),
    reports_root: Path = Path("reports"),
) -> dict[str, Any]:
    gt, consolidated, llm = load_layer_records(gt_root, consolidated_root, llm_root)
    _validate_real_population(gt, consolidated, llm)
    con_results = [
        evaluate_records(gt[scope], consolidated[scope], scope.program, scope.plan, "consolidated")
        for scope in gt
    ]
    llm_results = [
        evaluate_records(gt[scope], llm[scope], scope.program, scope.plan, "llm")
        for scope in gt
    ]
    con_result = _combine_layer_results(con_results, "consolidated")
    llm_result = _combine_layer_results(llm_results, "llm")
    comparison = compare_layers(gt, consolidated, llm)
    write_layer_report(con_result, reports_root / "eval_consolidated")
    write_layer_report(llm_result, reports_root / "eval_llm")
    _write_json(reports_root / "eval_comparison" / "comparison.json", comparison)
    delta_columns = ("field", "consolidated_accuracy", "llm_accuracy", "delta_percentage_points", "improvements", "regressions", "net_improvement")
    _write_csv(reports_root / "eval_comparison" / "field_delta.csv", comparison["field_deltas"], delta_columns)
    transition_columns = ("field", "comparison_mode", "correct_to_correct", "wrong_to_correct", "correct_to_wrong", "wrong_to_wrong")
    _write_csv(reports_root / "eval_comparison" / "transitions.csv", comparison["transitions"], transition_columns)
    return {"consolidated": con_result, "llm": llm_result, "comparison": comparison}


def _combine_layer_results(results: list[dict[str, Any]], layer: str) -> dict[str, Any]:
    combined = {
        "layer": layer,
        "gt_record_count": sum(result["gt_record_count"] for result in results),
        "output_record_count": sum(result["output_record_count"] for result in results),
        "matched_gt_rows": sum(result["matched_gt_rows"] for result in results),
        "missing_output_rows": sum(result["missing_output_rows"] for result in results),
        "extra_output_rows": sum(result["extra_output_rows"] for result in results),
        "ambiguous_rows": sum(result["ambiguous_rows"] for result in results),
        "field_metrics": {},
        "errors": [error for result in results for error in result["errors"]],
    }
    for field in FIELDS:
        metrics = _field_metric_template(field)
        for result in results:
            source = result["field_metrics"][field]
            metrics["denominator"] += source["denominator"]
            metrics["strict_correct"] += source["strict_correct"]
            metrics["strict_incorrect"] += source["strict_incorrect"]
            metrics["normalized_correct"] += source["normalized_correct"]
            metrics["normalized_incorrect"] += source["normalized_incorrect"]
            if field in TEXT_FIELDS:
                metrics["cer"] += source["cer"] * source["denominator"]
                metrics["wer"] += source["wer"] * source["denominator"]
        denominator = metrics["denominator"]
        metrics["strict_accuracy"] = round(metrics["strict_correct"] / denominator, 6) if denominator else 0.0
        metrics["normalized_accuracy"] = round(metrics["normalized_correct"] / denominator, 6) if denominator else 0.0
        if field in TEXT_FIELDS:
            metrics["cer"] = round(metrics["cer"] / denominator, 6) if denominator else 0.0
            metrics["wer"] = round(metrics["wer"] / denominator, 6) if denominator else 0.0
        combined["field_metrics"][field] = metrics
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Consolidated and LLM curriculum layers against shared Ground Truth")
    parser.add_argument("--reports-root", default="reports", help="Root directory for the three new report groups")
    args = parser.parse_args()
    result = run_dual_evaluation(reports_root=Path(args.reports_root))
    print(json.dumps({
        "consolidated": {key: result["consolidated"][key] for key in ("gt_record_count", "output_record_count", "matched_gt_rows", "missing_output_rows", "extra_output_rows", "ambiguous_rows")},
        "llm": {key: result["llm"][key] for key in ("gt_record_count", "output_record_count", "matched_gt_rows", "missing_output_rows", "extra_output_rows", "ambiguous_rows")},
        "non_name_layer_differences": len(result["comparison"]["non_name_layer_differences"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
