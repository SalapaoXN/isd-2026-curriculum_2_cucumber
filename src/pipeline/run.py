"""Single end-to-end pipeline entry point (clean rebuild).

Stages:
  ocr -> extract -> merge -> correct -> (evaluate) -> (build_index)

Final RAG-ready output: data/output/final/*_corrected.json (+ optional curriculum.db).
scripts/ask.py stays separate and is not run here.

Defaults (adjust only by flags, behavior unchanged):
  - evaluate.py IS included by default; pass --skip-eval to skip it.
  - build_index is OFF by default; pass --with-index to continue to
    curriculum.db, or --only-index to build only the DB from existing
    corrected files.

Intermediates (extracted/, page_ranges/, legacy_summaries/) go to a temp
directory and are deleted automatically. Pass --keep-intermediates for
debug to persist them under the output tree.

Resume points (--from):
  ocr          run everything from OCR (default)
  extracted    skip OCR and per-file extraction; merge from existing extracted/
  consolidated skip to LLM correction from existing consolidated full files
  corrected    skip to evaluate/index from existing corrected files
Resume modes read persisted data/output/ layout, so they pair with
--keep-intermediates output from a previous debug run.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from src.pipeline.tools.merge.consolidator import edition_filename_token

BASE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_PROGRAMS = ("ait", "bit", "dsba", "gened", "it")
FROM_CHOICES = ("ocr", "extracted", "consolidated", "corrected")
_ACADEMIC_YEAR_RE = re.compile(r"^[1-9]\d{3}(?:[-/]\d{2,4})?$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR -> extract -> merge -> correct -> (evaluate) -> (build_index).",
        epilog="Example: python -m src.pipeline.run --program it --with-index",
    )
    parser.add_argument("--program", choices=SUPPORTED_PROGRAMS, default="it")
    parser.add_argument("--pages", default=None, help="OCR pages/ranges, e.g. 32-38 or 32-34,328-330.")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluate.py stage.")
    parser.add_argument("--with-index", action="store_true", help="Continue to curriculum.db after correction.")
    parser.add_argument("--only-index", action="store_true", help="Only build curriculum.db from existing corrected files.")
    parser.add_argument("--from", dest="from_stage", choices=FROM_CHOICES, default="ocr")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--index-path", default=None)
    parser.add_argument(
        "--catalog-key",
        default=None,
        help="Explicit curriculum edition identity stored in corrected JSON metadata.",
    )
    parser.add_argument(
        "--academic-year",
        default=None,
        help="Explicit 4-digit academic year (optionally a range, e.g. 2568-2569).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _edition_metadata_from_args(args: argparse.Namespace) -> dict[str, str] | None:
    """Validate the optional, explicitly supplied curriculum edition identity."""
    catalog_key = getattr(args, "catalog_key", None)
    academic_year = getattr(args, "academic_year", None)
    if catalog_key is None and academic_year is None:
        return None
    if not isinstance(catalog_key, str) or not catalog_key.strip():
        raise ValueError("catalog_key and academic_year must be supplied together")
    if catalog_key != catalog_key.strip():
        raise ValueError("catalog_key must not have leading or trailing whitespace")

    if isinstance(academic_year, bool) or not isinstance(academic_year, (str, int)):
        raise ValueError("academic_year must be a 4-digit year or explicit year range")
    year_text = str(academic_year)
    if not _ACADEMIC_YEAR_RE.fullmatch(year_text):
        raise ValueError("academic_year must be a 4-digit year or explicit year range")
    return {"catalog_key": catalog_key, "academic_year": year_text}


def _apply_edition_metadata(
    paths: list[str | Path], metadata: dict[str, str] | None
) -> list[Path]:
    """Stamp canonical metadata and give explicitly scoped artifacts unique names."""
    if metadata is None:
        return [Path(path) for path in paths]
    if not paths:
        raise ValueError("no curriculum artifacts found for edition metadata")

    token = edition_filename_token(metadata["catalog_key"])
    updates: list[tuple[Path, Path, dict]] = []
    destinations: set[Path] = set()
    for path_value in paths:
        path = Path(path_value)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot apply edition metadata to {path}") from error
        if not isinstance(document, dict):
            raise ValueError(f"curriculum artifact must be a JSON object: {path}")
        catalog = document.get("catalog")
        if catalog is None:
            catalog = {}
        if not isinstance(catalog, dict):
            raise ValueError(f"curriculum artifact catalog metadata must be an object: {path}")
        existing_key = catalog.get("catalog_key")
        if existing_key is not None and existing_key != metadata["catalog_key"]:
            raise ValueError(
                f"curriculum artifact belongs to catalog {existing_key!r}, "
                f"not {metadata['catalog_key']!r}: {path}"
            )
        document["catalog"] = {**catalog, **metadata}
        destination = path
        marker = f"_{token}_"
        if marker not in f"_{path.stem}_":
            if path.stem.endswith("_corrected"):
                base = path.stem[: -len("_corrected")]
                destination = path.with_name(f"{base}_{token}_corrected{path.suffix}")
            else:
                destination = path.with_name(f"{path.stem}_{token}{path.suffix}")
        resolved_destination = destination.resolve()
        if resolved_destination in destinations:
            raise ValueError(f"duplicate edition artifact identity: {destination}")
        destinations.add(resolved_destination)
        if destination != path and destination.exists():
            raise FileExistsError(f"edition artifact already exists: {destination}")
        if destination != path and path.name.endswith("_corrected.json"):
            old_corrections = path.with_name(path.name.replace("_corrected.json", "_corrections.json"))
            new_corrections = destination.with_name(
                destination.name.replace("_corrected.json", "_corrections.json")
            )
            if old_corrections.exists() and new_corrections.exists():
                raise FileExistsError(f"edition artifact already exists: {new_corrections}")
        updates.append((path, destination, document))

    renamed: list[Path] = []
    for path, destination, document in updates:
        destination.write_text(
            json.dumps(document, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
        )
        if destination != path:
            path.unlink()
            corrections_path = path.with_name(path.name.replace("_corrected.json", "_corrections.json"))
            if corrections_path != path and corrections_path.exists():
                tokenized_corrections = destination.with_name(
                    destination.name.replace("_corrected.json", "_corrections.json")
                )
                if tokenized_corrections.exists():
                    raise FileExistsError(f"edition artifact already exists: {tokenized_corrections}")
                corrections_path.rename(tokenized_corrections)
        renamed.append(destination)
    return renamed


def _ensure_edition_corrected_targets_available(inputs: list[Path], llm_dir: Path) -> None:
    for path in inputs:
        corrected = llm_dir / f"{path.stem}_corrected.json"
        corrections = llm_dir / f"{path.stem}_corrections.json"
        if corrected.exists() or corrections.exists():
            raise FileExistsError(f"edition artifact already exists: {corrected}")


def _paths_for_program(paths: list[Path], program_key: str) -> list[Path]:
    wanted = program_key.upper()
    matching: list[Path] = []
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict) and str(document.get("program", "")).upper() == wanted:
            matching.append(path)
    return matching


def _program_scoped_prepare(
    program_key: str,
    ocr_root: Path,
    extracted_root: Path,
    consolidated_root: Path,
    edition_metadata: dict[str, str] | None = None,
) -> dict:
    """Run the prepare_data scope logic for one program only (no duplication)."""
    from src.pipeline.tools.preparation import tool as _pd

    key = program_key.casefold()
    if not _pd.prepare_program(
        key, ocr_root / key, extracted_root, consolidated_root, BASE_DIR, edition_metadata
    ):
        raise _pd.PreparationError(f"No usable OCR source for program '{key}' under {ocr_root}")
    return {"prepared_programs": [key]}


def _copy_final_full_files(
    consolidated_tmp: Path,
    consolidated_final: Path,
    *,
    edition_mode: bool = False,
) -> list[Path]:
    kept: list[Path] = []
    sources = sorted(consolidated_tmp.glob("**/full/merged_*_full.json"))
    destinations = [consolidated_final / src.relative_to(consolidated_tmp) for src in sources]
    if edition_mode:
        existing = [path for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(f"edition artifact already exists: {existing[0]}")
    for src, dest in zip(sources, destinations):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        kept.append(dest)
    return kept


def _corrected_paths_for_program(
    llm_dir: Path,
    program_key: str,
    *,
    catalog_key: str | None = None,
    plan: str | None = None,
) -> list[Path]:
    wanted = program_key.upper()
    kept: list[Path] = []
    for path in sorted(llm_dir.glob("*_corrected.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or str(doc.get("program", "")).upper() != wanted:
            continue
        if plan is not None and str(doc.get("plan", "")).casefold() != plan.casefold():
            continue
        if catalog_key is not None:
            catalog = doc.get("catalog")
            if not isinstance(catalog, dict) or catalog.get("catalog_key") != catalog_key:
                continue
            kept.append(path)
        else:
            kept.append(path)
    if catalog_key is not None:
        if not kept:
            raise FileNotFoundError(
                f"no corrected artifact for program={program_key}, plan={plan}, "
                f"catalog_key={catalog_key!r} under {llm_dir}"
            )
        if len(kept) > 1:
            raise ValueError(
                f"ambiguous corrected artifacts for program={program_key}, plan={plan}, "
                f"catalog_key={catalog_key!r}: {', '.join(map(str, kept))}"
            )
    return kept


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        edition_metadata = _edition_metadata_from_args(args)
    except ValueError as error:
        print(f"Invalid edition metadata: {error}", file=sys.stderr)
        return 2
    print("Stages: ocr -> extract -> merge -> correct -> (evaluate) -> (build_index)")
    print(f"Program={args.program} from={args.from_stage} keep_intermediates={args.keep_intermediates}")
    if args.dry_run:
        print("Dry run: no stages executed.")
        return 0
    if args.only_index and args.from_stage != "ocr":
        print("Note: --only-index ignores --from; building index only.", file=sys.stderr)

    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = BASE_DIR / output_base
    ocr_root = output_base / "ocr"
    extracted_final = output_base / "extracted"
    consolidated_final = output_base / "consolidated"
    llm_dir = output_base / "final"

    if args.only_index:
        from src.pipeline.tools.indexing.tool import run_build_index_stage

        # The default runtime database is unified.  Let the indexing stage
        # validate and discover the complete source set unless the caller
        # explicitly supplied a separate index path for a scoped build.
        paths = None
        if args.index_path is not None:
            paths = _corrected_paths_for_program(
                llm_dir,
                args.program,
                catalog_key=(edition_metadata or {}).get("catalog_key"),
            )
        print(run_build_index_stage(paths, args.index_path))
        return 0

    from_stage = args.from_stage
    do_ocr = from_stage == "ocr"
    do_extract_merge = from_stage in ("ocr", "extracted")
    do_correct = from_stage in ("ocr", "extracted", "consolidated")

    tmp_holder: tempfile.TemporaryDirectory | None = None
    try:
        if args.keep_intermediates:
            extracted_root = extracted_final
            consolidated_work = consolidated_final
        else:
            tmp_holder = tempfile.TemporaryDirectory(prefix="pipeline_")
            tmp_base = Path(tmp_holder.name)
            extracted_root = tmp_base / "extracted"
            consolidated_work = tmp_base / "consolidated"

        if do_ocr:
            from src.pipeline.tools.ocr.tool import run_ocr_stage

            run_ocr_stage(args.program, args.pages, args.no_gpu, BASE_DIR)

        if do_extract_merge:
            if from_stage == "extracted":
                # Merge only, from persisted extracted/ (debug resume path).
                from src.pipeline.tools.preparation import tool as _pd

                config = _pd.PROGRAM_CONFIG[args.program.casefold()]
                for scope in config.scopes:
                    _pd._run_merge(
                        extracted_final / args.program.casefold(),
                        consolidated_work,
                        config,
                        scope,
                        _pd._combined_pages(scope, True),
                        edition_metadata,
                    )
            else:
                _program_scoped_prepare(
                    args.program,
                    ocr_root,
                    extracted_root,
                    consolidated_work,
                    edition_metadata,
                )
            if not args.keep_intermediates:
                kept = _copy_final_full_files(
                    consolidated_work,
                    consolidated_final,
                    edition_mode=edition_metadata is not None,
                )
                print(f"Kept {len(kept)} final full file(s) under {consolidated_final}")

        if edition_metadata is not None:
            if from_stage == "corrected":
                metadata_paths = _corrected_paths_for_program(
                    llm_dir, args.program, catalog_key=edition_metadata["catalog_key"]
                )
            else:
                edition_token = edition_filename_token(edition_metadata["catalog_key"])
                consolidated_paths = sorted(
                    consolidated_final.glob("**/full/merged_*_full.json")
                )
                metadata_paths = [
                    path
                    for path in _paths_for_program(consolidated_paths, args.program)
                    if f"_{edition_token}_full" in path.name
                ]
            metadata_paths = _apply_edition_metadata(metadata_paths, edition_metadata)

        if do_correct:
            from src.pipeline.tools.correction.tool import run_correct_stage

            if from_stage == "consolidated":
                if edition_metadata is None:
                    inputs: list[Path] | None = None
                else:
                    token = edition_filename_token(edition_metadata["catalog_key"])
                    inputs = [
                        path
                        for path in sorted(
                            consolidated_final.glob("**/full/merged_*_full.json")
                        )
                        if token in path.stem
                        and str(
                            json.loads(path.read_text(encoding="utf-8")).get("program", "")
                        ).casefold()
                        == args.program.casefold()
                    ]
                    if not inputs:
                        raise FileNotFoundError(
                            f"no consolidated artifact for catalog_key={edition_metadata['catalog_key']!r}"
                        )
            else:
                full_files = sorted(consolidated_final.glob("**/full/merged_*_full.json"))
                wanted = [p for p in full_files if args.program.casefold() in p.as_posix().casefold()]
                if edition_metadata is not None:
                    token = edition_filename_token(edition_metadata["catalog_key"])
                    wanted = [path for path in wanted if token in path.stem]
                inputs = wanted or None
            if edition_metadata is not None and inputs is not None:
                _ensure_edition_corrected_targets_available(inputs, llm_dir)
            if inputs is None:
                corrected = run_correct_stage(consolidated_final, llm_dir)
            else:
                corrected = run_correct_stage(consolidated_final, llm_dir, inputs)
            print(f"Corrected {len(corrected)} file(s) under {llm_dir}")

        if not args.skip_eval:
            from src.pipeline.tools.evaluation.tool import run_evaluate_stage

            program_paths = _corrected_paths_for_program(
                llm_dir,
                args.program,
                catalog_key=(edition_metadata or {}).get("catalog_key"),
            ) if edition_metadata is not None else _corrected_paths_for_program(llm_dir, args.program)
            if program_paths:
                from src.pipeline.tools.evaluation.evaluate import discover_llm_evaluation_pairs

                all_pairs = discover_llm_evaluation_pairs(llm_dir)
                pairs = [(p, g) for p, g in all_pairs if Path(p) in set(program_paths)]
                run_evaluate_stage(pairs=pairs)
            else:
                print(f"No corrected files for {args.program}; skipping evaluation.")

        if args.with_index:
            from src.pipeline.tools.indexing.tool import run_build_index_stage

            program_paths = None
            if args.index_path is not None:
                program_paths = _corrected_paths_for_program(
                    llm_dir,
                    args.program,
                    catalog_key=(edition_metadata or {}).get("catalog_key"),
                )
            print(run_build_index_stage(program_paths or None, args.index_path))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()
    print(f"Done. RAG-ready files under {llm_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
