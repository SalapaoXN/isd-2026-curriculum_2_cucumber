"""Small standalone runtime-QA latency benchmark (presentation use).

Measures current runtime paths only; no OCR / build-index involved.
Does not modify production behavior. Uses time.perf_counter() throughout,
one warm-up run before measuring, and small sample counts.

- Local deterministic paths run in-process (read-only SQLite).
- /api/ask end-to-end runs against a live backend (default
  http://127.0.0.1:8001); start one first, e.g.
  .\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe scripts/benchmark_runtime.py [--runs 5] [--base-url URL]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.query_spec import parse_query_spec
from rag.structured.execute import execute_readonly

DB_PATH = PROJECT_ROOT / "cucumber_outputs" / "runtime" / "curriculum.db"

PARSE_CASES = (
    "06026212 กี่หน่วยกิต",
    "ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
    "วิชาไหนเรียนเกี่ยวกับ data",
)

SQL_TEXT = (
    "SELECT DISTINCT t1.catalog_key, t2.program, t2.course_code, "
    "t3.name_th, t3.name_en "
    "FROM catalogs AS t1 "
    "JOIN v_plan_courses AS t2 ON t2.program = 'DSBA' AND t2.year = 1 AND t2.semester = 1 "
    "JOIN courses AS t3 ON t3.course_id = t2.course_id AND t3.catalog_id = t1.catalog_id "
    "WHERE t1.catalog_key = 'dsba-2565'"
)

API_CASES = (
    {
        "name": "A-simple-factual",
        "question": "06026212 กี่หน่วยกิต",
        "conversation_context": {"program": "DSBA", "catalog_key": "dsba-2565"},
    },
    {
        "name": "B-sql-llm-backed",
        "question": "ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
        "conversation_context": {"program": "DSBA", "catalog_key": "dsba-2565"},
    },
    {
        "name": "C-fail-closed-retake",
        "question": (
            "ถ้าถอนวิชา FUNDAMENTAL WEB PROGRAMMING ตอนปี 2 เทอม 1 "
            "ต้องลงเรียนอีกทีตอนไหน เทอมไหน"
        ),
        "conversation_context": {"program": "DSBA", "catalog_key": "dsba-2565"},
    },
)


def summarize(samples: list[float]) -> dict[str, Any]:
    ms = [s * 1000.0 for s in samples]
    return {
        "runs": len(ms),
        "unit": "ms",
        "min": round(min(ms), 3),
        "mean": round(statistics.fmean(ms), 3),
        "median": round(statistics.median(ms), 3),
        "max": round(max(ms), 3),
    }


def bench_parse(runs: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for question in PARSE_CASES:
        parse_query_spec(question)  # warm-up
        samples = []
        for _ in range(runs):
            start = time.perf_counter()
            parse_query_spec(question)
            samples.append(time.perf_counter() - start)
        results[question] = summarize(samples)
    return results


def bench_sqlite(runs: int) -> dict[str, Any]:
    execute_readonly(str(DB_PATH), SQL_TEXT, catalog_key="dsba-2565", program="DSBA")  # warm-up
    samples = []
    rows = 0
    for _ in range(runs):
        start = time.perf_counter()
        _, fetched = execute_readonly(
            str(DB_PATH), SQL_TEXT, catalog_key="dsba-2565", program="DSBA"
        )
        samples.append(time.perf_counter() - start)
        rows = len(fetched)
    summary = summarize(samples)
    summary["rows_per_run"] = rows
    return {"scoped Y1S1 lookup (dsba-2565)": summary}


def post_ask(base_url: str, payload: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None, str | None]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/ask",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.loads(response.read().decode("utf-8")), None
    except Exception as exc:  # e.g. provider 503 / quota; never invent numbers
        return None, None, f"{type(exc).__name__}: {exc}"[:200]


def bench_api(base_url: str, runs: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/health", timeout=15):
            pass
    except Exception as exc:
        return {"unavailable": f"backend not reachable: {type(exc).__name__}"}
    for case in API_CASES:
        payload = {
            "question": case["question"],
            "conversation_context": case["conversation_context"],
        }
        post_ask(base_url, payload)  # warm-up (also absorbs cold provider latency)
        time.sleep(12)  # spacing to respect free-tier model quota
        samples: list[float] = []
        last_status: Any = None
        last_route: Any = None
        errors: list[str] = []
        for _ in range(runs):
            start = time.perf_counter()
            http, data, error = post_ask(base_url, payload)
            samples.append(time.perf_counter() - start)
            if error is not None:
                errors.append(error)
            else:
                last_status = (data or {}).get("status")
                last_route = (data or {}).get("route")
            time.sleep(12)
        if len(errors) > len(samples) // 2:
            results[case["name"]] = {
                "unavailable": f"{len(errors)}/{len(samples)} attempts failed",
                "errors": errors[:2],
            }
            continue
        summary = summarize(samples)
        summary["status"] = last_status
        summary["route"] = last_route
        if errors:
            summary["transport_errors"] = errors
        results[case["name"]] = summary
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime QA latency benchmark")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()

    report = {
        "runs_requested": args.runs,
        "deterministic_parsing": bench_parse(args.runs),
        "canonical_sqlite": bench_sqlite(args.runs),
        "api_ask_end_to_end": bench_api(args.base_url, args.runs),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
