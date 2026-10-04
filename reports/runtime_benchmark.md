# Runtime Benchmark (runtime QA latency only)

Standalone, presentation-ready snapshot. No OCR / build-index involved.
Method: `time.perf_counter()`, one warm-up run, 5 measured runs each.
Run: `.venv\Scripts\python.exe scripts/benchmark_runtime.py --runs 5`
against a local backend (`backend.main:app`, port 8001) + real Gemini provider.

## Local deterministic paths (in-process, read-only SQLite)

| Path | runs | min | mean | median | max | unit |
| --- | --- | --- | --- | --- | --- | --- |
| parse `06026212 กี่หน่วยกิต` | 5 | 0.122 | 0.135 | 0.125 | 0.171 | ms |
| parse `ปี 1 เทอม 1 มีวิชาอะไรบ้าง` | 5 | 0.173 | 0.237 | 0.236 | 0.304 | ms |
| parse `วิชาไหนเรียนเกี่ยวกับ data` | 5 | 0.108 | 0.222 | 0.254 | 0.330 | ms |
| Scoped SQLite Y1S1 lookup, dsba-2565 (7 rows) | 5 | 5.112 | 7.289 | 6.794 | 9.769 | ms |

## /api/ask end-to-end (DSBA / dsba-2565, live provider)

| Case | runs | min | mean | median | max | unit | status | route |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A simple factual (`06026212 กี่หน่วยกิต`) | 5 | 1584 | 1649 | 1646 | 1758 | ms | answer | llm_sql |
| B SQL+LLM-backed (`ปี 1 เทอม 1 มีวิชาอะไรบ้าง`) | 5 | 1621 | 1649 | 1636 | 1708 | ms | answer | llm_sql |
| C fail-closed retake-timing (guard, no model) | 5 | 5.6 | 7.8 | 8.2 | 9.1 | ms | insufficient_evidence | llm_sql |

## Unavailable measurements

None. All paths measured; no invented numbers. (API runs spaced 12 s
apart to respect free-tier model quota; one stale-server run was discarded
and re-measured after restarting the backend on the current tree.)

## Conclusion

Local parsing/SQLite operations are fast; most interactive latency comes from
external model calls.
