"""Audit canonical runtime artifacts for unresolved shared-course conflicts."""

from src.pipeline.tools.runtime_artifacts import main_preflight


if __name__ == "__main__":
    raise SystemExit(main_preflight())
