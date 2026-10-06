"""Semantic QA vNext: LLM language understanding over deterministic facts.

LANGUAGE → LLM interpreter. IDENTITY/SCOPE → deterministic resolver.
FACTS → canonical SQLite. TRUST → evidence + provenance.
PRESENTATION → bounded LLM answerer.

See docs/semantic-qa-vnext.md for the architecture and mode switch.
"""

from rag.semantic import (
    answerer,
    compiler,
    context,
    eval,
    executor,
    interpreter,
    modes,
    pipeline,
    planner,
    prompts,
    resolver,
    schema,
    trace,
    validation,
)
from rag.semantic.modes import (
    active_qa_mode,
    run_shadow_comparison,
    semantic_ask_response,
)
from rag.semantic.pipeline import SemanticPipelineResult, semantic_answer

__all__ = [
    "SemanticPipelineResult",
    "active_qa_mode",
    "answerer",
    "compiler",
    "context",
    "eval",
    "executor",
    "interpreter",
    "modes",
    "pipeline",
    "planner",
    "prompts",
    "resolver",
    "run_shadow_comparison",
    "schema",
    "semantic_answer",
    "semantic_ask_response",
    "trace",
    "validation",
]
