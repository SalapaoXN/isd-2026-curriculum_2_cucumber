"""Typed operational errors for the semantic pipeline.

A model-provider failure (timeout/transient/provider-side error) is
``provider_unavailable``; any other unexpected failure is ``error``.
Neither is evidence insufficiency: the backend maps them to distinct
public statuses and preserves the last validated safe context instead
of clearing it. Messages carry only the status, never prompts, secrets,
SQL, or exception internals.
"""

from __future__ import annotations


class SemanticOperationalError(Exception):
    """Typed operational failure escaping the semantic pipeline."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = (
            status if status in ("provider_unavailable", "error") else "error"
        )


def is_provider_error(exc: Exception) -> bool:
    """Return True only for known transport/provider-library failures.

    Conservative on purpose: anything that is not recognizably raised by
    the model-transport layer is classified as an unexpected system
    error, never silently relabeled as a transient provider issue.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        import httpx
    except ImportError:
        httpx = None
    if httpx is not None and isinstance(
        exc, (httpx.TimeoutException, httpx.NetworkError)
    ):
        return True
    try:
        from google.genai.errors import APIError
    except ImportError:
        return False
    return isinstance(exc, APIError)


__all__ = ["SemanticOperationalError", "is_provider_error"]
