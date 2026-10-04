"""Gemini provider adapter for injected model-callable interfaces."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any


DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_MS = 20_000
MAX_PROVIDER_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 0.25
_API_MESSAGE_MAX_LENGTH = 500

_logger = logging.getLogger(__name__)


def _is_transient_provider_error(exc: Exception) -> bool:
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
    if not isinstance(exc, APIError):
        return False

    code = getattr(exc, "code", None)
    return code == 429 or (isinstance(code, int) and 500 <= code <= 599)


def _api_error_diagnostics(
    exc: Exception,
    *,
    api_key: str,
    prompt: str,
) -> tuple[int | None, str | None]:
    try:
        from google.genai.errors import APIError
    except ImportError:
        return None, None
    if not isinstance(exc, APIError):
        return None, None

    code = getattr(exc, "code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        code = None
    message = getattr(exc, "message", None)
    if not isinstance(message, str):
        return code, None

    if prompt:
        message = message.replace(prompt, "[REDACTED PROMPT]")
    if api_key:
        message = message.replace(api_key, "[REDACTED CREDENTIAL]")
    message = re.sub(
        r"(?i)\bBearer\s+[^\s,;]+",
        "Bearer [REDACTED CREDENTIAL]",
        message,
    )
    message = re.sub(
        r"(?i)\b(api[_-]?key|key|authorization)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED CREDENTIAL]",
        message,
    )
    message = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED CREDENTIAL]", message)
    message = " ".join(message.split())
    return code, message[:_API_MESSAGE_MAX_LENGTH]


def _create_client(api_key: str) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def make_gemini_callable() -> Callable[..., str]:
    """Return a lazy Gemini callable with bounded timeout and transient retry."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is required to create the Gemini provider")

    client: Any = None

    def generate(
        prompt: str,
        *,
        response_mime_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        response_json_schema: dict[str, Any] | None = None,
    ) -> str:
        nonlocal client
        for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
            started = time.monotonic()
            stage = "client_creation"
            try:
                if client is None:
                    client = _create_client(api_key)
                stage = "generate_content"
                from google.genai import types

                config_options: dict[str, Any] = {"temperature": 0.0}
                if response_mime_type is not None:
                    config_options["response_mime_type"] = response_mime_type
                if response_schema is not None and response_json_schema is not None:
                    raise ValueError("provide response_schema or response_json_schema, not both")
                if response_schema is not None:
                    config_options["response_schema"] = types.Schema(**response_schema)
                if response_json_schema is not None:
                    config_options["response_json_schema"] = response_json_schema
                response = client.models.generate_content(
                    model=DEFAULT_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_options),
                )
                text = getattr(response, "text", None)
                if not isinstance(text, str):
                    raise RuntimeError("Gemini returned no text response")
                return text
            except Exception as exc:
                transient = _is_transient_provider_error(exc)
                elapsed = time.monotonic() - started
                api_code, api_message = _api_error_diagnostics(
                    exc,
                    api_key=api_key,
                    prompt=prompt,
                )
                exception_type = type(exc).__name__
                structured_output_requested = (
                    response_mime_type is not None
                    or response_schema is not None
                    or response_json_schema is not None
                )
                _logger.warning(
                    "Gemini request failed stage=%s attempt=%d/%d "
                    "exception_type=%s transient=%s elapsed=%.3fs "
                    "api_code=%s api_message=%s structured_output_requested=%s",
                    stage,
                    attempt,
                    MAX_PROVIDER_ATTEMPTS,
                    exception_type,
                    str(transient).lower(),
                    elapsed,
                    api_code,
                    api_message,
                    str(structured_output_requested).lower(),
                    extra={
                        "api_code": api_code,
                        "api_message": api_message,
                        "exception_type": exception_type,
                        "transient": transient,
                        "stage": stage,
                        "attempt": attempt,
                        "structured_output_requested": structured_output_requested,
                    },
                )
                if not transient or attempt >= MAX_PROVIDER_ATTEMPTS:
                    raise
                time.sleep(RETRY_BACKOFF_SECONDS)

    return generate


__all__ = ["DEFAULT_MODEL", "make_gemini_callable"]
