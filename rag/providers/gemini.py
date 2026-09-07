"""Gemini provider adapter for injected model-callable interfaces."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


DEFAULT_MODEL = "gemini-2.0-flash"


def _create_client(api_key: str) -> Any:
    from google import genai

    return genai.Client(api_key=api_key)


def make_gemini_callable() -> Callable[[str], str]:
    """Return a lazily initialized callable backed by Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is required to create the Gemini provider")

    client: Any = None

    def generate(prompt: str) -> str:
        nonlocal client
        if client is None:
            client = _create_client(api_key)
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise RuntimeError("Gemini returned no text response")
        return text

    return generate


__all__ = ["DEFAULT_MODEL", "make_gemini_callable"]
