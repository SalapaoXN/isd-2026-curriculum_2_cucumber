"""Deterministic text embeddings for retrieval chunks."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384
_COMPONENTS: tuple[Any, Any, Any] | None = None


def _load_components() -> tuple[Any, Any, Any]:
    """Load the tokenizer, model, and torch lazily on first non-empty batch."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model, torch


def _components() -> tuple[Any, Any, Any]:
    global _COMPONENTS
    if _COMPONENTS is None:
        _COMPONENTS = _load_components()
    return _COMPONENTS


def _hidden_states(model_output: Any) -> Any:
    if hasattr(model_output, "last_hidden_state"):
        return model_output.last_hidden_state
    if isinstance(model_output, dict) and "last_hidden_state" in model_output:
        return model_output["last_hidden_state"]
    return model_output[0]


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    """Embed texts in input order using attention-mask mean pooling."""
    if isinstance(texts, str):
        batch = [texts]
    else:
        batch = list(texts)

    if not batch:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    tokenizer, model, torch = _components()
    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        model_output = model(**encoded)

    token_embeddings = _hidden_states(model_output)
    mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return np.asarray(pooled.detach().cpu().numpy(), dtype=np.float32)


__all__ = ["EMBEDDING_DIMENSION", "MODEL_NAME", "embed_texts"]
