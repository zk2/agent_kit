"""Pluggable embeddings.

Two providers:
  - `hashing`  : dependency-free, offline. Feature-hashes tokens into a fixed-dim vector, so
                 cosine similarity reflects lexical overlap. Good enough to exercise the full
                 pipeline (incl. the no-answer threshold) without any network or model download.
                 Not production-quality semantics.
  - `fastembed`: real semantic embeddings via ONNX (BAAI/bge-small-en-v1.5, 384-dim). No torch.
                 Requires the optional `fastembed` dependency and a one-time model download.

All providers expose the same `Embedder` protocol and the same dimension, so the Postgres
`vector(dim)` column and the rest of the code are provider-agnostic.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol, runtime_checkable

from agentkit.config import settings

_TOKEN = re.compile(r"[a-z0-9]+")

# Dropping high-frequency function words keeps the hashing proxy's cosine focused on content
# words, which sharpens the gap between relevant and irrelevant queries for the no-answer gate.
_STOPWORDS = frozenset(
    (
        "a an and are as at be by for from has have in is it its of on or that the to was were "
        "what which who with you your this these those do does how when where why will can could"
    ).split()
)


@runtime_checkable
class Embedder(Protocol):
    provider: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic, offline embedder based on signed feature hashing of word tokens."""

    provider = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            if tok in _STOPWORDS:
                continue
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbedder:
    """Real semantic embeddings via the optional `fastembed` package (ONNX, CPU)."""

    provider = "fastembed"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        from fastembed import TextEmbedding  # type: ignore[import-not-found]  # optional dep

        self._model = TextEmbedding(model_name=model)
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._model.embed([text])))]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embedding_provider == "fastembed":
        return FastEmbedEmbedder(dim=settings.embedding_dim)
    return HashingEmbedder(dim=settings.embedding_dim)


def to_pgvector(vec: list[float]) -> str:
    """Render a vector as a pgvector literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
