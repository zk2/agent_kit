"""Document chunking.

Word-window chunking with overlap. Word boundaries avoid cutting tokens mid-word, and the
overlap keeps context that straddles a boundary retrievable from at least one chunk. Each chunk
carries its source and index so answers can cite a precise location.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    source: str
    content: str


@dataclass
class Chunk:
    source: str
    chunk_index: int
    content: str


def chunk_text(
    content: str,
    source: str,
    *,
    chunk_size: int = 120,
    overlap: int = 20,
) -> list[Chunk]:
    """Split `content` into overlapping word windows. Sizes are in words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = content.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    idx = 0
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_size])
        chunks.append(Chunk(source=source, chunk_index=idx, content=piece))
        idx += 1
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_document(doc: Document, **kwargs) -> list[Chunk]:
    return chunk_text(doc.content, doc.source, **kwargs)
