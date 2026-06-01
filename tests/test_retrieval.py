"""Unit tests for chunking and embeddings - offline, no DB."""

import math

from agentkit.retrieval.chunking import Document, chunk_document, chunk_text
from agentkit.retrieval.embeddings import HashingEmbedder, to_pgvector


def test_chunking_overlap_and_indices():
    words = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(words, "doc.md", chunk_size=120, overlap=20)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # step = 100 words -> 300 words -> chunks starting at 0,100,200
    assert len(chunks) == 3
    # overlap: last 20 words of chunk 0 reappear at the start of chunk 1
    tail = chunks[0].content.split()[-20:]
    head = chunks[1].content.split()[:20]
    assert tail == head


def test_chunking_empty_and_short():
    assert chunk_text("", "d") == []
    short = chunk_document(Document("d", "just a few words"))
    assert len(short) == 1
    assert short[0].chunk_index == 0


def test_hashing_embedder_dim_and_norm():
    emb = HashingEmbedder(dim=384)
    v = emb.embed_query("hybrid retrieval over pgvector")
    assert len(v) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


def test_hashing_embedder_lexical_overlap():
    """Shared words -> higher cosine than unrelated text. Drives the no-answer threshold."""
    emb = HashingEmbedder(dim=384)
    q = emb.embed_query("pgvector hybrid retrieval")
    related = emb.embed_query("retrieval uses pgvector for hybrid search")
    unrelated = emb.embed_query("the weather in tokyo is mild today")

    def cos(a, b):  # both are unit vectors
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cos(q, related) > cos(q, unrelated)


def test_to_pgvector_format():
    assert to_pgvector([0.5, -1.0]) == "[0.50000000,-1.00000000]"
