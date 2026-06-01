"""Postgres-backed hybrid retriever.

Storage: one `documents` table holding chunk text, a `vector(dim)` embedding (HNSW + cosine),
and a generated `tsvector` (GIN) for lexical search.

Search: vector and lexical candidates are fused in SQL with Reciprocal Rank Fusion (RRF), so no
merge logic lives in Python. The top vector cosine similarity is also returned so the graph can
make a no-answer decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

from psycopg import Connection
from psycopg.rows import TupleRow, dict_row
from psycopg_pool import ConnectionPool

from agentkit.config import settings
from agentkit.retrieval.chunking import Chunk
from agentkit.retrieval.embeddings import Embedder, get_embedder, to_pgvector

# The pool yields default (tuple-row) connections; read paths opt into dict_row per cursor.
_Pool = ConnectionPool[Connection[TupleRow]]

_RRF_K = 60  # RRF damping constant


def _schema_sql(dim: int) -> str:
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS documents (
        id          bigserial PRIMARY KEY,
        source      text NOT NULL,
        chunk_index int  NOT NULL,
        content     text NOT NULL,
        embedding   vector({dim}) NOT NULL,
        tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    );
    CREATE INDEX IF NOT EXISTS documents_tsv_idx ON documents USING gin (tsv);
    CREATE INDEX IF NOT EXISTS documents_embedding_idx
        ON documents USING hnsw (embedding vector_cosine_ops);
    """


_INSERT_SQL = """
    INSERT INTO documents (source, chunk_index, content, embedding)
    VALUES (%(source)s, %(chunk_index)s, %(content)s, %(embedding)s::vector)
"""

_SEARCH_SQL = """
WITH q AS (
    SELECT %(qvec)s::vector AS qvec,
           plainto_tsquery('english', %(qtext)s) AS qts
),
vec AS (
    SELECT d.id, d.source, d.chunk_index, d.content,
           1 - (d.embedding <=> q.qvec) AS vscore,
           row_number() OVER (ORDER BY d.embedding <=> q.qvec) AS vrank
    FROM documents d, q
    ORDER BY d.embedding <=> q.qvec
    LIMIT %(pool)s
),
lex AS (
    SELECT d.id, d.source, d.chunk_index, d.content,
           ts_rank(d.tsv, q.qts) AS lscore,
           row_number() OVER (ORDER BY ts_rank(d.tsv, q.qts) DESC) AS lrank
    FROM documents d, q
    WHERE q.qts @@ d.tsv
    ORDER BY lscore DESC
    LIMIT %(pool)s
)
SELECT
    COALESCE(vec.id, lex.id)                   AS id,
    COALESCE(vec.source, lex.source)           AS source,
    COALESCE(vec.chunk_index, lex.chunk_index) AS chunk_index,
    COALESCE(vec.content, lex.content)         AS content,
    COALESCE(vec.vscore, 0)::float             AS vscore,
    COALESCE(lex.lscore, 0)::float             AS lscore,
    (COALESCE(1.0 / (%(rrf_k)s + vec.vrank), 0)
     + COALESCE(1.0 / (%(rrf_k)s + lex.lrank), 0))::float AS rrf
FROM vec
FULL OUTER JOIN lex ON vec.id = lex.id
ORDER BY rrf DESC
LIMIT %(k)s
"""


@dataclass
class RetrievedChunk:
    id: int
    source: str
    chunk_index: int
    content: str
    vscore: float
    lscore: float
    rrf: float


class Retriever:
    def __init__(self, conninfo: str, embedder: Embedder) -> None:
        self._conninfo = conninfo
        self.embedder = embedder
        self._pool: _Pool | None = None

    @property
    def pool(self) -> _Pool:
        pool = self._pool
        if pool is None:
            created: _Pool = ConnectionPool(
                self._conninfo,
                max_size=5,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            created.open()
            self._pool = created
            pool = created
        return pool

    def ensure_schema(self) -> None:
        with self.pool.connection() as conn:
            # dim is a trusted int from config, not user input; safe to interpolate into DDL.
            conn.execute(_schema_sql(self.embedder.dim))  # type: ignore[arg-type]

    def clear(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("TRUNCATE documents RESTART IDENTITY")

    def ingest(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.content for c in chunks])
        params = [
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "content": c.content,
                "embedding": to_pgvector(v),
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(_INSERT_SQL, params)
        return len(params)

    def search(self, query: str, *, k: int = 5) -> list[RetrievedChunk]:
        qvec = to_pgvector(self.embedder.embed_query(query))
        candidate_pool = max(k * 4, 20)
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                _SEARCH_SQL,
                {
                    "qvec": qvec,
                    "qtext": query,
                    "pool": candidate_pool,
                    "k": k,
                    "rrf_k": _RRF_K,
                },
            )
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                id=r["id"],
                source=r["source"],
                chunk_index=r["chunk_index"],
                content=r["content"],
                vscore=r["vscore"],
                lscore=r["lscore"],
                rrf=r["rrf"],
            )
            for r in rows
        ]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    retriever = Retriever(settings.database_url, get_embedder())
    retriever.ensure_schema()
    return retriever


def chunk_to_dict(c: RetrievedChunk) -> dict:
    return asdict(c)
