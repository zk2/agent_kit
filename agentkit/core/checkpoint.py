"""Postgres-backed checkpointer lifecycle.

The checkpointer is what gives us recoverability and resume: every node transition is persisted
to Postgres keyed by `thread_id`, so a run survives a process restart and its intermediate
decisions can be inspected.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from agentkit.config import settings

# AsyncPostgresSaver requires autocommit connections with the dict row factory.
_CONNECTION_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Open a connection pool, ensure checkpoint tables exist, yield a saver."""
    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        max_size=10,
        open=False,
        kwargs=_CONNECTION_KWARGS,
    )
    await pool.open()
    try:
        # AsyncPostgresSaver accepts a pool at runtime; its stub types only a single connection.
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()  # idempotent: creates checkpoint tables if missing
        yield saver
    finally:
        await pool.close()
