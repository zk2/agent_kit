"""MCP server exposing agentkit's tools.

Run standalone over stdio (how the agent loads it):
    python -m agentkit.tools.server

Tools:
  - calculator : evaluate a basic arithmetic expression (deterministic, no DB)
  - chunk_stats: number of indexed chunks per source document (structured DB query)
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from agentkit.tools.calc import CalcError, safe_eval

mcp = FastMCP("agentkit")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. "2 + 3 * (4 - 1)".

    Supports + - * / // % ** and parentheses. Returns the numeric result as a string.
    """
    try:
        return str(safe_eval(expression))
    except CalcError as exc:
        return f"error: {exc}"


@mcp.tool()
def save_note(text: str) -> str:
    """Persist a note to the database. This has a side effect and requires human approval."""
    from agentkit.retrieval.store import get_retriever

    with get_retriever().pool.connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS notes (id bigserial PRIMARY KEY, text text)")
        row = conn.execute(
            "INSERT INTO notes (text) VALUES (%s) RETURNING id", (text,)
        ).fetchone()
    return f"saved note #{row['id']}"


@mcp.tool()
def chunk_stats() -> dict:
    """Return the number of indexed retrieval chunks per source document."""
    # Imported lazily so the server starts (and `calculator` works) without a DB.
    from agentkit.retrieval.store import get_retriever

    with get_retriever().pool.connection() as conn:
        rows = conn.execute(
            "SELECT source, count(*) AS chunks FROM documents GROUP BY source ORDER BY source"
        ).fetchall()
    return {r["source"]: r["chunks"] for r in rows}


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
