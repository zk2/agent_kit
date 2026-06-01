"""Manual integration check for Stage 3: real MCP server over stdio, no LLM/API key needed.

Loads the tools exactly as the app does (spawning `python -m agentkit.tools.server`), then:
  - calls `calculator` (deterministic, no DB)
  - calls `chunk_stats` (structured query against Postgres; requires ingested docs)

Run with Postgres up and docs ingested:
    python -m agentkit.retrieval.ingest data/docs --reset
    DATABASE_URL=postgresql://agentkit:agentkit@localhost:5432/agentkit \
        python tests/manual_mcp_check.py
"""

import asyncio

from agentkit.tools.client import load_mcp_tools


def _text(result) -> str:
    """MCP tools return a list of content blocks; flatten to their text."""
    if isinstance(result, list):
        return "".join(b.get("text", "") for b in result if isinstance(b, dict))
    return str(result)


async def main() -> None:
    tools = {t.name: t for t in await load_mcp_tools()}
    print("loaded tools:", sorted(tools))
    assert {"calculator", "chunk_stats"} <= set(tools)

    calc = _text(await tools["calculator"].ainvoke({"expression": "2 + 3 * (4 - 1)"}))
    print("calculator('2 + 3 * (4 - 1)') =", calc)
    assert calc.strip() == "11"

    stats = _text(await tools["chunk_stats"].ainvoke({}))
    print("chunk_stats() =", stats)
    assert "agentkit_overview.md" in stats

    print("OK: MCP tools loaded over stdio and executed (calc + DB-backed stats)")


if __name__ == "__main__":
    asyncio.run(main())
