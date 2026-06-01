# Manual walkthrough

Three ways to see every feature for yourself. The first two need **no API key**.

---

## A. One-command offline tour (no API key)

Runs every feature against a real Postgres, deterministically (a stub model stands in for Claude):

```bash
chmod +x scripts/demo.sh
./scripts/demo.sh
```

It brings up Postgres, ingests the corpus, then runs each integration check and the eval gate,
printing what each proves: checkpointing/restart, RAG citations + no-answer, MCP tools, HITL
pause/resume with a real DB side effect, the end-to-end trace, and the eval gate.

Run the test suite too (fully offline, no DB):

```bash
pip install -e ".[dev]"
ruff check . && pytest -q          # 35 tests
```

---

## B. Poke the HTTP API yourself (no API key)

Serve the API with the deterministic stub model so you can curl every endpoint offline.

```bash
docker compose -f infra/docker-compose.yml up -d postgres
export DATABASE_URL=postgresql://agentkit:agentkit@localhost:5432/agentkit
python -m agentkit.retrieval.ingest agentkit/eval/corpus --reset

APP_LLM=stub uvicorn agentkit.api.app:app --port 8000
```

In another terminal:

```bash
# RAG: answer with citations, plus latency/cost
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"How does agentkit combine vector and full text search?"}' | python -m json.tool

# No-answer fallback (out of the knowledge base)
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"What were avocado prices in Lisbon last quarter?"}' | python -m json.tool

# Chitchat (intent=chitchat, skips retrieval)
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Hello there, thanks!"}' | python -m json.tool

# Human-in-the-loop: a side-effecting tool pauses for approval
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Save a note about agentkit deployment with uvicorn on port 8000","thread_id":"demo"}' \
  | python -m json.tool        # -> status: awaiting_review, interrupt.pending_tool_calls

# Approve it (try reject, or edit with corrected tool_calls, too)
curl -s localhost:8000/runs/demo/resume -H 'content-type: application/json' \
  -d '{"action":"approve"}' | python -m json.tool      # -> status: done; note written to DB

# Inspect the run: status, full per-node decision trace, latency, cost
curl -s localhost:8000/runs/demo | python -m json.tool
```

What to look for: `citations`, `status` (`done` / `awaiting_review`), the `interrupt` payload,
and `trace` (one span per node with its decision and duration).

---

## C. The real thing (with an Anthropic API key)

```bash
cp .env.example .env            # set ANTHROPIC_API_KEY=..., leave APP_LLM=real
docker compose -f infra/docker-compose.yml up --build
python -m agentkit.retrieval.ingest data/docs --reset
```

Then the same curl calls as in B, but answers come from Claude. Also:

```bash
# Evaluate against the real model instead of the stub
EVAL_LLM=real python -m agentkit.eval.runner

# Run the same scenario on the second engine (Claude Agent SDK)
pip install -e ".[agent-sdk]"
ORCHESTRATOR=claude_agent_sdk uvicorn agentkit.api.app:app --port 8000
```

---

Stop everything: `docker compose -f infra/docker-compose.yml down`.
