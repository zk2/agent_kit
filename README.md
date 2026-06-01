# agentkit

A reusable agent platform: orchestration, tool calling, retrieval, evaluation, deployment,
monitoring and connectors - built along production lines.

The focus is the **hard part**, not "call an LLM": state management, recovery, evals,
observability, security boundaries, and a simple architecture another engineer can extend.

---

## Product scenario

**An assistant over a private document base.** Answers questions with citations from the
sources, can call a tool (DB lookup, calculation), and when context is insufficient it falls
back to `no-answer` or to human review. A single vertical slice covers RAG + tools + HITL +
eval at once.

---

## Capabilities

| Area | Where in the project |
|---|---|
| LangGraph: state, checkpointing, multi-agent, human-in-the-loop | `core/graph/`, `PostgresSaver`, `interrupt()` |
| Python / FastAPI / Postgres / Docker | `api/`, `infra/docker-compose.yml` |
| Agent end-to-end: orchestration -> tools -> retrieval -> eval | the whole vertical slice |
| RAG: hybrid retrieval, chunking, citation grounding, no-answer fallback | `retrieval/` |
| Evaluation: harness, release gating | `eval/`, CI gate |
| CI/CD, observability | GitHub Actions, `obs/` (tracing) |
| MCP tools | `tools/` exposed via an MCP server |
| Second orchestration engine | `Orchestrator` abstraction, optional Claude Agent SDK |
| Postgres extensions | `pgvector` for retrieval |

---

## Stack

- **Language:** Python 3.11
- **LLM:** Anthropic Claude (`claude-opus-4-8` / `claude-sonnet-4-6` / `claude-haiku-4-5`)
- **Orchestration:** LangGraph (Postgres checkpointer)
- **API:** FastAPI + Uvicorn
- **State:** Postgres + `pgvector` (retrieval and checkpoints in one DB)
- **Tools:** MCP server
- **Eval:** custom harness over a golden dataset
- **Deploy:** `docker-compose` (app + postgres), local
- **CI:** GitHub Actions (lint + tests + eval gate)

---

## Architecture

```
agentkit/
  api/            FastAPI: /chat, /runs/{id}, /runs/{id}/resume (HITL)
  core/
    graph/        LangGraph agent graph (nodes + routing)
    state.py      typed State (TypedDict / Pydantic)
    checkpoint.py PostgresSaver
    llm.py        Anthropic wrapper + retry / cost accounting
    orchestrator.py  engine abstraction (LangGraph | Agent SDK)
  retrieval/      chunking + hybrid (pgvector + BM25) + citations + no-answer
  tools/          MCP server with tools
  eval/           golden dataset + runner + metrics
  obs/            structured logging + tracing
  infra/          docker-compose, Dockerfile, DB migrations
  tests/
```

### Graph execution flow

```
classify -> plan -> retrieve -> (need_tool? -> tools) -> synthesize -> validate
                       |                                              |
                  no-answer <---------------------------------------/
                       |
                  human review (interrupt) --- resume --> synthesize
```

Each node reads/writes the typed `State`. Between runs the state is persisted by the
checkpointer in Postgres - hence recovery, resume and inspection of intermediate decisions.

---

## Principles

1. **Vertical slices.** Each roadmap stage leaves a working system, not a half-built one.
2. **Eval before shipping.** Releases are gated by metrics, not "the output looks fine".
3. **Design for failure.** Timeouts, retries, no-answer, explicit tool errors.
4. **Observability.** Every agent decision is traceable in logs/traces.
5. **Simple for the next engineer.** Typed state, documentation, minimal magic.

---

## Quick start

```bash
cp .env.example .env          # ANTHROPIC_API_KEY=...
docker compose -f infra/docker-compose.yml up --build -d postgres

# index some documents (chunk -> embed -> pgvector)
python -m agentkit.retrieval.ingest data/docs --reset

docker compose -f infra/docker-compose.yml up --build app
curl localhost:8000/chat \
  -d '{"message": "Where does agentkit store retrieval chunks?"}' \
  -H 'content-type: application/json'
# -> answer + citations; out-of-base questions return a no-answer fallback
```

Embeddings default to the offline `hashing` provider (works with no extra deps or network).
For real semantic retrieval: `pip install ".[embed]"` and set `EMBEDDING_PROVIDER=fastembed`.

## Evaluation

A golden-dataset harness runs cases through the real graph and retrieval, scores them, and gates
releases on metric regressions vs a committed baseline:

```bash
python -m agentkit.eval.runner                    # run + gate (exit != 0 on regression)
python -m agentkit.eval.runner --update-baseline  # refresh agentkit/eval/baseline.json
```

It uses a deterministic stub model by default (no API key, hermetic), so CI can gate it; set
`EVAL_LLM=real` to evaluate against the real model. Metrics: retrieval recall, citation grounding,
no-answer accuracy, tool-use correctness, answer correctness, intent accuracy, pass rate. CI
(`.github/workflows/ci.yml`) runs lint + tests + the eval gate against a Postgres service.

## Observability & engines

Every node execution records a span (decision, latency, token cost) into the run's `trace`, which
is checkpointed and returned by `GET /runs/{id}` - so any agent decision is observable end to end.
Spans are also emitted as structured logs; set `LANGCHAIN_TRACING_V2=true` to additionally export
full traces to LangSmith.

The API drives runs through an `Orchestrator` interface, so the engine is swappable:

- `LangGraphOrchestrator` (default) - the in-house graph: state, checkpointing, HITL, tracing.
- `ClaudeAgentSDKOrchestrator` - the same RAG+tools scenario on the Claude Agent SDK
  (`pip install '.[agent-sdk]'`, set `ORCHESTRATOR=claude_agent_sdk`).

Development plan - see [ROADMAP.md](./ROADMAP.md).
