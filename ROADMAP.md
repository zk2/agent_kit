# Roadmap

Each stage is a self-contained vertical slice: at the end the system runs and does something
useful. We move top to bottom. Every stage has a Definition of Done (DoD); until it is met, the
stage is not complete. All stages are implemented.

---

## Stage 0 - Repository scaffold

Basic hygiene so the rest stays focused.

- `git init`, `.gitignore` (venv, `.env`, `__pycache__`)
- `pyproject.toml` / dependencies: fastapi, uvicorn, langgraph, langchain-anthropic, psycopg,
  pydantic, pytest, ruff
- `.env.example` with `ANTHROPIC_API_KEY`, `DATABASE_URL`
- `infra/docker-compose.yml`: `postgres` service (image `pgvector/pgvector:pg16`)
- `ruff` + `pytest` configured, an empty test passes

DoD: `docker compose up postgres` brings up the DB; `pytest` is green.

---

## Stage 1 - Agent skeleton (LangGraph + FastAPI + checkpointing)

Core of the system: state / checkpointing / resume.

- `core/state.py`: typed `State` (message, intent, context, tool_outputs, decisions, errors,
  response)
- `core/graph/`: minimal `classify -> respond` graph on LangGraph
- `core/checkpoint.py`: `PostgresSaver`, checkpoint tables created by a migration
- `core/llm.py`: Anthropic (Claude) wrapper, retry, token/cost accounting
- `api/`: FastAPI with `POST /chat` (new run) and `GET /runs/{id}` (run state by `thread_id`)
- `Dockerfile` for the app, app added to `docker-compose`

DoD: a request to `/chat` runs the graph, state is saved in Postgres, `/runs/{id}` returns the
history; restarting the container does not lose the run.

---

## Stage 2 - RAG slice (retrieval + citations + no-answer)

The meatiest part of the requirements.

- `retrieval/chunking.py`: document splitting (with overlap, source metadata)
- `retrieval/index.py`: embeddings -> `pgvector`; ingest script for a folder of documents
- `retrieval/hybrid.py`: hybrid of vector search and BM25 / `ts_rank`, result fusion
- `retrieval/citations.py`: the answer references specific chunks/sources
- no-answer fallback: when retrieval score is low the agent declines to answer
- graph nodes: `retrieve -> synthesize -> validate`, with a `no-answer` branch

DoD: a question over indexed documents returns an answer with citations; a question outside the
base returns a correct refusal rather than a hallucination.

---

## Stage 3 - Tools via MCP

- `tools/server.py`: an MCP server with 1-2 tools (e.g. structured DB lookup, calculator)
- a `tools` node in the graph: tool selection, argument validation, error handling, iteration cap
  (anti-looping)
- `need_tool?` routing after `plan`

DoD: the agent selects the correct tool, passes valid arguments, survives a tool error
(retry/degrade), and does not loop forever.

---

## Stage 4 - Human-in-the-loop

- `interrupt()` in the graph at a "risky" step (e.g. before a side-effecting action)
- `POST /runs/{id}/resume`: an operator approves/edits, the graph continues from the checkpoint
- run statuses: `running | awaiting_review | done | failed`

DoD: the run stops at review and waits; after `/resume` it continues exactly from the interrupt,
without losing state.

---

## Stage 5 - Eval harness + gating

- `eval/dataset/`: a golden dataset (questions, expected answer properties, citation requirements,
  no-answer cases, regressions)
- `eval/runner.py`: run the dataset through the agent
- metrics: retrieval quality, citation grounding, answer correctness, hallucination rate, tool-use
  correctness, no-answer accuracy
- `eval/report.py`: comparison against the previous run (regressions)
- gate: GitHub Actions blocks merge if metrics drop or an old failure returns

DoD: `python -m eval.runner` produces a metrics report; CI fails on a regression.

---

## Stage 6 - Observability + second engine

- `obs/`: structured logging of agent decisions + tracing (OpenTelemetry or LangSmith)
- per-request latency and cost log/dashboard
- `core/orchestrator.py`: extract an engine interface and add a second implementation on the
  Claude Agent SDK for the same scenario - to show the "right call per client"

DoD: every agent decision is traced end to end; the same scenario runs on the second engine
through the shared interface.

---

## Deliberately out of scope (scope guard)

- Real cloud deployment (ECS/Cloud Run) - out of scope; everything runs locally in docker-compose.
- Fine-tuning (LoRA/PEFT) - a separate track, does not block the platform.
- Production-grade auth/multi-tenancy - simplified to an API key.
- Frontend - API + curl/scripts only.

These are candidates for a "phase 2" once the core works.
