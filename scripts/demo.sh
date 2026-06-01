#!/usr/bin/env bash
# Offline manual tour: runs every feature against a real Postgres, deterministically, no API key.
#
#   ./scripts/demo.sh
#
# It brings up Postgres, ingests the eval corpus, then runs each integration check and the eval
# gate, printing what each one proves.
set -euo pipefail
cd "$(dirname "$0")/.."

export DATABASE_URL="${DATABASE_URL:-postgresql://agentkit:agentkit@localhost:5432/agentkit}"
PY=./venv/bin/python
COMPOSE="docker compose -f infra/docker-compose.yml"

echo "== bringing up Postgres =="
$COMPOSE up -d postgres
until docker exec infra-postgres-1 pg_isready -U agentkit -d agentkit >/dev/null 2>&1; do sleep 1; done
echo "postgres ready"

echo; echo "== ingest corpus =="
$PY -m agentkit.retrieval.ingest agentkit/eval/corpus --reset

run() { echo; echo "== $1 =="; "${@:2}"; }

run "Stage 1 - checkpointing: state survives a pool restart"        $PY tests/manual_checkpoint_check.py
run "Stage 2 - RAG: citations on in-base, no-answer on out-of-base" $PY tests/manual_rag_check.py
run "Stage 3 - MCP tools over stdio (calculator + chunk_stats)"      $PY tests/manual_mcp_check.py
run "Stage 4 - HITL: pause, survive restart, resume, side effect"   $PY tests/manual_hitl_check.py
run "Stage 6 - observability: end-to-end decision trace"            $PY tests/manual_obs_check.py
run "Stage 5 - eval harness + gate"                                 $PY -m agentkit.eval.runner

echo; echo "== done. (stop Postgres with: $COMPOSE down) =="
