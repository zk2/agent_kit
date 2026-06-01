# agentkit overview

agentkit is a reusable agent platform. It provides orchestration with LangGraph, tool calling,
retrieval-augmented generation, evaluation, and deployment with Docker.

The graph state is checkpointed in Postgres. Each run is keyed by a thread_id, so a conversation
can be resumed after a restart and its intermediate decisions can be inspected.

Retrieval uses a hybrid of vector search over pgvector and lexical full-text search, fused with
reciprocal rank fusion. When retrieval confidence is low, the agent returns a no-answer response
instead of guessing.
