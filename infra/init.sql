-- Enabled at first DB init. pgvector is used by the retrieval layer (Stage 2).
-- LangGraph checkpoint tables are created at app startup via AsyncPostgresSaver.setup().
CREATE EXTENSION IF NOT EXISTS vector;
