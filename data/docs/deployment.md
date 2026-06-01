# Deployment

agentkit ships as a Docker image served by Uvicorn. The docker-compose stack runs two services:
a Postgres database with the pgvector extension, and the FastAPI application.

To run locally, copy .env.example to .env, set ANTHROPIC_API_KEY, then run
`docker compose -f infra/docker-compose.yml up --build`. The application listens on port 8000.

Database migrations for the checkpoint tables run automatically at startup. The pgvector
extension is enabled on first database initialization.
