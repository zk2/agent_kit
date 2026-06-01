# Deployment

agentkit ships as a docker image served by uvicorn. docker compose runs a postgres database with
the pgvector extension together with the fastapi application. the application listens on port 8000.
checkpoint tables are created automatically at startup.
