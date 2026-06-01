FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies first for better layer caching
COPY pyproject.toml ./
COPY agentkit ./agentkit
RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "agentkit.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
