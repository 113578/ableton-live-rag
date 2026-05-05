# Ableton Live RAG

> 🇷🇺 Доступно на русском: [docs/README.ru.md](docs/README.ru.md)

A retrieval-augmented generation (RAG) system over the Ableton documentation
corpus: Live 12 Reference Manual, Push 3 Manual, and *Making Music* by
Dennis DeSantis. Exposes a CLI, a FastAPI backend with SSE streaming, and a
Telegram bot.

## Features

- Hybrid retrieval (BM25 + dense vectors with reciprocal-rank fusion).
- Pluggable LLM backend: Ollama, OpenAI, or any OpenAI-compatible vLLM server.
- Guardrails: prompt-injection / off-topic classification and query rewriting.
- Semantic response cache and per-session chat memory backed by Redis.
- Streaming answers (SSE) with cited sources.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for local development.
- Docker and Docker Compose for the containerised deployment.

## Quick start (local)

```bash
git clone https://github.com/113578/ableton-rag.git
cd ableton-rag
uv sync
cp .env.example .env       # then edit values
```

Download the documentation corpus:

```bash
uv run dvc repro            # via DVC
# or
bash scripts/download_corpus.sh
```

Build the index and ask a question:

```bash
uv run rag ingest
uv run rag ask "How do I record a MIDI clip?"
```

## Deployment with Docker Compose

The bundled [`docker-compose.yml`](docker-compose.yml) launches four services:

| Service   | Image / Build         | Purpose                                      |
|-----------|-----------------------|----------------------------------------------|
| `qdrant`  | `qdrant/qdrant:v1.17` | Vector store for embeddings.                 |
| `redis`   | `redis:8.4.2`         | Chat history + semantic response cache.      |
| `backend` | local `Dockerfile`    | FastAPI app exposing `/ask`, `/chat`, `/search`. |
| `bot`     | local `Dockerfile`    | Telegram bot that talks to `backend`.        |

### 1. Configure environment

```bash
cp .env.example .env
```

Populate at least the variables required by your chosen LLM provider plus
`TELEGRAM_BOT_TOKEN` (only if you want the bot). `QDRANT_URL` and
`REDIS_URL` are overridden inside Compose to point at the in-network
services, so you can leave them unset.

### 2. Provide the corpus

The backend container expects PDFs under `./corpus`. Fetch them on the host
before starting the stack:

```bash
uv run dvc repro
# or
bash scripts/download_corpus.sh
```

### 3. Build and start the stack

```bash
docker compose up -d --build
```

On the first start the backend automatically runs the ingest pipeline if
the active Qdrant collection is empty. The healthcheck has a 20-minute
`start_period` to accommodate that one-time work.

## CLI usage

```bash
uv run rag ingest                                   # build/refresh the index
uv run rag ask "What is a rack?" --top-k 10         # answer with sources
uv run rag search "Audio routing" --top-k 5         # vector search only
uv run rag stats                                    # Qdrant collection info
uv run rag serve --host 0.0.0.0 --port 8000         # FastAPI app
uv run rag bot                                      # Telegram bot
```

## Project layout

```
ableton_rag/
  config.py        # Project settings (pydantic-settings)
  llm.py           # LlamaIndex provider wiring
  ingest.py        # PDF parsing and TOC extraction
  index.py         # Qdrant index management
  query.py         # Query pipeline with streaming
  guardrails.py    # Input moderation and query rewriting
  cli.py           # Typer CLI
  api/             # FastAPI app
  bot/             # Telegram bot
experiments/       # Retriever / reranker / generator evaluation harness
scripts/           # Index-building helpers and corpus download
tests/             # pytest suite
```

## Documentation

- English: [README.md](README.md) (this file)
- Русский: [docs/README.ru.md](docs/README.ru.md)
