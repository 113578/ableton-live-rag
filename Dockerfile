FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install uv==0.11.8

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --extra backend --extra bot --no-install-project

COPY . .

RUN uv sync --locked --extra backend --extra bot

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
