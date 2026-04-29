FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-cache

COPY ableton_live_rag/ ./ableton_live_rag/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
