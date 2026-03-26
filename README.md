# Ableton Live RAG

RAG-система по документации Ableton Live 12.

## Установка

```bash
git clone https://github.com/113578/ableton-live-rag.git
cd ableton-live-rag
uv sync
```

## Конфигурация

Создаём `.env` по примеру `.env.example`:
```bash
cp .env.example .env
```

Система поддерживает двух провайдеров LLM:
| Провайдер | `LLM_PROVIDER` | Описание |
|-----------|---------------|----------|
| Ollama    | `ollama`      | Локальная модель (по умолчанию) |
| vLLM      | `vllm`        | Любой OpenAI-совместимый API |

## Загрузка данных

Документация скачивается через DVC:
```bash
uv run dvc repro
```

Или вручную:
```bash
bash scripts/download_corpus.sh
```

## Использование

### 1. Индексировать документацию

```bash
uv run rag ingest
```

### 2. Задать вопрос

```bash
uv run rag ask "How do I record a MIDI clip?"
```

Можно указать количество фрагментов контекста:
```bash
uv run rag ask "What is a rack?" --top-k 10
```

### 3. Использовать только векторный поиск

```bash
uv run rag search "Audio routing" --top-k 5
```

### 4. Получить статистику Qdrant-коллекции

```bash
uv run rag stats
```

## Структура проекта

```
ableton_live_rag/
  config.py        # Конфигурация проекта
  llm.py           # Настройка LlamaIndex
  ingest.py        # Парсинг PDF и извлечение TOC
  index.py         # Управление Qdrant-индексом
  query.py         # Пайплайн запросов со стримингом
  cli.py           # CLI (Typer)
```
