# Ableton Live RAG

> 🇬🇧 Available in English: [../README.md](../README.md)

RAG-система по корпусу документации Ableton: Live 12 Reference Manual,
Push 3 Manual и *Making Music* Денниса ДеСантиса. Предоставляет CLI,
FastAPI-бэкенд со стримингом по SSE и Telegram-бота.

## Возможности

- Гибридный поиск (BM25 + плотные вектора с reciprocal-rank fusion).
- Подключаемые провайдеры LLM: Ollama, OpenAI или любой OpenAI-совместимый
  vLLM-сервер.
- Guardrails: классификация prompt-injection / off-topic и переписывание
  запросов.
- Семантический кэш ответов и память диалога по сессиям на базе Redis.
- Стриминг ответов (SSE) с указанием источников.

## Требования

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) для локальной разработки.
- Docker и Docker Compose для контейнерного развёртывания.

## Быстрый старт (локально)

```bash
git clone https://github.com/113578/ableton-rag.git
cd ableton-rag
uv sync
cp .env.example .env       # затем отредактируйте значения
```

Загрузите корпус документации:

```bash
uv run dvc repro            # через DVC
# или
bash scripts/download_corpus.sh
```

Постройте индекс и задайте вопрос:

```bash
uv run rag ingest
uv run rag ask "How do I record a MIDI clip?"
```

## Развёртывание через Docker Compose

Файл [`docker-compose.yml`](../docker-compose.yml) поднимает четыре сервиса:

| Сервис    | Образ / Сборка        | Назначение                                     |
|-----------|-----------------------|------------------------------------------------|
| `qdrant`  | `qdrant/qdrant:v1.17` | Векторное хранилище для эмбеддингов.           |
| `redis`   | `redis:8.4.2`         | История диалога + семантический кэш ответов.   |
| `backend` | локальный `Dockerfile`| FastAPI-приложение `/ask`, `/chat`, `/search`. |
| `bot`     | локальный `Dockerfile`| Telegram-бот, обращающийся к `backend`.        |

### 1. Настройка окружения

```bash
cp .env.example .env
```

Заполните как минимум переменные выбранного LLM-провайдера и
`TELEGRAM_BOT_TOKEN` (если нужен бот). `QDRANT_URL` и `REDIS_URL`
переопределяются внутри Compose на адреса сервисов в общей сети, поэтому
их можно не задавать.

### 2. Подготовьте корпус

Контейнер `backend` ожидает PDF-файлы в `./corpus`. Скачайте их на хосте
до старта стека:

```bash
uv run dvc repro
# или
bash scripts/download_corpus.sh
```

### 3. Сборка и запуск стека

```bash
docker compose up -d --build
```

При первом запуске бэкенд автоматически выполняет ingest, если активная
коллекция Qdrant пуста. У healthcheck выставлен `start_period` в 20 минут,
чтобы покрыть разовую загрузку.

## Использование CLI

```bash
uv run rag ingest                                   # построить/обновить индекс
uv run rag ask "What is a rack?" --top-k 10         # ответ с источниками
uv run rag search "Audio routing" --top-k 5         # только векторный поиск
uv run rag stats                                    # статистика коллекции Qdrant
uv run rag serve --host 0.0.0.0 --port 8000         # FastAPI-приложение
uv run rag bot                                      # Telegram-бот
```

## Структура проекта

```
ableton_rag/
  config.py        # Настройки проекта (pydantic-settings)
  llm.py           # Подключение LLM в LlamaIndex
  ingest.py        # Парсинг PDF и извлечение оглавления
  index.py         # Управление индексом Qdrant
  query.py         # Пайплайн запросов со стримингом
  guardrails.py    # Модерация ввода и переписывание запросов
  cli.py           # CLI на Typer
  api/             # FastAPI-приложение
  bot/             # Telegram-бот
experiments/       # Стенд оценки ретриверов / ранжировщиков / генераторов
scripts/           # Скрипты построения индексов и загрузки корпуса
tests/             # Тесты pytest
```

## Документация

- English: [../README.md](../README.md)
- Русский: [README.ru.md](README.ru.md) (этот файл)
