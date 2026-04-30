"""
Обработчики команд и сообщений Telegram-бота.
"""

import logging
import time

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from ableton_live_rag.bot.client import RAGClient

logger = logging.getLogger(__name__)

_WELCOME = (
    "Привет! Я помогу разобраться с документацией экосистемы Ableton.\n\n"
    "Просто задай вопрос — и я найду ответ в руководствах к Live 12, Push 3 "
    "и книге «Making Music».\n\n"
    "Команды:\n"
    "/new — начать новый диалог (сбросить историю)\n"
    "/help — показать эту справку"
)

_HELP = (
    "Задавай вопросы об экосистеме Ableton на русском или английском языке.\n\n"
    "Я веду контекст беседы: можешь уточнять и переспрашивать.\n\n"
    "/new — очистить историю и начать заново\n"
    "/help — эта справка"
)

_MIN_EDIT_INTERVAL = 1.2
_MAX_MESSAGE_LEN = 4096

_SOURCE_NAMES = {
    "live_12": "Live 12",
    "push_3": "Push 3",
    "making_music": "Making Music",
}


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""

    lines = ["📚 Источники:"]
    seen: set[str] = set()

    for s in sources:
        raw_source = s.get("metadata", {}).get("source") or ""
        source = _SOURCE_NAMES.get(raw_source, raw_source)
        chapter = s.get("chapter") or ""
        section = s.get("section") or ""
        page = s.get("page_start") or ""
        label = " › ".join(part for part in [source, chapter, section] if part)

        if label in seen:
            continue

        seen.add(label)
        line = f"• {label}"

        if page:
            line += f" (стр. {page})"

        lines.append(line)

    return "\n".join(lines)


def _split_text(text: str, limit: int = _MAX_MESSAGE_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks = []

    while text:
        chunks.append(text[:limit])
        text = text[limit:]

    return chunks


async def _safe_edit(msg, text: str, parse_mode: str | None = None) -> None:
    try:
        await msg.edit_text(text, parse_mode=parse_mode)

    except BadRequest as e:
        if "Message is not modified" in str(e):
            return

        if parse_mode:
            await msg.edit_text(text)
        else:
            raise


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(_WELCOME)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(_HELP)


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    client: RAGClient = context.bot_data["client"]
    session_id: str | None = context.user_data.get("session_id")

    if session_id:
        try:
            await client.delete_session(session_id)
        except Exception:
            pass

        context.user_data["session_id"] = None

    await update.message.reply_text("Начинаем новый диалог. Задавай вопрос!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    client: RAGClient = context.bot_data["client"]
    session_id: str | None = context.user_data.get("session_id")
    user_text = update.message.text or ""

    if not user_text.strip():
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    msg = await update.message.reply_text("⏳")

    text = ""
    sources: list[dict] = []
    last_edit = time.monotonic()

    try:
        async for event_type, content in client.chat(user_text, session_id=session_id):
            if event_type == "session_id":
                context.user_data["session_id"] = content

            elif event_type == "token":
                text += content
                now = time.monotonic()

                if now - last_edit >= _MIN_EDIT_INTERVAL and text:
                    display = (
                        text
                        if len(text) <= _MAX_MESSAGE_LEN
                        else text[-_MAX_MESSAGE_LEN:]
                    )

                    await _safe_edit(msg, display)

                    last_edit = now

            elif event_type == "sources":
                sources = content

        sources_block = _format_sources(sources)
        full = text

        if sources_block:
            full = full.rstrip() + "\n\n" + sources_block

        chunks = _split_text(full)

        await _safe_edit(msg, chunks[0], parse_mode=ParseMode.MARKDOWN)

        for chunk in chunks[1:]:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    except Exception:
        logger.exception(
            "Error handling message from user %s", update.effective_user.id
        )

        await _safe_edit(msg, "Не удалось получить ответ. Попробуй ещё раз.")
