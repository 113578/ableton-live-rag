"""
Telegram bot command and message handlers.
"""

import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from ableton_rag.bot.client import RAGClient

logger = logging.getLogger(__name__)

_WELCOME = (
    "👋 Hey! I'm your Ableton assistant.\n\n"
    "🎛 Ask me anything about *Live 12*, *Push 3*, or *Making Music* "
    "and I'll find the answer straight from the official documentation.\n\n"
    "💬 I remember the conversation, so feel free to follow up and dig deeper.\n\n"
    "⚡️ *Commands*\n"
    "/start — show start message\n"
    "/help — show help message"
)

_HELP = (
    "🎹 *Ableton Assistant — Help*\n\n"
    "Just send a question in plain text and I'll search the docs for you.\n\n"
    "📖 *Sources covered*\n"
    "• Ableton Live 12 Reference Manual\n"
    "• Push 3 Manual\n"
    "• Making Music by Dennis DeSantis\n\n"
    "🧠 *Context*\n"
    "I keep track of the conversation — you can ask follow-up questions "
    "without repeating yourself.\n\n"
    "⚡️ *Commands*\n"
    "/start — show start message\n"
    "/help — show help message"
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

    lines = ["📚 Sources:"]
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
            line += f" (p. {page})"

        lines.append(line)

    return "\n".join(lines)


def _sources_shown_keyboard(msg_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("🙈 Hide sources", callback_data=f"src:hide:{msg_id}")]]
    )


def _sources_hidden_keyboard(msg_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("📚 Sources", callback_data=f"src:show:{msg_id}")]]
    )


def _split_text(text: str, limit: int = _MAX_MESSAGE_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks = []

    while text:
        chunks.append(text[:limit])
        text = text[limit:]

    return chunks


async def _safe_edit(
    msg,
    text: str,
    parse_mode: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    try:
        await msg.edit_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
    except BadRequest as e:
        if "Message is not modified" in str(e):
            return
        if parse_mode:
            await msg.edit_text(text, reply_markup=reply_markup)
        else:
            raise


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Send the welcome message in response to the ``/start`` command.

    Parameters
    ----------
    update : Update
        Incoming update from Telegram.
    context : ContextTypes.DEFAULT_TYPE
        Handler context.
    """
    await update.message.reply_text(_WELCOME, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Send the help message in response to the ``/help`` command.

    Parameters
    ----------
    update : Update
        Incoming update from Telegram.
    context : ContextTypes.DEFAULT_TYPE
        Handler context.
    """
    await update.message.reply_text(_HELP, parse_mode=ParseMode.MARKDOWN)


async def sources_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Toggle source visibility when the inline button is pressed.

    Parameters
    ----------
    update : Update
        Incoming update with a callback query (``src:show:<id>`` or ``src:hide:<id>``).
    context : ContextTypes.DEFAULT_TYPE
        Handler context; the source cache lives in ``context.user_data``.
    """
    query = update.callback_query

    _, action, msg_id = query.data.split(":", 2)
    cache: dict = context.user_data.get("sources_cache", {}).get(msg_id)

    if not cache:
        await query.answer("Sources are no longer available.", show_alert=True)
        return

    await query.answer()

    if action == "show":
        full = cache["answer"].rstrip() + "\n\n" + cache["sources"]

        await _safe_edit(
            query.message,
            full,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_sources_shown_keyboard(msg_id),
        )
    else:
        await _safe_edit(
            query.message,
            cache["answer"],
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_sources_hidden_keyboard(msg_id),
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Stream the RAG answer for an incoming user message and show its sources.

    Parameters
    ----------
    update : Update
        Incoming update with the user's text message.
    context : ContextTypes.DEFAULT_TYPE
        Handler context; ``session_id`` is stored in ``context.user_data``.
    """
    client: RAGClient = context.bot_data["client"]
    session_id: str | None = context.user_data.get("session_id")
    user_text = update.message.text or ""

    if not user_text.strip():
        return

    logger.info(
        "User %s [session=%s]: %r", update.effective_user.id, session_id, user_text[:80]
    )

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
        msg_id = str(msg.message_id)

        if sources_block:
            context.user_data.setdefault("sources_cache", {})[msg_id] = {
                "answer": text,
                "sources": sources_block,
            }
            keyboard = _sources_hidden_keyboard(msg_id)
        else:
            keyboard = None

        chunks = _split_text(text)

        await _safe_edit(
            msg,
            chunks[0],
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
        )

        for chunk in chunks[1:]:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    except Exception:
        logger.exception(
            "Error handling message from user %s", update.effective_user.id
        )
        await _safe_edit(msg, "Failed to get a response. Please try again.")
