"""
Telegram bot entry point.
"""

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from ableton_rag.bot.client import RAGClient
from ableton_rag.bot.handlers import (
    handle_message,
    help_command,
    sources_callback,
    start_command,
)
from ableton_rag.config import get_logger

logger = get_logger(__name__)


def build_app(token: str, api_base_url: str) -> Application:
    """
    Build and configure the Telegram ``Application``.

    Parameters
    ----------
    token : str
        Telegram bot token.
    api_base_url : str
        URL of the FastAPI application.

    Returns
    -------
    Application
        Application with all handlers registered.
    """

    app = Application.builder().token(token).build()
    app.bot_data["client"] = RAGClient(api_base_url)

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(
        CallbackQueryHandler(sources_callback, pattern=r"^src:(show|hide):")
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app


def run(token: str, api_base_url: str) -> None:
    """
    Start the bot in long-polling mode.

    Parameters
    ----------
    token : str
        Telegram bot token.
    api_base_url : str
        Base URL of the FastAPI application.
    """

    logger.info("Starting Telegram bot (API: %s)...", api_base_url)

    app = build_app(token, api_base_url)
    app.run_polling(drop_pending_updates=True)
