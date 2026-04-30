"""
Точка входа Telegram-бота.
"""

import logging

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from ableton_live_rag.bot.client import RAGClient
from ableton_live_rag.bot.handlers import (
    handle_message,
    help_command,
    new_command,
    start_command,
)


def build_app(token: str, api_base_url: str) -> Application:
    """
    Сборка и настройка Telegram Application.

    Parameters
    ----------
    token : str
        Токен Telegram-бота.
    api_base_url : str
        URL FastAPI-приложения.

    Returns
    -------
    Application
        Готовое приложение с зарегистрированными обработчиками.
    """

    app = Application.builder().token(token).build()
    app.bot_data["client"] = RAGClient(api_base_url)

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("new", new_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app


def run(token: str, api_base_url: str) -> None:
    """
    Запуск бота в режиме long polling.

    Parameters
    ----------
    token : str
        Токен Telegram-бота.
    api_base_url : str
        Базовый URL FastAPI-приложения.
    """

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )

    app = build_app(token, api_base_url)
    app.run_polling(drop_pending_updates=True)
