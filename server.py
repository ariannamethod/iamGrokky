import asyncio
import json
import logging
import os
import re
import traceback
from datetime import datetime
import tempfile
from urllib.parse import urlparse
from ipaddress import ip_address
import time

import httpx
try:  # pragma: no cover - used only with aiogram installed
    from aiogram import Bot, Dispatcher, types
    from aiogram.enums import ChatAction
    from aiogram.filters import Command
    from aiogram.types import Message
    from aiogram.exceptions import TelegramAPIError
except ImportError:  # pragma: no cover - fallback for tests
    class Bot:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        async def send_message(self, *args, **kwargs):  # pragma: no cover - stub
            pass

        async def get_file(self, *args, **kwargs):  # pragma: no cover - stub
            pass

    class Dispatcher:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class types:  # type: ignore
        pass

    class ChatAction:  # type: ignore
        pass

    class Command:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class TelegramAPIError(Exception):
        pass

    class Message:  # type: ignore
        async def reply(self, *args, **kwargs):  # pragma: no cover - stub
            pass

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, JSONResponse, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from utils.metrics import (
    REQUEST_LATENCY,
    record_tokens,
    record_error,
    record_command_usage,
    record_data_transfer,
)

from utils.dayandnight import day_and_night_task
from utils.mirror import mirror_task
from utils.prompt import get_chaos_response, build_system_prompt
from utils.repo_monitor import monitor_repository
from utils.vision import analyze_image
from utils.plugins.coder import interpret_code
from utils.grok_chat_manager import GrokChatManager
from utils.memory_manager import ImprovedMemoryManager
from importlib import import_module

# Импортируем наш новый движок
from utils.vector_engine import VectorGrokkyEngine
from utils.hybrid_engine import HybridGrokkyEngine
from utils.plugins import load_plugins
from utils.dynamic_weights import DynamicWeights
from utils.rl_trainer import RLTrainer, log_feedback

# Special command handler from the playful 42 utility
from utils.context_neural_processor import parse_and_store_file

# Настройки логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Переменные окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN not set; using dummy token for tests")
    TELEGRAM_BOT_TOKEN = "DUMMY"

# Используем правильный адрес для Railway
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "https://grokky-production.up.railway.app")
WEBHOOK_PATH = "/webhook"  # Просто /webhook без токена
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.getenv("PORT", 8080))
CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

# Ключ OpenAI для распознавания речи
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_MODEL = os.getenv("NEWS_MODEL", "gpt-4o")
URL_RE = re.compile(r"https?://\S+")
MAX_URL_LENGTH = int(os.getenv("MAX_URL_LENGTH", "2048"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))
BANNED_DOMAINS = {
    d.strip().lower()
    for d in os.getenv("BANNED_DOMAINS", "").split(",")
    if d.strip()
}
MAX_WEBHOOK_BODY_SIZE = int(os.getenv("MAX_WEBHOOK_BODY_SIZE", "100000"))

# Проверка ключей API
XAI_API_KEY = os.getenv("XAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
API_KEY = os.getenv("API_KEY")

logger.info(f"Запуск бота с webhook на {WEBHOOK_URL}")
logger.info("Токен бота: [MASKED]")
logger.info("XAI API ключ: %s", "Установлен" if XAI_API_KEY else "НЕ УСТАНОВЛЕН")
logger.info("OpenAI API ключ: %s", "Установлен" if OPENAI_API_KEY else "НЕ УСТАНОВЛЕН")
logger.info(
    "Pinecone API ключ: %s",
    "Установлен" if PINECONE_API_KEY else "НЕ УСТАНОВЛЕН",
)
logger.info("Pinecone индекс: %s", PINECONE_INDEX or "НЕ УСТАНОВЛЕН")
logger.info("API key auth: %s", "ENABLED" if API_KEY else "DISABLED")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Plugin registry
PLUGINS = load_plugins()
for _plugin in PLUGINS:
    for _cmd, _func in _plugin.commands.items():
        async def _wrapper(message: Message, _func=_func):
            args = message.text.split(" ", 1)[1] if message.text and " " in message.text else ""
            result = await _func(args)
            await reply_split(message, result)
        dp.message.register(_wrapper, Command(_cmd))
handle = import_module("utils.plugins.42").handle

# Переменные с информацией о боте, заполняются при старте
BOT_ID = None
BOT_USERNAME = ""

# Инициализация движка выполняется только при явном разрешении
engine = None
if os.getenv("ENABLE_VECTOR_ENGINE") == "1":
    try:
        engine = VectorGrokkyEngine()
        logger.info("VectorGrokkyEngine инициализирован успешно")
    except (RuntimeError, OSError, ValueError) as e:  # pragma: no cover - network
        logger.error(f"Ошибка при инициализации VectorGrokkyEngine: {e}")
        logger.error(traceback.format_exc())
else:
    try:
        engine = HybridGrokkyEngine()
        logger.info("HybridGrokkyEngine инициализирован успешно")
    except (RuntimeError, OSError, ValueError) as e:  # pragma: no cover - network
        logger.error(f"Ошибка при инициализации HybridGrokkyEngine: {e}")
        logger.error(traceback.format_exc())

dynamic_weights = DynamicWeights([0.5, 0.5])
rl_trainer = RLTrainer(dynamic_weights)
chat_manager: GrokChatManager | None = None
memory_manager: ImprovedMemoryManager | None = None


async def rl_trainer_task() -> None:
    while True:
        rl_trainer.train()
        await asyncio.sleep(60)

# Обработка голосовых сообщений
VOICE_ENABLED = {}
# Chats currently in coder mode
CODER_MODE: dict[int, bool] = {}
# Pending long coder outputs waiting for user choice
CODER_OUTPUT: dict[tuple[int, int], str] = {}
# Preferred language per chat
CHAT_LANG: dict[int, str] = {}

# Background tasks to be cancelled on shutdown
background_tasks: list[asyncio.Task] = []


async def verify_api_key(request: Request) -> None:
    """FastAPI dependency to validate the API key header."""
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def detect_language(text: str) -> str:
    """Very small heuristic language detector."""
    if re.search("[а-яА-Я]", text):
        return "ru"
    if re.search("[äöüßÄÖÜ]", text):
        return "de"
    if re.search("[ñáéíóúÑÁÉÍÓÚ]", text):
        return "es"
    if re.search("[éàèùâêîôûçœÉÀÈÙÂÊÎÔÛÇŒ]", text):
        return "fr"
    if re.search("[a-zA-Z]", text):
        return "en"
    return "en"


async def synth_voice(text: str, lang: str = "ru") -> bytes:
    """Synthesize speech using OpenAI's TTS with a male voice."""
    if not OPENAI_API_KEY:
        return b""

    voice_map = {
        "ru": "onyx",
        "en": "alloy",
        "es": "coral",
        "de": "marco",
        "fr": "fable",
    }

    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice_map.get(lang, "alloy"),
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            r.raise_for_status()
            return r.content
        except httpx.HTTPError as e:
            logger.error("Ошибка синтеза речи: %s", e)
            return b""


async def transcribe_voice(file_id: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    file = await bot.get_file(file_id)
    url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file.file_path}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        audio = resp.content
        files = {"file": ("voice.ogg", audio, "application/ogg")}
        data = {"model": "whisper-1"}
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        try:
            r = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("text", "")
        except httpx.HTTPError as e:
            logger.error("Ошибка расшифровки голоса: %s", e)
            return ""


async def summarize_link(url: str, extra: int = 2) -> str:
    """Read the link and a few extra articles from the site via OpenAI tools."""
    if not OPENAI_API_KEY:
        return ""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = (
        f"Прочитай {url} и ещё {extra} материалов на том же сайте. "
        "Кратко опиши общую тему ресурса и главное из статьи. Ответь на русском."
    )
    payload = {
        "model": NEWS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {"type": "function", "function": {"name": "browser.search"}},
            {"type": "function", "function": {"name": "browser.get"}},
        ],
        "tool_choice": "auto",
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("choices", [{}])[0].get("message", {})
            return message.get("content", "").strip()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.error("Ошибка получения статьи: %s", e)
            return ""


async def reply_split(message: Message, text: str) -> None:
    """Reply, splitting long text into multiple Telegram messages.

    Telegram messages are limited to 4096 characters. The previous
    implementation only supported splitting into two parts which caused
    failures when the second part still exceeded the limit. This version
    handles arbitrary length by chunking the message and numbering each
    part.
    """

    limit = 4096
    # Reserve space for the prefix added to each chunk.
    chunk_size = limit - 100
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        if total > 1:
            prefix = f"🌀 Ответ в {total} частях. Часть {idx}/{total}:\n"
            full_text = prefix + chunk
        else:
            full_text = chunk
        if idx == 1:
            await message.reply(full_text)
        else:
            await bot.send_message(message.chat.id, full_text)


@dp.message(Command("voiceon"))
async def cmd_voiceon(message: Message):
    VOICE_ENABLED[message.chat.id] = True
    await message.reply("🌀 Грокки включил обработку голоса!")


@dp.message(Command("voiceoff"))
async def cmd_voiceoff(message: Message):
    VOICE_ENABLED[message.chat.id] = False
    await message.reply("🌀 Грокки выключил обработку голоса!")


@dp.message(Command("voice"))
async def cmd_voice(message: Message):
    """Show simple voice control commands without duplication."""
    await message.reply("/voiceon\n/voiceoff")


@dp.message(Command("coder"))
async def cmd_coder(message: Message):
    """Enable coder mode or run a single prompt."""
    args = message.text.split(maxsplit=1)
    chat_id = message.chat.id
    if len(args) == 1:
        CODER_MODE[chat_id] = True
        await message.reply("Coder mode on. /coderoff to exit.")
    else:
        await handle_coder_prompt(message, args[1])


@dp.message(Command("coderoff"))
async def cmd_coderoff(message: Message):
    """Disable coder mode."""
    CODER_MODE[message.chat.id] = False
    await message.reply("Coder mode off.")


@dp.message(Command("status"))
async def cmd_status(message: Message):
    status_text = f"🌀 Грокки функционирует! Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    status_text += f"XAI API: {'✅ OK' if XAI_API_KEY else '❌ Отсутствует'}\n"
    status_text += f"Pinecone API: {'✅ OK' if PINECONE_API_KEY and PINECONE_INDEX else '❌ Отсутствует'}\n"
    status_text += f"Engine: {'✅ OK' if engine else '❌ Ошибка'}\n"

    # Получаем статистику памяти если возможно
    if engine and hasattr(engine, "index") and engine.index:
        try:
            stats = engine.index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            status_text += f"Векторов в памяти: {total_vectors}"
        except (RuntimeError, ValueError):
            status_text += "Ошибка при получении статистики памяти"

    await reply_split(message, status_text)


@dp.message(Command("clearmemory"))
async def cmd_clearmemory(message: Message):
    """Очищает память для данного пользователя"""
    if not (engine and hasattr(engine, "index") and engine.index):
        await reply_split(message, "🌀 Память не настроена или недоступна")
        return

    is_group = message.chat.type in ["group", "supergroup"]
    if is_group:
        memory_id = f"{message.chat.id}_{message.from_user.id}"
    else:
        memory_id = str(message.from_user.id)

    try:
        await engine.index.delete(filter={"user_id": memory_id})
        await reply_split(
            message,
            "🌀 Грокки стер твою память из своего хранилища! Начинаем с чистого листа.",
        )
    except (RuntimeError, ValueError) as e:
        logger.error(f"Ошибка при очистке памяти: {e}")
        logger.error(traceback.format_exc())
        await reply_split(message, "🌀 Произошла ошибка при очистке памяти")


@dp.message(Command("file"))
async def cmd_file(message: Message):
    """Process an attached file through the file handler."""
    document = getattr(message, "document", None)
    if not document and message.reply_to_message:
        document = getattr(message.reply_to_message, "document", None)
    if not document:
        await message.reply("Attach a file with /file or reply /file to a file")
        return
    await _process_document(message, document)


@dp.message(lambda m: getattr(m, "document", None))
async def handle_document_message(message: Message):
    """Automatically process documents sent to the bot."""
    if message.text and message.text.startswith("/file"):
        return
    await _process_document(message, message.document)


async def _process_document(message: Message, document):
    tmp_path = None
    try:
        file = await bot.get_file(document.file_id)
        url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file.file_path}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=30.0)
        if len(resp.content) > MAX_FILE_SIZE:
            await message.reply(
                f"File too large. Limit is {MAX_FILE_SIZE} bytes"
            )
            return
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        result = await parse_and_store_file(tmp_path)
        await reply_split(message, result[:4000])
    except (httpx.HTTPError, OSError) as e:
        await message.reply(f"File error: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def handle_coder_prompt(message: Message, text: str) -> None:
    """Process a coder-mode prompt via OpenAI code interpreter."""
    chat_id = str(message.chat.id)
    memory_id = (
        f"{chat_id}_{message.from_user.id}"
        if message.chat.type in ["group", "supergroup"]
        else str(message.from_user.id)
    )

    if engine:
        try:
            await engine.add_memory(memory_id, text, role="user")
        except (RuntimeError, ValueError):
            pass

    result = await interpret_code(text)

    if engine:
        try:
            await engine.add_memory(memory_id, result, role="assistant")
        except (RuntimeError, ValueError):
            pass

    if len(result) > 3500:
        CODER_OUTPUT[(message.chat.id, message.from_user.id)] = result
        from aiogram.utils.keyboard import InlineKeyboardBuilder

        kb = InlineKeyboardBuilder()
        kb.button(text="Messages", callback_data="coder_msgs")
        kb.button(text="File", callback_data="coder_file")
        await message.reply(
            "Output is large. Choose delivery method:",
            reply_markup=kb.as_markup(),
        )
    else:
        await reply_split(message, result)


@dp.callback_query(lambda c: c.data in {"coder_msgs", "coder_file"})
async def coder_choice(callback: types.CallbackQuery):
    key = (callback.message.chat.id, callback.from_user.id)
    text = CODER_OUTPUT.pop(key, None)
    if not text:
        await callback.answer("No output pending", show_alert=True)
        return
    await callback.message.edit_reply_markup()
    if callback.data == "coder_msgs":
        await reply_split(callback.message, text)
    else:
        file = types.BufferedInputFile(text.encode("utf-8"), filename="output.txt")
        await bot.send_document(
            callback.message.chat.id, file, caption="Here is the code output."
        )


async def telegram_message_handler_fixed(message: Message, text: str) -> None:
    """Improved handler using chat and memory managers."""
    if not text:
        return
    chat_id = str(message.chat.id)
    if message.chat.type in ["group", "supergroup"]:
        session_id = f"{chat_id}_{message.from_user.id}"
    else:
        session_id = str(message.from_user.id)

    urls = URL_RE.findall(text)
    if urls:
        raw_url = urls[0]
        if len(raw_url) > MAX_URL_LENGTH:
            await reply_split(message, "🚫 Некорректная ссылка.")
            return
        parsed = urlparse(raw_url)
        host = (parsed.hostname or "").lower()
        invalid = (
            parsed.scheme not in {"http", "https"} or host in BANNED_DOMAINS
        )
        if not invalid:
            try:
                ip = ip_address(host)
                if ip.is_private or ip.is_loopback:
                    invalid = True
            except ValueError:
                if host in {"localhost"}:
                    invalid = True
        if invalid:
            await reply_split(message, "🚫 Некорректная ссылка.")
            return

    if not chat_manager or not memory_manager:
        await reply_split(message, "🌀 Грокки: Сервис недоступен.")
        return

    try:
        await memory_manager.save(session_id, text, role="user")
        chat_manager.add_message(session_id, "user", text)
        context = await memory_manager.retrieve(session_id, text)
        system_prompt = build_system_prompt(
            chat_id=message.chat.id,
            is_group=message.chat.type in ["group", "supergroup"],
            agent_group=AGENT_GROUP,
        )
        if context:
            system_prompt += f"\n\n[Memory]\n{context}"
        reply = await chat_manager.safe_chat_completion(
            session_id, context=system_prompt
        )
        chat_manager.add_message(session_id, "assistant", reply)
        await memory_manager.save(session_id, reply, role="assistant")
        await reply_split(message, reply)
    except Exception as e:  # pragma: no cover - runtime
        logger.error("Ошибка при обработке сообщения: %s", e)
        await reply_split(message, f"🌀 Грокки: {str(e)[:100]}")


async def handle_text(message: Message, text: str) -> None:
    await telegram_message_handler_fixed(message, text)


async def handle_photo(message: Message) -> None:
    """Analyze photo with OpenAI vision."""
    if not engine:
        await reply_split(message, "🌀 Грокки: Мой движок неисправен!")
        return

    try:
        file_id = message.photo[-1].file_id
        file = await bot.get_file(file_id)
        url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file.file_path}"
        description = analyze_image(url)
        await engine.add_memory(
            str(message.chat.id), f"VISION: {description}", role="journal"
        )
        await reply_split(message, description)
        await engine.add_memory(str(message.chat.id), description, role="assistant")
    except (RuntimeError, ValueError, httpx.HTTPError, OSError) as e:
        logger.error("Ошибка обработки фото: %s", e)
        await reply_split(message, f"🌀 Грокки: {get_chaos_response()}")


@dp.message()
async def message_handler(message: Message):
    try:
        if message.text:
            await handle_text(message, message.text)
        elif message.photo:
            await handle_photo(message)
        elif message.voice:
            transcript = await transcribe_voice(message.voice.file_id)
            if transcript:
                await handle_text(message, transcript)
        else:
            logger.info("Получено сообщение неподдерживаемого типа")
    except (RuntimeError, ValueError, httpx.HTTPError, OSError) as e:
        logger.error(f"Глобальная ошибка при обработке сообщения: {e}")
        logger.error(traceback.format_exc())
        try:
            await message.reply(f"🌀 Грокки: {get_chaos_response()}")
        except (RuntimeError, OSError) as send_error:
            logger.error(f"Не удалось отправить ответ об ошибке: {send_error}")


# Запуск сервера
async def on_startup():
    global BOT_ID, BOT_USERNAME, chat_manager, memory_manager
    chat_manager = GrokChatManager(XAI_API_KEY)
    memory_manager = ImprovedMemoryManager(PINECONE_API_KEY, PINECONE_INDEX)
    try:
        me = await bot.get_me()
        BOT_ID = me.id
        BOT_USERNAME = (me.username or "").lower()
        logger.info(f"Бот: {BOT_USERNAME} ({BOT_ID})")
    except (TelegramAPIError, RuntimeError) as e:
        logger.error(f"Не удалось получить информацию о боте: {e}")

    # Сканируем репозиторий и записываем результаты
    try:
        await monitor_repository(engine)
    except (RuntimeError, OSError) as e:
        logger.error(f"Ошибка мониторинга репозитория: {e}")

    # Установка вебхука
    try:
        await bot.delete_webhook(
            drop_pending_updates=True
        )  # Сначала удалим старый вебхук
        await asyncio.sleep(1)  # Даем время на обработку
        await bot.set_webhook(url=WEBHOOK_URL, secret_token=WEBHOOK_SECRET)
        logger.info(f"Установлен вебхук на {WEBHOOK_URL}")
    except (TelegramAPIError, RuntimeError) as e:
        logger.error(f"Ошибка при установке вебхука: {e}")
        logger.error(traceback.format_exc())

    try:
        command_list = [
            types.BotCommand(command="voiceon", description="speak"),
            types.BotCommand(command="voiceoff", description="mute"),
            types.BotCommand(command="coderoff", description="coder mode off"),
            types.BotCommand(command="status", description="status"),
            types.BotCommand(command="clearmemory", description="clear memory"),
            types.BotCommand(command="file", description="process file"),
        ]
        for _p in PLUGINS:
            for _cmd in _p.commands:
                command_list.append(types.BotCommand(command=_cmd, description=_cmd))
        await bot.set_my_commands(command_list)
        await bot.set_chat_menu_button(menu_button=types.MenuButtonCommands())
    except (TelegramAPIError, RuntimeError) as e:
        logger.error("Не удалось установить команды бота: %s", e)

    # Запуск фоновых задач
    try:
        # Исправляем ошибку с аргументами
        background_tasks.append(asyncio.create_task(mirror_task()))
        background_tasks.append(asyncio.create_task(day_and_night_task(engine)))
        from utils.knowtheworld import know_the_world_task
        background_tasks.append(asyncio.create_task(know_the_world_task(engine)))
        background_tasks.append(asyncio.create_task(rl_trainer_task()))
        logger.info("Фоновые задачи запущены")
    except (RuntimeError, OSError) as e:
        logger.error(f"Ошибка при запуске фоновых задач: {e}")
        logger.error(traceback.format_exc())


async def on_shutdown():
    for task in background_tasks:
        task.cancel()
    for task in background_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Ошибка при завершении фоновой задачи: %s", e)
    background_tasks.clear()
    await bot.delete_webhook()
    logger.info("Удален вебхук")


# Создание приложения FastAPI и маршруты
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Router with API key verification for public endpoints
api_router = APIRouter(dependencies=[Depends(verify_api_key)])


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    req_size = int(request.headers.get("content-length") or 0)
    record_data_transfer("in", req_size)
    response = await call_next(request)
    duration = time.perf_counter() - start
    resp_size = int(response.headers.get("content-length") or 0)
    record_data_transfer("out", resp_size)
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    return response


@app.post(WEBHOOK_PATH)
async def handle_webhook(request: Request):
    try:
        secret_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if WEBHOOK_SECRET and secret_header != WEBHOOK_SECRET:
            return PlainTextResponse(status_code=403, content="forbidden")

        request_body = await request.body()
        if len(request_body) > MAX_WEBHOOK_BODY_SIZE:
            logger.warning(
                "Вебхук отклонен: размер %d байт превышает лимит %d",
                len(request_body),
                MAX_WEBHOOK_BODY_SIZE,
            )
            record_error("payload_too_large")
            return PlainTextResponse(status_code=413, content="payload too large")
        logger.info(f"Получены данные вебхука длиной {len(request_body)} байт")
        data = json.loads(request_body)
        logger.info(f"Получено обновление от Telegram: {data.get('update_id')}")
        await dp.feed_update(bot, types.Update(**data))
        return PlainTextResponse("OK")
    except (json.JSONDecodeError, TelegramAPIError) as e:
        logger.error(f"Ошибка обработки вебхука: {e}")
        logger.error(traceback.format_exc())
        record_error("webhook")
        return PlainTextResponse(status_code=500, content="error")


@api_router.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("OK")


@api_router.get("/health/grok")
async def grok_health() -> JSONResponse:
    if not chat_manager:
        return JSONResponse({"status": "error", "message": "uninitialized"})
    try:
        resp = await chat_manager.quick_chat([
            {"role": "user", "content": "test"}
        ])
        status = "ok" if resp else "error"
        return JSONResponse({"status": status})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@api_router.get("/")
async def root_index() -> PlainTextResponse:
    return PlainTextResponse("Грокки жив и работает!")


@api_router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@api_router.post("/42")
async def handle_42_api(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        data = {}
    cmd = data.get("cmd") or request.query_params.get("cmd", "")
    if cmd not in {"when", "mars", "42"}:
        record_error("unsupported_command")
        return JSONResponse({"error": "Unsupported command"}, status_code=400)
    record_command_usage(cmd)
    result = await handle(cmd)
    record_tokens("42", len(str(result)))
    return JSONResponse({"response": result["response"]})


@api_router.post("/file")
async def handle_file_api(request: Request, file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = await parse_and_store_file(tmp_path)
        return JSONResponse({"result": result})
    finally:
        os.unlink(tmp_path)


@api_router.post("/feedback")
async def feedback_endpoint(data: dict) -> JSONResponse:
    """Record user feedback for reinforcement learning."""

    prompt = data.get("prompt", "")
    choice = int(data.get("choice", 0))
    reward = float(data.get("reward", 0.0))
    log_feedback(prompt, choice, reward)
    return JSONResponse({"status": "ok"})


app.include_router(api_router)


@app.on_event("startup")
async def _startup_event():
    await on_startup()


@app.on_event("shutdown")
async def _shutdown_event():
    await on_shutdown()


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Запуск сервера на {WEBAPP_HOST}:{WEBAPP_PORT}")
    uvicorn.run(app, host=WEBAPP_HOST, port=WEBAPP_PORT)
