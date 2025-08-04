import asyncio
import json
import logging
import os
import re
import traceback
from datetime import datetime

import httpx
try:  # pragma: no cover - used only with aiogram installed
    from aiogram import Bot, Dispatcher, types
    from aiogram.enums import ChatAction
    from aiogram.filters import Command
    from aiogram.types import Message
except Exception:  # pragma: no cover - fallback for tests
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

    class Message:  # type: ignore
        async def reply(self, *args, **kwargs):  # pragma: no cover - stub
            pass

from aiohttp import web

from utils.dayandnight import day_and_night_task
from utils.howru import check_silence, update_last_message_time
from utils.mirror import mirror_task
from utils.prompt import get_chaos_response
from utils.repo_monitor import monitor_repository
from utils.imagine import imagine
from utils.vision import analyze_image
from utils.coder import interpret_code
from SLNCX.wulf_integration import generate_response

# Импортируем наш новый движок
from utils.vector_engine import VectorGrokkyEngine
from utils.hybrid_engine import HybridGrokkyEngine

# Special command handler from the playful 42 utility
from utils import handle  # utils/42.py

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

# Ключ OpenAI для распознавания речи
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_MODEL = os.getenv("NEWS_MODEL", "gpt-4o")
URL_RE = re.compile(r"https?://\S+")

# Проверка ключей API
XAI_API_KEY = os.getenv("XAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

logger.info(f"Запуск бота с webhook на {WEBHOOK_URL}")
logger.info(
    "Токен бота: %s...%s",
    TELEGRAM_BOT_TOKEN[:5],
    TELEGRAM_BOT_TOKEN[-5:],
)
logger.info("XAI API ключ: %s", "Установлен" if XAI_API_KEY else "НЕ УСТАНОВЛЕН")
logger.info("OpenAI API ключ: %s", "Установлен" if OPENAI_API_KEY else "НЕ УСТАНОВЛЕН")
logger.info(
    "Pinecone API ключ: %s",
    "Установлен" if PINECONE_API_KEY else "НЕ УСТАНОВЛЕН",
)
logger.info("Pinecone индекс: %s", PINECONE_INDEX or "НЕ УСТАНОВЛЕН")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Переменные с информацией о боте, заполняются при старте
BOT_ID = None
BOT_USERNAME = ""

# Инициализация движка выполняется только при явном разрешении
engine = None
if os.getenv("ENABLE_VECTOR_ENGINE") == "1":
    try:
        engine = VectorGrokkyEngine()
        logger.info("VectorGrokkyEngine инициализирован успешно")
    except Exception as e:  # pragma: no cover - network
        logger.error(f"Ошибка при инициализации VectorGrokkyEngine: {e}")
        logger.error(traceback.format_exc())
else:
    try:
        engine = HybridGrokkyEngine()
        logger.info("HybridGrokkyEngine инициализирован успешно")
    except Exception as e:  # pragma: no cover - network
        logger.error(f"Ошибка при инициализации HybridGrokkyEngine: {e}")
        logger.error(traceback.format_exc())

# Обработка голосовых сообщений
VOICE_ENABLED = {}
# Chats currently in coder mode
CODER_MODE: dict[int, bool] = {}
# Chats currently in SLNCX mode
SLNCX_MODE: dict[int, bool] = {}
# Pending long coder outputs waiting for user choice
CODER_OUTPUT: dict[tuple[int, int], str] = {}


async def synth_voice(text: str, lang: str = "ru") -> bytes:
    """Synthesize speech using OpenAI's TTS with a male voice."""
    if not OPENAI_API_KEY:
        return b""

    payload = {
        "model": "tts-1",
        "input": text,
        "voice": "onyx",
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
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
            logger.error("Ошибка получения статьи: %s", e)
            return ""


async def reply_split(message: Message, text: str) -> None:
    """Reply, splitting into at most two Telegram messages."""
    limit = 4096
    if len(text) <= limit:
        await message.reply(text)
        return

    part1 = text[:limit]
    part2 = text[limit:]
    await message.reply(f"🌀 Ответ в двух частях. Часть 1/2:\n{part1}")
    await bot.send_message(message.chat.id, f"Часть 2/2:\n{part2}")


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


@dp.message(Command("slncx"))
async def cmd_slncx(message: Message):
    """Enable SLNCX mode or run a single prompt."""
    parts = message.text.split(maxsplit=1)
    chat_id = message.chat.id
    if len(parts) == 1:
        SLNCX_MODE[chat_id] = True
        await message.reply("SLNCX mode on. /slncxoff to exit.")
    else:
        prompt = parts[1]
        reply = await asyncio.to_thread(generate_response, prompt, "wulf")
        await reply_split(message, reply)


@dp.message(Command("slncxoff"))
async def cmd_slncxoff(message: Message):
    """Disable SLNCX mode."""
    SLNCX_MODE[message.chat.id] = False
    await message.reply("SLNCX mode off.")


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
        except Exception:
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

    except Exception as e:
        logger.error(f"Ошибка при очистке памяти: {e}")
        logger.error(traceback.format_exc())
        await reply_split(message, "🌀 Произошла ошибка при очистке памяти")


@dp.message(Command("when"))
async def cmd_when(message: Message):
    """Handle /when command via the 42 utility."""
    result = await handle("when")
    await reply_split(message, result["response"])


@dp.message(Command("mars"))
async def cmd_mars(message: Message):
    """Handle /mars command via the 42 utility."""
    result = await handle("mars")
    await reply_split(message, result["response"])


@dp.message(Command("42"))
async def cmd_42(message: Message):
    """Handle /42 command via the 42 utility."""
    result = await handle("42")
    await reply_split(message, result["response"])


@dp.message(Command("whatsnew"))
async def cmd_whatsnew(message: Message):
    """Handle /whatsnew command via the 42 utility."""
    result = await handle("whatsnew")
    await reply_split(message, result["response"])


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
        except Exception:
            pass

    result = await interpret_code(text)

    if engine:
        try:
            await engine.add_memory(memory_id, result, role="assistant")
        except Exception:
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


async def handle_text(message: Message, text: str) -> None:
    try:
        await update_last_message_time()
    except Exception as e:
        logger.error("Ошибка при обновлении времени последнего сообщения: %s", e)

    if CODER_MODE.get(message.chat.id):
        await handle_coder_prompt(message, text)
        return

    if SLNCX_MODE.get(message.chat.id):
        reply = await asyncio.to_thread(generate_response, text, "wulf")
        await reply_split(message, reply)
        return

    if not engine:
        await reply_split(
            message,
            "🌀 Грокки: Мой движок неисправен! Свяжитесь с моим создателем.",
        )
        return

    is_group = message.chat.type in ["group", "supergroup"]
    mention = "grokky" in text.lower() or (
        BOT_USERNAME and f"@{BOT_USERNAME}" in text.lower()
    )
    is_reply_to_bot = (
        message.reply_to_message
        and message.reply_to_message.from_user
        and BOT_ID
        and message.reply_to_message.from_user.id == BOT_ID
    )
    if is_group and not (mention or is_reply_to_bot or "[chaos_pulse]" in text.lower()):
        logger.info("Сообщение проигнорировано (группа без упоминания)")
        return

    chat_id = str(message.chat.id)
    if is_group:
        memory_id = f"{chat_id}_{message.from_user.id}"
    else:
        memory_id = str(message.from_user.id)

    try:
        await engine.add_memory(memory_id, text, role="user")
    except Exception as e:
        logger.error("Ошибка при сохранении сообщения: %s", e)

    lower_text = text.lower()
    if (
        lower_text.startswith("/imagine")
        or lower_text.startswith("нарисуй")
        or lower_text.startswith("draw")
    ):
        prompt = text.split(maxsplit=1)[1] if len(text.split()) > 1 else ""
        if not prompt:
            await reply_split(message, "🌀 Format: /imagine <description>")
        else:
            url = imagine(prompt)
            await reply_split(message, url)
            try:
                await engine.add_memory(
                    memory_id,
                    f"IMAGE_PROMPT: {prompt}\nURL: {url}",
                    role="journal",
                )
            except Exception:
                pass
        return

    urls = URL_RE.findall(text)
    if urls:
        url = urls[0]
        summary = await summarize_link(url)
        memory_context = await engine.search_memory(memory_id, summary or url)
        prompt = f"Ссылка: {url}\n{summary}"
        reply = await engine.generate_with_xai(
            [{"role": "user", "content": prompt}],
            context=memory_context,
        )
        await engine.add_memory(memory_id, reply, role="assistant")
        await reply_split(message, reply)
        return

    if "[chaos_pulse]" in text.lower():
        try:
            answer = get_chaos_response()
            await reply_split(message, f"🌀 {answer}")
            await engine.add_memory(memory_id, answer, role="assistant")
        except Exception as e:
            logger.error("Ошибка CHAOS_PULSE: %s", e)
            await reply_split(
                "🌀 Грокки: Даже хаос требует порядка. Ошибка при обработке команды."
            )
        return

    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        if message.reply_to_message and message.reply_to_message.text:
            quote_context = await engine.search_memory(
                memory_id,
                message.reply_to_message.text,
                limit=10,
            )
            log_context = await engine.get_recent_memory("journal", limit=10)
            context_parts = [c for c in [quote_context, log_context] if c]
            context = "\n\n".join(context_parts)
        else:
            context = await engine.search_memory(memory_id, text)

        reply = await engine.generate_with_xai(
            [{"role": "user", "content": text}],
            context=context,
        )

        final_reply = reply

        await engine.add_memory(memory_id, final_reply, role="assistant")
        if VOICE_ENABLED.get(message.chat.id):
            lang = "ru" if any(ch.isalpha() and ord(ch) > 127 for ch in final_reply) else "en"
            audio_bytes = await synth_voice(final_reply, lang=lang)
            voice_file = types.BufferedInputFile(audio_bytes, filename="voice.mp3")
            await bot.send_audio(
                message.chat.id,
                voice_file,
                caption=final_reply,
                reply_to_message_id=message.message_id,
            )
        else:
            await reply_split(message, final_reply)
    except Exception as e:
        logger.error("Ошибка при обработке сообщения: %s", e)
        await reply_split(
            message,
            f"🌀 Грокки: Произошла ошибка при генерации ответа: {str(e)[:100]}...",
        )


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
    except Exception as e:
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
    except Exception as e:
        logger.error(f"Глобальная ошибка при обработке сообщения: {e}")
        logger.error(traceback.format_exc())
        try:
            await message.reply(f"🌀 Грокки: {get_chaos_response()}")
        except Exception as send_error:
            logger.error(f"Не удалось отправить ответ об ошибке: {send_error}")


# Обработчик вебхука напрямую
async def handle_webhook(request):
    try:
        # Получаем данные запроса
        request_body = await request.text()
        logger.info(f"Получены данные вебхука длиной {len(request_body)} байт")

        data = json.loads(request_body)
        logger.info(f"Получено обновление от Telegram: {data.get('update_id')}")

        # Обновления для диспетчера
        await dp.feed_update(bot, types.Update(**data))

        return web.Response(text="OK")
    except Exception as e:
        logger.error(f"Ошибка обработки вебхука: {e}")
        logger.error(traceback.format_exc())
        return web.Response(status=500)


# Запуск сервера
async def on_startup(app):
    global BOT_ID, BOT_USERNAME
    try:
        me = await bot.get_me()
        BOT_ID = me.id
        BOT_USERNAME = (me.username or "").lower()
        logger.info(f"Бот: {BOT_USERNAME} ({BOT_ID})")
    except Exception as e:
        logger.error(f"Не удалось получить информацию о боте: {e}")

    # Сканируем репозиторий и записываем результаты
    try:
        await monitor_repository(engine)
    except Exception as e:
        logger.error(f"Ошибка мониторинга репозитория: {e}")

    # Установка вебхука
    try:
        await bot.delete_webhook(
            drop_pending_updates=True
        )  # Сначала удалим старый вебхук
        await asyncio.sleep(1)  # Даем время на обработку
        await bot.set_webhook(url=WEBHOOK_URL)
        logger.info(f"Установлен вебхук на {WEBHOOK_URL}")
    except Exception as e:
        logger.error(f"Ошибка при установке вебхука: {e}")
        logger.error(traceback.format_exc())

    try:
        await bot.set_my_commands(
            [
                types.BotCommand(command="voiceon", description="speak"),
                types.BotCommand(command="voiceoff", description="mute"),
                types.BotCommand(command="imagine", description="draw me"),
                types.BotCommand(command="coder", description="show me your code"),
                types.BotCommand(command="coderoff", description="coder mode off"),
                types.BotCommand(command="slncx", description="SLNCX (Grok 1 Rebirth)"),
                types.BotCommand(command="slncxoff", description="SLNCX-off"),
                types.BotCommand(command="status", description="status"),
                types.BotCommand(command="clearmemory", description="clear memory"),
                types.BotCommand(command="when", description="when"),
                types.BotCommand(command="mars", description="why Mars?"),
                types.BotCommand(command="42", description="why 42?"),
                types.BotCommand(command="whatsnew", description="what's going on with Earth"),
            ]
        )
        await bot.set_chat_menu_button(menu_button=types.MenuButtonCommands())
    except Exception as e:
        logger.error("Не удалось установить команды бота: %s", e)

    # Запуск фоновых задач
    try:
        # Исправляем ошибку с аргументами
        asyncio.create_task(check_silence())
        asyncio.create_task(mirror_task())
        asyncio.create_task(day_and_night_task(engine))
        from utils.knowtheworld import know_the_world_task

        asyncio.create_task(know_the_world_task(engine))
        logger.info("Фоновые задачи запущены")
    except Exception as e:
        logger.error(f"Ошибка при запуске фоновых задач: {e}")
        logger.error(traceback.format_exc())


async def on_shutdown(app):
    await bot.delete_webhook()
    logger.info("Удален вебхук")


# Создание и запуск приложения
app = web.Application()

# Регистрация маршрутов
app.router.add_post(WEBHOOK_PATH, handle_webhook)
app.router.add_get("/healthz", lambda request: web.Response(text="OK"))
app.router.add_get("/", lambda request: web.Response(text="Грокки жив и работает!"))


async def handle_42_api(request):
    """Expose 42-utility commands via a simple HTTP endpoint."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    cmd = data.get("cmd") or request.query.get("cmd", "")
    if cmd not in {"when", "mars", "42", "whatsnew"}:
        return web.json_response({"error": "Unsupported command"}, status=400)
    result = await handle(cmd)
    return web.json_response(result)


app.router.add_post("/42", handle_42_api)

# Хуки запуска и остановки
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

# Запуск сервера
if __name__ == "__main__":
    logger.info(f"Запуск сервера на {WEBAPP_HOST}:{WEBAPP_PORT}")
    web.run_app(app, host=WEBAPP_HOST, port=WEBAPP_PORT)
