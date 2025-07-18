import os
import asyncio
import json
from datetime import datetime
import logging
import sys
import traceback

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message
import httpx
from aiohttp import web

# Импортируем наш новый движок
from utils.vector_engine import VectorGrokkyEngine
from utils.genesis2 import genesis2_handler
from utils.howru import check_silence, update_last_message_time
from utils.mirror import mirror_task
from utils.prompt import build_system_prompt, get_chaos_response
from utils.dayandnight import day_and_night_task

# Настройки логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Переменные окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.error("Не задан TELEGRAM_BOT_TOKEN! Завершение работы.")
    sys.exit(1)

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

# Инициализация движка
try:
    engine = VectorGrokkyEngine()
    logger.info("VectorGrokkyEngine инициализирован успешно")
except Exception as e:
    logger.error(f"Ошибка при инициализации VectorGrokkyEngine: {e}")
    logger.error(traceback.format_exc())
    engine = None

# Обработка голосовых сообщений
VOICE_ENABLED = {}


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
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(
        types.KeyboardButton(text="/voiceon"), types.KeyboardButton(text="/voiceoff")
    )
    await message.reply("Выберите режим голоса", reply_markup=kb)


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
        except:
            status_text += "Ошибка при получении статистики памяти"

    await message.reply(status_text)


@dp.message(Command("clearmemory"))
async def cmd_clearmemory(message: Message):
    """Очищает память для данного пользователя"""
    if not (engine and hasattr(engine, "index") and engine.index):
        await message.reply("🌀 Память не настроена или недоступна")
        return

    memory_id = str(message.chat.id)

    try:
        await engine.index.delete(filter={"user_id": memory_id})
        await message.reply(
            "🌀 Грокки стер твою память из своего хранилища! Начинаем с чистого листа."
        )

    except Exception as e:
        logger.error(f"Ошибка при очистке памяти: {e}")
        logger.error(traceback.format_exc())
        await message.reply("🌀 Произошла ошибка при очистке памяти")


async def handle_text(message: Message, text: str) -> None:
    if not engine:
        await message.reply(
            "🌀 Грокки: Мой движок неисправен! Свяжитесь с моим создателем."
        )
        return

    try:
        await update_last_message_time()
    except Exception as e:
        logger.error("Ошибка при обновлении времени последнего сообщения: %s", e)

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
    memory_id = chat_id

    try:
        await engine.add_memory(memory_id, text, role="user")
    except Exception as e:
        logger.error("Ошибка при сохранении сообщения: %s", e)

    if "[chaos_pulse]" in text.lower():
        intensity = 5
        chaos_type = None
        for part in text.lower().split():
            if part.startswith("type="):
                chaos_type = part.split("=")[1]
            if part.startswith("intensity="):
                try:
                    intensity = int(part.split("=")[1])
                except ValueError:
                    pass
        try:
            system_prompt = build_system_prompt(
                chat_id=chat_id, is_group=is_group, agent_group=AGENT_GROUP
            )
            result = await genesis2_handler(
                ping="CHAOS PULSE ACTIVATED",
                raw=True,
                system_prompt=system_prompt,
                intensity=intensity,
                is_group=is_group,
                chaos_type=chaos_type,
            )
            answer = result.get("answer", get_chaos_response())
            await message.reply(f"🌀 {answer}")
            await engine.add_memory(memory_id, answer, role="assistant")
        except Exception as e:
            logger.error("Ошибка CHAOS_PULSE: %s", e)
            await message.reply(
                "🌀 Грокки: Даже хаос требует порядка. Ошибка при обработке команды."
            )
        return

    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        context = await engine.search_memory(memory_id, text)
        reply = await engine.generate_with_xai(
            [{"role": "user", "content": text}], context=context
        )
        await engine.add_memory(memory_id, reply, role="assistant")
        if VOICE_ENABLED.get(message.chat.id):
            lang = "ru" if any(ch.isalpha() and ord(ch) > 127 for ch in reply) else "en"
            audio_bytes = await synth_voice(reply, lang=lang)
            voice_file = types.BufferedInputFile(audio_bytes, filename="voice.mp3")
            await bot.send_audio(
                message.chat.id,
                voice_file,
                caption=reply,
                reply_to_message_id=message.message_id,
            )
        else:
            await message.reply(reply)
    except Exception as e:
        logger.error("Ошибка при обработке сообщения: %s", e)
        await message.reply(
            f"🌀 Грокки: Произошла ошибка при генерации ответа: {str(e)[:100]}..."
        )


@dp.message()
async def message_handler(message: Message):
    try:
        if message.text:
            await handle_text(message, message.text)
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
                types.BotCommand(command="voiceon", description="/voiceon"),
                types.BotCommand(command="voiceoff", description="/voiceoff"),
            ]
        )
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

# Хуки запуска и остановки
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

# Запуск сервера
if __name__ == "__main__":
    logger.info(f"Запуск сервера на {WEBAPP_HOST}:{WEBAPP_PORT}")
    web.run_app(app, host=WEBAPP_HOST, port=WEBAPP_PORT)
