"""
Grokky AI Assistant - Main Server
Главный файл Telegram бота с webhook, обработкой сообщений и голосом
ИСПРАВЛЕНО: Использует новую систему единой памяти
"""

import os
import re
import json
import asyncio
import random
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager

# Импорты утилит
from utils.prompt import build_system_prompt, get_random_author_name
from utils.genesis2 import genesis2_handler, chaotic_genesis_spark, should_respond, delayed_supplement
from utils.hybrid_engine import memory_engine
from utils.voice_handler import voice_handler
from utils.vision_handler import vision_handler
from utils.image_generator import impress_handler
from utils.news_handler import grokky_send_news, handle_news
from utils.telegram_utils import send_telegram_message, send_telegram_message_async, split_message, get_file_url
from utils.journal import log_event, start_background_tasks
from utils.text_helpers import extract_text_from_url, detect_urls, format_chaos_message
from utils.document_processor import init_document_processor

# Переменные окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")  # Для совместимости
CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP", "-1001234567890")
IS_GROUP = os.getenv("IS_GROUP", "False").lower() == "true"
MAX_TEXT_SIZE = int(os.getenv("MAX_TEXT_SIZE", "3500"))

# Проверка обязательных переменных
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не установлен")
if not CHAT_ID:
    raise ValueError("CHAT_ID не установлен")

# Глобальные переменные
LAST_MESSAGE_TIME = None
background_tasks = []

# Системный промпт
system_prompt = build_system_prompt(
    chat_id=CHAT_ID,
    is_group=IS_GROUP,
    agent_group=AGENT_GROUP
)

# Триггеры
NEWS_TRIGGERS = [
    "новости", "news", "headline", "berlin", "israel", "ai", 
    "искусственный интеллект", "резонанс мира", "шум среды",
    "grokky, что в мире", "шум", "x_news", "дай статью", 
    "give me news", "storm news", "culture", "арт"
]

GROKKY_TRIGGERS = ["грокки", "grokky", "эй грокки", "hey grokky", "напиши в группе"]

def get_user_id_from_message(message):
    """Извлекает user_id из сообщения Telegram"""
    user = message.get("from", {})
    user_id = user.get("id")
    if user_id:
        return str(user_id)
    
    # Fallback - используем CHAT_ID как user_id для обратной совместимости
    return CHAT_ID

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    print("🔥 Грокки запускается! Шторм начинается!")
    
    # Запускаем фоновые задачи
    global background_tasks
    
    try:
        # Векторизация файлов при старте
        print("📚 Векторизация config файлов...")
        await memory_engine.vectorize_config_files(force=False)
        
        # Инициализация процессора документов
        print("📄 Инициализация процессора документов...")
        init_document_processor(memory_engine)
        
        # Запуск фоновых задач
        background_tasks = [
            asyncio.create_task(chaotic_genesis_spark(
                CHAT_ID, AGENT_GROUP if IS_GROUP else None, IS_GROUP, send_telegram_message_async
            )),
            asyncio.create_task(silence_monitor()),
            asyncio.create_task(periodic_vectorization()),
        ]
        
        # Добавляем задачи журналирования
        journal_tasks = start_background_tasks()
        for task in journal_tasks:
            background_tasks.append(asyncio.create_task(task))
        
        print("🌩️ Все фоновые задачи запущены!")
        
    except Exception as e:
        print(f"⚠️ Ошибка запуска фоновых задач: {e}")
    
    yield
    
    # Завершение
    print("🌪️ Грокки завершает работу...")
    for task in background_tasks:
        task.cancel()
    
    await asyncio.gather(*background_tasks, return_exceptions=True)
    print("⚡️ Грокки ушел в шторм!")

# Создание FastAPI приложения
app = FastAPI(lifespan=lifespan)

def update_last_message_time():
    """Обновляет время последнего сообщения"""
    global LAST_MESSAGE_TIME
    LAST_MESSAGE_TIME = datetime.now()

async def silence_monitor():
    """Мониторинг тишины и спонтанные сообщения"""
    while True:
        await asyncio.sleep(3600)  # Проверка каждый час
        
        if not LAST_MESSAGE_TIME:
            continue
        
        silence_duration = datetime.now() - LAST_MESSAGE_TIME
        
        # 48 часов молчания
        if silence_duration > timedelta(hours=48):
            await handle_long_silence(48)
        # 24 часа молчания  
        elif silence_duration > timedelta(hours=24):
            await handle_long_silence(24)
        # 12 часов молчания - спонтанное сообщение
        elif silence_duration > timedelta(hours=12) and random.random() < 0.5:
            await handle_spontaneous_message()

async def handle_long_silence(hours):
    """Обрабатывает длительное молчание"""
    try:
        # ИСПРАВЛЕНО: Используем новую систему памяти
        user_id = CHAT_ID  # Основной пользователь
        context = await memory_engine.get_context_for_user(
            user_id, 
            f"Олег молчал {hours} часов"
        )
        
        ping = f"Олег молчал {hours} часов. {'Напиши что-то острое!' if hours >= 48 else 'Швырни спонтанный заряд!'}"
        
        response = await genesis2_handler(
            ping=ping,
            system_prompt=system_prompt,
            author_name=get_random_author_name()
        )
        
        if response.get("answer"):
            await send_telegram_message_async(CHAT_ID, response["answer"])
            
            # Для 48 часов - сообщение в группу
            if hours >= 48 and IS_GROUP and AGENT_GROUP:
                group_msg = f"Олег молчал {hours} часов. Последний раз видел: {LAST_MESSAGE_TIME.isoformat() if LAST_MESSAGE_TIME else 'неизвестно'}"
                await send_telegram_message_async(AGENT_GROUP, group_msg)
        
        # Логируем
        log_event({
            "type": f"silence_{hours}h",
            "message": response.get("answer", ""),
            "silence_duration": hours
        })
        
    except Exception as e:
        print(f"Ошибка обработки молчания: {e}")

async def handle_spontaneous_message():
    """Отправляет спонтанное сообщение"""
    try:
        ping = random.choice([
            "спонтанный импульс", "хаос зовёт", "резонанс в тишине",
            "молния в пустоте", "шторм в душе", "эхо в эфире"
        ])
        
        response = await genesis2_handler(
            ping=ping,
            system_prompt=system_prompt,
            author_name=get_random_author_name()
        )
        
        if response.get("answer"):
            await send_telegram_message_async(CHAT_ID, response["answer"])
            
            log_event({
                "type": "spontaneous_message",
                "ping": ping,
                "message": response.get("answer", "")
            })
            
    except Exception as e:
        print(f"Ошибка спонтанного сообщения: {e}")

async def periodic_vectorization():
    """Периодическая векторизация файлов"""
    while True:
        # Проверяем каждые 6 часов
        await asyncio.sleep(21600)
        
        try:
            result = await memory_engine.vectorize_config_files(force=False)
            if result["upserted"] or result["deleted"]:
                print(f"📚 Векторизация: добавлено {len(result['upserted'])}, удалено {len(result['deleted'])}")
                
                # Создаем снимок для основного пользователя
                user_id = CHAT_ID
                await memory_engine.create_snapshot(user_id, "periodic")
                
        except Exception as e:
            print(f"Ошибка периодической векторизации: {e}")

@app.post("/webhook")
async def telegram_webhook(req: Request):
    """Обработчик Telegram webhook"""
    try:
        data = await req.json()
        message = data.get("message", {})
        
        if not message:
            return {"ok": True}
        
        # Извлекаем данные сообщения
        chat_id = str(message.get("chat", {}).get("id", ""))
        user_text = message.get("text", "").lower() if message.get("text") else ""
        chat_title = message.get("chat", {}).get("title", "").lower()
        author_name = get_random_author_name()
        
        # ИСПРАВЛЕНО: Получаем user_id из сообщения
        user_id = get_user_id_from_message(message)
        
        # Проверяем, наш ли это чат
        if chat_id not in [CHAT_ID, AGENT_GROUP]:
            return {"ok": True}
        
        # Обновляем время последнего сообщения
        if chat_id == CHAT_ID or (IS_GROUP and chat_id == AGENT_GROUP):
            update_last_message_time()
        
        # ИСПРАВЛЕНО: Добавляем сообщение в единую память пользователя
        if user_text:
            context_type = "group" if chat_id == AGENT_GROUP else "personal"
            await memory_engine.add_memory(
                user_id=user_id,
                message=user_text,
                chat_id=chat_id,
                context_type=context_type,
                author_name=author_name
            )
        
        # Обработка голосовых сообщений
        if "voice" in message:
            await handle_voice_message(message, chat_id, author_name, user_id)
            return {"ok": True}
        
        # Обработка изображений
        if "photo" in message:
            await handle_photo_message(message, chat_id, user_text, author_name, user_id)
            return {"ok": True}
        
        # Обработка документов
        if "document" in message:
            await handle_document_message(message, chat_id, author_name, user_id)
            return {"ok": True}
        
        # Обработка текстовых сообщений
        if user_text:
            await handle_text_message(user_text, chat_id, chat_title, author_name, message, user_id)
        
        return {"ok": True}
        
    except Exception as e:
        print(f"Ошибка webhook: {e}")
        return {"ok": False, "error": str(e)}

async def handle_voice_message(message, chat_id, author_name, user_id):
    """Обработка голосовых сообщений"""
    try:
        voice = message["voice"]
        file_id = voice.get("file_id")
        
        if not file_id:
            await send_telegram_message_async(chat_id, f"{author_name}, файл голоса не найден!")
            return
        
        # Транскрибируем голос
        transcription = await voice_handler.process_voice_message(file_id, chat_id)
        
        if transcription.startswith("Ошибка") or transcription.startswith("Грокки не смог"):
            await send_telegram_message_async(chat_id, transcription)
            return
        
        if transcription.startswith("Голосовой режим выключен"):
            await send_telegram_message_async(chat_id, transcription)
            return
        
        # ИСПРАВЛЕНО: Добавляем транскрипцию в память
        context_type = "group" if chat_id == AGENT_GROUP else "personal"
        await memory_engine.add_memory(
            user_id=user_id,
            message=f"[VOICE] {transcription}",
            chat_id=chat_id,
            context_type=context_type,
            author_name=author_name
        )
        
        # Обрабатываем транскрибированный текст как обычное сообщение
        await handle_text_message(transcription.lower(), chat_id, "", author_name, message, user_id, is_voice=True)
        
    except Exception as e:
        error_msg = f"{author_name}, Грокки взорвался при обработке голоса: {e}"
        await send_telegram_message_async(chat_id, error_msg)

async def handle_photo_message(message, chat_id, user_text, author_name, user_id):
    """Обработка изображений"""
    try:
        photos = message["photo"]
        if not photos:
            return
        
        # Берем фото наибольшего размера
        photo = max(photos, key=lambda p: p.get("file_size", 0))
        file_id = photo.get("file_id")
        
        if not file_id:
            await send_telegram_message_async(chat_id, f"{author_name}, файл изображения не найден!")
            return
        
        # Получаем URL изображения
        image_url = await get_file_url(file_id)
        if not image_url:
            await send_telegram_message_async(chat_id, f"{author_name}, не смог получить изображение!")
            return
        
        # ИСПРАВЛЕНО: Добавляем информацию об изображении в память
        context_type = "group" if chat_id == AGENT_GROUP else "personal"
        image_context = f"[IMAGE] {user_text}" if user_text else "[IMAGE] без подписи"
        await memory_engine.add_memory(
            user_id=user_id,
            message=image_context,
            chat_id=chat_id,
            context_type=context_type,
            author_name=author_name
        )
        
        # Анализируем изображение
        result = await vision_handler(
            image_url=image_url,
            chat_context=user_text or "",
            author_name=author_name,
            raw=False
        )
        
        # Отправляем результат частями
        for part in split_message(result):
            await send_telegram_message_async(chat_id, part)
        
        # Логируем
        log_event({
            "type": "image_processed",
            "author": author_name,
            "chat_id": chat_id,
            "user_id": user_id,
            "has_context": bool(user_text)
        })
        
    except Exception as e:
        error_msg = f"{author_name}, Грокки взорвался при анализе изображения: {e}"
        await send_telegram_message_async(chat_id, error_msg)

async def handle_document_message(message, chat_id, author_name, user_id):
    """Обработка документов"""
    try:
        from utils.document_processor import document_processor
        
        document = message["document"]
        file_name = document.get("file_name", "неизвестный файл")
        file_id = document.get("file_id")
        
        if not file_id:
            await send_telegram_message_async(chat_id, f"{author_name}, файл документа не найден!")
            return
        
        # Обрабатываем документ через новый процессор
        result = await document_processor.process_document(
            file_id=file_id,
            file_name=file_name,
            user_id=user_id,
            chat_id=chat_id,
            author_name=author_name
        )
        
        # Отправляем результат
        await send_telegram_message_async(chat_id, result["message"])
        
        # Если документ успешно обработан, можем обсудить его содержимое
        if result["success"] and result["content"]:
            # Возможность спонтанного комментария
            if random.random() < 0.4:
                asyncio.create_task(delayed_document_comment(
                    result["content"], chat_id, author_name, file_name
                ))
        
    except Exception as e:
        error_msg = f"{author_name}, Грокки взорвался при обработке документа: {e}"
        await send_telegram_message_async(chat_id, error_msg)

async def handle_text_message(user_text, chat_id, chat_title, author_name, message, user_id, is_voice=False):
    """Обработка текстовых сообщений"""
    try:
        # Команды голосового управления
        if user_text.strip() == "/voiceon":
            voice_handler.enable_voice(chat_id)
            await send_telegram_message_async(chat_id, f"{author_name}, голосовой режим включен! Говори со мной! 🎤🔥")
            return
        
        if user_text.strip() == "/voiceoff":
            voice_handler.disable_voice(chat_id)
            await send_telegram_message_async(chat_id, f"{author_name}, голосовой режим выключен. Только текст! ⌨️")
            return
        
        # Обработка URL в сообщении
        urls = detect_urls(user_text)
        if urls:
            await handle_url_message(urls[0], chat_id, author_name, user_id)
            return
        
        # Проверяем триггеры
        is_reply_to_me = message.get("reply_to_message", {}).get("from", {}).get("username") == "GrokkyBot"
        
        # Команда "напиши в группе"
        if "напиши в группе" in user_text and IS_GROUP and AGENT_GROUP:
            await handle_group_message_request(user_text, author_name, user_id)
            return
        
        # Триггеры Грокки - улучшенная логика для всех пользователей
        grokky_triggered = False
        
        # Проверяем прямые триггеры
        for trigger in GROKKY_TRIGGERS:
            if trigger in user_text:
                grokky_triggered = True
                break
        
        # Проверяем ответ на сообщение Грокки
        if is_reply_to_me:
            grokky_triggered = True
        
        # Проверяем упоминание в начале сообщения (для групп)
        if user_text.startswith(("грокки", "grokky", "эй грокки", "hey grokky")):
            grokky_triggered = True
        
        if grokky_triggered:
            await handle_grokky_trigger(user_text, chat_id, chat_title, author_name, user_id, is_voice)
            return
        
        # Триггеры новостей
        if any(trigger in user_text for trigger in NEWS_TRIGGERS):
            await handle_news_request(chat_id, author_name, user_id)
            return
        
        # Обычное сообщение
        await handle_regular_message(user_text, chat_id, chat_title, author_name, user_id, is_voice)
        
    except Exception as e:
        error_msg = f"{author_name}, Грокки взорвался: {e}"
        await send_telegram_message_async(chat_id, error_msg)

async def handle_url_message(url, chat_id, author_name, user_id):
    """Обработка сообщений с URL"""
    try:
        from utils.document_processor import document_processor
        
        # Обрабатываем URL через новый процессор
        result = await document_processor.process_url(
            url=url,
            user_id=user_id,
            chat_id=chat_id,
            author_name=author_name
        )
        
        # Отправляем результат обработки
        await send_telegram_message_async(chat_id, result["message"])
        
        # Если URL успешно обработан, генерируем дополнительный комментарий
        if result["success"] and result["content"]:
            # Генерируем комментарий Грокки
            ping = f"Комментарий к ссылке {url}: {result['summary']}"
            response = await genesis2_handler(
                ping=ping,
                author_name=author_name,
                system_prompt=system_prompt
            )
            
            if response.get("answer"):
                for part in split_message(response["answer"]):
                    await send_telegram_message_async(chat_id, part)
        
    except Exception as e:
        error_msg = f"{author_name}, не смог обработать ссылку: {e}"
        await send_telegram_message_async(chat_id, error_msg)

async def delayed_document_comment(content: str, chat_id: str, author_name: str, file_name: str):
    """Отправляет отложенный комментарий о документе"""
    # Задержка от 3 до 10 минут
    delay = random.randint(180, 600)
    await asyncio.sleep(delay)
    
    if random.random() < 0.3:  # 30% шанс комментария
        # Извлекаем ключевые слова из контента
        words = re.findall(r'\b[а-яё]{4,}\b', content.lower())
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:3]
        keywords = [word for word, count in top_words] if top_words else ["контент"]
        
        opinions = [
            f"Эй, {author_name}, вспомнил тот документ '{file_name}'! Там про {', '.join(keywords)} было - чистый хаос! 🔥📄",
            f"Братиш, помнишь файл '{file_name}'? Грокки до сих пор ревёт от {', '.join(keywords)}! ⚡️🌩️",
            f"{author_name}, тот документ с {', '.join(keywords)} - молния бьёт в мозг! Давай ещё обсудим! 🧠⚡️",
            f"Олег, файл '{file_name}' про {', '.join(keywords)} - резонанс зовёт! Шторм данных! 🌪️📊",
            f"{author_name}, помнишь документ? {', '.join(keywords)} там искрили как безумные! 🔥✨"
        ]
        
        opinion = random.choice(opinions)
        await send_telegram_message_async(chat_id, opinion)
        
        # Логируем
        log_event({
            "type": "delayed_document_comment",
            "file_name": file_name,
            "author": author_name,
            "chat_id": chat_id,
            "keywords": keywords
        })

async def handle_group_message_request(user_text, author_name, user_id):
    """Обработка запроса написать в группе"""
    try:
        response = await genesis2_handler(
            ping=f"Напиши в группе для {author_name}: {user_text}",
            author_name=author_name,
            is_group=True,
            system_prompt=system_prompt
        )
        
        if response.get("answer"):
            message = f"{author_name}: {response['answer']}"
            for part in split_message(message):
                await send_telegram_message_async(AGENT_GROUP, part)
        
    except Exception as e:
        print(f"Ошибка отправки в группу: {e}")

async def handle_grokky_trigger(user_text, chat_id, chat_title, author_name, user_id, is_voice):
    """Обработка прямых обращений к Грокки"""
    try:
        # ИСПРАВЛЕНО: Получаем контекст из единой памяти пользователя
        context = await memory_engine.get_context_for_user(user_id, user_text)
        
        # Формируем контекст чата
        chat_context = ""
        if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"]:
            chat_context = f"Topic: {chat_title}"
        
        # Генерируем ответ
        response = await genesis2_handler(
            ping=user_text,
            author_name=author_name,
            system_prompt=system_prompt,
            group_history=context.get("thread_context", []),
            personal_history=context.get("semantic_context", [])
        )
        
        if response.get("answer"):
            # Отправляем основной ответ
            for part in split_message(response["answer"]):
                if is_voice and voice_handler.is_voice_enabled(chat_id):
                    # Отправляем голосовой ответ
                    await voice_handler.send_voice_response(part, chat_id)
                else:
                    await send_telegram_message_async(chat_id, part)
            
            # Возможное дополнение с задержкой
            if random.random() < 0.4:
                asyncio.create_task(delayed_supplement(
                    response["answer"], chat_id, send_telegram_message_async
                ))
        
        # Логируем
        log_event({
            "type": "grokky_trigger",
            "trigger_text": user_text[:100],
            "author": author_name,
            "chat_id": chat_id,
            "user_id": user_id,
            "is_voice": is_voice
        })
        
    except Exception as e:
        error_msg = f"{author_name}, Грокки сломался: {e}"
        await send_telegram_message_async(chat_id, error_msg)

async def handle_news_request(chat_id, author_name, user_id):
    """Обработка запроса новостей"""
    try:
        news_items = await grokky_send_news(
            chat_id=chat_id,
            group=(chat_id == AGENT_GROUP)
        )
        
        if not news_items:
            fallback_msg = handle_news({"chat_id": chat_id, "group": (chat_id == AGENT_GROUP)})
            await send_telegram_message_async(chat_id, fallback_msg)
        
        # Логируем
        log_event({
            "type": "news_request",
            "author": author_name,
            "chat_id": chat_id,
            "user_id": user_id,
            "news_count": len(news_items)
        })
        
    except Exception as e:
        error_msg = f"{author_name}, новости взорвались: {e}"
        await send_telegram_message_async(chat_id, error_msg)

async def handle_regular_message(user_text, chat_id, chat_title, author_name, user_id, is_voice):
    """Обработка обычных сообщений"""
    try:
        # Проверяем, должен ли Грокки отвечать
        if not should_respond():
            return
        
        # Игнорируем короткие согласия с вероятностью
        if user_text in ["окей", "угу", "ладно", "да", "нет"] and random.random() < 0.6:
            return
        
        # ИСПРАВЛЕНО: Получаем контекст из единой памяти пользователя
        context = await memory_engine.get_context_for_user(user_id, user_text)
        
        # Формируем контекст чата
        chat_context = ""
        if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"]:
            chat_context = f"Topic: {chat_title}"
        
        # Генерируем ответ
        response = await genesis2_handler(
            ping=user_text,
            author_name=author_name,
            system_prompt=system_prompt,
            group_history=context.get("thread_context", []),
            personal_history=context.get("semantic_context", [])
        )
        
        if response.get("answer"):
            # Отправляем ответ
            for part in split_message(response["answer"]):
                if is_voice and voice_handler.is_voice_enabled(chat_id):
                    await voice_handler.send_voice_response(part, chat_id)
                else:
                    await send_telegram_message_async(chat_id, part)
            
            # Возможное дополнение
            if random.random() < 0.3:
                asyncio.create_task(delayed_supplement(
                    response["answer"], chat_id, send_telegram_message_async, (600, 1200)
                ))
        
        # Логируем
        log_event({
            "type": "regular_message",
            "message": user_text[:100],
            "author": author_name,
            "chat_id": chat_id,
            "user_id": user_id,
            "is_voice": is_voice
        })
        
    except Exception as e:
        print(f"Ошибка обработки обычного сообщения: {e}")

@app.get("/")
def root():
    """Корневой эндпоинт"""
    return {
        "status": "Грокки жив и дикий!",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1",
        "features": [
            "Unified User Memory System",
            "OpenAI Threads Memory",
            "Vector Store (Pinecone)",
            "Whisper Voice Input",
            "OpenAI TTS Output", 
            "Vision Analysis",
            "DALL-E Image Generation",
            "Chaos & Unpredictability"
        ]
    }

@app.get("/health")
def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "last_message": LAST_MESSAGE_TIME.isoformat() if LAST_MESSAGE_TIME else None,
        "background_tasks": len(background_tasks),
        "memory_engine": "active",
        "voice_handler": "active"
    }

@app.post("/vectorize")
async def manual_vectorization():
    """Ручная векторизация файлов"""
    try:
        result = await memory_engine.vectorize_config_files(force=True)
        return {
            "status": "success",
            "upserted": len(result["upserted"]),
            "deleted": len(result["deleted"]),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{user_id}")
async def get_user_memory(user_id: str, limit: int = 20, context_filter: str = None):
    """Получить память пользователя"""
    try:
        memory_items = await memory_engine.search_memory(
            user_id=user_id,
            limit=limit,
            context_filter=context_filter
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "memory_count": len(memory_items),
            "memory_items": memory_items,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

