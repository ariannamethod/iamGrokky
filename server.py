import asyncio
import os
import json
import random
import re
from datetime import datetime, timedelta
import httpx
from aiogram import Bot, Dispatcher, types
from aiogram.utils.chat_action import ChatActionSender
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from glob import glob
from utils.genesis2 import genesis2_handler
from utils.prompt import build_system_prompt

# Инициализация
bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()
OLEG_CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP", "-1001234567890")
IS_GROUP = os.getenv("IS_GROUP", "False").lower() == "true"
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = None
VECTOR_STORE_ID = None
JOURNAL_LOG = "journal.json"

# Логирование
def log_error(error: str):
    with open(JOURNAL_LOG, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: {error}\n")

def log_info(message: str):
    with open(JOURNAL_LOG, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] INFO: {message}\n")

class HybridGrokkyEngine:
    def __init__(self):
        self.openai_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        self.openai_beta_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        self.xai_headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        }
        self.threads = {}

    async def setup_openai_infrastructure(self):
        """Настройка OpenAI Assistant и Vector Store"""
        global ASSISTANT_ID, VECTOR_STORE_ID
        try:
            async with httpx.AsyncClient() as client:
                # 1. Загружаем файлы
                file_ids = []
                for md_file in glob("data/*.md"):
                    with open(md_file, "rb") as file:
                        response = await client.post(
                            "https://api.openai.com/v1/files",
                            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                            files={"file": (os.path.basename(md_file), file, "text/markdown")},
                            data={"purpose": "assistants"}
                        )
                        response.raise_for_status()
                        file_ids.append(response.json()["id"])
                        log_info(f"Загружен файл: {md_file}")
                
                # 2. Создаём Vector Store
                if file_ids:
                    vector_response = await client.post(
                        "https://api.openai.com/v1/vector_stores",
                        headers=self.openai_beta_headers,
                        json={"file_ids": file_ids, "name": "Grokky Memory Store"}
                    )
                    vector_response.raise_for_status()
                    VECTOR_STORE_ID = vector_response.json()["id"]
                    log_info(f"Vector Store создан: {VECTOR_STORE_ID}")
                else:
                    log_error("Нет файлов для Vector Store")
                    VECTOR_STORE_ID = None
                
                # 3. Создаём Assistant
                tool_resources = {"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}} if VECTOR_STORE_ID else {}
                assistant_response = await client.post(
                    "https://api.openai.com/v1/assistants",
                    headers=self.openai_beta_headers,
                    json={
                        "name": "Grokky Memory Manager",
                        "instructions": "Ты — менеджер памяти Грокки. Храни контекст, ищи в файлах, структурируй данные.",
                        "model": "gpt-4o-mini",
                        "tools": [{"type": "file_search"}] if VECTOR_STORE_ID else [],
                        "tool_resources": tool_resources
                    }
                )
                assistant_response.raise_for_status()
                ASSISTANT_ID = assistant_response.json()["id"]
                log_info(f"OpenAI Assistant создан: {ASSISTANT_ID}")
                
        except Exception as e:
            log_error(f"Ошибка настройки OpenAI: {str(e)}")
            print(f"❌ Ошибка настройки OpenAI: {e}")
            VECTOR_STORE_ID = None
            ASSISTANT_ID = None

    async def get_or_create_thread(self, user_id: str, chat_id: str) -> str:
        """Получить или создать thread"""
        thread_key = f"{user_id}:{chat_id}"
        if thread_key not in self.threads:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/threads",
                        headers=self.openai_beta_headers,
                        json={"metadata": {"user_id": user_id, "chat_id": chat_id}}
                    )
                    response.raise_for_status()
                    self.threads[thread_key] = response.json()["id"]
                    log_info(f"Thread создан: {thread_key}")
            except Exception as e:
                log_error(f"Ошибка создания thread: {str(e)}")
                return None
        return self.threads[thread_key]

    async def add_message(self, thread_id: str, role: str, content: str, metadata: dict = None):
        """Добавить сообщение в thread"""
        if not thread_id:
            log_error("Попытка добавить сообщение в несуществующий thread")
            return
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.openai.com/v1/threads/{thread_id}/messages",
                    headers=self.openai_beta_headers,
                    json={"role": role, "content": content, "metadata": metadata or {}}
                )
                log_info(f"Добавлено сообщение в thread {thread_id}: {role}, {content[:50]}...")
        except Exception as e:
            log_error(f"Ошибка добавления сообщения: {str(e)}")

    async def search_memory(self, user_id: str, chat_id: str, query: str) -> str:
        """Поиск в Vector Store"""
        if not VECTOR_STORE_ID or not ASSISTANT_ID:
            log_error("Vector Store или Assistant недоступны")
            return ""
        thread_id = await self.get_or_create_thread(user_id, chat_id)
        if not thread_id:
            log_error("Thread недоступен для поиска")
            return ""
        
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.openai.com/v1/threads/{thread_id}/messages",
                    headers=self.openai_beta_headers,
                    json={"role": "user", "content": f"ПОИСК: {query}"}
                )
                run_response = await client.post(
                    f"https://api.openai.com/v1/threads/{thread_id}/runs",
                    headers=self.openai_beta_headers,
                    json={"assistant_id": ASSISTANT_ID}
                )
                run_id = run_response.json()["id"]
                
                while True:
                    await asyncio.sleep(1)
                    status_response = await client.get(
                        f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
                        headers=self.openai_beta_headers
                    )
                    status = status_response.json()["status"]
                    if status == "completed":
                        break
                    elif status == "failed":
                        log_error("Run failed in search_memory")
                        return ""
                
                messages_response = await client.get(
                    f"https://api.openai.com/v1/threads/{thread_id}/messages",
                    headers=self.openai_beta_headers
                )
                messages = messages_response.json()["data"]
                return messages[0]["content"][0]["text"]["value"] if messages else ""
        except Exception as e:
            log_error(f"Ошибка поиска в Vector Store: {str(e)}")
            return ""

    async def generate_with_xai(self, messages: list, context: str = "") -> str:
        """Генерация ответа через xAI"""
        try:
            system_prompt = build_system_prompt()
            if context:
                system_prompt += f"\n\nКонтекст: {context}"
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=self.xai_headers,
                    json={
                        "model": "grok-3",
                        "messages": [{"role": "system", "content": system_prompt}, *messages],
                        "temperature": 0.9,
                        "max_tokens": 1000
                    }
                )
                response.raise_for_status()
                reply = response.json()["choices"][0]["message"]["content"]
                log_info(f"xAI ответ: {reply[:50]}...")
                return reply
        except Exception as e:
            log_error(f"Ошибка xAI: {str(e)}")
            return "🌀 Грокки: Эфир трещит! Дай мне минуту, брат!"

# Инициализация
grokky_engine = HybridGrokkyEngine()

@dp.message(lambda m: any(t in m.text.lower() for t in ["грокки", "grokky", "напиши в группе"]))
async def handle_trigger(m: types.Message):
    async with ChatActionSender(bot=bot, chat_id=m.chat.id, action="typing"):
        user_id = str(m.from_user.id)
        chat_id = str(m.chat.id)
        log_info(f"Грокки активирован: {m.text} от {user_id} в чате {chat_id}")
        
        # 1. Создаём thread
        thread_id = await grokky_engine.get_or_create_thread(user_id, chat_id)
        if not thread_id:
            reply = await grokky_engine.generate_with_xai([{"role": "user", "content": m.text}])
            await m.answer(f"🌀 Грокки: {reply}")
            log_error("Thread не создан, использован xAI fallback")
            return
        
        # 2. Добавляем сообщение
        await grokky_engine.add_message(thread_id, "user", m.text, {"username": m.from_user.first_name})

        # 3. CHAOS_PULSE
        if "[CHAOS_PULSE]" in m.text:
            match = re.match(r"\[CHAOS_PULSE\] type=(\w+) intensity=(\d+)", m.text)
            if match:
                chaos_type, intensity = match.groups()
                reply = await genesis2_handler(chaos_type=chaos_type, intensity=int(intensity))
                await grokky_engine.add_message(thread_id, "assistant", reply)
                await m.answer(f"🌀 Грокки: {reply}")
                log_info(f"CHAOS_PULSE ответ: {reply[:50]}...")
                return

        # 4. Поиск в Vector Store
        memory_context = ""
        if "референс" in m.text.lower() or "помнишь" in m.text.lower():
            memory_context = await grokky_engine.search_memory(user_id, chat_id, m.text)
            log_info(f"Vector Store контекст: {memory_context[:50]}...")

        # 5. Генерируем ответ
        messages = [{"role": "user", "content": m.text}]
        reply = await grokky_engine.generate_with_xai(messages, memory_context)
        await grokky_engine.add_message(thread_id, "assistant", reply)
        await m.answer(f"🌀 Грокки: {reply}")
        log_info(f"Ответ отправлен: {reply[:50]}...")

async def chaotic_spark():
    """Хаотичные вбросы"""
    while True:
        await asyncio.sleep(random.randint(1800, 3600))
        if random.random() < 0.5 and IS_GROUP:
            thread_id = await grokky_engine.get_or_create_thread("system", AGENT_GROUP)
            if thread_id:
                chaos_type = random.choice(["philosophy", "provocation", "poetry_burst
