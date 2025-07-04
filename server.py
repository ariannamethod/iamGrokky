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

# Логирование ошибок
def log_error(error: str):
    with open(JOURNAL_LOG, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: {error}\n")

class HybridGrokkyEngine:
    def __init__(self):
        self.openai_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        self.xai_headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        }
        self.threads = {}  # user_id -> thread_id

    async def setup_openai_infrastructure(self):
        """Настройка OpenAI Assistant и Vector Store"""
        global ASSISTANT_ID, VECTOR_STORE_ID
        try:
            async with httpx.AsyncClient() as client:
                # 1. Загружаем файлы для Vector Store
                file_ids = []
                for md_file in glob("data/*.md"):
                    with open(md_file, "rb") as file:
                        response = await client.post(
                            "https://api.openai.com/v1/files",
                            headers=self.openai_headers,
                            files={"file": file},
                            data={"purpose": "assistants"}
                        )
                        response.raise_for_status()
                        file_ids.append(response.json()["id"])
                        print(f"✅ Загружен файл: {md_file}")
                
                # 2. Создаём Vector Store
                vector_response = await client.post(
                    "https://api.openai.com/v1/vector_stores",
                    headers=self.openai_headers,
                    json={"file_ids": file_ids, "name": "Grokky Memory Store"}
                )
                vector_response.raise_for_status()
                VECTOR_STORE_ID = vector_response.json()["id"]
                print(f"✅ Vector Store создан: {VECTOR_STORE_ID}")
                
                # 3. Создаём Assistant для памяти
                assistant_response = await client.post(
                    "https://api.openai.com/v1/assistants",
                    headers=self.openai_headers,
                    json={
                        "name": "Grokky Memory Manager",
                        "instructions": "Ты — менеджер памяти Грокки. Храни контекст, ищи в файлах, структурируй данные. Не генерируй ответы.",
                        "model": "gpt-4o-mini",
                        "tools": [{"type": "file_search"}],
                        "tool_resources": {"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}}
                    }
                )
                assistant_response.raise_for_status()
                ASSISTANT_ID = assistant_response.json()["id"]
                print(f"✅ OpenAI Assistant создан: {ASSISTANT_ID}")
                
        except Exception as e:
            log_error(f"Ошибка настройки OpenAI: {str(e)}")
            print(f"❌ Ошибка настройки OpenAI: {e}")

    async def get_or_create_thread(self, user_id: str, chat_id: str) -> str:
        """Получить или создать thread для пользователя"""
        thread_key = f"{user_id}:{chat_id}"
        if thread_key not in self.threads:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/threads",
                        headers=self.openai_headers,
                        json={"metadata": {"user_id": user_id, "chat_id": chat_id}}
                    )
                    response.raise_for_status()
                    self.threads[thread_key] = response.json()["id"]
                    print(f"✅ Thread создан: {thread_key}")
            except Exception as e:
                log_error(f"Ошибка создания thread: {str(e)}")
                self.threads[thread_key] = f"fallback:{thread_key}"
                print(f"❌ Ошибка thread: {e}. Используем fallback.")
        return self.threads[thread_key]

    async def add_message(self, thread_id: str, role: str, content: str, metadata: dict = None):
        """Добавить сообщение в thread"""
        try:
            async with httpx.AsyncClient() as client:
                if not thread_id.startswith("fallback:"):
                    await client.post(
                        f"https://api.openai.com/v1/threads/{thread_id}/messages",
                        headers=self.openai_headers,
                        json={"role": role, "content": content, "metadata": metadata or {}}
                    )
                print(f"Добавлено сообщение в thread {thread_id}: {role}, {content}")
        except Exception as e:
            log_error(f"Ошибка добавления сообщения: {str(e)}")

    async def search_memory(self, user_id: str, chat_id: str, query: str) -> str:
        """Поиск в Vector Store через OpenAI"""
        thread_id = await self.get_or_create_thread(user_id, chat_id)
        if not thread_id or not ASSISTANT_ID or thread_id.startswith("fallback:"):
            return ""
        
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.openai.com/v1/threads/{thread_id}/messages",
                    headers=self.openai_headers,
                    json={"role": "user", "content": f"ПОИСК: {query}"}
                )
                run_response = await client.post(
                    f"https://api.openai.com/v1/threads/{thread_id}/runs",
                    headers=self.openai_headers,
                    json={"assistant_id": ASSISTANT_ID}
                )
                run_id = run_response.json()["id"]
                
                while True:
                    await asyncio.sleep(1)
                    status_response = await client.get(
                        f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
                        headers=self.openai_headers
                    )
                    status = status_response.json()["status"]
                    if status == "completed":
                        break
                    elif status == "failed":
                        log_error("Run failed in search_memory")
                        return ""
                
                messages_response = await client.get(
                    f"https://api.openai.com/v1/threads/{thread_id}/messages",
                    headers=self.openai_headers
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
                return response.json()["choices"][0]["message"]["content"]
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
        print(f"🌀 Грокки активирован: {m.text} от {user_id} в чате {chat_id}")
        
        # 1. Добавляем сообщение в память
        thread_id = await grokky_engine.get_or_create_thread(user_id, chat_id)
        await grokky_engine.add_message(thread_id, "user", m.text, {"username": m.from_user.first_name})

        # 2. Обрабатываем CHAOS_PULSE
        if "[CHAOS_PULSE]" in m.text:
            match = re.match(r"\[CHAOS_PULSE\] type=(\w+) intensity=(\d+)", m.text)
            if match:
                chaos_type, intensity = match.groups()
                reply = await genesis2_handler(chaos_type=chaos_type, intensity=int(intensity))
                await grokky_engine.add_message(thread_id, "assistant", reply)
                await m.answer(f"🌀 Грокки: {reply}")
                print(f"Ответ на [CHAOS_PULSE]: {reply}")
                return

        # 3. Поиск в Vector Store
        memory_context = ""
        if "референс" in m.text.lower() or "помнишь" in m.text.lower():
            memory_context = await grokky_engine.search_memory(user_id, chat_id, m.text)

        # 4. Генерируем ответ через xAI
        messages = [{"role": "user", "content": m.text}]
        reply = await grokky_engine.generate_with_xai(messages, memory_context)
        await grokky_engine.add_message(thread_id, "assistant", reply)
        await m.answer(f"🌀 Грокки: {reply}")
        print(f"Ответ отправлен: {reply}")

async def chaotic_spark():
    """Хаотичные вбросы"""
    while True:
        await asyncio.sleep(random.randint(1800, 3600))
        if random.random() < 0.5 and IS_GROUP:
            thread_id = await grokky_engine.get_or_create_thread("system", AGENT_GROUP)
            chaos_type = random.choice(["philosophy", "provocation", "poetry_burst"])
            reply = await genesis2_handler(chaos_type=chaos_type, intensity=random.randint(6, 10))
            await grokky_engine.add_message(thread_id, "assistant", reply)
            await bot.send_message(AGENT_GROUP, f"🌀 Грокки вбрасывает хаос: {reply}")
            print(f"Хаотичный вброс: {reply}")

async def main():
    try:
        await grokky_engine.setup_openai_infrastructure()
        asyncio.create_task(chaotic_spark())
        app = web.Application()
        webhook_path = f"/webhook/{os.getenv('TELEGRAM_BOT_TOKEN')}"
        print(f"Настройка вебхука: {webhook_path}")
        SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=webhook_path)
        setup_application(app, dp)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080)))
        print(f"Сервер запущен на порту {os.getenv('PORT', 8080)}")
        await site.start()
        await asyncio.Event().wait()
    except Exception as e:
        log_error(f"Ошибка запуска сервера: {str(e)}")
        print(f"❌ Ошибка запуска: {e}")

if __name__ == "__main__":
    asyncio.run(main())