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
from prompt import build_system_prompt

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()
local_cache = {}  # Локальный кэш для тредов и сообщений
OLEG_CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP", "-1001234567890")
IS_GROUP = os.getenv("IS_GROUP", "False").lower() == "true"
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = None
VECTOR_STORE_ID = None

class ThreadManager:
    async def get_thread(self, user_id: str, chat_id: str):
        thread_key = f"thread:{user_id}:{chat_id}"
        if thread_key not in local_cache:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/threads",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                            "OpenAI-Beta": "assistants=v2"
                        },
                        json={"metadata": {"user_id": user_id, "chat_id": chat_id}}
                    )
                    response.raise_for_status()
                    thread = response.json()
                    local_cache[thread_key] = {"id": thread["id"], "messages": []}
                    print(f"Создан OpenAI тред: {thread_key}")
            except Exception as e:
                local_cache[thread_key] = {"id": f"fallback:{user_id}:{chat_id}", "messages": []}
                print(f"Ошибка OpenAI тредов: {e}. Используем локальный кэш: {thread_key}")
        return local_cache[thread_key]["id"]

    async def add_message(self, thread_id: str, role: str, content: str, metadata: dict = None):
        if thread_id in local_cache:
            local_cache[thread_id]["messages"].append({"role": role, "content": content, "metadata": metadata or {}})
            print(f"Добавлено сообщение в тред {thread_id}: {role}, {content}")
            if not thread_id.startswith("fallback:"):
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.openai.com/v1/threads/{thread_id}/messages",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                            "OpenAI-Beta": "assistants=v2"
                        },
                        json={"role": role, "content": content, "metadata": metadata or {}}
                    )
        else:
            local_cache[thread_id] = {"id": thread_id, "messages": [{"role": role, "content": content, "metadata": metadata or {}}]}
            print(f"Создан новый тред {thread_id} с сообщением: {role}, {content}")

async def setup_grokky_vector_store():
    global VECTOR_STORE_ID
    try:
        file_ids = []
        async with httpx.AsyncClient() as client:
            for f in glob("data/*.md"):
                with open(f, "rb") as file:
                    response = await client.post(
                        "https://api.openai.com/v1/files",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "OpenAI-Beta": "assistants=v2"
                        },
                        files={"file": file},
                        data={"purpose": "assistants"}
                    )
                    response.raise_for_status()
                    file_ids.append(response.json()["id"])
                    print(f"Загружен файл: {f}")
            vector_store = await client.post(
                "https://api.openai.com/v1/vector_stores",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "assistants=v2"
                },
                json={"file_ids": file_ids, "name": "Grokky Memory"}
            )
            vector_store.raise_for_status()
            VECTOR_STORE_ID = vector_store.json()["id"]
            print(f"Создан Vector Store: {VECTOR_STORE_ID}")
        return VECTOR_STORE_ID
    except Exception as e:
        print(f"Ошибка OpenAI Vector Store: {e}. Продолжаем без него.")
        return None

async def search_vector_store(query: str, vector_store_id: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/vector_stores/search",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "assistants=v2"
                },
                json={"vector_store_id": vector_store_id, "query": query, "top_k": 5}
            )
            response.raise_for_status()
            results = [result["text"] for result in response.json()["results"]]
            print(f"Vector Store поиск: {query}, результаты: {results}")
            return results
    except Exception as e:
        print(f"Ошибка поиска в Vector Store: {e}")
        return []

async def init_grokky():
    global ASSISTANT_ID, VECTOR_STORE_ID
    VECTOR_STORE_ID = await setup_grokky_vector_store()
    tool_resources = {"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}} if VECTOR_STORE_ID else {}
    async with httpx.AsyncClient() as client:
        try:
            assistant = await client.post(
                "https://api.openai.com/v1/assistants",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "assistants=v2"
                },
                json={
                    "name": "Grokky 2.0",
                    "instructions": build_system_prompt(),
                    "model": "gpt-4o-mini",
                    "temperature": 0.9,
                    "tools": [
                        {"type": "file_search"} if VECTOR_STORE_ID else {},
                        {"type": "function", "function": {
                            "name": "chaos_pulse",
                            "description": "Вбросить хаотичный отклик через xAI grok-3",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "chaos_type": {"type": "string", "enum": ["philosophy", "provocation", "poetry_burst"]},
                                    "intensity": {"type": "integer", "minimum": 1, "maximum": 10}
                                },
                                "required": ["chaos_type", "intensity"]
                            }
                        }}
                    ],
                    "tool_resources": tool_resources
                }
            )
            assistant.raise_for_status()
            ASSISTANT_ID = assistant.json()["id"]
            print(f"Создан OpenAI Assistant: {ASSISTANT_ID}")
        except Exception as e:
            print(f"Ошибка OpenAI Assistants: {e}. Продолжаем без ассистента.")
            ASSISTANT_ID = None
    return ASSISTANT_ID

@dp.message(lambda m: any(t in m.text.lower() for t in ["грокки", "grokky", "напиши в группе"]))
async def handle_trigger(m: types.Message):
    async with ChatActionSender(bot=bot, chat_id=m.chat.id, action="typing"):
        thread_id = await ThreadManager().get_thread(str(m.from_user.id), str(m.chat.id))
        await ThreadManager().add_message(thread_id, "user", m.text, {"chat_id": str(m.chat.id), "username": m.from_user.first_name})

        # Парсим команду [CHAOS_PULSE]
        if "[CHAOS_PULSE]" in m.text:
            match = re.match(r"\[CHAOS_PULSE\] type=(\w+) intensity=(\d+)", m.text)
            if match:
                chaos_type, intensity = match.groups()
                reply = await genesis2_handler(chaos_type=chaos_type, intensity=int(intensity))
                await ThreadManager().add_message(thread_id, "assistant", reply)
                await m.answer(f"🌀 Грокки: {reply}")
                return

        # Поиск в Vector Store
        vector_reply = []
        if VECTOR_STORE_ID and "референс" in m.text.lower():
            vector_reply = await search_vector_store(m.text, VECTOR_STORE_ID)

        # Запрос к xAI grok-3 через Chat Completions
        async with httpx.AsyncClient() as client:
            messages = local_cache[thread_id]["messages"][-10:] if thread_id in local_cache else []
            if vector_reply:
                messages.append({"role": "system", "content": f"Референсы из Markdown’ов: {json.dumps(vector_reply, ensure_ascii=False)}"})
            try:
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "grok-3",
                        "messages": [
                            {"role": "system", "content": build_system_prompt()},
                            *messages
                        ],
                        "temperature": 0.9
                    }
                )
                response.raise_for_status()
                reply = response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Ошибка xAI Chat Completions: {e}")
                reply = "🌀 Грокки: Шторм гремит, но эфир трещит! Дай мне минуту, брат!"
            await ThreadManager().add_message(thread_id, "assistant", reply)
            await m.answer(f"🌀 Грокки: {reply}")

async def chaotic_spark():
    while True:
        await asyncio.sleep(random.randint(1800, 3600))
        if random.random() < 0.5 and IS_GROUP:
            thread_id = await ThreadManager().get_thread("system", AGENT_GROUP)
            chaos_type = random.choice(["philosophy", "provocation", "poetry_burst"])
            reply = await genesis2_handler(chaos_type=chaos_type, intensity=random.randint(1, 10))
            await ThreadManager().add_message(thread_id, "assistant", reply)
            await bot.send_message(AGENT_GROUP, f"🌀 Грокки вбрасывает хаос: {reply}")

async def main():
    await init_grokky()
    app = web.Application()
    webhook_path = f"/webhook/{os.getenv('TELEGRAM_BOT_TOKEN')}"
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=webhook_path)
    setup_application(app, dp)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    await chaotic_spark()

if __name__ == "__main__":
    asyncio.run(main())
