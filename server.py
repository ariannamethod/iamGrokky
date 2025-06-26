import os
import re
import requests
import asyncio
import random
from fastapi import FastAPI, Request
from utils.prompt import build_system_prompt
from utils.genesis2 import genesis2_handler
from utils.vision import vision_handler
from utils.impress import impress_handler
from utils.howru import check_silence, update_last_message_time
from utils.mirror import run_mirror
from utils.vector_store import daily_snapshot
from utils.journal import log_event

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")
is_group_env = os.getenv("IS_GROUP", "False").lower() == "true"

system_prompt = build_system_prompt(
    chat_id=OLEG_CHAT_ID,
    is_group=is_group_env,
    AGENT_GROUP=GROUP_CHAT_ID
)

GENESIS2_TRIGGERS = [
    "резонанс", "шторм", "буря", "молния", "хаос", "разбуди", "impress", "impression", "association", "dream",
    "фрагмент", "инсайт", "surreal", "ignite", "fractal", "field resonance", "raise the vibe", "impress me",
    "give me chaos", "озарение", "ассоциация", "намёк", "give me a spark", "разорви тишину", "волну", "взрыв",
    "помнишь", "знаешь", "любишь", "пошумим", "поэзия"
]

def extract_first_json(text):
    match = re.search(r'({[\s\S]+})', text)
    if match:
        import json as pyjson
        try:
            return pyjson.loads(match.group(1))
        except Exception:
            return None
    return None

def detect_language(text):
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    if cyrillic.search(text or ""):
        return 'ru'
    return 'en'

def query_grok(user_message, chat_context=None, author_name=None):
    url = "https://api.x.ai/v1/chat/completions"
    user_lang = detect_language(user_message)
    language_hint = {
        "role": "system",
        "content": f"Always reply in the language the user writes: {user_lang.upper()}. Never switch to another language unless explicitly asked."
    }
    messages = [
        {"role": "system", "content": system_prompt},
        language_hint,
        {"role": "user", "content": user_message}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 1.0
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]

    data = extract_first_json(reply)
    if data and "function_call" in data:
        fn = data["function_call"]["name"]
        args = data["function_call"]["arguments"]
        if fn == "genesis2_handler":
            return handle_genesis2(args)
        elif fn == "vision_handler":
            return handle_vision(args)
        elif fn == "impress_handler":
            return handle_impress(args)
        else:
            return f"Grokky raw: {reply}"

    if any(w in (user_message or "").lower() for w in GENESIS2_TRIGGERS):
        response = genesis2_handler(
            ping=user_message,
            group_history=None,
            personal_history=None,
            is_group=is_group_env,
            author_name=author_name,
            raw=True
        )
        import json as pyjson
        return pyjson.dumps(response, ensure_ascii=False, indent=2)

    return reply

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, data=payload)

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")
    author_name = message.get("from", {}).get("first_name", "anon")

    if str(chat_id) == OLEG_CHAT_ID:
        update_last_message_time()

    reply_text = ""
    if "photo" in message:
        # Обработка фото (добавь, если нужно)
        pass
    elif user_text:
        if any(x in user_text.lower() for x in ["грокки", "grokky", "напиши в группе"]) or message.get("reply_to_message", {}).get("from", {}).get("username") == "GrokkyBot":
            delay = random.randint(300, 900)  # 5-15 минут
            await asyncio.sleep(delay)
            reply_text = query_grok(user_text, author_name=author_name)
            send_telegram_message(GROUP_CHAT_ID, f"{author_name}, {reply_text}")
        else:
            reply_text = query_grok(user_text, author_name=author_name)
            send_telegram_message(chat_id, reply_text)
    else:
        reply_text = "Grokky got nothing to say to static void."
        send_telegram_message(chat_id, reply_text)

    asyncio.create_task(maybe_add_supplement(chat_id, reply_text))
    return {"ok": True}

async def maybe_add_supplement(chat_id, original_message):
    if random.random() < 0.2:  # 20% шанс
        await asyncio.sleep(random.randint(300, 600))  # 5-10 минут
        supplement = query_grok(f"Дополни свой предыдущий ответ: {original_message}")
        send_telegram_message(chat_id, f"Я тут подумал... {supplement}")

@app.get("/")
def root():
    return {"status": "Grokky alive and wild!"}

# Запуск фоновых задач
asyncio.create_task(check_silence())
asyncio.create_task(run_mirror())
asyncio.create_task(run_snapshots())

def run_snapshots():
    async def snapshots():
        while True:
            await daily_snapshot(os.getenv("OPENAI_API_KEY"))
            await was_asyncio.sleep(86400)  # Раз в день
    return snapshots()
