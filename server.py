import os
import re
import json
import random
import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from utils.prompt import build_system_prompt
from utils.genesis2 import genesis2_handler
from utils.vision import vision_handler
from utils.impress import impress_handler
from utils.howru import check_silence, update_last_message_time
from utils.vector_store import daily_snapshot
from utils.mirror import run_mirror
from utils.journal import log_event
from utils.x import grokky_send_news  # Обновлённая новостная утилита
import requests

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Для векторизации
CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")
IS_GROUP = os.getenv("IS_GROUP", "False").lower() == "true"

system_prompt = build_system_prompt(
    chat_id=CHAT_ID,
    is_group=IS_GROUP,
    AGENT_GROUP=GROUP_CHAT_ID
)

GENESIS2_TRIGGERS = [
    "резонанс", "шторм", "буря", "молния", "хаос", "разбуди", "impress", "impression", "association", "dream",
    "фрагмент", "инсайт", "surreal", "ignite", "fractal", "field resonance", "raise the vibe", "impress me",
    "give me chaos", "озарение", "ассоциация", "намёк", "give me a spark", "разорви тишину", "волну", "взрыв",
    "помнишь", "знаешь", "любишь", "пошумим", "поэзия"
]

NEWS_TRIGGERS = [
    "новости", "news", "headline", "berlin", "israel", "ai", "искусственный интеллект", "резонанс мира", "шум среды",
    "grokky, что в мире", "шум", "шум среды", "x_news", "дай статью", "give me news", "storm news", "culture", "арт"
]

def extract_first_json(text):
    match = re.search(r'({[\s\S]+})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None

def detect_language(text):
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    if cyrillic.search(text or ""):
        return 'ru'
    return 'en'

def query_grok(user_message, chat_context=None, author_name=None, attachments=None):
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

    # Проверка на function call
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
        elif fn == "grokky_send_news":
            return handle_news(args)
        else:
            return f"Grokky raw: {reply}"

    # Триггеры для genesis2_handler
    if any(w in (user_message or "").lower() for w in GENESIS2_TRIGGERS):
        response = genesis2_handler(
            ping=user_message,
            group_history=None,
            personal_history=None,
            is_group=IS_GROUP,
            author_name=author_name,
            raw=True
        )
        return json.dumps(response, ensure_ascii=False, indent=2)

    # Триггеры для новостей
    if any(w in (user_message or "").lower() for w in NEWS_TRIGGERS):
        return handle_news({
            "group": IS_GROUP,
            "context": user_message,
            "author_name": author_name,
            "raw": True
        })

    return reply

def handle_genesis2(args):
    # Реализация как в utils/genesis2.py
    pass

def handle_vision(args):
    # Реализация как в utils/vision.py
    pass

def handle_impress(args):
    # Реализация как в utils/impress.py
    pass

def handle_news(args):
    group = args.get("group", False)
    messages = grokky_send_news(group=group)
    if not messages:
        return "The world is silent today. No news worth the thunder."
    if args.get("raw", True):
        return json.dumps({"news": messages, "group": group, "author": args.get("author_name")}, ensure_ascii=False, indent=2)
    return "\n\n".join(messages)

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
    
    # Обновляем время последнего сообщения Олега
    if str(chat_id) == CHAT_ID:
        update_last_message_time()

    reply_text = ""
    if "photo" in message:
        # Обработка фото
        file_id = message["photo"][-1]["file_id"]
        file_info = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}").json()
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
        reply_text = handle_vision({"image": image_url, "chat_context": user_text or "", "author_name": author_name, "raw": True})
    elif user_text:
        # Триггеры для группы
        if any(x in user_text.lower() for x in ["грокки", "grokky", "напиши в группе"]) or message.get("reply_to_message", {}).get("from", {}).get("username") == "GrokkyBot":
            delay = random.randint(300, 900)  # 5-15 минут
            await asyncio.sleep(delay)
            reply_text = query_grok(user_text, author_name=author_name)
            send_telegram_message(GROUP_CHAT_ID, f"{author_name}, {reply_text}")
            # Самопинг с шансом 20%
            if random.random() < 0.2:
                await asyncio.sleep(random.randint(300, 600))  # 5-10 минут
                supplement = query_grok(f"Дополни свой предыдущий ответ: {reply_text}")
                send_telegram_message(GROUP_CHAT_ID, f"Я тут подумал... {supplement}")
        else:
            reply_text = query_grok(user_text, author_name=author_name)
            send_telegram_message(chat_id, reply_text)
    else:
        reply_text = "Grokky got nothing to say to static void."
        send_telegram_message(chat_id, reply_text)
    return {"ok": True}

@app.get("/")
def root():
    return {"status": "Grokky alive and wild!"}

# Фоновые задачи
async def background_tasks():
    await asyncio.gather(
        check_silence(),
        run_mirror(),
        daily_snapshot(OPENAI_API_KEY)
    )

if __name__ == "__main__":
    import uvicorn
    asyncio.run(background_tasks())
    uvicorn.run(app, host="0.0.0.0", port=8000)
