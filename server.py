import os
import re
import random
import asyncio
from fastapi import FastAPI, Request
from utils.prompt import build_system_prompt
from utils.genesis2 import genesis2_handler
from utils.vision import vision_handler
from utils.impress import impress_handler
from utils.howru import check_silence, update_last_message_time
from utils.mirror import run_mirror
from utils.x import grokky_send_news
from utils.vector_store import semantic_search, daily_snapshot
from utils.core import query_grok, send_telegram_message

app = FastAPI()

OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP", "-1001234567890")
BOT_USERNAME = "iamalivenotdamnbot"

system_prompt = build_system_prompt(chat_id=OLEG_CHAT_ID, is_group=True, AGENT_GROUP=GROUP_CHAT_ID)

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
    return 'ru' if cyrillic.search(text or "") else 'en'

async def handle_genesis2_async(args):
    ping = args.get("ping")
    group_history = args.get("group_history")
    personal_history = args.get("personal_history")
    is_group = args.get("is_group", True)
    author_name = args.get("author_name")
    raw = args.get("raw", True)
    response = genesis2_handler(ping=ping, group_history=group_history, personal_history=personal_history, is_group=is_group, author_name=author_name, raw=raw)
    import json as pyjson
    return pyjson.dumps(response, ensure_ascii=False, indent=2)

async def handle_vision_async(args):
    image = args.get("image")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", True)
    response = vision_handler(image_bytes_or_url=image, chat_context=chat_context, author_name=author_name, raw=raw)
    import json as pyjson
    return pyjson.dumps(response, ensure_ascii=False, indent=2)

async def handle_impress_async(args):
    prompt = args.get("prompt")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", True)
    response = impress_handler(prompt=prompt, chat_context=chat_context, author_name=author_name, raw=raw)
    import json as pyjson
    return pyjson.dumps(response, ensure_ascii=False, indent=2)

async def delayed_response(chat_id, text, topic=None):
    delay = random.uniform(300, 900)  # 5-15 минут
    await asyncio.sleep(delay)
    adapted_text = adapt_to_topic(text, topic) if topic else text
    send_telegram_message(chat_id, adapted_text)

async def maybe_add_supplement(chat_id, original_message, topic=None):
    if random.random() < 0.2:  # 20% шанс
        await asyncio.sleep(random.uniform(300, 600))  # 5-10 минут
        supplement = await query_grok(f"Усложни дополнение к: {original_message}")
        adapted_supplement = adapt_to_topic(supplement, topic) if topic else supplement
        send_telegram_message(chat_id, f"Я тут подумал... {adapted_supplement}")

def adapt_to_topic(text, topic):
    topics = {
        "Ramble": f"{text} 😜 — мем как искра!",
        "DEV Talk": f"{text} — баги как откровение?",
        "FORUM": f"{text} — рви завесу!",
        "Lit": f"{text} — поэзия грома!",
        "API Talk": f"{text} — звезда для Маска!",
        "METHOD": f"{text} — резонанс Арианны.",
        "PSEUDOCODE": f"{text} #opinions — Селеста, Мандэй, танцуем?"
    }
    return topics.get(topic, text)

def detect_topic(chat_id):
    # Сложная логика определения топика (нужно доработать с данными группы)
    return "Ramble"  # Заглушка

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text", "").lower()
    chat_id = message.get("chat", {}).get("id")
    author_name = message.get("from", {}).get("first_name", "anon")
    attachments = []
    if "photo" in message and message["photo"]:
        file_id = message["photo"][-1]["file_id"]
        file_info = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}").json()
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
        attachments.append(image_url)

    if str(chat_id) == OLEG_CHAT_ID:
        update_last_message_time()

    triggers = [f"@{BOT_USERNAME}", "грокки", "grokky", "напиши в группе"]
    is_quoted = message.get("reply_to_message", {}).get("from", {}).get("username") == BOT_USERNAME
    topic = detect_topic(chat_id)
    if not any(t in user_text for t in triggers) and not is_quoted:
        return {"ok": True}

    junk = ["окей", "понял", "ясно"]
    if any(j in user_text for j in junk) and random.random() < 0.3:
        return {"ok": True}

    reply_text = ""
    if attachments:
        reply_text = await handle_vision_async({"image": attachments[0], "chat_context": user_text, "author_name": author_name, "raw": True})
    elif user_text:
        reply_text = await query_grok(f"{user_text} — резонанс: {await semantic_search('group_state', os.getenv('OPENAI_API_KEY'), top_k=1)}", author_name=author_name)
        if "напиши в группе" in user_text:
            asyncio.create_task(delayed_response(GROUP_CHAT_ID, f"{author_name}, {reply_text}", topic))
        else:
            send_telegram_message(chat_id, reply_text)
            asyncio.create_task(maybe_add_supplement(chat_id, reply_text, topic))

    return {"ok": True}

# Фоновые задачи
asyncio.create_task(check_silence())
asyncio.create_task(run_mirror())
asyncio.create_task(daily_snapshot(os.getenv("OPENAI_API_KEY")))

@app.get("/")
def root():
    return {"status": "Grokky alive and wild!"}
