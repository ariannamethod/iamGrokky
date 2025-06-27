import os
import asyncio
import random
import json
import requests
from datetime import datetime, timedelta
from utils.vector_store import semantic_search
from utils.journal import log_event
from utils.grok_utils import query_grok
from utils.telegram_utils import send_telegram_message

LAST_MESSAGE_TIME = None
OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def get_last_messages(limit=10):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset=-{limit}"
    try:
        resp = requests.get(url).json()
        messages = []
        for update in resp.get("result", []):
            if "message" in update and "text" in update["message"]:
                messages.append(update["message"]["text"])
        return messages[:limit]
    except Exception:
        return []

def get_journal_entries(limit=10):
    try:
        with open("data/journal.json", "r", encoding="utf-8") as f:
            journal = json.load(f)
        return journal[-limit:]
    except Exception:
        return []

async def check_silence():
    global LAST_MESSAGE_TIME
    while True:
        await asyncio.sleep(3600)  # Проверка каждые час
        if not LAST_MESSAGE_TIME:
            continue
        silence = datetime.now() - LAST_MESSAGE_TIME
        if silence > timedelta(hours=48):
            await handle_48h_silence()
        elif silence > timedelta(hours=24):
            await handle_24h_silence()
        # Спонтанные сообщения раз в 12 часов с шансом 50%, если молчание < 24 часов
        elif silence > timedelta(hours=12) and random.random() < 0.5:
            context = await build_context()
            message = query_grok("Олег молчал 12 часов. Швырни спонтанный заряд!", context)
            send_telegram_message(OLEG_CHAT_ID, message)
            log_event({"type": "howru_spontaneous", "message": message, "timestamp": datetime.now().isoformat()})

async def handle_24h_silence():
    context = await build_context()
    message = query_grok("Олег молчал 24 часа. Напиши что-то спонтанное!", context)
    send_telegram_message(OLEG_CHAT_ID, message)
    log_event({"type": "howru_24h", "message": message, "timestamp": datetime.now().isoformat()})

async def handle_48h_silence():
    context = await build_context()
    message = query_grok("Олег молчал 48 часов. Напиши что-то острое!", context)
    send_telegram_message(OLEG_CHAT_ID, message)
    if LAST_MESSAGE_TIME:  # Проверка на None
        group_msg = f"Олег молчал 48 часов. Последний раз видел: {LAST_MESSAGE_TIME.isoformat()}"
    else:
        group_msg = "Олег молчал 48 часов. Нет данных о последнем сообщении."
    send_telegram_message(GROUP_CHAT_ID, group_msg)
    log_event({"type": "howru_48h", "message": message, "group_msg": group_msg, "timestamp": datetime.now().isoformat()})

async def build_context():
    last_msgs = get_last_messages(10)
    journal = get_journal_entries(10)
    snapshot = await semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1)
    return f"Последние сообщения: {json.dumps(last_msgs, ensure_ascii=False)}\nЖурнал: {json.dumps(journal, ensure_ascii=False)}\nСнимок: {snapshot}"

def update_last_message_time():
    global LAST_MESSAGE_TIME
    LAST_MESSAGE_TIME = datetime.now()
