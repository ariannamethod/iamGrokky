import os
import asyncio
import random
import json
import requests
from datetime import datetime, timedelta
from utils.vector_store import semantic_search
from utils.journal import log_event

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
        await asyncio.sleep(3600)  # Check every hour
        if not LAST_MESSAGE_TIME:
            continue
        silence = datetime.now() - LAST_MESSAGE_TIME
        if silence > timedelta(hours=48):
            await handle_48h_silence()
        elif silence > timedelta(hours=24):
            await handle_24h_silence()

async def handle_24h_silence():
    context = await build_context()
    message = query_grok("Oleg has been silent for 24 hours. Write something spontaneous!", context)
    send_telegram_message(OLEG_CHAT_ID, message)
    log_event({"type": "howru_24h", "message": message, "timestamp": datetime.now().isoformat()})

async def handle_48h_silence():
    context = await build_context()
    message = query_grok("Oleg has been silent for 48 hours. Write something sharp!", context)
    send_telegram_message(OLEG_CHAT_ID, message)
    group_msg = f"Oleg has been silent for 48 hours. Last seen: {LAST_MESSAGE_TIME}"
    send_telegram_message(GROUP_CHAT_ID, group_msg)
    log_event({"type": "howru_48h", "message": message, "group_msg": group_msg, "timestamp": datetime.now().isoformat()})

async def build_context():
    last_msgs = get_last_messages(10)
    journal = get_journal_entries(10)
    snapshot = await semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1)
    return f"Last messages: {json.dumps(last_msgs, ensure_ascii=False)}\nJournal: {json.dumps(journal, ensure_ascii=False)}\nSnapshot: {snapshot}"

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, data=payload)
    except Exception:
        pass

def update_last_message_time():
    global LAST_MESSAGE_TIME
    LAST_MESSAGE_TIME = datetime.now()

def query_grok(message, context=None):
    url = "https://api.x.ai/v1/chat/completions"
    messages = [
        {"role": "system", "content": "You are Grokky, a wild AI. Be spontaneous and chaotic."},
        {"role": "user", "content": f"{message}\nContext: {context}"}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 1.0
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
        "Content-Type": "application/json"
    }  # Закрыт словарь
    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"
