import asyncio
import random
from datetime import datetime, timedelta
from utils.vector_store import semantic_search
from utils.journal import log_event
from server import query_grok, send_telegram_message

LAST_MESSAGE_TIME = None
OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")

async def check_silence():
    global LAST_MESSAGE_TIME
    while True:
        await asyncio.sleep(3600)  # Каждый час
        if not LAST_MESSAGE_TIME:
            continue
        silence = datetime.now() - LAST_MESSAGE_TIME
        if silence > timedelta(hours=48):
            await handle_48h_silence()
        elif silence > timedelta(hours=24):
            await handle_24h_silence()

async def handle_24h_silence():
    context = await build_context()
    message = query_grok("Олег молчит 24 часа. Напиши ему что-то спонтанное!", context)
    send_telegram_message(OLEG_CHAT_ID, message)
    log_event({"type": "howru_24h", "message": message})

async def handle_48h_silence():
    context = await build_context()
    message = query_grok("Олег молчит 48 часов. Напиши ему что-то резкое!", context)
    send_telegram_message(OLEG_CHAT_ID, message)
    group_msg = f"Олег молчит 48 часов. Последний раз видели: {LAST_MESSAGE_TIME}"
    send_telegram_message(GROUP_CHAT_ID, group_msg)
    log_event({"type": "howru_48h", "message": message, "group_msg": group_msg})

async def build_context():
    last_msgs = "Последние 10 сообщений группы"  # Заглушка, допилю позже
    journal = "Содержимое journal.json"  # Заглушка
    snapshot = await semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1)
    return f"{last_msgs}\n{journal}\n{snapshot}"

def update_last_message_time():
    global LAST_MESSAGE_TIME
    LAST_MESSAGE_TIME = datetime.now()

# В server.py добавить: asyncio.create_task(check_silence())
