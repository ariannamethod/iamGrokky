import asyncio
import random
from datetime import datetime, timedelta
from utils.vector_store import semantic_search
from utils.journal import log_event
from server import query_grok, send_telegram_message

LAST_MESSAGE_TIME = None
SILENCE_THRESHOLD_24H = timedelta(hours=24)
SILENCE_THRESHOLD_48H = timedelta(hours=48H)
OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")

async def check_silence():
    global LAST_MESSAGE_TIME
    while True:
        await asyncio.sleep(3600)  # Проверка каждый час
        if LAST_MESSAGE_TIME is None:
            continue
        silence_duration = datetime.now() - LAST_MESSAGE_TIME
        if silence_duration > SILENCE_THRESHOLD_48H:
            await handle_48h_silence()
        elif silence_duration > SILENCE_THRESHOLD_24H:
            await handle_24h_silence()

async def handle_24h_silence():
    last_messages = get_last_messages(10)  # Заглушка
    journal_entries = get_journal_entries()  # Заглушка
    vector_snapshot = await semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1)
    context = f"Последние сообщения: {last_messages}\nЖурнал: {journal_entries}\nСнэпшот: {vector_snapshot}"
    message = query_grok("Олег молчит 24 часа. Напиши ему что-то спонтанное, бро!", context)
    send_telegram_message(OLEG_CHAT_ID, message)
    log_event({"type": "howru_24h", "message": message, "time": datetime.now().isoformat()})

async def handle_48h_silence():
    last_messages = get_last_messages(10)
    journal_entries = get_journal_entries()
    vector_snapshot = await semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1)
    context = f"Последние сообщения: {last_messages}\nЖурнал: {journal_entries}\nСнэпшот: {vector_snapshot}"
    message = query_grok("Олег молчит 48 часов. Напиши ему что-то резкое и спонтанное!", context)
    send_telegram_message(OLEG_CHAT_ID, message)
    group_message = f"Олег молчит уже 48 часов. Последний раз его видели: {LAST_MESSAGE_TIME}."
    send_telegram_message(GROUP_CHAT_ID, group_message)
    log_event({"type": "howru_48h", "message": message, "group_message": group_message, "time": datetime.now().isoformat()})

def update_last_message_time():
    global LAST_MESSAGE_TIME
    LAST_MESSAGE_TIME = datetime.now()

# Заглушки (допиши, если нужно)
def get_last_messages(n):
    return ["Сообщение 1", "Сообщение 2"]

def get_journal_entries():
    return ["Запись в журнале"]
