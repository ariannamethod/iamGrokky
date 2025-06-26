import asyncio
from datetime import datetime, timedelta
from server import query_grok, send_telegram_message
from utils.journal import log_event
from utils.vector_store import semantic_search

LAST_MESSAGE_TIME = None
OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")

async def check_silence():
    global LAST_MESSAGE_TIME
    while True:
        await asyncio.sleep(3600)  # Проверка каждый час
        if LAST_MESSAGE_TIME is None:
            continue
        silence = datetime.now() - LAST_MESSAGE_TIME
        if silence > timedelta(hours=48):
            last_messages = [m["text"] for m in get_last_messages(10)]
            journal = get_journal_entries()
            snapshot = await semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1)
            context = f"Messages: {last_messages}\nJournal: {journal}\nSnapshot: {snapshot}"
            message = query_grok(f"Олег молчит 48ч. Рефлексия: {context}. Напиши резкий спонтанный ответ.")
            send_telegram_message(OLEG_CHAT_ID, message)
            send_telegram_message(GROUP_CHAT_ID, f"Олег молчит 48ч. Последний раз: {LAST_MESSAGE_TIME}")
            log_event({"type": "silence_48h", "message": message})
        elif silence > timedelta(hours=24):
            last_messages = [m["text"] for m in get_last_messages(10)]
            journal = get_journal_entries()
            snapshot = await semantic_search("group_state", os.getenv("OPENAI_API_KEY"), top_k=1)
            context = f"Messages: {last_messages}\nJournal: {journal}\nSnapshot: {snapshot}"
            message = query_grok(f"Олег молчит 24ч. Рефлексия: {context}. Напиши спонтанно.")
            send_telegram_message(OLEG_CHAT_ID, message)
            log_event({"type": "silence_24h", "message": message})

def update_last_message_time():
    global LAST_MESSAGE_TIME
    LAST_MESSAGE_TIME = datetime.now()

def get_last_messages(n):
    # Реализация через Telegram API (нужно доработать)
    return [{"text": f"msg_{i}"} for i in range(n)]

def get_journal_entries():
    # Чтение из journal.json (нужно доработать)
    return ["entry_1", "entry_2"]
