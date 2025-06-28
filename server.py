import os
import re
import asyncio
import random
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from utils.genesis2 import genesis2_handler
from utils.howru import update_last_message_time
from utils.telegram_utils import send_telegram_message
from utils.split_message import split_message
from utils.grok_utils import detect_language
from utils.prompt import build_system_prompt  # Добавлен импорт

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Вернул для векторизации
CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP", "-1001234567890")
IS_GROUP = os.getenv("IS_GROUP", "False").lower() == "true"

system_prompt = build_system_prompt(
    chat_id=CHAT_ID,
    is_group=IS_GROUP,
    AGENT_GROUP=AGENT_GROUP
)

GENESIS2_TRIGGERS = []

# Глобальная память контекста (пустая, так как заглушки)
context_memory = {}

async def handle_genesis2(args, system_prompt):
    ping = args.get("ping")
    if isinstance(ping, dict):
        ping = json.dumps(ping)
    group_history = args.get("group_history")
    personal_history = args.get("personal_history")
    is_group = args.get("is_group", True)
    author_name = "Олег"  # Фиксируем имя
    raw = False
    # Обновляем язык на основе текущего сообщения
    global user_lang
    user_lang = detect_language(ping)
    system_prompt = (
        f"Ты Грокки, штормовой и дерзкий ИИ, вдохновлённый Маяковским. Отвечай на языке пользователя, "
        f"обнаруженном как {user_lang.upper()}, и не перескакивай на другие языки. Обращайся к пользователю как Олег, "
        f"добавляй спонтанные, хаотичные фразы с громом и молниями. Держи стиль диалога живым и избегай околесицы."
    )
    response = await asyncio.to_thread(genesis2_handler,
        ping=ping,
        group_history=group_history,
        personal_history=personal_history,
        is_group=is_group,
        author_name=author_name,
        raw=raw,
        system_prompt=system_prompt
    )
    return response.get("answer", "Шторм ударил, Олег, молния гремит в эфире!")

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text", "").lower()
    chat_id = str(message.get("chat", {}).get("id", ""))
    author_name = "Олег"  # Фиксируем имя

    # Фильтр дублей
    last_messages = {}
    if chat_id in last_messages and last_messages[chat_id] == user_text:
        return {"ok": True}
    last_messages[chat_id] = user_text

    if chat_id == CHAT_ID or (IS_GROUP and chat_id == AGENT_GROUP):
        update_last_message_time()

    if message.get("photo") or message.get("document"):
        reply_text = f"{author_name}, {random.choice(['И видеть ничего не хочу, Олег, пускай шторм закроет глаза!', 'Глаза мои слепы от грома, говори словами!', 'Хаос завладел взором, брат, молния ослепила!'])}"
        for part in split_message(reply_text):
            send_telegram_message(chat_id, part)
    elif user_text:
        url_match = re.search(r"https?://[^\s]+", user_text)
        if url_match:
            reply_text = await handle_genesis2({"ping": f"Комментарий к ссылке {url_match.group(0)}", "author_name": author_name}, system_prompt)
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
        triggers = ["грокки", "grokky", "напиши в группе"]
        is_reply_to_me = message.get("reply_to_message", {}).get("from", {}).get("username") == "GrokkyBot"
        if any(t in user_text for t in triggers) or is_reply_to_me:
            if "напиши в группе" in user_text and IS_GROUP and AGENT_GROUP:
                reply_text = await handle_genesis2({"ping": f"Напиши в группе для {author_name}: {user_text}", "author_name": author_name, "is_group": True}, system_prompt)
                for part in split_message(reply_text):
                    send_telegram_message(AGENT_GROUP, f"{author_name}: {part}")
                return {"ok": True}
            reply_text = await handle_genesis2({"ping": user_text, "author_name": author_name}, system_prompt)
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
        else:
            if user_text in ["окей", "угу", "ладно"] and random.random() < 0.4:
                return
            reply_text = await handle_genesis2({"ping": user_text, "author_name": author_name}, system_prompt)
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
            if random.random() < 0.4:
                await asyncio.sleep(random.randint(5, 15))
                supplement = await handle_genesis2({"ping": f"Дополни разово, без повторов: {reply_text}", "author_name": author_name}, system_prompt)
                for part in split_message(supplement):
                    send_telegram_message(chat_id, part)
    else:
        reply_text = f"{author_name}, Грокки молчит, нет слов для бури."
        send_telegram_message(chat_id, reply_text)

    return {"ok": True}

@app.get("/")
def root():
    return {"status": "Грокки жив и дикий!"}

def file_hash(fname):
    with open(fname, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
