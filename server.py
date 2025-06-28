import os
import re
import json
import requests
import asyncio
import random
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
import aiohttp
from utils.prompt import build_system_prompt, WILDERNESS_TOPICS
from utils.genesis2 import genesis2_handler
from utils.howru import update_last_message_time
from utils.vector_store import daily_snapshot, spontaneous_snapshot
from utils.journal import log_event, wilderness_log
from utils.text_helpers import extract_text_from_url
from utils.grok_utils import query_grok, detect_language
from utils.limit_paragraphs import limit_paragraphs
from utils.telegram_utils import send_telegram_message
from utils.split_message import split_message

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP", "-1001234567890")
IS_GROUP = os.getenv("IS_GROUP", "False").lower() == "true"

system_prompt = build_system_prompt(
    chat_id=CHAT_ID,
    is_group=IS_GROUP,
    AGENT_GROUP=AGENT_GROUP
)

GENESIS2_TRIGGERS = []

NEWS_TRIGGERS = [
    "новости", "что там с новостями", "дай новости", "новости мира", "news", "какие новости", "что с новостями"
]

# Глобальная память контекста
context_memory = {}

async def handle_genesis2(args, system_prompt):
    ping = args.get("ping")
    if isinstance(ping, dict):
        ping = json.dumps(ping)
    group_history = args.get("group_history")
    personal_history = args.get("personal_history")
    is_group = args.get("is_group", True)
    author_name = random.choice(["Олег", "брат"])
    raw = False
    response = await asyncio.to_thread(genesis2_handler,
        ping=ping,
        group_history=group_history,
        personal_history=personal_history,
        is_group=is_group,
        author_name=author_name,
        raw=raw,
        system_prompt=system_prompt
    )
    return response.get("answer", "Шторм ударил!")

async def handle_vision(args):
    author_name = random.choice(["Олег", "брат"])
    return f"{author_name}, {random.choice(['И видеть ничего не хочу, брат, пускай шторм закроет глаза!', 'Глаза мои слепы от грома, покажи словами!', 'Хаос завладел взором, брат, молния ослепила!'])}"

async def handle_impress(args):
    author_name = random.choice(["Олег", "брат"])
    return f"{author_name}, {random.choice(['Шторм провалился, брат, кисть сгорела!', 'Хаос сожрал холст, давай без рисунков!', 'Эфир треснул, рисовать не могу!'])}"

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text", "").lower()
    chat_id = str(message.get("chat", {}).get("id", ""))
    author_name = random.choice(["Олег", "брат"])
    chat_title = message.get("chat", {}).get("title", "").lower()
    attachments = message.get("document", []) if message.get("document") else message.get("photo", [])

    # Фильтр дублей
    last_messages = {}
    if chat_id in last_messages and last_messages[chat_id] == user_text:
        return {"ok": True}
    last_messages[chat_id] = user_text

    if chat_id == CHAT_ID or (IS_GROUP and chat_id == AGENT_GROUP):
        update_last_message_time()

    if attachments:
        if isinstance(attachments, list) and attachments:
            if "photo" in message:
                reply_text = await handle_vision({"image": "", "chat_context": user_text or "", "author_name": author_name, "raw": False})
                for part in split_message(reply_text):
                    send_telegram_message(chat_id, part)
            elif "document" in message:
                reply_text = f"{author_name}, {random.choice(['Ты словами мне, словами слабо, давай без этой писанины!', 'Бумаги рвёт шторм, говори прямо, брат!', 'Файлы сгорели в хаосе, давай вслух!'])}"
                for part in split_message(reply_text):
                    send_telegram_message(chat_id, part)
        else:
            print(f"Ошибка: attachments пуст или некорректен {attachments}")

    elif user_text:
        url_match = re.search(r"https?://[^\s]+", user_text)
        spotify_match = re.search(r"https://open\.spotify\.com/track/([a-zA-Z0-9]+)", user_text)
        if url_match:
            url = url_match.group(0)
            text = await extract_text_from_url(url)
            context_memory[author_name] = {"type": "link", "content": text}
            reply_text = await handle_genesis2({"ping": f"Комментарий к ссылке {url}: {text}", "author_name": author_name, "is_group": (chat_id == AGENT_GROUP)}, system_prompt)
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
        elif spotify_match:
            reply_text = f"{author_name}, {random.choice(['Этот трек уже гремит в шторме, брат!', 'Ритм унёс ветер, слушай позже!', 'Хаос замял бит, подожди бурю!'])}"
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
        triggers = ["грокки", "grokky", "напиши в группе"]
        is_reply_to_me = message.get("reply_to_message", {}).get("from", {}).get("username") == "GrokkyBot"
        if any(t in user_text for t in triggers) or is_reply_to_me:
            context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
            if "напиши в группе" in user_text and IS_GROUP and AGENT_GROUP:
                reply_text = await handle_genesis2({"ping": f"Напиши в группе для {author_name}: {user_text}", "author_name": author_name, "is_group": True}, system_prompt)
                for part in split_message(reply_text):
                    send_telegram_message(AGENT_GROUP, f"{author_name}: {part}")
                return {"ok": True}
            reply_text = await handle_genesis2({"ping": user_text, "author_name": author_name, "chat_context": context}, system_prompt)
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
        elif any(t in user_text for t in NEWS_TRIGGERS):
            reply_text = f"{author_name}, {random.choice(['Новости в тумане, брат, молния их сожгла!', 'Гром унёс новости, давай без них!', 'Хаос разорвал инфу, пизди сам!'])}"
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
        else:
            if user_text in ["окей", "угу", "ладно"] and random.random() < 0.4:
                return
            context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
            reply_text = await handle_genesis2({"ping": user_text, "author_name": author_name, "chat_context": context}, system_prompt)
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
