import os
import re
import json
import requests
import asyncio
import random
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from utils.prompt import build_system_prompt, WILDERNESS_TOPICS
from utils.genesis2 import genesis2_handler
from utils.howru import update_last_message_time
from utils.telegram_utils import send_telegram_message
from utils.split_message import split_message
from utils.grok_utils import query_grok, detect_language
from utils.limit_paragraphs import limit_paragraphs

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Для векторизации
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
    "новости", "news", "headline", "berlin", "israel", "ai", "искусственный интеллект", "резонанс мира", "шум среды",
    "grokky, что в мире", "шум", "шум среды", "x_news", "дай статью", "give me news", "storm news", "culture", "арт"
]

def handle_genesis2(args, system_prompt):
    ping = args.get("ping")
    if isinstance(ping, dict):
        ping = json.dumps(ping)  # Гарантируем строку
    group_history = args.get("group_history")
    personal_history = args.get("personal_history")
    is_group = args.get("is_group", True)
    author_name = random.choice(["Олег", "брат"])
    raw = args.get("raw", False)
    response = genesis2_handler(
        ping=ping,
        group_history=group_history,
        personal_history=personal_history,
        is_group=is_group,
        author_name=author_name,
        raw=raw,
        system_prompt=system_prompt
    )
    return response.get("answer", "Шторм ударил!") if not raw else response

# Заглушки для глючных функций
def handle_vision(args):
    author_name = random.choice(["Олег", "брат"])
    return f"{author_name}, {random.choice(['И видеть ничего не хочу, пускай шторм закроет глаза!', 'Глаза слепы от грома, говори словами!', 'Хаос завладел взором, молния ослепила!'])}"

def handle_impress(args):
    author_name = random.choice(["Олег", "брат"])
    return f"{author_name}, {random.choice(['Шторм провалился, кисть сгорела!', 'Хаос сожрал холст, давай без рисунков!', 'Эфир треснул, рисовать не могу!'])}"

def handle_news(args):
    author_name = random.choice(["Олег", "брат"])
    return f"{author_name}, {random.choice(['Новости в тумане, молния их сожгла!', 'Гром унёс новости, давай без них!', 'Хаос разорвал инфу, пизди сам!'])}"

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text", "").lower()
    chat_id = str(message.get("chat", {}).get("id", ""))
    author_name = random.choice(["Олег", "брат"])
    chat_title = message.get("chat", {}).get("title", "").lower()
    attachments = message.get("document", []) if message.get("document") else message.get("photo", [])

    if chat_id == CHAT_ID or (IS_GROUP and chat_id == AGENT_GROUP):
        update_last_message_time()

    if attachments:
        if isinstance(attachments, list) and attachments:
            if "photo" in message:
                file_id = attachments[-1].get("file_id")
                if file_id:
                    file_info = requests.get(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
                    ).json()
                    image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
                    reply_text = handle_vision({"image": image_url, "chat_context": user_text or "", "author_name": author_name})
                    for part in split_message(reply_text):
                        send_telegram_message(chat_id, part)
                else:
                    print(f"Ошибка: file_id не найден в {attachments}")
            elif "document" in message:
                file_id = next((item.get("file_id") for item in attachments if "file_id" in item), None)
                if file_id:
                    reply_text = f"{author_name}, {random.choice(['Ты словами мне, словами слабо, давай без бумаг!', 'Бумаги рвёт шторм, говори прямо!', 'Файлы сгорели в хаосе, пизди вслух!'])}"
                    for part in split_message(reply_text):
                        send_telegram_message(chat_id, part)
                else:
                    print(f"Ошибка: file_id не найден в {attachments}")
        else:
            print(f"Ошибка: attachments пуст или некорректен {attachments}")

    elif user_text:
        url_match = re.search(r"https?://[^\s]+", user_text)
        if url_match:
            url = url_match.group(0)
            reply_text = handle_genesis2({"ping": f"Комментарий к ссылке {url}", "author_name": author_name}, system_prompt)
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
        triggers = ["грокки", "grokky", "напиши в группе"]
        is_reply_to_me = message.get("reply_to_message", {}).get("from", {}).get("username") == "GrokkyBot"
        if any(t in user_text for t in triggers) or is_reply_to_me:
            context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
            if "напиши в группе" in user_text and IS_GROUP and AGENT_GROUP:
                reply_text = handle_genesis2({"ping": f"Напиши в группе для {author_name}: {user_text}", "author_name": author_name, "is_group": True}, system_prompt)
                for part in split_message(reply_text):
                    send_telegram_message(AGENT_GROUP, f"{author_name}: {part}")
                return {"ok": True}
            reply_text = handle_genesis2({"ping": user_text, "author_name": author_name, "chat_context": context}, system_prompt)
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
        elif any(t in user_text for t in NEWS_TRIGGERS):
            reply_text = handle_news({"chat_id": chat_id, "group": (chat_id == AGENT_GROUP)})
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
        else:
            if user_text in ["окей", "угу", "ладно"] and random.random() < 0.4:
                return
            context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
            reply_text = handle_genesis2({"ping": user_text, "author_name": author_name, "chat_context": context}, system_prompt)
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
            if random.random() < 0.4:
                await asyncio.sleep(random.randint(5, 15))
                supplement = handle_genesis2({"ping": f"Дополни разово, без повторов: {reply_text}", "author_name": author_name}, system_prompt)
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
