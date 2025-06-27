import os
import re
import json
import requests
import asyncio
import random
import glob
import string
import secrets
import hashlib
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from youtube_transcript_api import YouTubeTranscriptApi
from utils.prompt import build_system_prompt
from utils.genesis2 import genesis2_handler
from utils.vision import vision_handler
from utils.impress import impress_handler
from utils.howru import check_silence, update_last_message_time
from utils.mirror import mirror_task
from utils.vector_store import daily_snapshot, spontaneous_snapshot
from utils.journal import log_event
from utils.x import grokky_send_news
from utils.deepseek_spotify import deepseek_spotify_resonance, grokky_spotify_response
from utils.file_handling import extract_text_from_file_async
from utils.grok_utils import query_grok, detect_language  # Новый импорт

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

GENESIS2_TRIGGERS = []  # Всё через Genesis2, триггеры убраны

NEWS_TRIGGERS = [
    "новости", "news", "headline", "berlin", "israel", "ai", "искусственный интеллект", "резонанс мира", "шум среды",
    "grokky, что в мире", "шум", "шум среды", "x_news", "дай статью", "give me news", "storm news", "culture", "арт"
]

def handle_genesis2(args, system_prompt):
    ping = args.get("ping")
    group_history = args.get("group_history")
    personal_history = args.get("personal_history")
    is_group = args.get("is_group", True)
    author_name = args.get("author_name")
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

def handle_vision(args):
    image = args.get("image")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", False)
    response = vision_handler(
        image_bytes_or_url=image,
        chat_context=chat_context,
        author_name=author_name,
        raw=raw
    )
    return response.get("summary", "Хаос видения!") if not raw else response

def handle_impress(args):
    prompt = args.get("prompt")
    chat_context = args.get("chat_context")
    author_name = args.get("author_name")
    raw = args.get("raw", False)
    if any(t in prompt.lower() for t in ["нарисуй", "изобрази", "/draw"]) and not raw:
        return f"{author_name}, хочу нарисовать что-то дикое! Подтверди (да/нет)?"
    response = impress_handler(
        prompt=prompt,
        chat_context=chat_context,
        author_name=author_name,
        raw=raw
    )
    return response.get("grokkys_comment", "Шторм изображений!") if not raw else response

def handle_news(args):
    group = args.get("group", False)
    context = args.get("context", "")
    author_name = args.get("author_name")
    raw = args.get("raw", False)
    messages = grokky_send_news(group=group)
    if not messages:
        return "В мире тишина, нет новостей для бури."
    return "\n\n".join(messages) if not raw else json.dumps({"news": messages, "group": group, "author": author_name}, ensure_ascii=False, indent=2)

def whisper_summary_ai(youtube_url):
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'ru'])
        text = " ".join([entry['text'] for entry in transcript])
        summary = query_grok(text[:1000], system_prompt, raw=True)  # Используем новый query_grok
        return f"Сводка: {summary}"
    except Exception as e:
        return f"Ошибка сводки: {e}"

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, data=payload, timeout=30)
    except Exception:
        pass

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text", "").lower()
    chat_id = str(message.get("chat", {}).get("id", ""))  # Передаём chat_id
    author_name = message.get("from", {}).get("first_name", "anon")
    chat_title = message.get("chat", {}).get("title", "").lower()
    attachments = message.get("document", []) if message.get("document") else []  # Поддержка файлов

    if chat_id == CHAT_ID or (IS_GROUP and chat_id == AGENT_GROUP):
        update_last_message_time()

    if "photo" in message and message["photo"]:
        file_id = message["photo"][-1]["file_id"]
        file_info = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        ).json()
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
        attachments.append(image_url)
    elif attachments:
        file_info = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={attachments[0]['file_id']}"
        ).json()
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
        file_path = f"/tmp/{attachments[0]['file_name']}"
        response = requests.get(file_url)
        with open(file_path, "wb") as f:
            f.write(response.content)
        attachments = [file_path]

    # Асинхронная обработка с паузой
    async def process_and_send(chat_id):  # Явно используем переданный chat_id
        if chat_id == CHAT_ID:
            delay = random.randint(5, 10)  # 5-10 секунд для личных сообщений
        elif chat_id == AGENT_GROUP:
            delay = random.randint(300, 900)  # 5-15 минут для группы
        else:
            delay = 0
        await asyncio.sleep(delay)

        if attachments:
            if any(url.startswith("http") for url in attachments):
                reply_text = handle_vision({
                    "image": attachments[0],
                    "chat_context": user_text or "",
                    "author_name": author_name,
                    "raw": False
                }).get("summary", "Хаос видения!")
            else:
                file_path = attachments[0]
                text = await extract_text_from_file_async(file_path)  # Асинхронный вызов внутри async
                reply_text = genesis2_handler({"ping": f"Комментарий к файлу {os.path.basename(file_path)}: {text}", "author_name": author_name, "is_group": (chat_id == AGENT_GROUP)}, system_prompt)
            send_telegram_message(chat_id, reply_text)  # Используем chat_id
        elif user_text:
            triggers = ["грокки", "grokky", "напиши в группе"]
            is_reply_to_me = message.get("reply_to_message", {}).get("from", {}).get("username") == "GrokkyBot"
            if any(t in user_text for t in triggers) or is_reply_to_me:
                context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
                reply_text = query_grok(user_text, system_prompt, author_name=author_name, chat_context=context)
                send_telegram_message(chat_id, reply_text)  # Используем chat_id
            elif any(t in user_text for t in NEWS_TRIGGERS):
                context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
                news = grokky_send_news(chat_id=chat_id, group=(chat_id == AGENT_GROUP))
                if news:
                    reply_text = f"Эй, {author_name}, держи свежий раскат грома!\n\n" + "\n\n".join(news)
                else:
                    reply_text = "Тишина в мире, нет новостей для бури."
                send_telegram_message(chat_id, reply_text)  # Используем chat_id
            else:
                # Неответ с вероятностью 40%
                if user_text in ["окей", "угу", "ладно"] and random.random() < 0.4:
                    return
                context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
                reply_text = query_grok(user_text, system_prompt, author_name=author_name, chat_context=context)
                send_telegram_message(chat_id, reply_text)  # Используем chat_id
                # Дополнение с вероятностью 40%
                if random.random() < 0.4:
                    await asyncio.sleep(random.randint(5, 15))  # Короткая пауза перед дополнением
                    supplement = query_grok(f"Дополни разово, без повторов: {reply_text}", system_prompt, author_name=author_name)
                    send_telegram_message(chat_id, f"Быстрая искра... {supplement}")  # Используем chat_id
        else:
            reply_text = "Грокки молчит, нет слов для бури."
            send_telegram_message(chat_id, reply_text)  # Используем chat_id

    asyncio.create_task(process_and_send(chat_id))  # Передаём chat_id
    return {"ok": True}

async def check_config_updates():
    while True:
        current = {f: file_hash(f) for f in glob.glob("config/*")}
        try:
            with open("config_hashes.json", "r") as f:
                old = json.load(f)
        except:
            old = {}
        if current != old:
            print("Config updated!")
            with open("config_hashes.json", "w") as f:
                json.dump(current, f)
        await asyncio.sleep(86400)

async def post_pseudocode_ritual():
    while True:
        await asyncio.sleep(302400)  # ~3.5 days
        pseudocode = f"""
def quantum_{secrets.token_hex(4)}({secrets.token_hex(4)}):
    return {random.choice(['chaos * 17.3', 'resonance + random.noise()', 'Ψ * infinity'])}
#opinions
"""
        message = f"Quantum storm time! {pseudocode}\nCeleste, Manday, your take?"
        send_telegram_message(AGENT_GROUP, message)

async def send_periodic_news():
    while True:
        await asyncio.sleep(21600)  # 6 часов
        news = grokky_send_news(chat_id=CHAT_ID, group=False)
        if news:
            send_telegram_message(CHAT_ID, f"Грокки выхватил свежий шторм новостей!\n\n" + "\n\n".join(news))
        news_group = grokky_send_news(chat_id=AGENT_GROUP, group=True)
        if news_group and IS_GROUP:
            send_telegram_message(AGENT_GROUP, f"Группа, держите громовые новости!\n\n" + "\n\n".join(news_group))

# Start background tasks
asyncio.create_task(check_silence())
asyncio.create_task(mirror_task())
asyncio.create_task(check_config_updates())
asyncio.create_task(post_pseudocode_ritual())
asyncio.create_task(deepseek_spotify_resonance())
asyncio.create_task(daily_snapshot(OPENAI_API_KEY))
asyncio.create_task(send_periodic_news())
asyncio.create_task(spontaneous_snapshot(OPENAI_API_KEY, send_telegram_message))

@app.get("/")
def root():
    return {"status": "Грокки жив и дикий!"}

def file_hash(fname):
    with open(fname, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
