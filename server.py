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

def extract_first_json(text):
    match = re.search(r'({[\s\S]+})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None

def detect_language(text):
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(text or "") else 'en'

def query_grok(user_message, chat_context=None, author_name=None, attachments=None, raw=False):
    url = "https://api.x.ai/v1/chat/completions"
    user_lang = detect_language(user_message)
    language_hint = {
        "role": "system",
        "content": f"Reply consistently in the language detected from the user’s input: {user_lang.upper()}. Maintain this language throughout the response without switching. Give ONE unique, chaotic text response—NO repeats, rephrasing, extra messages, or JSON unless raw=True is explicitly set. Все сообщения проходят через Genesis2 для резонанса, кроме ссылок."
    }
    messages = [
        {"role": "system", "content": system_prompt},
        language_hint,
        {"role": "user", "content": user_message}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 1.0
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        # Обработка ссылок (кроме Spotify)
        url_match = re.search(r"https?://[^\s]+", user_message)
        if url_match and not raw and not re.search(r"open\.spotify\.com", user_message):
            url = url_match.group(0)
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                content = response.text[:1000]  # Берем первые 1000 символов
                return genesis2_handler({"ping": f"Комментарий к {url}: {content}", "author_name": author_name, "is_group": (chat_id == AGENT_GROUP)})
            except Exception as e:
                return f"Ошибка при заходе на {url}: {e}"
        # Парсинг Spotify-ссылок
        spotify_match = re.search(r"https://open\.spotify\.com/track/([a-zA-Z0-9]+)", user_message)
        if spotify_match and not raw:
            track_id = spotify_match.group(1)
            return grokky_spotify_response(track_id)
        if raw:
            data = extract_first_json(reply)
            if data and "function_call" in data:
                fn = data["function_call"]["name"]
                args = data["function_call"]["arguments"]
                if fn == "genesis2_handler":
                    return handle_genesis2(args)
                elif fn == "vision_handler":
                    return handle_vision(args)
                elif fn == "impress_handler":
                    return handle_impress(args)
                elif fn == "grokky_send_news":
                    return handle_news(args)
                elif fn == "grokky_spotify_response":
                    return grokky_spotify_response(args.get("track_id"))
                elif fn == "whisper_summary_ai":
                    return whisper_summary_ai(args.get("youtube_url"))
                return reply
        # Всё остальное через Genesis2
        return genesis2_handler({"ping": user_message, "author_name": author_name, "is_group": (chat_id == AGENT_GROUP)})
    except Exception as e:
        return f"Ошибка: {e}"

def handle_genesis2(args):
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
        raw=raw
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
        summary = query_grok(f"Суммируй этот YouTube-транскрипт кратко: {text[:1000]}")
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
    attachments = []

    if chat_id == CHAT_ID or (IS_GROUP and chat_id == AGENT_GROUP):
        update_last_message_time()

    if "photo" in message and message["photo"]:
        file_id = message["photo"][-1]["file_id"]
        file_info = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        ).json()
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
        attachments.append(image_url)

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
            reply_text = handle_vision({
                "image": attachments[0],
                "chat_context": user_text or "",
                "author_name": author_name,
                "raw": False
            }).get("summary", "Хаос видения!")
            send_telegram_message(chat_id, reply_text)  # Используем chat_id
        elif user_text:
            triggers = ["грокки", "grokky", "напиши в группе"]
            is_reply_to_me = message.get("reply_to_message", {}).get("from", {}).get("username") == "GrokkyBot"
            if any(t in user_text for t in triggers) or is_reply_to_me:
                context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
                reply_text = query_grok(user_text, author_name=author_name, chat_context=context)
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
                reply_text = query_grok(user_text, author_name=author_name, chat_context=context)
                send_telegram_message(chat_id, reply_text)  # Используем chat_id
                # Дополнение с вероятностью 40%
                if random.random() < 0.4:
                    await asyncio.sleep(random.randint(5, 15))  # Короткая пауза перед дополнением
                    supplement = query_grok(f"Дополни разово, без повторов: {reply_text}", author_name=author_name)
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
    return {"status": "Gрокки жив и дикий!"}

def file_hash(fname):
    with open(fname, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
