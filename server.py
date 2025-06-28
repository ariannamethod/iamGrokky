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
from utils.vision import vision_handler
from utils.impress import impress_handler
from utils.howru import update_last_message_time
from utils.vector_store import daily_snapshot, spontaneous_snapshot
from utils.journal import log_event, wilderness_log
from utils.x import grokky_send_news
from utils.deepseek_spotify import grokky_spotify_response
from utils.file_handling import extract_text_from_file_async
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
    "новости", "что там с новостями", "дай новости", "новости мира", "news", "какие новости"
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
    image = args.get("image")
    chat_context = args.get("chat_context")
    author_name = random.choice(["Олег", "брат"])
    raw = False
    if isinstance(image, str):
        try:
            response = await vision_handler(
                image_bytes_or_url=image,
                chat_context=chat_context,
                author_name=author_name,
                raw=raw
            )
            # Сохраняем контекст
            context_memory[author_name] = {"type": "image", "content": response}
            return response
        except Exception as e:
            return f"{author_name}, Грокки взорвался: {e}"
    return f"{author_name}, что-то пошло не так с изображением!"

async def handle_impress(args):
    prompt = args.get("prompt")
    chat_context = args.get("chat_context")
    author_name = random.choice(["Олег", "брат"])
    raw = False
    if any(t in prompt.lower() for t in ["нарисуй", "изобрази", "/draw", "нарисуй мне", "draw me"]):
        if not raw:
            return f"{author_name}, хочу нарисовать что-то дикое! Подтверди (да/нет)?"
        response = await impress_handler(prompt=prompt, chat_context=chat_context, author_name=author_name, raw=raw)
        if isinstance(response, dict) and "image_url" in response:
            send_telegram_message(chat_id, f"{author_name}, держи шторм! {response['image_url']}\n{response['grokkys_comment']}")
            context_memory[author_name] = {"type": "image", "content": response["image_url"]}
            return response['grokkys_comment']
        return response.get("grokkys_comment", f"{author_name}, шторм изображений!")
    return response.get("grokkys_comment", f"{author_name}, шторм изображений!") if not raw else response

def handle_news(args):
    group = args.get("group", False)
    context = args.get("context", "")
    author_name = random.choice(["Олег", "брат"])
    raw = False
    messages = grokky_send_news(chat_id=args.get("chat_id"), group=group)
    if not messages:
        return f"{author_name}, в мире тишина, нет новостей для бури."
    # Фильтр дублирования новостей
    unique_news = []
    seen_titles = set()
    for msg in messages:
        title = msg.split('\n', 1)[0]
        if title not in seen_titles:
            unique_news.append(msg)
            seen_titles.add(title)
    return "\n\n".join(unique_news)

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
                file_id = attachments[-1].get("file_id")
                if file_id:
                    file_info = requests.get(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
                    ).json()
                    image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
                    reply_text = await handle_vision({"image": image_url, "chat_context": user_text or "", "author_name": author_name, "raw": False})
                    for part in split_message(reply_text):
                        send_telegram_message(chat_id, part)
                    if "видишь ли ты картинку" in user_text:
                        reply_text = await handle_impress({"prompt": "Оцени изображение", "chat_context": user_text, "author_name": author_name, "raw": False})
                        for part in split_message(reply_text):
                            send_telegram_message(chat_id, part)
                else:
                    print(f"Ошибка: file_id не найден в {attachments}")
            elif "document" in message:
                file_id = next((item.get("file_id") for item in attachments if "file_id" in item), None)
                if file_id:
                    file_info = requests.get(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
                    ).json()
                    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
                    file_path = f"/tmp/{next((item.get('file_name', 'unknown') for item in attachments if 'file_name' in item), 'unknown')}"
                    response = requests.get(file_url)
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    text = await extract_text_from_file_async(file_path)
                    context_memory[author_name] = {"type": "file", "content": text}
                    reply_text = await handle_genesis2({"ping": f"Комментарий к файлу {os.path.basename(file_path)}: {text}", "author_name": author_name, "is_group": (chat_id == AGENT_GROUP)}, system_prompt)
                    for part in split_message(reply_text):
                        send_telegram_message(chat_id, part)
                else:
                    print(f"Ошибка: file_id не найден в {attachments}")
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
            track_id = spotify_match.group(1)
            asyncio.create_task(grokky_spotify_response(track_id))
            reply_text = f"{author_name}, слушаю трек, ща разберусь!"
            for part in split_message(reply_text):
                send_telegram_message(chat_id, part)
            context_memory[author_name] = {"type": "track", "content": track_id}
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
            context = f"Topic: {chat_title}" if chat_title in ["ramble", "dev talk", "forum", "lit", "api talk", "method", "pseudocode"] else ""
            news = handle_news({"chat_id": chat_id, "group": (chat_id == AGENT_GROUP)})
            if news:
                reply_text = f"{author_name}, держи свежий раскат грома!\n\n{news}"
            else:
                reply_text = f"{author_name}, тишина в мире, нет новостей для бури."
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
