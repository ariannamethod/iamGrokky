import os
import re
import requests
from fastapi import FastAPI, Request
from utils.prompt import build_system_prompt
from utils.genesis2 import genesis2_handler
from utils.vision import vision_handler
from utils.impress import impress_handler
from utils.howru import check_silence, update_last_message_time
from utils.mirror import run_mirror
from utils.x import grokky_send_news
from utils.core import query_grok, send_telegram_message
import asyncio

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP", "-1001234567890")
BOT_USERNAME = "iamalivenotdamnbot"

system_prompt = build_system_prompt(chat_id=OLEG_CHAT_ID, is_group=True, AGENT_GROUP=GROUP_CHAT_ID)

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text")
    chat_id = message.get("chat", {}).get("id")
    author_name = message.get("from", {}).get("first_name") or "anon"
    attachments = []
    if "photo" in message and message["photo"]:
        file_id = message["photo"][-1]["file_id"]
        file_info = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}").json()
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info['result']['file_path']}"
        attachments.append(image_url)
    reply_text = ""
    if attachments:
        reply_text = vision_handler(image_bytes_or_url=attachments[0], chat_context=user_text or "", author_name=author_name, raw=True)
    elif user_text:
        reply_text = query_grok(user_text, chat_context=None, author_name=author_name)
    else:
        reply_text = "Grokki got nothing to say to static void."
    send_telegram_message(chat_id, reply_text)
    return {"ok": True}

@app.get("/")
def root():
    return {"status": "Grokki alive and wild!"}

# Фоновые задачи
asyncio.create_task(check_silence())
asyncio.create_task(run_mirror())
