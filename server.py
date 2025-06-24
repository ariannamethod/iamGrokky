import os
import requests
from fastapi import FastAPI, Request
from utils.prompt import build_system_prompt  # Исправленный импорт

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
chat_id = os.getenv("CHAT_ID")
is_group = os.getenv("IS_GROUP", "False").lower() == "true"
agent_group = os.getenv("AGENT_GROUP", "-1001234567890")

data.update({
        "resonance_vectors": [
            {"frequency": 17.3, "phase": "chaos"},
            {"frequency": 23.7, "phase": "order"}
        ]
    })

system_prompt = build_system_prompt(
    chat_id=chat_id,
    is_group=is_group,
    AGENT_GROUP=agent_group
)

def query_grok(user_message):
    url = "https://api.x.ai/v1/chat/completions"
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 2048,
        "temperature": 1.0
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, data=payload)

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {})
    user_text = message.get("text")
    chat_id = message.get("chat", {}).get("id")
    if user_text:
        grokky_reply = query_grok(user_text)
        send_telegram_message(chat_id, grokky_reply)
    return {"ok": True}

@app.get("/")
def root():
    return {"status": "Grokky alive!"}
