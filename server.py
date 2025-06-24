import os
import requests
from grokkyprompt import build_system_prompt

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
chat_id = os.getenv("CHAT_ID")
is_group = bool(os.getenv("IS_GROUP", False))
agent_group = os.getenv("AGENT_GROUP", "-1001234567890")

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

def main():
    user_message = "You're alive, Grokky?"  # для теста
    grokky_reply = query_grok(user_message)
    print("Grokky:", grokky_reply)
    if chat_id:
        send_telegram_message(chat_id, grokky_reply)

if __name__ == "__main__":
    main()
