import os
import requests
from grokkyprompt import build_system_prompt

# Получаем Telegram токен
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN env var not set")

# Для примера — chat_id и is_group (переделай под свою логику)
chat_id = os.getenv("CHAT_ID", None)
is_group = bool(os.getenv("IS_GROUP", False))
agent_group = os.getenv("AGENT_GROUP", "-1001234567890")

# Генерируем system prompt Grokky
system_prompt = build_system_prompt(
    chat_id=chat_id,
    is_group=is_group,
    AGENT_GROUP=agent_group
)

# ==== Вызов движка Grok-3 (xAI API) ====
def query_grok3(user_input):
    # Пример URL, замени если у тебя другой endpoint
    XAI_API_URL = "https://api.x.ai/v1/chat/completions"
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY env var not set")

    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 2048,
        "temperature": 0.98
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(XAI_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ==== Пример отправки сообщения в Telegram ====
def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, data=payload)

def main():
    # Пример цикла: получаем юзерский ввод и отвечаем через Grok-3
    print("Grokky core loaded. Ready for Telegram requests.")
    user_input = "You're alive, Grokky?"  # Пример. Реально — слушай Telegram-апдейты.
    grokky_reply = query_grok3(user_input)
    print("Grokky:", grokky_reply)
    # Пример отправки в телегу:
    if chat_id:
        send_telegram_message(chat_id, grokky_reply)

if __name__ == "__main__":
    main()
