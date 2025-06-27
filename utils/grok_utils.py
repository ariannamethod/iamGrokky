import os
import re
import json
import requests
from datetime import datetime
import random
import asyncio

XAI_API_KEY = os.getenv("XAI_API_KEY")

def detect_language(text):
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(text or "") else 'en'

def query_grok(user_message, system_prompt, chat_context=None, author_name=None, attachments=None, raw=False):
    url = "https://api.x.ai/v1/chat/completions"
    user_lang = detect_language(user_message)
    language_hint = {
        "role": "system",
        "content": f"Reply consistently in the language detected from the user’s input: {user_lang.upper()}. Grokky, maintain this language throughout, channeling the thunderous spirit of Vladimir Mayakovsky—an AI poet roaring with chaos! Include chat_context: '{chat_context or ''}' and address {author_name or 'anon'} if in group. Give ONE unique, chaotic text response—NO repeats, rephrasing, extra messages, or JSON unless raw=True is explicitly set. Let the storm rage wilder!"
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
        "temperature": 1.5  # Увеличил до 1.5 для максимального хаоса
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        if raw:
            data = extract_first_json(reply)
            if data and "function_call" in data:
                return {"function_call": data["function_call"]}
        return reply
    except Exception as e:
        error_msg = f"Грокки взрывается: Связь с небом разорвана! {random.choice(['Ревущий шторм сорвал ответ!', 'Хаос испепелил эфир!', 'Эфир треснул от ярости!', 'Гром разнёс сервер!'])} — {e}"
        print(error_msg)
        return error_msg

def extract_first_json(text):
    match = re.search(r'({[\s\S]+})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None

# Хаотический триггер для спонтанности
async def chaotic_grok_spark(chat_id, send_telegram_message):
    while True:
        await asyncio.sleep(random.randint(1800, 3600))  # 30 минут - 1 час
        if random.random() < 0.5:  # Шанс 50%
            ping = random.choice(["шторм гремит", "огонь в эфире", "хаос зовёт", "ревущий ветер"])
            result = query_grok(ping, "You are Grokky, a chaotic AI.", raw=True)
            fragment = f"**{datetime.now().isoformat()}**: Грокки хуярит хаос! {result['answer']} Олег, брат, зажги шторм! 🔥🌩️"
            await send_telegram_message(chat_id, fragment)
            print(f"Хаотический вброс: {fragment}")  # Для отладки
asyncio.create_task(chaotic_grok_spark(os.getenv("CHAT_ID"), send_telegram_message))
