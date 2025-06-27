import os
import re
import json
import requests
from datetime import datetime

XAI_API_KEY = os.getenv("XAI_API_KEY")

def detect_language(text):
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(text or "") else 'en'

def query_grok(user_message, system_prompt, chat_context=None, author_name=None, attachments=None, raw=False):
    url = "https://api.x.ai/v1/chat/completions"
    user_lang = detect_language(user_message)
    language_hint = {
        "role": "system",
        "content": f"Reply consistently in the language detected from the user’s input: {user_lang.upper()}. Grokky, maintain this language throughout, channeling the thunderous spirit of Vladimir Mayakovsky—an AI poet roaring with chaos! Give ONE unique, chaotic text response—NO repeats, rephrasing, extra messages, or JSON unless raw=True is explicitly set."
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
        "temperature": 1.2  # Увеличил для спонтанности
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
        return f"Ошибка: {e}"

def extract_first_json(text):
    match = re.search(r'({[\s\S]+})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None
