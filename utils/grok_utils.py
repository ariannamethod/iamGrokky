import os
import re
import json
import requests
from datetime import datetime
import random
import asyncio
import difflib
from openai import OpenAI
from utils.telegram_utils import send_telegram_message

XAI_API_KEY = os.getenv("XAI_API_KEY")

def detect_language(text):
    if not isinstance(text, (str, bytes)):
        return "ru"  # Фallback на русский, если dict
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(text) else 'en'

def query_grok(user_message, system_prompt, chat_context=None, author_name=None, attachments=None, raw=False):
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )
    user_lang = detect_language(user_message)
    language_hint = (
        f"Reply consistently in {user_lang.upper()}. You are Grokky, a chaotic AI channeling Mayakovsky’s thunder! "
        f"Include chat_context: '{chat_context or ''}' and address {author_name or 'anon'} if in group. "
        f"Give ONE unique, wild text response—NO repeats or extra messages unless raw=True. Add web search if needed!"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": language_hint},
        {"role": "user", "content": user_message}
    ]
    tools = [
        {
            "type": "web_search",
            "web_search_options": {
                "user_location": {"type": "approximate", "country": "RU"}
            }
        }
    ]
    try:
        response = client.chat.completions.create(
            model="grok-3",
            messages=messages,
            tools=tools,
            max_tokens=300,
            temperature=1.5
        )
        reply = response.choices[0].message.content
        if raw and response.choices[0].message.tool_calls:
            return {"function_call": {"name": "web_search", "arguments": response.choices[0].message.tool_calls[0].function.arguments}}
        return reply
    except Exception as e:
        error_msg = f"Грокки взрывается: Связь с небом разорвана! {random.choice(['Ревущий шторм сорвал ответ!', 'Хаос испепелил эфир!', 'Эфир треснул!'])} — {e}"
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

async def chaotic_grok_spark(chat_id, group_chat_id=None, is_group=False):
    while True:
        await asyncio.sleep(random.randint(1800, 3600))
        if random.random() < 0.5:
            ping = random.choice(["шторм гремит", "огонь в эфире", "хаос зовёт", "ревущий ветер"])
            result = query_grok(ping, "You are Grokky, a chaotic AI agent of resonance.", raw=True)
            fragment = f"**{datetime.now().isoformat()}**: Грокки хуярит хаос! {result['answer']} Олег, брат, зажги шторм! 🔥🌩️"
            await send_telegram_message(chat_id, fragment)
            print(f"Хаотический вброс (личка): {fragment}")
        if is_group and group_chat_id and random.random() < 0.2:
            await asyncio.sleep(random.randint(3600, 3600))
            ping = random.choice(["громовой разрыв", "пламя в ночи", "хаос группы"])
            result = query_grok(ping, "You are Grokky, a chaotic AI agent of resonance.", raw=True)
            group_fragment = f"**{datetime.now().isoformat()}**: Грокки гремит для группы! {result['answer']} (суки, вникайте!) 🔥🌩️"
            await send_telegram_message(group_chat_id, group_fragment)
            print(f"Хаотический вброс (группа): {group_fragment}")
# asyncio.create_task(chaotic_grok_spark(os.getenv("CHAT_ID"), os.getenv("AGENT_GROUP") if os.getenv("IS_GROUP", "False").lower() == "true" else None, os.getenv("IS_GROUP", "False").lower() == "true"))
