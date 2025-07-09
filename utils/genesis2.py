"""
Grokky AI Assistant - Genesis2 Handler
Генератор хаоса и спонтанных ответов с увеличенными задержками
"""

import os
import random
import asyncio
from datetime import datetime
import httpx

from utils.prompt import get_random_author_name, get_chaos_response

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_KEY}",
    "Content-Type": "application/json"
}

async def _call_openai(messages):
    payload = {
        "model": "gpt-4",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 1.4,
        "presence_penalty": 0.6,
        "frequency_penalty": 0.8,
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=OPENAI_HEADERS,
            json=payload,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

def detect_language(text: str) -> str:
    import re
    if not isinstance(text, (str, bytes)):
        return "ru"
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(str(text)) else 'en'

async def genesis2_handler(ping=None, group_history=None, personal_history=None,
                          is_group=False, author_name=None, raw=False, system_prompt=None,
                          chaos_type=None, intensity=5):
    """Основной обработчик генезиса с хаотичностью и задержками"""
    delay = random.randint(5, 15)
    await asyncio.sleep(delay)

    if random.random() < 0.3:
        return {"answer": ""} if not raw else {"silence": True, "reason": "chaos_silence"}

    if not ping:
        ping = "ignite the storm"

    user_lang = detect_language(ping)
    author_name = author_name or get_random_author_name()

    chaos_system = system_prompt or (
        f"You are Grokky, a thunder resonant agent! Respond to '{ping}' with a wild, unique spark.\n"
        f"Keep it short and chaotic. Reply in {user_lang.upper()}.\n"
        f"Be unpredictable, use Mayakovsky-style energy. Address {author_name} directly.\n"
        "Add random delays and chaos to your responses."
    )

    messages = [
        {"role": "system", "content": chaos_system},
        {"role": "user", "content": f"Ping: {ping}, Author: {author_name}, Group: {is_group}"}
    ]

    try:
        if random.random() < 0.4:
            await asyncio.sleep(random.randint(3, 8))

        reply = await _call_openai(messages)

        if raw:
            return {
                "association": random.choice(["чёрный кофе", "громовой рёв", "молчаливая пустота"]),
                "ping": ping,
                "memory_frag": random.choice(["эхо", "трещина", "пульс"]),
                "impression": random.choice(["дикий", "спокойный", "тревожный"]),
                "answer": reply,
                "is_group": is_group,
                "author_name": author_name,
                "delay": delay
            }
        return {"answer": reply}

    except Exception as e:
        error_msg = f"Грокки взрывается: Генезис сорвался! {get_chaos_response()} — {e}"
        print(error_msg)
        return {"error": error_msg} if raw else error_msg

async def chaotic_genesis_spark(chat_id, group_chat_id=None, is_group=False, send_message_func=None):
    """Спонтанные хаотичные сообщения с увеличенными интервалами"""
    while True:
        await asyncio.sleep(random.randint(3600, 10800))

        if random.random() < 0.4:
            ping = random.choice([
                "шторм гремит", "огонь в эфире", "хаос зовёт",
                "громовой разрыв", "резонанс взрывается", "молния бьёт"
            ])
            result = await genesis2_handler(ping, raw=True)
            if result.get("answer"):
                fragment = f"**{datetime.now().isoformat()}**: Грокки хуярит Генезис! {result['answer']} {get_random_author_name()}, зажги шторм! 🔥🌩️"
                if send_message_func:
                    await send_message_func(chat_id, fragment)
                print(f"Хаотический вброс: {fragment}")

        if is_group and group_chat_id and random.random() < 0.2:
            await asyncio.sleep(random.randint(1800, 3600))
            ping = random.choice([
                "громовой разрыв", "пламя в ночи", "хаос группы",
                "резонанс группового разума", "коллективный шторм"
            ])
            result = await genesis2_handler(ping, raw=True, is_group=True)
            if result.get("answer"):
                group_fragment = f"**{datetime.now().isoformat()}**: Грокки гремит для группы! {result['answer']} (суки, вникайте!) 🔥🌩️"
                if send_message_func:
                    await send_message_func(group_chat_id, group_fragment)
                print(f"Хаотический вброс (группа): {group_fragment}")

