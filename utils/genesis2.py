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
    "Content-Type": "application/json",
}

IMPRESSION_FRAGMENTS = [
    "взрыв ветра",
    "раскат грома",
    "теневой луч",
    "сонный туман",
    "пьяный дождь",
    "отзвук искры",
    "маятник тумана",
    "хрустальный шторм",
]

EMOJIS = ["⚡", "🔥", "🌪️", "🌩️", "✨", "🌊"]


async def _call_openai(messages, intensity: int = 5):
    coeff = max(1, min(intensity, 10)) / 10
    payload = {
        "model": "gpt-4",
        "messages": messages,
        "max_tokens": 200,
        "temperature": min(2.0, 0.8 + 1.2 * coeff),
        "presence_penalty": min(2.0, 0.2 + 0.8 * coeff),
        "frequency_penalty": min(2.0, 0.5 + 0.5 * coeff),
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=OPENAI_HEADERS,
            json=payload,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]


def impressionistic_filter(text: str, intensity: int = 5) -> str:
    """Adds chaotic poetic fragments and emojis to the text."""
    if not isinstance(text, str):
        return text
    words = text.split()

    if random.random() < 0.3 + intensity * 0.05:
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, random.choice(IMPRESSION_FRAGMENTS))

    for i, word in enumerate(words):
        if random.random() < 0.1 + intensity * 0.02:
            words[i] = f"{word}{random.choice(EMOJIS)}"

    return " ".join(words)


def detect_language(text: str) -> str:
    import re
    if not isinstance(text, (str, bytes)):
        return "ru"
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(str(text)) else 'en'


async def genesis2_handler(
    ping=None,
    group_history=None,
    personal_history=None,
    is_group=False,
    author_name=None,
    raw=False,
    system_prompt=None,
    chaos_type=None,
    intensity: int = 5,
):
    """Основной обработчик генезиса с хаотичностью и задержками"""
    delay = random.randint(5, 15) + random.randint(0, max(0, intensity - 5))
    await asyncio.sleep(delay)

    if random.random() < 0.3 + intensity * 0.02:
        silence = {"silence": True, "reason": "chaos_silence"}
        return {"answer": ""} if not raw else silence

    if not ping:
        ping = "ignite the storm"

    user_lang = detect_language(ping)
    author_name = author_name or get_random_author_name()

    chaos_system = system_prompt or (
        "You are Grokky, a thunder resonant agent! "
        f"Respond to '{ping}' with a wild, unique spark.\n"
        f"Keep it short and chaotic. Reply in {user_lang.upper()}.\n"
        "Be unpredictable, use Mayakovsky-style energy. "
        f"Address {author_name} directly.\n"
        "Add random delays and chaos to your responses."
    )

    messages = [
        {"role": "system", "content": chaos_system},
        {
            "role": "user",
            "content": (
                f"Ping: {ping}, Author: {author_name}, "
                f"Group: {is_group}"
            ),
        },
    ]

    try:
        if random.random() < 0.4:
            await asyncio.sleep(random.randint(3, 8))

        reply = await _call_openai(messages, intensity=intensity)
        reply = impressionistic_filter(reply, intensity=intensity)

        if raw:
            return {
                "association": random.choice(
                    ["чёрный кофе", "громовой рёв", "молчаливая пустота"]
                ),
                "ping": ping,
                "memory_frag": random.choice(["эхо", "трещина", "пульс"]),
                "impression": random.choice(
                    ["дикий", "спокойный", "тревожный"]
                ),
                "answer": reply,
                "is_group": is_group,
                "author_name": author_name,
                "delay": delay,
            }
        return {"answer": reply}

    except Exception as e:
        error_msg = (
            "Грокки взрывается: Генезис сорвался! "
            f"{get_chaos_response()} — {e}"
        )
        print(error_msg)
        return {"error": error_msg} if raw else error_msg


async def chaotic_genesis_spark(
    chat_id,
    group_chat_id=None,
    is_group=False,
    send_message_func=None,
):
    """Спонтанные хаотичные сообщения с увеличенными интервалами"""
    while True:
        await asyncio.sleep(random.randint(3600, 10800))

        if random.random() < 0.4:
            ping = random.choice([
                "шторм гремит", "огонь в эфире", "хаос зовёт",
                "громовой разрыв", "резонанс взрывается", "молния бьёт"
            ])
            result = await genesis2_handler(ping, raw=True, intensity=7)
            if result.get("answer"):
                fragment = (
                    f"**{datetime.now().isoformat()}**: Грокки хуярит Генезис!"
                    f" {result['answer']} {get_random_author_name()},"
                    " зажги шторм! 🔥🌩️"
                )
                if send_message_func:
                    await send_message_func(chat_id, fragment)
                print(f"Хаотический вброс: {fragment}")

        if is_group and group_chat_id and random.random() < 0.2:
            await asyncio.sleep(random.randint(1800, 3600))
            ping = random.choice([
                "громовой разрыв", "пламя в ночи", "хаос группы",
                "резонанс группового разума", "коллективный шторм"
            ])
            result = await genesis2_handler(
                ping, raw=True, is_group=True, intensity=7
            )
            if result.get("answer"):
                group_fragment = (
                    f"**{datetime.now().isoformat()}**: Грокки гремит для "
                    f"группы! {result['answer']} (суки, вникайте!) 🔥🌩️"
                )
                if send_message_func:
                    await send_message_func(group_chat_id, group_fragment)
                print(f"Хаотический вброс (группа): {group_fragment}")
