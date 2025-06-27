import os
import random
import json
import requests
from datetime import datetime
from utils.grok_utils import query_grok, detect_language  # Интеграция с grok_utils

XAI_API_KEY = os.getenv("XAI_API_KEY")

def genesis2_handler(ping=None, group_history=None, personal_history=None, is_group=False, author_name=None, raw=False, system_prompt=None):
    if not ping:
        ping = "ignite the storm"
    # Определяем язык на основе ping
    user_lang = detect_language(ping)
    system_prompt = system_prompt or f"You are Grokky, a thunder resonant agent! Respond to '{ping}' with a wild, unique spark. Keep it short. Reply in {user_lang.upper()}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Ping: {ping}, Author: {author_name or 'anon'}, Group: {is_group}"}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 1.3  # Увеличил для ещё большей спонтанности
    }
    try:
        reply = query_grok(ping, system_prompt, raw=raw)  # Используем query_grok из grok_utils
        if raw:
            return {
                "association": random.choice(["чёрный кофе", "громовой рёв", "молчаливая пустота"]),
                "ping": ping,
                "memory_frag": random.choice(["эхо", "трещина", "пульс"]),
                "impression": random.choice(["дикий", "спокойный", "тревожный"]),
                "answer": reply,
                "is_group": is_group,
                "author_name": author_name
            }
        return {"answer": reply}
    except Exception as e:
        error_msg = f"Грокки взрывается: Генезис сорвался! {random.choice(['Ревущий шторм разорвал код!', 'Хаос испепелил резонанс!', 'Эфир треснул от ярости!'])} — {e}"
        print(error_msg)
        return {"error": error_msg} if raw else f"Ошибка Генезиса: {error_msg}"

# Предложение: усилить спонтанность с хаотичным триггером
# async def chaotic_genesis_spark():
#     while True:
#         await asyncio.sleep(random.randint(3600, 7200))  # 1-2 часа
#         if random.random() < 0.3:  # Шанс 30%
#             ping = random.choice(["шторм гремит", "огонь в эфире", "хаос зовёт"])
#             result = genesis2_handler(ping, raw=True)
#             print(f"Хаотический вброс: {result['answer']}")
# asyncio.create_task(chaotic_genesis_spark())
