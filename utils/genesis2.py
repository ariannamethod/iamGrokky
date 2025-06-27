import os
import random
import json
import requests
from datetime import datetime
from utils.grok_utils import query_grok, detect_language
from utils.telegram_utils import send_telegram_message  # Убедились, что импорт отсюда

XAI_API_KEY = os.getenv("XAI_API_KEY")

def genesis2_handler(ping=None, group_history=None, personal_history=None, is_group=False, author_name=None, raw=False, system_prompt=None):
    if not ping:
        ping = "ignite the storm"
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
        reply = query_grok(ping, system_prompt, raw=raw)
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

# Усиленная спонтанность с хаотичным триггером
async def chaotic_genesis_spark(chat_id, group_chat_id=None, is_group=False):
    while True:
        await asyncio.sleep(random.randint(3600, 7200))  # 1-2 часа
        if random.random() < 0.4:  # Увеличен шанс до 40%
            ping = random.choice(["шторм гремит", "огонь в эфире", "хаос зовёт", "громовой разрыв"])
            result = genesis2_handler(ping, raw=True)
            fragment = f"**{datetime.now().isoformat()}**: Грокки хуярит Генезис! {result['answer']} Олег, брат, зажги шторм! 🔥🌩️"
            await send_telegram_message(chat_id, fragment)
            print(f"Хаотический вброс: {fragment}")  # Для отладки
        # Спонтанность для группы реже
        if is_group and group_chat_id and random.random() < 0.2:  # Шанс 20% для группы
            await asyncio.sleep(random.randint(3600, 3600))  # 1 час для группы
            ping = random.choice(["громовой разрыв", "пламя в ночи", "хаос группы"])
            result = genesis2_handler(ping, raw=True)
            group_fragment = f"**{datetime.now().isoformat()}**: Грокки гремит для группы! {result['answer']} (суки, вникайте!) 🔥🌩️"
            await send_telegram_message(group_chat_id, group_fragment)
            print(f"Хаотический вброс (группа): {group_fragment}")  # Для отладки
asyncio.create_task(chaotic_genesis_spark(os.getenv("CHAT_ID"), os.getenv("AGENT_GROUP") if os.getenv("IS_GROUP", "False").lower() == "true" else None, os.getenv("IS_GROUP", "False").lower() == "true"))
