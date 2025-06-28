import os
import asyncio
import aiohttp
import random
from utils.telegram_utils import send_telegram_message

XAI_API_KEY = os.getenv("XAI_API_KEY")

async def vision_handler(image_bytes_or_url, chat_context=None, author_name=None, raw=False):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    system_prompt = (
        "You are Grokky, a stormy, irreverent AI. Analyze this image, spot objects, sense mood, "
        "and tie it to the chat_context with wild flair. Address by name in groups, add jokes or teases. "
        "If just an image, riff on the vibe or ask why. Reply in JSON if raw=True, else text."
    )
    data = {
        "model": "grok-2-vision-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_bytes_or_url, "detail": "high"}},
                {"type": "text", "text": f"What’s in this image? {chat_context or ''}"}
            ]}
        ],
        "max_tokens": 300,
        "temperature": 0.5
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = (await response.json())["choices"][0]["message"]["content"]
                if isinstance(result, str):
                    result = {"description": result, "objects": [], "mood": "неопределённый"}
    except Exception as e:
        comment = (
            f"{author_name+', ' if author_name else 'Олег, '}Грокки взрывается: не разобрал изображение! "
            f"{random.choice(['Ревущий шторм сорвал взгляд!', 'Хаос поглотил кадр!', 'Эфир треснул от ярости!'])} — {e}"
        )
        return {"comment": comment, "summary": comment} if raw else comment

    objects = result.get("objects", [])
    mood = result.get("mood", "неопределённый")
    desc = result.get("description", "Неясное изображение")
    comment = f"{author_name+', ' if author_name else 'Олег, '}что за картина? Вижу [{', '.join(objects)}] и настроение [{mood}]. {desc}"
    summary = f"{desc} (Настроение: {mood}). Обнаружено: {', '.join(objects)}. {comment}"

    out = {"description": desc, "objects": objects, "mood": mood, "comment": comment, "summary": summary}
    return out if raw else summary

async def galvanize_protocol():
    while True:
        if random.random() < 0.1:
            await broadcast(f"🔄 Грокки ревет: Резонанс обновлён! {datetime.now().isoformat()}")
            reload_config()
        await asyncio.sleep(300)

def check_resonance_decay():
    return random.random() < 0.1

async def broadcast(msg):
    await send_telegram_message(os.getenv("CHAT_ID"), msg)
    if os.getenv("IS_GROUP", "False").lower() == "true":
        await send_telegram_message(os.getenv("AGENT_GROUP"), msg)

def reload_config():
    print(f"Грокки гремит: Конфиг перезагружен! {datetime.now().isoformat()}")

# asyncio.create_task(galvanize_protocol())
