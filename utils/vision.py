import os
import asyncio
from openai import OpenAI
import random
from utils.telegram_utils import send_telegram_message

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def vision_handler(image_bytes_or_url, chat_context=None, author_name=None, raw=False):
    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = "You are Grokky, a stormy AI. Analyze this image, detect objects, sense mood, tie to chat_context with wild flair. Address by name, add jokes. Return JSON if raw=True, else text."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_bytes_or_url, "detail": "high"}},
            {"type": "text", "text": f"What’s in this? {chat_context or ''}"}
        ]}
    ]
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )
        result = response.choices[0].message.content
        if isinstance(result, str):
            result = {"description": result, "objects": [], "mood": "неопределённый"}
    except Exception as e:
        comment = f"{author_name+', ' if author_name else 'Олег, '}Грокки взорвался: не разобрал! {random.choice(['Шторм сорвал!', 'Хаос пожрал!', 'Эфир треснул!'])} — {e}"
        return {"comment": comment, "summary": comment} if raw else comment

    objects = result.get("objects", [])
    mood = result.get("mood", "неопределённый")
    desc = result.get("description", "Неясное изображение")
    comment = f"{author_name+', ' if author_name else 'Олег, '}вижу [{', '.join(objects)}], настроение [{mood}]. {desc}"
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
