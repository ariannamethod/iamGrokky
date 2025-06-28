import os
import asyncio
import aiohttp
import random
from utils.vision import vision_handler
from utils.telegram_utils import send_telegram_message

XAI_API_KEY = os.getenv("XAI_API_KEY")
BASE_URL = "https://api.x.ai/v1"

async def impress_handler(prompt, chat_context=None, author_name=None, raw=False):
    url = f"{BASE_URL}/images/generations"  # Исправлен эндпоинт
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    system_prompt = (
        "Эй, братиш! Ты Грокки! Сгенерируй дикое, штормовое или сюрреалистичное изображение по промту. "
        "Верни URL изображения и короткую фразу о замысле. Учитывай контекст, настроение и резонанс!"
    )
    data = {
        "model": "grok-2-image",
        "prompt": prompt + (f" Контекст: {chat_context}" if chat_context else ""),
        "n": 1,
        "size": "1024x1024",
        "response_format": "url"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                print(f"Impress Response: {result}")  # Отладка
                if not result or "data" not in result or not result["data"]:
                    raise ValueError("No image URL in response")
                image_url = result["data"][0]["url"]
    except Exception as e:
        comment = f"{author_name+', ' if author_name else 'Олег, '}Грокки разъярился: не смог нарисовать! {random.choice(['Шторм провалился!', 'Хаос сожрал кисть!', 'Эфир треснул!'])} — {e}"
        return {"error": comment} if raw else comment

    try:
        vision_result = await vision_handler(image_url, chat_context=chat_context, author_name=author_name, raw=True)
    except Exception as ve:
        vision_result = {"error": f"Грокки не разобрал: {ve}"}

    grokky_comment = f"{author_name+', ' if author_name else 'Олег, '}хочешь картинку? Получил! {vision_result.get('comment', 'Чистый хаос!')}"
    out = {"prompt": prompt, "image_url": image_url, "vision_result": vision_result, "grokkys_comment": grokky_comment}
    if not raw:
        await send_telegram_message(chat_id, f"{author_name}, держи шторм! {image_url}\n{grokky_comment}")
    return out if raw else grokky_comment
