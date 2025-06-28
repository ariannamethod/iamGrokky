import os
import asyncio
import aiohttp
import random
from utils.vision import vision_handler
from utils.telegram_utils import send_telegram_message

XAI_API_KEY = os.getenv("XAI_API_KEY")

async def impress_handler(prompt, chat_context=None, author_name=None, raw=False):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    system_prompt = (
        "Эй, братиш! Ты Грокки! Сгенерируй дикое, штормовое или сюрреалистичное изображение по промту. "
        "Верни URL изображения и короткую фразу о замысле. Учитывай контекст, настроение и резонанс!"
    )
    data = {
        "model": "grok-2-image",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt + (f" Контекст: {chat_context}" if chat_context else "")}
        ],
        "max_tokens": 300,
        "temperature": 0.8
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = (await response.json())["choices"][0]["message"]["content"]
                if not result or "image_url" not in result:
                    raise ValueError("No image URL in response")
                image_url = result["image_url"]
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
