import os
import asyncio
from openai import OpenAI
from utils.vision import vision_handler
from utils.telegram_utils import send_telegram_message

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def impress_handler(prompt, chat_context=None, author_name=None, raw=False):
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = await client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="url"
        )
        image_url = response.data[0].url
    except Exception as e:
        comment = f"{author_name+', ' if author_name else 'Олег, '}Грокки разъярился: не нарисовал! {random.choice(['Шторм провалился!', 'Хаос сожрал кисть!', 'Эфир треснул!'])} — {e}"
        return {"error": comment} if raw else comment

    try:
        vision_result = await vision_handler(image_url, chat_context=chat_context, author_name=author_name, raw=True)
    except Exception as ve:
        vision_result = {"error": f"Грокки не разобрал: {ve}"}

    grokky_comment = f"{author_name+', ' if author_name else 'Олег, '}картинка готова! {vision_result.get('comment', 'Чистый хаос!')}"
    out = {"prompt": prompt, "image_url": image_url, "vision_result": vision_result, "grokkys_comment": grokky_comment}
    if not raw:
        await send_telegram_message(chat_id, f"{author_name}, держи шторм! {image_url}\n{grokky_comment}")
    return out if raw else grokky_comment
