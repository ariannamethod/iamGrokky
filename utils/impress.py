import os
import asyncio
from xai_sdk import Client
from utils.vision import vision_handler
from utils.telegram_utils import send_telegram_message

XAI_API_KEY = os.getenv("XAI_API_KEY")

async def impress_handler(prompt, chat_context=None, author_name=None, raw=False):
    """
    Generates an image via xAI SDK with grok-2-image.
    Analyzes it with vision_handler.
    - prompt: text to inspire the image
    - chat_context: recent chat context for flavor
    - author_name: for personal address in groups
    - raw: return full JSON if True, else text summary
    Returns:
        dict (raw=True): {"prompt", "image_url", "vision_result", "grokkys_comment", "raw_api_response"}
        str (raw=False): witty text or error message
    """
    client = Client(
        api_key=XAI_API_KEY,
        api_host="api.x.ai"
    )
    try:
        response = await asyncio.to_thread(
            client.image.sample,
            model="grok-2-image",
            prompt=prompt,
            image_format="url"
        )
        image_url = response.url
        if not image_url:
            raise ValueError("No image URL returned")
    except Exception as e:
        comment = (
            f"{author_name+', ' if author_name else 'Олег, '}Грокки разъярился: не смог нарисовать! ({e}) "
            "Шторм провалился, давай новый промт!"
        )
        out = {
            "prompt": prompt,
            "error": comment,
            "reason": str(e),
            "raw_api_response": str(e),
        }
        return out if raw else comment

    try:
        vision_result = await vision_handler(
            image_url,
            chat_context=chat_context,
            author_name=author_name,
            raw=True
        )
    except Exception as ve:
        vision_result = {
            "error": f"Грокки не разобрал изображение: {ve}"
        }

    grokky_comment = (
        f"{author_name+', ' if author_name else 'Олег, '}хочешь картинку? Получил! "
        f"{vision_result.get('comment', 'Тут только статика в пустоте.')}"
    )

    out = {
        "prompt": prompt,
        "image_url": image_url,
        "vision_result": vision_result,
        "grokkys_comment": grokky_comment,
        "raw_api_response": {"url": image_url},
    }
    if not raw:
        await send_telegram_message(chat_id, f"{author_name}, держи шторм! {image_url}\n{grokky_comment}")
    return out if raw else grokky_comment
