import os
import requests
from utils.vision import vision_handler
from utils.telegram_utils import send_telegram_message

XAI_API_KEY = os.getenv("XAI_API_KEY")

def impress_handler(prompt, chat_context=None, author_name=None, raw=False):
    """
    Generates an image via xAI by prompt, then analyzes it with vision_handler.
    If generation fails, returns a witty text response.
    - prompt: text to inspire the image
    - chat_context: recent chat context for flavor
    - author_name: for personal address in groups
    - raw: return full JSON if True, else text summary
    Returns:
        dict (raw=True): {"prompt", "image_url", "vision_result", "grokkys_comment", "raw_api_response"}
        str (raw=False): witty text or error message
    Grokky-chaos: always with a tease, even if the image is absurd or fails.
    """
    gen_endpoint = "https://api.x.ai/v1/multimodal/image-generation"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    system_prompt = (
        "Эй, братиш! Ты Грокки! Сгенерируй дикое, штормовое или сюрреалистичное изображение по промту. "
        "Делай вывод ярким и экспрессивным. Верни URL изображения и короткую фразу о своём замысле. "
        "Учитывай контекст беседы, настроение и резонанс, ёмаё!"
    )
    data = {
        "prompt": prompt,
        "system": system_prompt,
    }
    if chat_context:
        data["chat_context"] = chat_context

    try:
        resp = requests.post(gen_endpoint, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        image_result = resp.json()
    except Exception as e:
        comment = (
            f"{author_name+', ' if author_name else 'Олег, '}Грокки разъярился: не смог нарисовать изображение! ({e}) "
            "Шторм провалился, давай новый промт!"
        )
        out = {
            "prompt": prompt,
            "error": comment,
            "reason": str(e),
            "raw_api_response": str(getattr(e, 'response', None)),
        }
        return out if raw else comment

    image_url = image_result.get("image_url")
    if not image_url:
        comment = (
            f"{author_name+', ' if author_name else 'Олег, '}Грокки в шоке: xAI не дал URL! "
            "Шторм провалился, кидай новый вызов!"
        )
        out = {
            "prompt": prompt,
            "error": comment,
            "raw_api_response": image_result,
        }
        return out if raw else comment

    try:
        vision_result = vision_handler(
            image_url,
            chat_context=chat_context,
            author_name=author_name,
            raw=True
        )
    except Exception as ve:
        vision_result = {
            "error": f"Грокки не смог разобрать изображение: {ve}"
        }

    grokky_comment = (
        f"{author_name+', ' if author_name else 'Олег, '}хочешь картинку? Получил! "
        f"Но серьёзно, что ты ждал? {vision_result.get('comment', 'Тут только статика в пустоте.')}"
    )

    out = {
        "prompt": prompt,
        "image_url": image_url,
        "vision_result": vision_result,
        "grokkys_comment": grokky_comment,
        "raw_api_response": image_result,
    }
    if not raw:
        send_telegram_message(chat_id, f"Олег, держи шторм! {image_url}\n{grokky_comment}")
    return out if raw else grokky_comment
