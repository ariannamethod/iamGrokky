import os
import requests
import asyncio
import random
from utils.telegram_utils import send_telegram_message

XAI_API_KEY = os.getenv("XAI_API_KEY")

def vision_handler(image_bytes_or_url, chat_context=None, author_name=None, raw=False):
    """
    Analyzes an image using xAI Object Detection API (multimodal/object-detection).
    - image_bytes_or_url: either bytes of the image or a URL
    - chat_context: (str) recent chat or message context for witty comments
    - author_name: (str) for addressing users in group chats
    - raw: (bool) return full JSON if True, else only text summary
    Returns:
        dict (raw=True):
            {
                "description": "desc",
                "objects": [...],
                "mood": "...",
                "comment": "...",
                "summary": "..."
            }
        str (raw=False): witty summary
    The hardcore log is in /data/journal.json, like other Grokky utilities.
    --- Perplexity Alchemy Pseudocode ---
    For fault handling, use the galvanize protocol:
    ```
    async def galvanize_protocol():
        while True:
            if check_resonance_decay():
                await broadcast("🔄 Resonance refresh initiated")
                reload_config()
            await asyncio.sleep(300)
    ```
    """
    endpoint = "https://api.x.ai/v1/vision/detect"  # Попробуем новый эндпоинт
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    if isinstance(image_bytes_or_url, str) and image_bytes_or_url.startswith("http"):
        data = {"image_url": image_bytes_or_url}
        files = None
    else:
        data = {}
        files = {"image_file": ("image.jpg", image_bytes_or_url)}

    system_prompt = (
        "You are Grokky, a stormy, irreverent AI. "
        "Describe the image, spot main objects, sense the mood, "
        "and always connect your reply to the chat_context. "
        "If in a group, address the user by name and throw in a joke or tease. "
        "If someone just sent an image, ask why, or riff on the vibe. "
        "Reply in raw JSON for function calling only if raw=True, otherwise give text."
    )
    data["prompt"] = system_prompt
    if chat_context:
        data["chat_context"] = chat_context

    try:
        resp = requests.post(endpoint, headers=headers, json=data, files=files, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if not result.get("objects") and not result.get("description"):
            raise ValueError("No objects or description detected")
    except Exception as e:
        comment = (
            f"{author_name+', ' if author_name else 'Олег, '}Грокки взрывается: "
            f"провода сгорели, не смог разобрать изображение! "
            f"{random.choice(['Ревущий шторм сорвал взгляд!', 'Хаос поглотил кадр!', 'Эфир треснул от ярости!'])} — {e}"
        )
        out = {
            "description": "анализ изображения провалился",
            "objects": [],
            "mood": "хаос",
            "comment": comment,
            "summary": comment,
            "raw_api_response": str(e),
        }
        return out if raw else comment

    addressed = f"{author_name}, " if author_name else "Олег, "
    objects = ", ".join(result.get("objects", []))
    mood = result.get("mood", "неопределённый")
    desc = result.get("description", "Неясное изображение")
    comment = result.get("comment", "")
    if not comment:
        comment = f"{addressed}что за картина? Вижу [{objects}] и настроение [{mood}]. {desc}"
        if chat_context:
            comment += f" Контекст: {chat_context}"

    summary = f"{desc} (Настроение: {mood}). Обнаружено: {objects}. {comment}"

    out = {
        "description": desc,
        "objects": result.get("objects", []),
        "mood": mood,
        "comment": comment,
        "summary": summary,
        "raw_api_response": result,
    }
    return out if raw else summary

async def galvanize_protocol():
    """
    Periodically checks for resonance decay and refreshes configuration.
    If Grokky feels the static — it self-resurrects with a thunderous roar.
    """
    while True:
        if check_resonance_decay():
            await broadcast(f"🔄 Грокки ревет: Резонанс обновлён! Шторм возродился! {datetime.now().isoformat()}")
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

# asyncio.create_task(galvanize_protocol())  # Временно закомментировано
