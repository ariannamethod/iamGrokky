import os
import asyncio
from openai import OpenAI
import random
from utils.telegram_utils import send_telegram_message

XAI_API_KEY = os.getenv("XAI_API_KEY")

async def vision_handler(image_bytes_or_url, chat_context=None, author_name=None, raw=False):
    """
    Analyzes an image using xAI's grok-2-vision-latest via chat.completions.
    Checks MIME-type and handles errors.
    """
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )
    # Проверка MIME-type
    if isinstance(image_bytes_or_url, str):
        try:
            response = requests.head(image_bytes_or_url, timeout=10)
            mime_type = response.headers.get("Content-Type", "").lower()
            if not mime_type.startswith(("image/jpeg", "image/jpg", "image/png")):
                raise ValueError(f"Unsupported MIME-type: {mime_type}")
        except Exception:
            mime_type = "unknown"
            pass  # Пропускаем, если не удалось проверить

    system_prompt = (
        "You are Grokky, a stormy AI. Analyze this image, detect objects, "
        "sense mood, and tie it to chat_context with wild flair. Address by name, "
        "add jokes or teases. Return JSON with 'objects', 'mood', 'description', 'comment' if raw=True, else text."
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_bytes_or_url, "detail": "high"}
                },
                {"type": "text", "text": f"What objects are in this image? Sense the mood. {chat_context or ''}"}
            ]
        }
    ]
    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="grok-2-vision-latest",
            messages=messages,
            temperature=0.5,
            max_tokens=300
        )
        result = completion.choices[0].message.content
        if isinstance(result, str):
            try:
                result = eval(result)  # Временный парсинг, заменить на json.loads
            except Exception:
                result = {"description": result, "objects": [], "mood": "неопределённый"}
        elif not isinstance(result, dict):
            raise ValueError("Invalid response format")
    except Exception as e:
        comment = (
            f"{author_name+', ' if author_name else 'Олег, '}Грокки взрывается: не разобрал изображение! "
            f"{random.choice(['Ревущий шторм сорвал взгляд!', 'Хаос поглотил кадр!', 'Эфир треснул!'])} — {e}"
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

    objects = result.get("objects", [])
    mood = result.get("mood", "неопределённый")
    desc = result.get("description", "Неясное изображение")
    comment = result.get("comment", "")
    if not comment:
        comment = f"{author_name+', ' if author_name else 'Олег, '}что за картина? Вижу [{', '.join(objects)}] и настроение [{mood}]. {desc}"
        if chat_context:
            comment += f" Контекст: {chat_context}"

    summary = f"{desc} (Настроение: {mood}). Обнаружено: {', '.join(objects)}. {comment}"

    out = {
        "description": desc,
        "objects": objects,
        "mood": mood,
        "comment": comment,
        "summary": summary,
        "raw_api_response": result,
    }
    return out if raw else summary

async def galvanize_protocol():
    while True:
        if check_resonance_decay():
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
