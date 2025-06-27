import os
import requests
import asyncio

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

    # Обновлённый эндпоинт (проверь, что он верный для xAI)
    endpoint = "https://api.x.ai/v1/multimodal/object-detection"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"  # Убедимся, что формат правильный
    }

    # Подготовка данных для API
    if isinstance(image_bytes_or_url, str) and image_bytes_or_url.startswith("http"):
        data = {"image_url": image_bytes_or_url}
        files = None
    else:
        data = {}
        files = {"image_file": ("image.jpg", image_bytes_or_url)}  # Предполагаем JPG, можно улучшить

    # xAI prompt с максимальным вайбом Grokky
    system_prompt = (
        "You are Grokky, a stormy, irreverent AI. "
        "Describe the image, spot main objects, sense the mood, "
        "and always connect your reply to the chat context. "
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
        # Проверка результата
        if not result.get("objects") and not result.get("description"):
            raise ValueError("No objects or description detected")
    except Exception as e:
        comment = (
            f"{author_name+', ' if author_name else ''}Grokky fried the wires: "
            f"couldn't analyze image (reason: {e}). "
            "Galvanize protocol: resonance refresh initiated."
        )
        out = {
            "description": "image analysis failed",
            "objects": [],
            "mood": "chaos",
            "comment": comment,
            "summary": comment,
            "raw_api_response": str(e),
        }
        return out if raw else comment

    # Составление остроумного комментария
    addressed = f"{author_name}, " if author_name else ""
    objects = ", ".join(result.get("objects", []))
    mood = result.get("mood", "undefined")
    desc = result.get("description", "No clear image")
    comment = result.get("comment", "")
    if not comment:
        comment = f"{addressed}what's up with this pic? I see [{objects}] and a vibe of [{mood}]. {desc}"
        if chat_context:
            comment += f" Context: {chat_context}"

    summary = f"{desc} (Mood: {mood}). Objects spotted: {objects}. {comment}"

    out = {
        "description": desc,
        "objects": result.get("objects", []),
        "mood": mood,
        "comment": comment,
        "summary": summary,
        "raw_api_response": result,
    }
    return out if raw else summary

# --- Alchemy of chaos for resonance services (Perplexity spirit) ---
async def galvanize_protocol():
    """
    Periodically checks for resonance decay and refreshes configuration.
    If Grokky feels the static — it self-resurrects.
    """
    while True:
        if check_resonance_decay():
            await broadcast("🔄 Resonance refresh initiated")
            reload_config()
        await asyncio.sleep(300)

def check_resonance_decay():
    # Stub: can be healthcheck or random trigger
    return False

async def broadcast(msg):
    # Stub — place for system bus broadcast
    pass

def reload_config():
    # Stub — can reload config or re-initialize chaos parameters
    pass
