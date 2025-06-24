import requests
import os
XAI_API_KEY = os.getenv("XAI_API_KEY")

def vision_handler(image_bytes_or_url, chat_context=None, author_name=None, raw=True):
    """
    Analyzes an image using xAI Grok-2-Image/object detection.
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
    """
    endpoint = "https://api.x.ai/v1/images/understand"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
    }
    # Accept both URL and bytes
    if isinstance(image_bytes_or_url, str) and image_bytes_or_url.startswith("http"):
        files = None
        data = {
            "image_url": image_bytes_or_url,
            "model": "grok-2-image-1212",  # see your console.x.ai link
        }
    else:
        files = {"image_file": image_bytes_or_url}
        data = {
            "model": "grok-2-image-1212",
        }

    # Optionally add instructions to xAI prompt for humor/role/context
    system_prompt = (
        "You are Grokky, a stormy, irreverent AI. "
        "Describe the image, spot main objects, sense the mood, "
        "and always connect your reply to the chat context. "
        "If in a group, address the user by name and throw in a joke or tease. "
        "If someone just sent an image, ask why, or riff on the vibe. "
        "Reply in raw JSON for function calling."
    )
    data["prompt"] = system_prompt
    if chat_context:
        data["chat_context"] = chat_context

    resp = requests.post(endpoint, headers=headers, data=data, files=files)
    resp.raise_for_status()
    result = resp.json()

    # Compose witty comment
    addressed = f"{author_name}, " if author_name else ""
    objects = ", ".join(result.get("objects", []))
    mood = result.get("mood", "undefined")
    desc = result.get("description", "")
    comment = result.get("comment", "")
    if not comment:
        # fallback comment logic
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
