import requests
import os
from utils.vision import vision_handler

XAI_API_KEY = os.getenv("XAI_API_KEY")

def impress_handler(prompt, chat_context=None, author_name=None, raw=True):
    """
    Generates an image via xAI by prompt, then immediately analyzes it with vision_handler.
    Returns raw JSON: prompt, image_url, vision_result, grokkys_comment.
    """
    gen_endpoint = "https://api.x.ai/v1/images/generate"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    # Prompt for image generation
    system_prompt = (
        "You are Grokky. Generate a wild, stormy, or surreal image based on the prompt. "
        "Always make the output raw and expressive. "
        "Return the image URL and a short phrase describing your intent."
    )
    data = {
        "prompt": prompt,
        "model": "grok-2-image-1212",  # adjust if needed
        "system": system_prompt,
    }
    if chat_context:
        data["chat_context"] = chat_context

    resp = requests.post(gen_endpoint, headers=headers, json=data)
    resp.raise_for_status()
    image_result = resp.json()
    image_url = image_result.get("image_url")
    if not image_url:
        raise Exception("No image_url in response from xAI!")

    # Call vision_handler for self-roast/analysis
    vision_result = vision_handler(image_url, chat_context=chat_context, author_name=author_name, raw=True)

    # Compose Grokky's comment
    grokky_comment = (
        f"{author_name+', ' if author_name else ''}wanted an image? Here you go! "
        f"But seriously, what did you expect? {vision_result.get('comment')}"
    )

    out = {
        "prompt": prompt,
        "image_url": image_url,
        "vision_result": vision_result,
        "grokkys_comment": grokky_comment,
        "raw_api_response": image_result,
    }
    return out if raw else f"Image: {image_url}\n{grokky_comment}"
