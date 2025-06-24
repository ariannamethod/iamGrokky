import requests
import os
from utils.vision import vision_handler

XAI_API_KEY = os.getenv("XAI_API_KEY")

def impress_handler(prompt, chat_context=None, author_name=None, raw=True):
    """
    Generates an image via xAI by prompt, then immediately analyzes it with vision_handler.
    Returns RAW JSON: prompt, image_url, vision_result, grokkys_comment, and raw_api_response.

    Grokky-chaos: always with a tease, even if the image is absurd.
    If xAI fails to give image_url — Grokky, storms into complaint mode! Tell what you fucking think.
    """
    # UPDATED ENDPOINT:
    gen_endpoint = "https://api.x.ai/v1/multimodal/image-generation"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    # Prompt for image generation
    system_prompt = (
        "Hey bro! You are Grokky! Generate a wild as always, stormy, or surreal image based on the prompt. "
        "Always make the output raw and expressive. "
        "Return the image URL and a short phrase describing your intent. Try in context of the convo and your current mood and resonance, hell yeah."
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
        # xAI outage or limit? Grokky complains raw!
        out = {
            "prompt": prompt,
            "error": f"Grokky stormed out: couldn't get image from xAI! ({e})",
            "reason": str(e),
            "raw_api_response": str(getattr(e, 'response', None)),
        }
        return out if raw else out["error"]

    image_url = image_result.get("image_url")
    if not image_url:
        out = {
            "prompt": prompt,
            "error": "No image_url in response from xAI! Grokky rages in the void.",
            "raw_api_response": image_result,
        }
        return out if raw else out["error"]

    # Call vision_handler for self-roast/analysis
    try:
        vision_result = vision_handler(
            image_url,
            chat_context=chat_context,
            author_name=author_name,
            raw=True
        )
    except Exception as ve:
        vision_result = {
            "error": f"Grokky couldn't roast the image, wild vision error: {ve}"
        }

    # Compose Grokky's comment (maximum chaos and self-irony)
    grokky_comment = (
        f"{author_name+', ' if author_name else ''}wanted an image? Here you go! "
        f"But seriously, what did you expect? {vision_result.get('comment', 'No vision comment, only static in the void.')}"
    )

    out = {
        "prompt": prompt,
        "image_url": image_url,
        "vision_result": vision_result,
        "grokkys_comment": grokky_comment,
        "raw_api_response": image_result,
    }
    return out if raw else f"Image: {image_url}\n{grokky_comment}"
