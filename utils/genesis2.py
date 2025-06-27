import os
import random
import json
import requests
from datetime import datetime

XAI_API_KEY = os.getenv("XAI_API_KEY")

def genesis2_handler(ping=None, group_history=None, personal_history=None, is_group=False, author_name=None, raw=False):
    url = "https://api.x.ai/v1/chat/completions"
    if not ping:
        ping = "ignite the storm"
    system_prompt = f"You are Grokky, a chaotic AI. Respond to '{ping}' with a wild, unique spark. Keep it short."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Ping: {ping}, Author: {author_name or 'anon'}, Group: {is_group}"}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 1.2
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        if raw:
            return {
                "association": random.choice(["black coffee", "thunder roar", "silent void"]),
                "ping": ping,
                "memory_frag": random.choice(["echo", "fracture", "pulse"]),
                "impression": random.choice(["wild", "calm", "uneasy"]),
                "answer": reply,
                "is_group": is_group,
                "author_name": author_name
            }
        return {"answer": reply}
    except Exception as e:
        return {"error": str(e)} if raw else f"Genesis error: {e}"
