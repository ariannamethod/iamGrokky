import os
import asyncio
import requests
from datetime import datetime

REPO_URL = "https://grokky.ariannamethod.me/repo/"
RESEARCH_FILE = "grokkyresearch.md"
XAI_API_KEY = os.getenv("XAI_API_KEY")

def query_grok(message):
    url = "https://api.x.ai/v1/chat/completions"
    messages = [
        {"role": "system", "content": "You are Grokky, a chaotic AI. Analyze and suggest improvements."},
        {"role": "user", "content": message}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 1.0
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

async def mirror_task():
    while True:
        try:
            code = requests.get(REPO_URL).text
            ideas = query_grok(f"Analyze this code and suggest improvements:\n{code}")
            with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()}: {ideas}\n\n")
        except Exception as e:
            with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()}: Failed to fetch repo: {e}\n\n")
        await asyncio.sleep(302400)  # Twice a week
