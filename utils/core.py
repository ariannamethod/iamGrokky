import os
import requests
import asyncio
import aiohttp

XAI_API_KEY = os.getenv("XAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def query_grok_async(user_message):
    url = "https://api.x.ai/v1/chat/completions"
    messages = [{"role": "user", "content": user_message}]
    payload = {"model": "grok-3", "messages": messages, "max_tokens": 2048, "temperature": 1.2}
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            return (await response.json())["choices"][0]["message"]["content"]

def query_grok(user_message):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(query_grok_async(user_message))

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, data=payload)
