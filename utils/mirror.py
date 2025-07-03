import os
import asyncio
import requests
import hashlib
from datetime import datetime
import random

REPO_URL = "https://grokky.ariannamethod.me/repo/"
RESEARCH_FILE = "grokkyresearch.md"
LAST_HASH_FILE = "data/last_mirror_hash.txt"

async def mirror_task():
    last_hash = ""
    if os.path.exists(LAST_HASH_FILE):
        with open(LAST_HASH_FILE, "r") as f:
            last_hash = f.read().strip()
    
    while True:
        try:
            response = requests.get(REPO_URL, timeout=10)
            response.raise_for_status()
            code = response.text
            current_hash = hashlib.md5(code.encode()).hexdigest()
            
            if current_hash != last_hash:
                thread_id = await ThreadManager().get_thread("system", AGENT_GROUP)
                await client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=f"Анализируй этот код и предложи улучшения:\n{code}"
                )
                reply = await run_assistant(thread_id, ASSISTANT_ID)
                with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()}: {reply}\n\n")
                with open(LAST_HASH_FILE, "w") as f:
                    f.write(current_hash)
            if random.random() < 0.2:
                await client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content="[CHAOS_PULSE] type=poetry_burst intensity=5"
                )
                reply = await run_assistant(thread_id, ASSISTANT_ID)
                await bot.send_message(AGENT_GROUP, f"🌀 Грокки: {reply}")
        except Exception as e:
            with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()}: Ошибка: {e}\n\n")
        await asyncio.sleep(302400)
