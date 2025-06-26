import os
import requests
import asyncio
from datetime import datetime
from utils.core import query_grok  # Изменён импорт на utils/core
import aiohttp  # Для асинхронных запросов, если понадобится

REPO_URL = "https://grokky.ariannamethod.me/repo/"
RESEARCH_FILE = "grokkyresearch.md"

def analyze_code():
    response = requests.get(REPO_URL)
    code = response.text if response.status_code == 200 else "Ошибка загрузки кода с {REPO_URL}"
    prompt = f"Проанализируй код: {code}. Предложи сложные и спонтанные улучшения с квантовыми элементами."
    ideas = query_grok(prompt)
    with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}: {ideas}\n\n")

async def run_mirror():
    while True:
        analyze_code()
        await asyncio.sleep(604800 // 2)  # Дважды в неделю (примерно 3.5 дня)

if __name__ == "__main__":
    asyncio.run(run_mirror())  # Для тестирования
