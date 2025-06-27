import os
import asyncio
import requests
import hashlib
from datetime import datetime
from server import query_grok, detect_language  # Импорт из server.py

REPO_URL = "https://grokky.ariannamethod.me/repo/"
RESEARCH_FILE = "grokkyresearch.md"
XAI_API_KEY = os.getenv("XAI_API_KEY")
LAST_HASH_FILE = "data/last_mirror_hash.txt"

def file_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

async def mirror_task():
    last_hash = ""
    if os.path.exists(LAST_HASH_FILE):
        with open(LAST_HASH_FILE, "r") as f:
            last_hash = f.read().strip()
    
    while True:
        try:
            # Получаем код и хэш
            response = requests.get(REPO_URL, timeout=10)
            response.raise_for_status()
            code = response.text
            current_hash = file_hash(code)
            
            if current_hash != last_hash:  # Анализируем только если код изменился
                user_lang = detect_language(code)  # Определяем язык кода
                language_hint = f"Отвечай на языке кода: {user_lang.upper()}. Предлагай улучшения хаотично."
                ideas = query_grok(f"Анализируй этот код и предложи улучшения:\n{code}\n{language_hint}")
                with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()}: {ideas}\n\n")
                with open(LAST_HASH_FILE, "w") as f:
                    f.write(current_hash)
            else:
                with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()}: Код не изменился, пропускаю анализ.\n\n")
        except Exception as e:
            with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()}: Не удалось получить репозиторий: {e}\n\n")
        await asyncio.sleep(302400)  # Дважды в неделю

# Предложение: спонтанные проверки раз в 12-24 часа с шансом 20%
# async def spontaneous_mirror():
#     while True:
#         await asyncio.sleep(random.randint(43200, 86400))  # 12-24 часа
#         if random.random() < 0.2:  # Шанс 20%
#             await mirror_task()  # Вызываем зеркало спонтанно
# asyncio.create_task(spontaneous_mirror())
