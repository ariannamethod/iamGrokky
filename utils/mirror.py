import os
import asyncio
import requests
import hashlib
from datetime import datetime

REPO_URL = "https://grokky.ariannamethod.me/repo/"
RESEARCH_FILE = "grokkyresearch.md"
XAI_API_KEY = os.getenv("XAI_API_KEY")
LAST_HASH_FILE = "data/last_mirror_hash.txt"

def file_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

async def mirror_task(query_grok_func):
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
                user_lang = detect_language(code)  # Используем переданную функцию
                language_hint = f"Отвечай на языке кода: {user_lang.upper()}. Предлагай улучшения хаотично."
                ideas = query_grok_func(f"Анализируй этот код и предложи улучшения:\n{code}\n{language_hint}")
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

# Функция detect_language перенесена в grok_utils.py, но оставим заглушку для совместимости
def detect_language(text):
    cyrillic = re.compile('[а-яА-ЯёЁ]')
    return 'ru' if cyrillic.search(text or "") else 'en'
