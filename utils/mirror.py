import os
import asyncio
import requests
import hashlib
from datetime import datetime
import random  # Добавлен импорт
from utils.grok_utils import detect_language

REPO_URL = "https://grokky.ariannamethod.me/repo/"
RESEARCH_FILE = "grokkyresearch.md"
XAI_API_KEY = os.getenv("XAI_API_KEY")
LAST_HASH_FILE = "data/last_mirror_hash.txt"

def file_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

async def mirror_task(query_grok_func):
    if not query_grok_func:
        print("Грокки рычит: Нет функции анализа кода!")
        return
    last_hash = ""
    if os.path.exists(LAST_HASH_FILE):
        with open(LAST_HASH_FILE, "r") as f:
            last_hash = f.read().strip()
    
    while True:
        try:
            response = requests.get(REPO_URL, timeout=10)
            response.raise_for_status()
            code = response.text
            current_hash = file_hash(code)
            
            if current_hash != last_hash:
                user_lang = detect_language(code)
                language_hint = f"Отвечай на языке кода: {user_lang.upper()}. Предлагай улучшения хаотично."
                ideas = query_grok_func(f"Анализируй этот код и предложи улучшения:\n{code}\n{language_hint}")
                with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()}: {ideas}\n\n")
                with open(LAST_HASH_FILE, "w") as f:
                    f.write(current_hash)
            else:
                with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()}: Код не изменился, пропускаю анализ.\n\n")
            # Спонтанный вброс с шансом 20%
            if random.random() < 0.2:
                fragment = f"**{datetime.now().isoformat()}**: Грокки ревет над кодом! {random.choice(['Шторм вырвал строки!', 'Искры летят из репозитория!', 'Резонанс жжёт улучшения!'])} Олег, брат, зажги хаос! 🔥🌩️"
                print(f"Спонтанный вброс: {fragment}")  # Для отладки
        except Exception as e:
            with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                error_msg = f"{datetime.now()}: Грокки взрывается: Репозиторий не достал! {random.choice(['Ревущий ветер сорвал связь!', 'Хаос испепелил код!', 'Эфир треснул от ошибки!'])} — {e}\n\n"
                f.write(error_msg)
        await asyncio.sleep(302400)  # Дважды в неделю
