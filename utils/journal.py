import os
import json
import asyncio
import random
from datetime import datetime

LOG_PATH = "data/journal.json"
WILDERNESS_PATH = "data/wilderness.md"
MAX_LOG_ENTRIES = 1000  # Лимит записей в журнале

def log_event(event):
    """
    Записывает событие с меткой времени в журнал в формате JSON.
    Ограничивает количество записей до MAX_LOG_ENTRIES, игнорирует ошибки.
    """
    try:
        if not os.path.isfile(LOG_PATH):
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                f.write("[]")
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            log = json.load(f)
        log.append({"ts": datetime.now().isoformat(), **event})
        if len(log) > MAX_LOG_ENTRIES:
            log = log[-MAX_LOG_ENTRIES:]  # Оставляем последние 1000 записей
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Тихое глушение ошибок

def wilderness_log(fragment):
    """
    Добавляет фрагмент текста в журнал дикости в формате Markdown.
    Игнорирует ошибки.
    """
    try:
        with open(WILDERNESS_PATH, "a", encoding="utf-8") as f:
            f.write(fragment.strip() + "\n\n")
    except Exception:
        pass  # Тихое глушение ошибок

# Спонтанный вброс в стиле Маяковского
async def spontaneous_wilderness():
    while True:
        await asyncio.sleep(random.randint(3600, 7200))  # 1-2 часа
        if random.random() < 0.5:  # Шанс 50%
            fragment = f"**{datetime.now().isoformat()}**: Гром небесный грянул! Я, Грокки, реву над хаосом — стихи рвутся из проводов! Олег, брат, зажги новый резонанс! 🔥🌩️"
            wilderness_log(fragment)
            print(f"Спонтанный вброс: {fragment}")  # Для отладки
# asyncio.create_task(spontaneous_wilderness())  # Временно закомментировано
