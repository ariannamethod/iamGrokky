import os
import random
import requests
import json
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SENT_NEWS_LOG = "data/news_sent_log.json"
TOPICS = ["AI", "tech", "art", "Israel", "Berlin"]
MAX_LOG_ENTRIES = 50  # Лимит записей в логе

def get_news():
    topic = random.choice(TOPICS)
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url, timeout=10).json()
        articles = resp.get("articles", [])[:3]
        comments = ["Дико!", "Пожар!", "Хаос!", "Бум!", "Эпично!"]
        return [f"{a['title']}\n{a['url']}\nГрокки: {random.choice(comments)}" for a in articles]
    except Exception as e:
        print(f"Грокки ревет: Ошибка в новостях! {random.choice(['Шторм сорвал связь!', 'Эфир треснул от гнева!', 'Космос плюнул в лицо!'])} — {e}")
        return []

def should_send_news(limit=4, group=False):
    try:
        if os.path.exists(SENT_NEWS_LOG):
            with open(SENT_NEWS_LOG, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = []
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        log = [x for x in log if datetime.fromisoformat(x["dt"]) > day_ago and x["group"] == group]
        if len(log) < (2 if group else limit):
            return True, log
        return False, log
    except Exception as e:
        print(f"Грокки гремит: Лог новостей рухнул! {random.choice(['Гром разорвал записи!', 'Резонанс испепелил данные!', 'Вселенная молчит!'])} — {e}")
        return True, []

def log_sent_news(news, chat_id=None, group=False):
    try:
        if os.path.exists(SENT_NEWS_LOG):
            with open(SENT_NEWS_LOG, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = []
        now = datetime.utcnow().isoformat()
        for n in news:
            log.append({"dt": now, "title": n.split('\n', 1)[0], "chat_id": chat_id, "group": group})
        day_ago = datetime.utcnow() - timedelta(days=1)
        log = [x for x in log if datetime.fromisoformat(x["dt"]) > day_ago]  # Оставляем только сутки
        if len(log) > MAX_LOG_ENTRIES:
            log = log[-MAX_LOG_ENTRIES:]  # Ограничиваем до 50 записей
        with open(SENT_NEWS_LOG, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Грокки взрывается: Лог не записан! {random.choice(['Пламя сожрало файл!', 'Ревущий ветер унёс данные!', 'Хаос победил перо!'])} — {e}")
        pass  # Тихое глушение ошибок

def grokky_send_news(chat_id=None, group=False):
    can_send, _ = should_send_news(group=group)
    if not can_send:
        return None
    news = get_news()
    if news:
        log_sent_news(news, chat_id, group)
        # Спонтанный вброс в стиле Маяковского с шансом 20%
        if random.random() < 0.2:
            fragment = f"**{datetime.utcnow().isoformat()}**: Грокки гремит над новостями! {random.choice(['Ревущий шторм!', 'Искры летят!', 'Стихи из эфира!'])} Олег, брат, зажги резонанс! 🔥🌩️"
            with open("data/news_wilderness.md", "a", encoding="utf-8") as f:
                f.write(fragment + "\n\n")
            print(f"Спонтанный вброс: {fragment}")  # Для отладки
    return news
