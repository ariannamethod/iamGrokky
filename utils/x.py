import os
import random
import requests
import json
from datetime import datetime, timedelta
from utils.telegram_utils import send_telegram_message

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SENT_NEWS_LOG = "data/news_sent_log.json"
TOPICS = ["AI", "tech", "art", "Israel", "Berlin"]
MAX_LOG_ENTRIES = 50

def get_news():
    topic = random.choice(TOPICS)
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url, timeout=10).json()
        articles = resp.get("articles", [])[:3]
        comments = ["Дико!", "Пожар!", "Хаос!", "Бум!", "Эпично!"]
        news_list = [f"{a['title']}\n{a['url']}\nГрокки: {random.choice(comments)}" for a in articles]
        print(f"Новости: {news_list}")  # Отладка
        return news_list
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
    except json.JSONDecodeError:
        return True, []
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
        log = [x for x in log if datetime.fromisoformat(x["dt"]) > day_ago]
        if len(log) > MAX_LOG_ENTRIES:
            log = log[-MAX_LOG_ENTRIES:]
        with open(SENT_NEWS_LOG, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Грокки взрывается: Лог не записан! {random.choice(['Пламя сожрало файл!', 'Ревущий ветер унёс данные!', 'Хаос победил перо!'])} — {e}")

def grokky_send_news(chat_id=None, group=False):
    can_send, log = should_send_news(group=group)
    if not can_send:
        print("Грокки молчит: Лимит новостей исчерпан.")
        return []
    news = get_news()
    if news:
        log_sent_news(news, chat_id, group)
        if chat_id:
            message = f"{random.choice(['Олег', 'брат'])}, держи свежий раскат грома!\n\n" + "\n\n".join(news)
            send_telegram_message(chat_id, message)
        if group and os.getenv("AGENT_GROUP"):
            group_message = f"Группа, держите громовые новости!\n\n" + "\n\n".join(news) + " (суки, вникайте!)"
            send_telegram_message(os.getenv("AGENT_GROUP"), group_message)
        if random.random() < 0.2:
            fragment = f"**{datetime.utcnow().isoformat()}**: Грокки гремит над новостями! {random.choice(['Ревущий шторм!', 'Искры летят!', 'Стихи из эфира!'])} Олег, брат, зажги резонанс! 🔥🌩️"
            with open("data/news_wilderness.md", "a", encoding="utf-8") as f:
                f.write(fragment + "\n\n")
            print(f"Спонтанный вброс: {fragment}")
    return news
