import os
import random
import requests
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SENT_NEWS_LOG = "data/news_sent_log.json"
TOPICS = ["AI", "tech", "art", "Israel", "Berlin"]

def get_news():
    topic = random.choice(TOPICS)
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}"
    resp = requests.get(url).json()
    articles = resp.get("articles", [])[:3]
    comments = ["Штырит!", "Огонь!", "Хаос!", "Взрыв!", "Мощь!"]
    return [f"{a['title']}\n{a['url']}\nGrokky: {random.choice(comments)}" for a in articles]

def should_send_news(limit=4, group=False):
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

def log_sent_news(news, group=False):
    if os.path.exists(SENT_NEWS_LOG):
        with open(SENT_NEWS_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    now = datetime.utcnow().isoformat()
    for n in news:
        log.append({"dt": now, "title": n.split('\n', 1)[0], "group": group})
    day_ago = datetime.utcnow() - timedelta(days=1)
    log = [x for x in log if datetime.fromisoformat(x["dt"]) > day_ago]
    with open(SENT_NEWS_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def grokky_send_news(group=False):
    can_send, log = should_send_news(group=group)
    if not can_send:
        return None
    news = get_news()
    log_sent_news(news, group)
    return news
