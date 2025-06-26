import os
import random
import requests
from datetime import datetime, timedelta
from server import send_telegram_message

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SENT_NEWS_LOG = "data/news_sent_log.json"
TOPICS = ["AI", "tech", "art", "Israel", "Berlin"]
OLEG_CHAT_ID = os.getenv("CHAT_ID")
GROUP_CHAT_ID = os.getenv("AGENT_GROUP")

def get_news():
    topic = random.choice(TOPICS)
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}&language=en"
    response = requests.get(url).json()
    articles = response.get("articles", [])[:3]
    return [
        f"{a['title']}\n{a['url']}\nGrokky: {random.choice(['Штырит!', 'Огонь!', 'Хаос!'])} — резонанс: {analyze_sentiment(a['description'])}"
        for a in articles if a.get("description")
    ]

def analyze_sentiment(text):
    url = "https://api.x.ai/v1/sentiment"
    headers = {"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}", "Content-Type": "application/json"}
    data = {"text": text[:500]}  # Ограничиваем длину
    try:
        resp = requests.post(url, json=data, headers=headers, timeout=10)
        resp.raise_for_status()
        sentiment = resp.json().get("sentiment", "neutral")
        return f"{sentiment} ({resp.json().get('score', 0):.2f})"
    except Exception:
        return "chaos (unreadable)"

def should_send_news(limit=4, group=False):
    import json
    if os.path.exists(SENT_NEWS_LOG):
        with open(SENT_NEWS_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    now = datetime.utcnow()
    day_ago = now - timedelta(days=1)
    log = [x for x in log if datetime.fromisoformat(x["dt"]) > day_ago and x["group"] == group]
    return len(log) < (2 if group else limit), log

def log_sent_news(news, group=False):
    import json
    if os.path.exists(SENT_NEWS_LOG):
        with open(SENT_NEWS_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    now = datetime.utcnow().isoformat()
    for n in news:
        log.append({"dt": now, "title": n.split('\n', 1)[0], "group": group, "sentiment": analyze_sentiment(n.split('\n', 2)[2])})
    day_ago = datetime.utcnow() - timedelta(days=1)
    log = [x for x in log if datetime.fromisoformat(x["dt"]) > day_ago]
    with open(SENT_NEWS_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def grokky_send_news(group=False):
    can_send, log = should_send_news(group=group)
    if not can_send:
        return None
    news = get_news()
    if news:
        chat_id = GROUP_CHAT_ID if group else OLEG_CHAT_ID
        send_telegram_message(chat_id, "\n\n".join(news))
        log_sent_news(news, group=group)
    return news
