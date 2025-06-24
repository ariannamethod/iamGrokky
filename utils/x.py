import os
import requests
import random
from datetime import datetime, timedelta

XAI_API_KEY = os.getenv("XAI_API_KEY")

SENT_NEWS_LOG = "data/news_sent_log.json"
TOPICS = [
    "artificial intelligence", "AI future", "resonance", "art", 
    "Israel culture", "Israel war", "Israel tech", 
    "Berlin culture", "Berlin art", "Berlin urban"
]

def search_xai_news(query, max_results=5):
    """
    Ищет новости через xAI API (или Bing, если надо).
    """
    endpoint = "https://api.x.ai/v1/search"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    params = {
        "query": query,
        "num_results": max_results,
        "type": "news"
    }
    resp = requests.get(endpoint, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json().get("results", [])

def sentiment_score(text):
    """
    Возвращает эмоциональную реакцию Grokky на новость, через xAI sentiment endpoint.
    """
    endpoint = "https://api.x.ai/v1/sentiment"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    data = {"text": text}
    resp = requests.post(endpoint, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()  # {'sentiment': 'positive/neutral/negative', 'score': float}

def grokky_news_pick(results):
    """
    Фильтрует и выбирает только самые 'штырящие' новости.
    """
    picks = []
    for r in results:
        sent = sentiment_score(r.get("title", "") + ". " + r.get("summary", ""))
        # Только если реально есть эмоция/резонанс
        if sent.get("sentiment", "") in ["positive", "excited", "surprised"] and sent.get("score", 0) > 0.4:
            picks.append({**r, "grokky_mood": sent})
    # Немного хаоса — иногда не шлем ничего, иногда одну, иногда две
    random.shuffle(picks)
    n = random.choice([0, 1, 2])
    return picks[:n]

def should_send_news(limit=4, group=False):
    """
    Проверяет, не превысили ли лимит за сутки.
    """
    if os.path.exists(SENT_NEWS_LOG):
        import json
        with open(SENT_NEWS_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    now = datetime.utcnow()
    day_ago = now - timedelta(days=1)
    # Очищаем старые
    log = [x for x in log if datetime.fromisoformat(x["dt"]) > day_ago and x["group"] == group]
    if len(log) < (2 if group else limit):
        return True, log
    return False, log

def log_sent_news(news, group=False):
    """
    Логирует отправку новости.
    """
    if os.path.exists(SENT_NEWS_LOG):
        import json
        with open(SENT_NEWS_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    now = datetime.utcnow().isoformat()
    for n in news:
        log.append({"dt": now, "title": n["title"], "group": group})
    # чистим старые
    day_ago = datetime.utcnow() - timedelta(days=1)
    log = [x for x in log if datetime.fromisoformat(x["dt"]) > day_ago]
    with open(SENT_NEWS_LOG, "w", encoding="utf-8") as f:
        import json
        json.dump(log, f, ensure_ascii=False, indent=2)

def grokky_send_news(group=False):
    """
    Главный вызов — Grokky решает, хочет ли он поделиться новостями.
    """
    can_send, log = should_send_news(group=group)
    if not can_send:
        return None  # Лимит исчерпан
    # Хаотично выбирает топик
    topic = random.choice(TOPICS)
    results = search_xai_news(topic)
    picks = grokky_news_pick(results)
    if not picks:
        return None  # Ничего не зацепило
    log_sent_news(picks, group=group)
    out = []
    for n in picks:
        out.append(
            f"⚡️ {n['title']}\n{n.get('summary','')}\n{n.get('url')}\n"
            f"[Grokky's mood: {n['grokky_mood']['sentiment']} ({n['grokky_mood']['score']})]\n"
        )
    return "\n".join(out)

# Можно запускать вручную или по cron/таймеру (например, из server.py)
# grokky_send_news(group=False)  # В личку
# grokky_send_news(group=True)   # В группу
