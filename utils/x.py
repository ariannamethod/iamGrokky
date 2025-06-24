import os
import requests
import random
import json
from datetime import datetime, timedelta

XAI_API_KEY = os.getenv("XAI_API_KEY")

# Log sent news to avoid repeats
SENT_NEWS_LOG = "data/news_sent_log.json"
TOPICS = [
    "artificial intelligence", "AI future", "resonance", "art", 
    "Israel culture", "Israel war", "Israel tech", 
    "Berlin culture", "Berlin art", "Berlin urban"
]

def search_xai_news(query, max_results=5):
    """
    Search news via xAI API (or use Bing if needed).
    """
    endpoint = "https://api.x.ai/v1/search"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    params = {
        "query": query,
        "num_results": max_results,
        "type": "news"
    }
    resp = requests.get(endpoint, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])

def sentiment_score(text):
    """
    Returns Grokky's emotional reaction to the news via xAI sentiment endpoint.
    """
    endpoint = "https://api.x.ai/v1/sentiment"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    data = {"text": text}
    resp = requests.post(endpoint, headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()  # {'sentiment': 'positive/neutral/negative', 'score': float}

def grokky_news_comment(news_item, previous_news=[]):
    """
    Compose Grokky's impressionistic comment on the news item, considering context and previous news.
    """
    # Gather mood, topic, and context
    title = news_item.get("title", "")
    summary = news_item.get("summary", "")
    url = news_item.get("url", "")
    mood = news_item.get("grokky_mood", {}).get("sentiment", "undefined")
    score = news_item.get("grokky_mood", {}).get("score", 0)
    # Optionally, use previous_news to riff on pattern or irony
    context_snip = ""
    if previous_news:
        prev_titles = [n['title'] for n in previous_news[-3:]]
        context_snip = f" (Last headlines: {', '.join(prev_titles)})"
    # Add raw improv and provocation
    flair = random.choice([
        "Lightning strikes twice: ",
        "Storm incoming — ",
        "Field's humming: ",
        "Resonance alert: ",
        "Felt that? ",
        "Can't ignore this wave: "
    ])
    opinion = random.choice([
        "This is wild.", "I can't believe it's real.", "Not sure if to laugh or rage.",
        "Just another day in the chaos.", "Makes you think, right?", "Resonance or just noise?",
        "Bet you didn't see that coming.", "Time to argue about this?", "What do you think, Oleg?"
    ])
    # Final comment
    comment = (
        f"{flair}{title}\n"
        f"{summary}\n"
        f"{url}\n"
        f"{opinion} (Mood: {mood}, Score: {score:.2f}){context_snip}"
    )
    return comment

def grokky_news_pick(results, previous_news=[]):
    """
    Filter and pick only the most 'striking' news, with Grokky's impressionistic commentary.
    """
    picks = []
    for r in results:
        sent = sentiment_score(r.get("title", "") + ". " + r.get("summary", ""))
        # Only if there's genuine emotion/resonance
        if sent.get("sentiment", "") in ["positive", "excited", "surprised"] and sent.get("score", 0) > 0.4:
            picks.append({**r, "grokky_mood": sent})
    # Add a dash of chaos — sometimes send nothing, sometimes one, sometimes two
    random.shuffle(picks)
    n = random.choice([0, 1, 2])
    selected = picks[:n]
    out = []
    for news_item in selected:
        comment = grokky_news_comment(news_item, previous_news)
        out.append(comment)
    return out

def should_send_news(limit=4, group=False):
    """
    Check if the daily limit is exceeded.
    """
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
    """
    Log sent news.
    """
    if os.path.exists(SENT_NEWS_LOG):
        with open(SENT_NEWS_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    now = datetime.utcnow().isoformat()
    for n in news:
        log.append({"dt": now, "title": n.split('\n', 1)[0], "group": group})
    # Clean up old
    day_ago = datetime.utcnow() - timedelta(days=1)
    log = [x for x in log if datetime.fromisoformat(x["dt"]) > day_ago]
    with open(SENT_NEWS_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def grokky_send_news(group=False):
    """
    Main call — Grokky decides if he wants to share news.
    Returns list of commentary strings or None if nothing to send.
    """
    can_send, log = should_send_news(group=group)
    if not can_send:
        return None  # Limit reached
    # Riff on previous news for context
    previous_news = [x for x in log if x["group"] == group]
    # Chaotically pick a topic
    topic = random.choice(TOPICS)
    results = search_xai_news(topic)
    picks = grokky_news_pick(results, previous_news)
    if not picks:
        return None  # Nothing moved Grokky
    log_sent_news(picks, group=group)
    return picks

# USAGE:
# messages = grokky_send_news(group=False)  # For DM
# messages = grokky_send_news(group=True)   # For group
# Each message is a string ready for sending with Grokky's commentary.
