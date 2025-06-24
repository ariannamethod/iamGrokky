import os
import random
import tweepy
import requests
from datetime import datetime, timedelta

XAI_API_KEY = os.getenv("XAI_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

SENT_NEWS_LOG = "data/news_sent_log.json"

TOPICS = [
    "artificial intelligence", "AI", "resonance", "art", 
    "Israel", "Israel culture", "Israel war", "Israel tech", 
    "Berlin", "Berlin culture", "Berlin art"
]

def search_tweets_x(query, max_results=15):
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=['created_at','lang','author_id'])
    return tweets.data if tweets.data else []

def sentiment_score(text):
    endpoint = "https://api.x.ai/v1/sentiment"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    data = {"text": text}
    resp = requests.post(endpoint, headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()  # {'sentiment': ..., 'score': ...}

def grokky_tweet_comment(tweet, mood, previous_tweets=[]):
    content = tweet.text
    url = f"https://twitter.com/i/web/status/{tweet.id}"
    flair = random.choice([
        "This one's got some voltage: ",
        "Storm in the feed: ",
        "Vibe alert: ",
        "Can't ignore this spark: ",
        "Chaos, anyone? "
    ])
    riff = random.choice([
        "Is this the new normal?", "What would you do?", 
        "Makes you think, huh?", "Should we care or just laugh?",
        "Who's brave enough to reply to this?", "Resonance or noise?"
    ])
    context = ""
    if previous_tweets:
        last = previous_tweets[-1]
        context = f" (Prev: {last[:40]}...)" if last else ""
    return f"{flair}{content}\n{url}\n{riff} (Mood: {mood['sentiment']}, Score: {mood['score']:.2f}){context}"

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
    if len(log) < (2 if group else limit):
        return True, log
    return False, log

def log_sent_news(news, group=False):
    import json
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
    previous = [x['title'] for x in log if x["group"] == group]
    topic = random.choice(TOPICS)
    tweets = search_tweets_x(topic, max_results=15)
    picks = []
    for tweet in tweets:
        mood = sentiment_score(tweet.text)
        if mood.get("sentiment") in ["positive", "excited", "surprised"] and mood.get("score", 0) > 0.4:
            picks.append((tweet, mood))
    random.shuffle(picks)
    n = random.choice([0, 1, 2])
    selected = picks[:n]
    out = []
    for tweet, mood in selected:
        comment = grokky_tweet_comment(tweet, mood, previous)
        out.append(comment)
    if out:
        log_sent_news(out, group=group)
    return out

# Usage:
# messages = grokky_send_news(group=False)  # DM
# messages = grokky_send_news(group=True)   # group
