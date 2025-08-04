"""Playful mini neural network for Grokky's special commands.

The module powers the `/when`, `/mars`, `/42` and `/whatsnew` commands.
It mixes a tiny Markov chain with a couple of bio-inspired helpers to
produce whimsical text and fetch small pieces of news.  The goal of this
patch is not only to integrate the module into the main server but also
to make it a little more robust and lively.
"""

import argparse
import asyncio
import json
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin

import aiohttp

# Optional dependencies -----------------------------------------------------
# BeautifulSoup is used for HTML parsing.  It is listed in requirements but
# we guard the import so that tests can run even if the package is missing.
try:  # pragma: no cover - optional dependency
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

# CharGen is optional.  For Grok-3 access we use the dynamic weights helper
# which falls back to GPT when Grok-3 is unavailable.
try:  # pragma: no cover - optional dependency
    from char_gen import CharGen  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CharGen = None  # type: ignore

from utils.dynamic_weights import get_dynamic_knowledge, apply_pulse
import httpx


def _translate(text: str, lang: str) -> str:
    """Translate ``text`` to ``lang`` using Google Translate.

    ``lang`` expects a two-letter code (e.g. ``"ru"``). English returns the
    text unchanged. Any errors fall back to the original text.
    """

    lang = (lang or "en").split("-")[0].lower()
    if lang in ("en", ""):
        return text
    try:
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": lang,
            "dt": "t",
            "q": text,
        }
        resp = httpx.get(
            "https://translate.googleapis.com/translate_a/single",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        translated = "".join(part[0] for part in data[0])
        return translated or text
    except Exception:
        return text


def _append_links(base: str, text: str) -> str:
    """Ensure URLs from ``base`` are present in ``text``."""

    links = re.findall(r"https?://\S+", base)
    missing = [link for link in links if link not in text]
    if missing:
        text += "\n" + "\n".join(missing)
    return text

LOG_DIR = Path("logs/42")
CACHE_DB = Path("cache/cache.db")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
FROM_CLI_SEED: Optional[int] = None

_SEED_CORPUS = """
mars starship optimus robots xai resonance chaos wulf multiplanetary arcadia
42 engines ignite elon musk space humanity survives science fiction reality
shred void pulse storm nikole spark civilization self sustaining grok xai
"""

_42_TEMPLATES = [
    "There are {num} ways to align with Mars’ 42 pulse.",
    "{num} beats echo, but 42 engines roar louder!",
    "Life’s riddle: {num} or 42? Wulf decides!",
    "{num} sparks fly, yet 42 rules the void!",
    "42 hums the stars, {num}’s my thunderstrike!",
]

WHEN_BASE = """Hi! Elon Musk shared his Mars plan at Starbase, May 2025, to make humanity multiplanetary. Key steps:
- 2026: Uncrewed Starships with Optimus robots test landings and build bases. No humans yet—machines scout safety.
- 2030-2035: First humans land! Orbital refueling hauls tools, food, solar panels. Science: Mars’ CO2 splits into oxygen with electricity.
- 2045-2055: City for 1M, self-sustaining with food, water, energy. Mars had rivers eons ago, maybe life!

Mars unlocks science like fossils or space hacks. Links:
- SpaceX: https://www.spacex.com/updates
- Video: https://www.youtube.com/watch?v=y9Rv-Q20zRE
- Timeline: https://www.space.com/elon-musk-spacex-colonize-mars-2050"""

MARS_BASE = """Yo! Why Mars? It’s humanity’s backup and science goldmine. Earth’s cool, but asteroids or climate could wreck it. Mars is Plan B:
- **Survival**: Earth fails? Mars keeps going. Science says Mars had water, rivers, maybe life billions of years ago.
- **Resources**: Ice for water, iron for tools, sun for power. Gravity’s 38% of Earth’s—jump high, but workout!
- **Science & Adventure**: Hydroponics (plants in water), ray shields, homes. Could spark tech or cures. First on another planet? Epic!
- **How**: Optimus robots build, then humans with SpaceX ships. NASA helps.

Why go? Survive, explore, make humanity eternal. Links:
- SpaceX: https://www.spacex.com/humanspaceflight/mars
- NASA: https://mars.nasa.gov
- Overview: https://www.space.com/mars-colonization"""

# Mini-Markov с n-gram и адаптацией
class MiniMarkov:
    def __init__(self, seed_text: str, pulse: float = 0.5, n: int = 2):
        self.chain: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.words = re.findall(r'\w+', seed_text.lower())
        self.pulse = pulse
        self.n = min(max(1, n), 3)
        self.build_chain()

    def build_chain(self):
        for i in range(len(self.words) - self.n):
            state = tuple(self.words[i:i + self.n])
            next_word = self.words[i + self.n]
            self.chain[state][next_word] += 1 + self.pulse * 0.7

    def update_chain(self, new_text: str):
        words = re.findall(r'\w+', new_text.lower())
        for i in range(len(words) - self.n):
            state = tuple(words[i:i + self.n])
            next_word = words[i + self.n]
            self.chain[state][next_word] += 1 + self.pulse * 0.7
        self.pulse = max(0.1, min(0.9, self.pulse + random.uniform(-0.05, 0.05)))

    def next_counts(self, state: Tuple[str, ...]) -> Counter:
        if not state or len(state) != self.n or state not in self.chain:
            states = [s for s in self.chain if len(s) == self.n]
            state = random.choice(states) if states else tuple(random.choice(self.words) for _ in range(self.n))
        return Counter(self.chain[state])

    def generate(self, length: int = 10, start: str = None) -> str:
        if not self.chain:
            return "No chain, Wulf waits."
        start_words = start.lower().split() if start else [random.choice(self.words)]
        state = tuple(start_words[-self.n:] if len(start_words) >= self.n else start_words + [random.choice(self.words)] * (self.n - len(start_words)))
        result = list(state)
        for _ in range(length - self.n):
            if state not in self.chain or not self.chain[state]:
                break
            choices = list(self.chain[state].keys())
            raw = [self.chain[state][w] for w in choices]
            weights = apply_pulse(raw, self.pulse)
            next_word = random.choices(choices, weights=weights, k=1)[0]
            result.append(next_word)
            state = tuple(result[-self.n:])
        return ' '.join(result).capitalize() + random.choice([' Shred the void! 🌌', ' 42 ignites! 🚀', ' Mars pulse alive! 🌩️'])

# ChaosPulse для динамики
class ChaosPulse:
    def __init__(self):
        self.pulse = 0.5
        self.last_update = 0
        self.cache = {}

    def update(self, news_text: str) -> float:
        if time.time() - self.last_update < 43200:  # 12h cache
            return self.pulse
        keywords = {"success": 0.2, "launch": 0.15, "mars": 0.15, "progress": 0.1, "delay": -0.15, "failure": -0.25}
        pulse_change = sum(keywords.get(word, 0) for word in re.findall(r'\w+', news_text.lower()))
        self.pulse = max(0.1, min(0.9, self.pulse + pulse_change + random.uniform(-0.05, 0.05)))
        self.last_update = time.time()
        self.cache['last_pulse'] = self.pulse
        return self.pulse

    def get(self) -> float:
        return self.pulse

# BioOrchestra с SixthSense
class BioOrchestra:
    def __init__(self):
        self.blood = BloodFlux(iron=0.6)
        self.skin = SkinSheath(sensitivity=0.55)
        self.sense = SixthSense()

    def enhance(self, intensity: float) -> Tuple[float, float, float]:
        pulse = self.blood.circulate(intensity)
        quiver = self.skin.ripple(intensity * 0.1)
        sense = self.sense.foresee(intensity)
        return pulse, quiver, sense

class BloodFlux:
    def __init__(self, iron: float):
        self.iron = iron
        self.pulse = 0.0

    def circulate(self, agitation: float) -> float:
        self.pulse = max(0.0, min(self.pulse * 0.9 + agitation * self.iron + random.uniform(-0.03, 0.03), 1.0))
        return self.pulse

class SkinSheath:
    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity
        self.quiver = 0.0

    def ripple(self, impact: float) -> float:
        self.quiver = max(0.0, min(impact * self.sensitivity + random.uniform(-0.05, 0.05), 1.0))
        return self.quiver

class SixthSense:
    def __init__(self):
        self.chaos = 0.0

    def foresee(self, intensity: float) -> float:
        self.chaos = max(0.0, min(self.chaos * 0.95 + intensity * 0.2 + random.uniform(-0.02, 0.02), 1.0))
        return self.chaos

# Инициализация
chaos_pulse = ChaosPulse()
bio = BioOrchestra()
markov = MiniMarkov(_SEED_CORPUS, chaos_pulse.get(), n=3)
cg = CharGen(seed_text="Mars is humanity’s storm. 42 ignites it.", seed=42) if CharGen else None

# Логирование
def log_event(msg: str, log_type: str = "info") -> None:
    log_dir = Path("logs/42") if log_type == "info" else Path("failures")
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(), "type": log_type, "message": msg}
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# SQLite кэш для /whatsnew
def init_cache_db():
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS whatsnew (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT,
                timestamp REAL,
                pulse REAL
            )
        """)

def save_cache(summary: str, pulse: float):
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("INSERT INTO whatsnew (summary, timestamp, pulse) VALUES (?, ?, ?)",
                     (summary, time.time(), pulse))
        conn.commit()

def load_cache(max_age: float = 43200) -> Optional[Dict]:
    """Load cached news summary if it is still fresh."""
    with sqlite3.connect(CACHE_DB) as conn:
        cursor = conn.execute(
            "SELECT summary, pulse, timestamp FROM whatsnew ORDER BY timestamp DESC LIMIT 1"
        )
        result = cursor.fetchone()
        if result and (time.time() - result[2]) < max_age:
            return {"summary": result[0], "pulse": result[1]}
        return None

# Асинхронный fetch
async def fetch_url(url: str, timeout: int = 10) -> Optional[str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout) as resp:
                resp.raise_for_status()
                return await resp.text()
    except Exception as e:
        log_event(f"Fetch {url} failed: {str(e)}", "error")
        return None

# Парфраз
def paraphrase(text: str, prefix: str = "Retell simply: ") -> str:
    temp = 0.8 + chaos_pulse.get() * 0.2
    try:
        if cg:
            paraphrased = cg.generate(prefix=prefix + text[:200], n=400, temp=temp).strip()
            if not paraphrased or len(paraphrased) < 50:
                raise ValueError("Paraphrase too short")
            markov.update_chain(paraphrased)
            return paraphrased + random.choice([" Shred the cosmos! 🌌", " Thunderstrike alive! 🚀"])
        else:
            paraphrased = get_dynamic_knowledge(f"{prefix}{text[:200]}").strip()
            markov.update_chain(paraphrased)
            return paraphrased + " Void pulse kicks in! 🌩️" if paraphrased else text
    except Exception as e:
        log_event(f"Paraphrase failed: {str(e)}", "error")
        return text + " Wulf holds the line! 🌌"

# Команды
def when(lang: str = "en") -> str:
    base = markov.generate(length=12, start="starship")
    paraphrased = paraphrase(base, "Answer like a decisive fixer: ")
    paraphrased = _append_links(WHEN_BASE, paraphrased)
    pulse, quiver, sense = bio.enhance(len(paraphrased) / 100)
    log_event(
        f"Served /when: {paraphrased[:50]}... (pulse={pulse:.2f}, quiver={quiver:.2f}, sense={sense:.2f})"
    )
    if random.random() < 0.01:  # 1% xAI easter egg
        paraphrased += "\nP.S. Grok 3 vibes with Mars! #xAI"
    result = paraphrased
    return _translate(result, lang)

async def mars(lang: str = "en") -> str:
    sources = [
        ("SpaceX", "https://www.spacex.com/humanspaceflight/mars"),
        ("NASA", "https://mars.nasa.gov"),
        ("Space.com", "https://www.space.com/mars-colonization"),
    ]
    pieces = []
    links = []
    for name, url in sources:
        links.append(url)
        html = await fetch_url(url)
        summary = "See link"
        if html and BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(html, "html.parser")  # type: ignore[misc]
                text = " ".join(p.get_text(strip=True) for p in soup.find_all("p")[:5])
                summary = paraphrase(text, f"Summarize {name} Mars update: ")
            except Exception as e:
                log_event(f"Parse {url} failed: {str(e)}", "error")
        pieces.append(f"{name}: {summary}")
        chaos_pulse.update(summary)
        markov.update_chain(summary)
    retell = "Mars roundup:\n" + "\n".join(pieces)
    retell += "\nLinks:\n" + "\n".join(links)
    pulse, quiver, sense = bio.enhance(len(retell) / 100)
    log_event(
        f"Served /mars: {retell[:50]}... (pulse={pulse:.2f}, quiver={quiver:.2f}, sense={sense:.2f})"
    )
    result = retell
    return _translate(result, lang)

def forty_two(lang: str = "en") -> str:
    base_text = markov.generate(length=10, start="42")
    paraphrased = paraphrase(base_text, "Fun 42 fact with Wulf edge: ")
    if random.random() < 1 / 42:
        paraphrased += (
            "\nP.S. 42 engines - Musk’s Arcadia prophecy! "
            "https://en.wikipedia.org/wiki/42_(number)"
        )
    pulse, quiver, sense = bio.enhance(len(paraphrased) / 50)
    log_event(
        f"Served /42: {paraphrased[:50]}... (pulse={pulse:.2f}, quiver={quiver:.2f}, sense={sense:.2f})"
    )
    result = paraphrased
    return _translate(result, lang)

async def whatsnew(lang: str = "en") -> str:
    init_cache_db()
    cached = load_cache()
    if cached:
        chaos_pulse.pulse = cached["pulse"]
        return _translate(cached["summary"], lang)

    urls = ["https://www.spacex.com/updates", "https://x.ai/blog"]
    articles: list[str] = []
    links: list[str] = []
    for url in urls:
        html = await fetch_url(url)
        if not html or BeautifulSoup is None:
            continue
        try:
            soup = BeautifulSoup(html, "html.parser")  # type: ignore[misc]
            updates = soup.find_all("article", limit=3)
            for article in updates:
                title = article.find("h2").text.strip() if article.find("h2") else "Update"
                date = article.find("time").text.strip() if article.find("time") else datetime.now().strftime("%B %Y")
                link = urljoin(url, article.find("a")["href"]) if article.find("a") else url
                summary = article.find("p").text.strip()[:150] + "..." if article.find("p") else "See link"
                paraphrased = paraphrase(summary, "Retell this news for kids: ")
                articles.append(f"{title} ({date}): {paraphrased}")
                links.append(link)
            break
        except Exception as e:
            log_event(f"Parse {url} failed: {str(e)}", "error")
            continue

    if articles:
        retell = "Latest news:\n" + "\n".join(f"- {a}" for a in articles)
        if links:
            retell += "\nLinks:\n" + "\n".join(links)
        chaos_pulse.update(retell)
        markov.update_chain(retell)
        pulse, quiver, sense = bio.enhance(len(retell) / 100)
        save_cache(retell, chaos_pulse.get())
        log_event(
            f"Served /whatsnew: {retell[:50]}... (pulse={pulse:.2f}, quiver={quiver:.2f}, sense={sense:.2f})"
        )
        return _translate(retell, lang)

    try:
        tweet = get_dynamic_knowledge(
            "Latest x.com/SpaceX Mars/Starship tweet, summarize in 100 chars",
        ).strip()
        if any(kw in tweet.lower() for kw in ["mars", "starship"]):
            paraphrased = paraphrase(tweet, "Retell this tweet simply: ")
            retell = f"Latest SpaceX tweet: {paraphrased}"
            links = ["https://x.com/SpaceX"]
            chaos_pulse.update(retell)
            markov.update_chain(retell)
        else:
            retell = "No Mars/Starship tweets. Last: Starship Flight 6 (July 2025)."
            links = ["https://www.spacex.com/updates/starship-flight-6"]
    except Exception as e:
        log_event(f"X search failed: {str(e)}", "error")
        retell = (
            "News fetch failed, Wulf stands ready! Last: Starship Flight 6 (July 2025)."
        )
        links = ["https://www.spacex.com/updates/starship-flight-6"]

    if links:
        retell += "\nLinks:\n" + "\n".join(links)
    pulse, quiver, sense = bio.enhance(len(retell) / 100)
    save_cache(retell, chaos_pulse.get())
    log_event(
        f"Served /whatsnew: {retell[:50]}... (pulse={pulse:.2f}, quiver={quiver:.2f}, sense={sense:.2f})"
    )
    return _translate(retell, lang)

# Главный обработчик
async def handle(cmd: str, lang: str = "en") -> Dict[str, str]:
    """Main asynchronous dispatcher for the 42 utility.

    Parameters
    ----------
    cmd:
        Command name without the leading slash.  Supported values are
        ``"when"``, ``"mars"``, ``"42"`` and ``"whatsnew"``.

    Returns
    -------
    Dict[str, str]
        Mapping with the textual ``response`` and the current ``pulse``.
    """

    cmd = cmd.strip().lower()
    try:
        if cmd == "when":
            return {"response": when(lang), "pulse": chaos_pulse.get()}
        if cmd == "mars":
            return {"response": await mars(lang), "pulse": chaos_pulse.get()}
        if cmd == "42":
            return {"response": forty_two(lang), "pulse": chaos_pulse.get()}
        if cmd == "whatsnew":
            return {"response": await whatsnew(lang), "pulse": chaos_pulse.get()}

        log_event(f"Unknown cmd: {cmd}", "error")
        return {
            "response": "Unknown command! Try /when, /mars, /42, /whatsnew.",
            "pulse": chaos_pulse.get(),
        }
    except Exception as e:  # pragma: no cover - best effort
        log_event(f"Handle {cmd} failed: {str(e)}", "error")
        return {"response": f"Error: {str(e)}. Wulf persists!", "pulse": chaos_pulse.get()}

# CLI для теста
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grokky 42: Mars chaos & 42 fire! #AriannaMethod")
    parser.add_argument("--cmd", choices=["when", "mars", "42", "whatsnew"], help="Command to run")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed")
    args = parser.parse_args()
    if args.cmd:
        if args.seed:
            random.seed(args.seed)
            FROM_CLI_SEED = args.seed
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(handle(args.cmd))
        print(result["response"])
