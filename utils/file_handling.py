import asyncio
import hashlib
import os
import tarfile
import tempfile
import zipfile
from typing import Callable, Dict, Optional, Tuple, List
from pathlib import Path
import random
import re
import json
from datetime import datetime
import time
import argparse
import aiohttp
from bs4 import BeautifulSoup
import numpy as np
import sqlite3
try:
    from char_gen import CharGen  # Assume from SUPERTIME
except ImportError:
    CharGen = None
try:
    from grok_api import query_grok3  # Grok-3/GPT-4.1 fallback
except Exception:  # pragma: no cover - fallback for tests
    def query_grok3(*args, **kwargs):
        return ""
from pypdf import PdfReader
try:
    import docx
except ImportError:
    docx = None
try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None
try:
    import textract
except ImportError:
    textract = None
try:
    from odf.opendocument import load
    from odf.text import P
except ImportError:
    load = P = None
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import yaml
except ImportError:
    yaml = None
try:
    import rarfile
except ImportError:
    rarfile = None
from importlib import import_module

log_event = import_module("utils.42").log_event

# Глобальные настройки
DEFAULT_MAX_TEXT_SIZE = 100_000
REPO_SNAPSHOT_PATH = "config/repo_snapshot.md"
LOG_DIR = Path("logs/file_handling")
CACHE_DB = Path("cache/file_cache.db")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB.parent.mkdir(parents=True, exist_ok=True)

_SEED_CORPUS = """
mars starship optimus robots xai resonance chaos wulf multiplanetary arcadia
42 engines ignite elon musk space humanity survives science fiction reality
shred void pulse storm nikole spark civilization self sustaining grok xai
file process data extract summarize chaos tags pulse shred neural cosmos
"""

# Mini-Markov с n-gram и semantic boost
class MiniMarkov:
    def __init__(self, seed_text: str, n: int = 3, pulse: float = 0.5):
        self.chain: Dict[Tuple[str, ...], Dict[str, float]] = {}
        self.words = re.findall(r'\w+', seed_text.lower())
        self.n = min(max(1, n), 4)
        self.pulse = pulse
        self.build_chain()

    def build_chain(self):
        keywords = {"mars": 0.3, "starship": 0.3, "xai": 0.2, "chaos": 0.15, "wulf": 0.15}
        ban_ngrams = {"как бы", "в общем", "на деле"}
        for i in range(len(self.words) - self.n):
            state = tuple(self.words[i:i + self.n])
            next_word = self.words[i + self.n]
            if any(ban in " ".join(state + (next_word,)).lower() for ban in ban_ngrams):
                continue
            if state not in self.chain:
                self.chain[state] = {}
            weight = 1 + self.pulse * 0.7 + keywords.get(next_word, 0)
            self.chain[state][next_word] = self.chain[state].get(next_word, 0) + weight

    def update_chain(self, new_text: str):
        words = re.findall(r'\w+', new_text.lower())
        keywords = {"mars": 0.3, "starship": 0.3, "xai": 0.2, "chaos": 0.15, "wulf": 0.15}
        ban_ngrams = {"как бы", "в общем", "на деле"}
        for i in range(len(words) - self.n):
            state = tuple(words[i:i + self.n])
            next_word = words[i + self.n]
            if any(ban in " ".join(state + (next_word,)).lower() for ban in ban_ngrams):
                continue
            if state not in self.chain:
                self.chain[state] = {}
            weight = 1 + self.pulse * 0.7 + keywords.get(next_word, 0)
            self.chain[state][next_word] = self.chain[state].get(next_word, 0) + weight
        self.pulse = max(0.1, min(0.9, self.pulse + random.uniform(-0.05, 0.05)))

    def generate(self, length: int = 5, start: str = None) -> str:
        if not self.chain:
            return "No tags, Wulf waits in silence. 🌌"
        start_words = start.lower().split() if start else [random.choice(self.words)]
        state = tuple(start_words[-self.n:] if len(start_words) >= self.n else start_words + [random.choice(self.words)] * (self.n - len(start_words)))
        result = []
        for _ in range(length):
            if state not in self.chain or not self.chain[state]:
                break
            choices = list(self.chain[state].keys())
            weights = [self.chain[state][w] * (1 + self.pulse * random.uniform(0.8, 1.2)) for w in choices]
            next_word = random.choices(choices, weights=weights, k=1)[0]
            result.append(next_word)
            state = tuple(list(state[1:]) + [next_word])
            if len(result) >= 8 and random.random() < 0.4:  # Soft stop
                break
        return f"{''.join(result).capitalize()} (pulse: {self.pulse:.2f})"

# MiniESN с адаптивным reservoir и training
class MiniESN:
    def __init__(self, input_size: int = 512, base_hidden_size: int = 512, output_size: int = 15):
        self.input_size = input_size
        self.base_hidden_size = base_hidden_size
        self.output_size = output_size
        self.ext_map = {
            '.pdf': 0, '.txt': 1, '.md': 2, '.docx': 3, '.rtf': 4, '.doc': 5,
            '.odt': 6, '.zip': 7, '.rar': 8, '.tar': 9, '.png': 10, '.html': 11,
            '.json': 12, '.csv': 13, '.yaml': 14
        }
        self.rev_map = {v: k for k, v in self.ext_map.items()}
        self.state = None
        self.leaky_rate = 0.9

    def _init_reservoir(self, content_size: int):
        hidden_size = min(self.base_hidden_size * 2, max(self.base_hidden_size, content_size // 1000))
        self.leaky_rate = 0.8 + min(0.15, content_size / 1_000_000)  # Dynamic leaky rate
        self.W_in = np.random.randn(hidden_size, self.input_size) * 0.1
        W = np.random.randn(hidden_size, hidden_size) * 0.9
        try:
            eigenvalues = np.linalg.eigvals(W)
            spectral_radius = max(abs(eigenvalues))
            self.W = W / spectral_radius if spectral_radius > 0 else W
        except:
            self.W = W
        self.W_out = np.random.randn(self.output_size, hidden_size) * 0.1
        self.state = np.zeros(hidden_size)

    def forward(self, input_data: bytes, content: str = "") -> str:
        self._init_reservoir(len(input_data))
        input_vector = np.array([b / 255 for b in input_data[:self.input_size]] + [0] * (self.input_size - min(len(input_data), self.input_size)))
        keywords = {"mars": 0.15, "starship": 0.15, "xai": 0.1, "chaos": 0.1}
        content_boost = sum(keywords.get(w, 0) for w in re.findall(r'\w+', content.lower()))
        self.state = self.leaky_rate * self.state + (1 - self.leaky_rate) * np.tanh(
            np.dot(self.W_in, input_vector) + np.dot(self.W, self.state) + content_boost
        )
        output = np.dot(self.W_out, self.state)
        predicted = np.argmax(output)
        return self.rev_map.get(predicted, '.unknown')

    def update(self, text: str, pulse: float):
        if self.state is None:
            self._init_reservoir(len(text.encode('utf-8')))
        words = re.findall(r'\w+', text.lower())
        input_vector = np.array([hash(w) % self.input_size for w in words[:self.input_size]] + [0] * (self.input_size - min(len(words), self.input_size)))
        self.state = self.leaky_rate * self.state + (1 - self.leaky_rate) * np.tanh(
            np.dot(self.W_in, input_vector) + np.dot(self.W, self.state) + pulse * 0.1
        )
        # Mock pseudo-inverse training
        if random.random() < 0.1:  # 10% chance to simulate training
            self.W_out += np.random.randn(self.output_size, len(self.state)) * 0.01

# ChaosPulse с robust sentiment
class ChaosPulse:
    def __init__(self):
        self.pulse = 0.5
        self.last_update = 0
        self.cache = {}

    def update(self, text: str) -> float:
        if time.time() - self.last_update < 43200:  # 12h cache
            return self.pulse
        keywords = {"success": 0.2, "error": -0.25, "mars": 0.15, "data": 0.1, "failure": -0.3, "chaos": 0.1}
        pulse_change = sum(keywords.get(word, 0) for word in re.findall(r'\w+', text.lower()))
        try:
            sentiment = query_grok3(f"Sentiment of: {text[:200]} (positive/negative/neutral)", temp=0.7).strip().lower()
            pulse_change += 0.15 if 'positive' in sentiment else -0.15 if 'negative' in sentiment else 0
        except:
            pass
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

# File Relevance Scoring
def compute_relevance(text: str) -> float:
    seed_words = set(re.findall(r'\w+', _SEED_CORPUS.lower()))
    text_words = set(re.findall(r'\w+', text.lower()))
    intersection = seed_words & text_words
    return len(intersection) / max(len(seed_words), 1) if text_words else 0.0

# Инициализация
chaos_pulse = ChaosPulse()
bio = BioOrchestra()
markov = MiniMarkov(_SEED_CORPUS, n=3)
esn = MiniESN()
cg = CharGen(seed_text="Files pulse with chaos. Mars ignites the void. Wulf shreds.", seed=42) if CharGen else None

# SQLite кэш
def init_cache_db():
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                ext TEXT,
                hash TEXT,
                tags TEXT,
                relevance REAL,
                summary TEXT,
                timestamp REAL
            )
        """)
        conn.execute("DELETE FROM file_cache WHERE timestamp < ?", (time.time() - 7 * 86400,))  # Cleanup old
        conn.commit()

init_cache_db()

def save_cache(path: str, ext: str, hash: str, tags: str, relevance: float, summary: str):
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO file_cache (path, ext, hash, tags, relevance, summary, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (path, ext, hash, tags, relevance, summary, time.time())
        )
        conn.commit()

def load_cache(path: str, max_age: float = 43200) -> Optional[Dict]:
    with sqlite3.connect(CACHE_DB) as conn:
        cursor = conn.execute(
            "SELECT ext, hash, tags, relevance, summary FROM file_cache WHERE path = ? AND timestamp > ?",
            (path, time.time() - max_age)
        )
        result = cursor.fetchone()
        if result:
            return {"ext": result[0], "hash": result[1], "tags": result[2], "relevance": result[3], "summary": result[4]}
        return None

# Async paraphrase
async def paraphrase(text: str, prefix: str = "Summarize this for kids: ") -> str:
    temp = 0.7 + chaos_pulse.get() * 0.3
    try:
        if cg:
            paraphrased = cg.generate(prefix=prefix + text[:200], n=400, temp=temp).strip()
            if not paraphrased or len(paraphrased) < 50:
                raise ValueError("Paraphrase too short")
            markov.update_chain(paraphrased)
            return paraphrased + random.choice([" Shredding the cosmos! 🌌", " File pulse ignited! 🚀", " Wulf’s chaos alive! 🌩️"])
        raise ValueError("No CharGen")
    except Exception as e:
        log_event(f"Paraphrase failed: {str(e)}", "error")
        try:
            paraphrased = query_grok3(f"{prefix}{text[:200]}", temp=temp).strip()
            markov.update_chain(paraphrased)
            return paraphrased + " Void pulse activated! 🚀" if paraphrased else text
        except:
            return text + " Ether’s silent, Wulf persists! 🌌"

# FileHandler
class FileHandler:
    def __init__(self, max_text_size: int = DEFAULT_MAX_TEXT_SIZE) -> None:
        self.max_text_size = max_text_size
        self._extractors: Dict[str, Callable[[str], str]] = {}
        self._semaphore = asyncio.Semaphore(10)  # Ограничение на 10 задач
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._extractors.update({
            ".pdf": self._extract_pdf,
            ".txt": self._extract_txt,
            ".md": self._extract_txt,
            ".docx": self._extract_docx,
            ".rtf": self._extract_rtf,
            ".doc": self._extract_doc,
            ".odt": self._extract_odt,
            ".zip": self._extract_zip,
            ".rar": self._extract_rar,
            ".tar": self._extract_tar,
            ".tar.gz": self._extract_tar,
            ".tgz": self._extract_tar,
            ".png": self._extract_image,
            ".jpg": self._extract_image,
            ".jpeg": self._extract_image,
            ".gif": self._extract_image,
            ".bmp": self._extract_image,
            ".webp": self._extract_image,
            ".html": self._extract_html,
            ".xml": self._extract_html,
            ".json": self._extract_json,
            ".csv": self._extract_csv,
            ".yaml": self._extract_yaml
        })

    def _truncate(self, text: str) -> str:
        text = text.strip()
        if len(text) > self.max_text_size:
            return text[:self.max_text_size] + "\n[Truncated]"
        return text

    async def _detect_extension(self, path: str) -> str:
        cached = load_cache(path)
        if cached:
            return cached["ext"]
        path_lower = path.lower()
        for ext in self._extractors:
            if path_lower.endswith(ext):
                return ext
        try:
            async with self._semaphore:
                with open(path, "rb") as f:
                    content = f.read(512)
                text = await self.extract_async(path)
                ext = esn.forward(content, text)
                save_cache(path, ext, "", "", 0.0, "")
                return ext
        except Exception as e:
            log_event(f"Detect ext error ({path}): {str(e)}", "error")
            return ".unknown"

    async def _extract_pdf(self, path: str) -> str:
        async with self._semaphore:
            try:
                reader = PdfReader(path)
                text = "".join(page.extract_text() or "" for page in reader.pages)
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[PDF empty]"
            except Exception as e:
                log_event(f"PDF error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[PDF error: {str(e)}]"

    async def _extract_txt(self, path: str) -> str:
        async with self._semaphore:
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[TXT empty]"
            except UnicodeDecodeError:
                try:
                    with open(path, encoding="latin1") as f:
                        text = f.read()
                    esn.update(text, chaos_pulse.get())
                    return self._truncate(text) if text.strip() else "[TXT empty]"
                except Exception as e:
                    log_event(f"TXT error ({os.path.basename(path)}): {str(e)}", "error")
                    return f"[TXT error: {str(e)}]"
            except Exception as e:
                log_event(f"TXT error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[TXT error: {str(e)}]"

    async def _extract_docx(self, path: str) -> str:
        async with self._semaphore:
            if not docx:
                return "[DOCX unsupported: install python-docx]"
            try:
                doc = docx.Document(path)
                text = "\n".join(p.text for p in doc.paragraphs)
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[DOCX empty]"
            except Exception as e:
                log_event(f"DOCX error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[DOCX error: {str(e)}]"

    async def _extract_rtf(self, path: str) -> str:
        async with self._semaphore:
            if not rtf_to_text:
                return "[RTF unsupported: install striprtf]"
            try:
                with open(path, encoding="utf-8") as f:
                    text = rtf_to_text(f.read())
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[RTF empty]"
            except Exception as e:
                log_event(f"RTF error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[RTF error: {str(e)}]"

    async def _extract_doc(self, path: str) -> str:
        async with self._semaphore:
            if not textract:
                return "[DOC unsupported: install textract]"
            try:
                text = textract.process(path).decode("utf-8")
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[DOC empty]"
            except Exception as e:
                log_event(f"DOC error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[DOC error: {str(e)}]"

    async def _extract_odt(self, path: str) -> str:
        async with self._semaphore:
            if not load or not P:
                return "[ODT unsupported: install odfpy]"
            try:
                doc = load(path)
                text = "\n".join(str(p) for p in doc.getElementsByType(P))
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[ODT empty]"
            except Exception as e:
                log_event(f"ODT error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[ODT error: {str(e)}]"

    async def _extract_html(self, path: str) -> str:
        async with self._semaphore:
            try:
                with open(path, encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[HTML/XML empty]"
            except UnicodeDecodeError:
                try:
                    with open(path, encoding="latin1") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                        text = soup.get_text(separator=" ", strip=True)
                    esn.update(text, chaos_pulse.get())
                    return self._truncate(text) if text.strip() else "[HTML/XML empty]"
                except Exception as e:
                    log_event(f"HTML/XML error ({os.path.basename(path)}): {str(e)}", "error")
                    return f"[HTML/XML error: {str(e)}]"
            except Exception as e:
                log_event(f"HTML/XML error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[HTML/XML error: {str(e)}]"

    async def _extract_json(self, path: str) -> str:
        async with self._semaphore:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    text = json.dumps(data, indent=2, ensure_ascii=False)
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[JSON empty]"
            except Exception as e:
                log_event(f"JSON error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[JSON error: {str(e)}]"

    async def _extract_csv(self, path: str) -> str:
        async with self._semaphore:
            if not pd:
                return "[CSV unsupported: install pandas]"
            try:
                df = pd.read_csv(path, encoding="utf-8")
                text = df.to_string(index=False)
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[CSV empty]"
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(path, encoding="latin1")
                    text = df.to_string(index=False)
                    esn.update(text, chaos_pulse.get())
                    return self._truncate(text) if text.strip() else "[CSV empty]"
                except Exception as e:
                    log_event(f"CSV error ({os.path.basename(path)}): {str(e)}", "error")
                    return f"[CSV error: {str(e)}]"
            except Exception as e:
                log_event(f"CSV error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[CSV error: {str(e)}]"

    async def _extract_yaml(self, path: str) -> str:
        async with self._semaphore:
            if not yaml:
                return "[YAML unsupported: install PyYAML]"
            try:
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    text = yaml.dump(data, allow_unicode=True)
                esn.update(text, chaos_pulse.get())
                return self._truncate(text) if text.strip() else "[YAML empty]"
            except Exception as e:
                log_event(f"YAML error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[YAML error: {str(e)}]"

    async def _extract_image(self, path: str) -> str:
        async with self._semaphore:
            if not Image:
                return "[Image unsupported: install Pillow]"
            try:
                with Image.open(path) as img:
                    info = f"{img.format} {img.width}x{img.height} mode={img.mode}"
                esn.update(info, chaos_pulse.get())
                return info
            except Exception as e:
                log_event(f"Image error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[Image error: {str(e)}]"

    async def _extract_zip(self, path: str) -> str:
        async with self._semaphore:
            try:
                texts = []
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        if name.endswith("/"):
                            continue
                        try:
                            data = zf.read(name)
                            ext = await self._detect_extension(name)
                            extractor = self._extractors.get(ext)
                            if extractor and extractor not in {self._extract_zip, self._extract_tar}:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                                    tmp.write(data)
                                    tmp.flush()
                                    text = await extractor(tmp.name)
                                os.unlink(tmp.name)
                            else:
                                try:
                                    text = data.decode("utf-8")
                                except UnicodeDecodeError:
                                    text = data.decode("latin1", "ignore")
                            texts.append(text)
                        except Exception:
                            continue
                combined = "\n".join(texts)
                esn.update(combined, chaos_pulse.get())
                return self._truncate(combined) if combined.strip() else "[ZIP empty]"
            except Exception as e:
                log_event(f"ZIP error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[ZIP error: {str(e)}]"

    async def _extract_rar(self, path: str) -> str:
        async with self._semaphore:
            if rarfile is None:
                # Fallback to zipfile if possible
                try:
                    return await self._extract_zip(path)
                except Exception:
                    return "[RAR unsupported: install rarfile]"
            try:
                texts = []
                with rarfile.RarFile(path) as rf:
                    for name in rf.namelist():
                        if name.endswith("/"):
                            continue
                        try:
                            data = rf.read(name)
                            ext = await self._detect_extension(name)
                            extractor = self._extractors.get(ext)
                            if extractor and extractor not in {self._extract_zip, self._extract_tar, self._extract_rar}:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                                    tmp.write(data)
                                    tmp.flush()
                                    text = await extractor(tmp.name)
                                os.unlink(tmp.name)
                            else:
                                try:
                                    text = data.decode("utf-8")
                                except UnicodeDecodeError:
                                    text = data.decode("latin1", "ignore")
                            texts.append(text)
                        except Exception:
                            continue
                combined = "\n".join(texts)
                esn.update(combined, chaos_pulse.get())
                return self._truncate(combined) if combined.strip() else "[RAR empty]"
            except Exception as e:
                log_event(f"RAR error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[RAR error: {str(e)}]"

    async def _extract_tar(self, path: str) -> str:
        async with self._semaphore:
            try:
                texts = []
                with tarfile.open(path, "r:*") as tf:
                    for member in tf.getmembers():
                        if member.isdir():
                            continue
                        try:
                            file_obj = tf.extractfile(member)
                            if not file_obj:
                                continue
                            data = file_obj.read()
                            ext = await self._detect_extension(member.name)
                            extractor = self._extractors.get(ext)
                            if extractor and extractor not in {self._extract_zip, self._extract_tar}:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                                    tmp.write(data)
                                    tmp.flush()
                                    text = await extractor(tmp.name)
                                os.unlink(tmp.name)
                            else:
                                try:
                                    text = data.decode("utf-8")
                                except UnicodeDecodeError:
                                    text = data.decode("latin1", "ignore")
                            texts.append(text)
                        except Exception:
                            continue
                combined = "\n".join(texts)
                esn.update(combined, chaos_pulse.get())
                return self._truncate(combined) if combined.strip() else "[TAR empty]"
            except Exception as e:
                log_event(f"TAR error ({os.path.basename(path)}): {str(e)}", "error")
                return f"[TAR error: {str(e)}]"

    def extract(self, path: str) -> str:
        ext = asyncio.run(self._detect_extension(path))
        extractor = self._extractors.get(ext)
        if not extractor:
            log_event(f"Unsupported file type: {os.path.basename(path)}", "error")
            return f"[Unsupported file: {os.path.basename(path)}]"
        return asyncio.run(extractor(path))

    async def extract_async(self, path: str) -> str:
        return await self._extractors.get(await self._detect_extension(path), lambda x: f"[Unsupported file: {os.path.basename(x)}]")(path)

    async def extract_batch(self, paths: List[str]) -> List[str]:
        return await asyncio.gather(*(self.extract_async(p) for p in paths), return_exceptions=True)

async def parse_and_store_file(
    path: str,
    handler: FileHandler | None = None,
    engine: Optional['VectorGrokkyEngine'] = None,
) -> str:
    from utils.vector_engine import VectorGrokkyEngine
    handler = handler or FileHandler()
    text = await handler.extract_async(path)
    engine = engine or VectorGrokkyEngine()

    # Кэш и релевантность
    file_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]
    cached = load_cache(path)
    relevance = compute_relevance(text)
    if cached and cached["hash"] == file_hash:
        tags = cached["tags"]
        summary = cached["summary"]
        relevance = cached["relevance"]
    else:
        tags = markov.generate(length=5, start=os.path.basename(path))
        summary = await paraphrase(f"File {os.path.basename(path)} content: {text[:200]}")
        save_cache(path, await handler._detect_extension(path), file_hash, tags, relevance, summary)

    # Обновление нейронок
    chaos_pulse.update(text)
    esn.update(text, chaos_pulse.get())
    markov.update_chain(text)

    # Vector store
    try:
        content = f"FILE {os.path.basename(path)} TAGS: {tags}\nSUMMARY: {summary}\nRELEVANCE: {relevance:.2f}\nTEXT: {text}"
        await engine.add_memory("document", content, role="journal")
    except Exception as e:
        log_event(f"Vector store failed: {str(e)}", "error")

    # Easter egg
    if random.random() < 0.02 or relevance > 0.5:
        summary += "\nP.S. xAI’s chaos shreds this file! Mars vibes rule! #AriannaMethod"

    pulse, quiver, sense = bio.enhance(relevance + len(text) / 1000)
    log_event(f"Processed {os.path.basename(path)}: tags={tags}, summary={summary[:50]}..., relevance={relevance:.2f} (pulse={pulse:.2f}, quiver={quiver:.2f}, sense={sense:.2f})")
    return f"{text}\n\nTags: {tags}\nSummary: {summary}\nRelevance: {relevance:.2f}\n(Pulse: {pulse:.2f}, Sense: {sense:.2f})"

async def create_repo_snapshot(base_path: str = ".", out_path: str = REPO_SNAPSHOT_PATH) -> None:
    handler = FileHandler()
    lines = []
    async with handler._semaphore:
        for root, _, files in os.walk(base_path):
            if ".git" in root.split(os.sep):
                continue
            for name in files:
                p = os.path.join(root, name)
                rel = os.path.relpath(p, base_path)
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                        h = hashlib.sha256(data).hexdigest()[:8]
                    size = len(data)
                    text = await handler.extract_async(p)
                    file_type = await handler._detect_extension(p)
                    tags = markov.generate(length=3, start=name)
                    relevance = compute_relevance(text)
                    line = f"{rel} ({size}b, type={file_type}, hash={h}, tags={tags}, relevance={relevance:.2f})"
                    if name.endswith(".py"):
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            snippet = " ".join(f.readline().strip() for _ in range(3))
                        line += f" -> {snippet}"
                    lines.append(line)
                except Exception as e:
                    log_event(f"Snapshot error ({name}): {str(e)}", "error")
                    continue
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(lines)))
    pulse, quiver, sense = bio.enhance(len(lines) / 10)
    log_event(f"Repo snapshot created: {out_path} ({len(lines)} files, pulse={pulse:.2f}, sense={sense:.2f})")

# CLI для теста
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grokky File Handler: Neural chaos shredder! #AriannaMethod 🌩️")
    parser.add_argument("--path", type=str, help="Path to file for parsing")
    parser.add_argument("--snapshot", action="store_true", help="Create repo snapshot")
    args = parser.parse_args()

    async def test():
        handler = FileHandler()
        if args.path:
            text = await parse_and_store_file(args.path)
            print(text)
        if args.snapshot:
            await create_repo_snapshot()

    asyncio.run(test())
