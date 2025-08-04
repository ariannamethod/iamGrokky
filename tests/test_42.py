import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
u42 = importlib.import_module('utils.42')


def log_result(name: str, status: str) -> None:
    log_dir = Path('logs/tests')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    entry = {"timestamp": datetime.now().isoformat(), "test": name, "status": status}
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


@pytest.mark.asyncio
async def test_when_pulse(monkeypatch):
    try:
        monkeypatch.setattr(u42, 'get_dynamic_knowledge', lambda prompt: 'ok')
        monkeypatch.setattr(u42, 'paraphrase', lambda text, prefix='': text)
        result = await u42.handle('when')
        assert 0.1 <= result['pulse'] <= 0.9
        log_result('42_when_pulse', 'pass')
    except Exception:
        log_result('42_when_pulse', 'fail')
        raise


@pytest.mark.asyncio
async def test_42_easter_egg(monkeypatch):
    try:
        monkeypatch.setattr(u42, 'get_dynamic_knowledge', lambda prompt: 'base')
        monkeypatch.setattr(u42, 'paraphrase', lambda text, prefix='': text)
        monkeypatch.setattr(u42.random, 'random', lambda: 0.0)
        result = await u42.handle('42')
        assert '42_(number)' in result['response']
        log_result('42_easter_egg', 'pass')
    except Exception:
        log_result('42_easter_egg', 'fail')
        raise


@pytest.mark.asyncio
async def test_whatsnew_parsing(monkeypatch):
    try:
        async def fake_fetch(url: str, timeout: int = 10) -> str:
            return '<html><body><article><h2>Starship Test</h2><time>May 2025</time><a href="/x"></a><p>Launch success</p></article></body></html>'

        monkeypatch.setattr(u42, 'fetch_url', fake_fetch)
        monkeypatch.setattr(u42, 'paraphrase', lambda text, prefix='': text)
        result = await u42.handle('whatsnew')
        assert 'Starship Test' in result['response']
        log_result('42_whatsnew', 'pass')
    except Exception:
        log_result('42_whatsnew', 'fail')
        raise
