import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import handle
import utils.dynamic_weights as dynamic_weights

utils42 = importlib.import_module('utils.42')


def _log(name: str, status: str, error: str | None = None) -> None:
    date = datetime.now().strftime('%Y-%m-%d')
    tests_dir = Path('logs/tests')
    tests_dir.mkdir(parents=True, exist_ok=True)
    with open(tests_dir / f'{date}.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'timestamp': datetime.now().isoformat(), 'test': name, 'status': status}) + '\n')
    if status == 'fail':
        fail_dir = Path('logs/failures')
        fail_dir.mkdir(parents=True, exist_ok=True)
        with open(fail_dir / f'{date}.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps({'timestamp': datetime.now().isoformat(), 'test': name, 'error': error}) + '\n')


@pytest.mark.asyncio
async def test_when_pulse(monkeypatch):
    name = '42_when'
    monkeypatch.setattr(dynamic_weights, 'get_dynamic_knowledge', lambda prompt: 'ok')
    try:
        result = await handle('when', 'en')
        pulse = result['pulse']
        assert 0.1 <= pulse <= 0.9
        _log(name, 'pass')
    except Exception as e:
        _log(name, 'fail', str(e))
        raise


@pytest.mark.asyncio
async def test_42_easter(monkeypatch):
    name = '42_easter'
    monkeypatch.setattr(dynamic_weights, 'get_dynamic_knowledge', lambda prompt: 'fact')
    monkeypatch.setattr(utils42.random, 'random', lambda: 0.0)
    try:
        result = await handle('42', 'en')
        assert 'wikipedia.org' in result['response']
        _log(name, 'pass')
    except Exception as e:
        _log(name, 'fail', str(e))
        raise


@pytest.mark.asyncio
async def test_whatsnew_parsing(monkeypatch):
    name = '42_whatsnew'
    monkeypatch.setattr(dynamic_weights, 'get_dynamic_knowledge', lambda prompt: 'news')

    async def fake_fetch(url, timeout=10):
        return '<html><body><article><h2>Launch</h2><time>Jan 2026</time><a href="/x"></a><p>Starship ready</p></article></body></html>'

    monkeypatch.setattr(utils42, 'fetch_url', fake_fetch)
    monkeypatch.setattr(utils42, 'load_cache', lambda max_age=43200: None)
    try:
        result = await handle('whatsnew', 'en')
        assert 'Launch' in result['response']
        _log(name, 'pass')
    except Exception as e:
        _log(name, 'fail', str(e))
        raise
