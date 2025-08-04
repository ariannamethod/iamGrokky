import importlib
import pytest

mod = importlib.import_module('utils.42')


@pytest.mark.asyncio
async def test_when_pulse_range(monkeypatch):
    monkeypatch.setattr(mod, 'paraphrase', lambda text, prefix='': text)
    result = await mod.handle('when')
    assert 0.1 <= result['pulse'] <= 0.9


@pytest.mark.asyncio
async def test_42_easter_egg(monkeypatch):
    monkeypatch.setattr(mod, 'paraphrase', lambda text, prefix='': text)
    monkeypatch.setattr(mod.random, 'random', lambda: 0.0)
    result = await mod.handle('42')
    assert 'wikipedia.org/wiki/42' in result['response']


@pytest.mark.asyncio
async def test_whatsnew_parsing(monkeypatch):
    async def fake_fetch(url, timeout=10):
        return '<article><h2>Test Update</h2><time>2025</time><a href="/test"></a><p>Starship chaos</p></article>'
    monkeypatch.setattr(mod, 'fetch_url', fake_fetch)
    monkeypatch.setattr(mod, 'paraphrase', lambda text, prefix='': text)
    result = await mod.handle('whatsnew')
    assert 'Test Update' in result['response']
    assert 'Starship chaos' in result['response']
