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


