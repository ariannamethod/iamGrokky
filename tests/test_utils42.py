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


def test_markov_uses_apply_pulse(monkeypatch):
    calls = []

    def fake_apply(weights, pulse):
        calls.append((weights, pulse))
        return [1 / len(weights) for _ in weights]

    monkeypatch.setattr(mod, 'apply_pulse', fake_apply)
    mm = mod.MiniMarkov('mars starship chaos', n=1)
    mm.generate(5, start='mars')
    assert calls


def test_paraphrase_uses_dynamic_knowledge(monkeypatch):
    called = {}

    def fake_get(prompt):
        called['prompt'] = prompt
        return 'dynamic'

    monkeypatch.setattr(mod, 'cg', None)
    monkeypatch.setattr(mod, 'get_dynamic_knowledge', fake_get)
    result = mod.paraphrase('hello world')
    assert 'dynamic' in result
    assert called


