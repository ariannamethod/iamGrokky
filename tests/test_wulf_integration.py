import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from SLNCX import wulf_integration


class FakePiece:
    def __init__(self, text):
        self.type = "output_text"
        self.text = text


class FakeMessage:
    def __init__(self, text):
        self.content = [FakePiece(text)]


class FakeResponse:
    def __init__(self, text):
        self.output = [FakeMessage(text)]


def test_generate_response_handles_response_objects(monkeypatch):
    def fake_get_dynamic(prompt, api_key=None):
        return FakeResponse("ok")

    monkeypatch.setattr(wulf_integration, "get_dynamic_knowledge", fake_get_dynamic)
    assert wulf_integration.generate_response("hi", "grok3") == "ok"


def test_generate_response_uses_wulf_model(monkeypatch):
    def fake_generate(prompt, ckpt_path="out/ckpt.pt", api_key=None):
        return "wolf"

    monkeypatch.setattr(wulf_integration, "run_wulf", fake_generate)
    assert wulf_integration.generate_response("hi", "wulf") == "wolf"


class MemoryEngine:
    """Simple in-memory engine for testing memory integration."""

    def __init__(self):
        self.mem: list[tuple[str, str]] = []

    async def search_memory(self, user_id, query, limit=5):  # pragma: no cover - simple
        return "\n".join(m for m, _ in self.mem)

    async def add_memory(self, user_id, content, role="user"):
        self.mem.append((content, role))


def test_memory_affects_prompt(monkeypatch):
    engine = MemoryEngine()
    prompts: list[str] = []

    def fake_run_wulf(prompt, ckpt_path="out/ckpt.pt", api_key=None):
        prompts.append(prompt)
        return "resp"

    monkeypatch.setattr(wulf_integration, "run_wulf", fake_run_wulf)

    # First call populates memory
    wulf_integration.generate_response(
        "hello", "wulf", user_id="u1", engine=engine
    )
    assert engine.mem == [("hello", "user"), ("resp", "assistant")]

    # Second call should include previous memory in the prompt
    wulf_integration.generate_response(
        "next", "wulf", user_id="u1", engine=engine
    )
    assert any("hello" in p for p in prompts[1:])


def test_retrieved_context_alters_answer(monkeypatch):
    class EngineStub:
        async def search_memory(self, user_id, query, limit=5):
            return "snippet"

        async def add_memory(self, user_id, content, role="user"):
            pass

    def fake_run_wulf(prompt, ckpt_path="out/ckpt.pt", api_key=None):
        return "used snippet" if "snippet" in prompt else "no snippet"

    monkeypatch.setattr(wulf_integration, "VectorGrokkyEngine", lambda: EngineStub())
    monkeypatch.setattr(wulf_integration, "run_wulf", fake_run_wulf)

    monkeypatch.setenv("WULF_SNIPPET_LIMIT", "1")
    with_snippet = wulf_integration.generate_response("hi", "wulf")

    monkeypatch.setenv("WULF_SNIPPET_LIMIT", "0")
    without_snippet = wulf_integration.generate_response("hi", "wulf")

    assert with_snippet == "used snippet"
    assert without_snippet == "no snippet"
