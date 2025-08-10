import os
import zipfile
from datetime import datetime
from pathlib import Path
import importlib
import io
import sys
import types

import pytest
from PIL import Image
from reportlab.pdfgen import canvas

import utils.context_neural_processor as cnp
from utils.context_neural_processor import (
    FileHandler,
    compute_relevance,
    create_repo_snapshot,
)


@pytest.mark.asyncio
async def test_pdf_processing(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(10, 800, "Mars mission 2026")
    c.save()
    handler = FileHandler()
    text = await handler.extract_async(str(pdf_path))
    assert "Mars mission 2026" in text


@pytest.mark.asyncio
async def test_txt_keywords(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Starship chaos", encoding="utf-8")
    handler = FileHandler()
    text = await handler.extract_async(str(txt_path))
    assert text.strip() == "Starship chaos"
    assert compute_relevance(text) > 0


@pytest.mark.asyncio
async def test_unsupported_file(tmp_path):
    bad_path = tmp_path / "bad.xyz"
    bad_path.write_text("data", encoding="utf-8")
    handler = FileHandler()
    res = await handler.extract_async(str(bad_path))
    assert "unsupported" in res.lower()
    fail_log = Path("logs/failures") / f"{datetime.utcnow().date()}.jsonl"
    assert fail_log.exists()


@pytest.mark.asyncio
async def test_batch_processing(tmp_path):
    pdf_path = tmp_path / "a.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(10, 800, "Mars")
    c.save()
    txt_path = tmp_path / "b.txt"
    txt_path.write_text("Mars", encoding="utf-8")
    img_path = tmp_path / "c.png"
    Image.new("RGB", (2, 2), color="red").save(img_path)
    handler = FileHandler()
    results = await handler.extract_batch([str(pdf_path), str(txt_path), str(img_path)])
    assert len(results) == 3


@pytest.mark.asyncio
async def test_repo_snapshot(tmp_path):
    (tmp_path / "one.txt").write_text("mars starship", encoding="utf-8")
    (tmp_path / "two.txt").write_text("xai chaos", encoding="utf-8")
    out = tmp_path / "snap.md"
    await create_repo_snapshot(base_path=str(tmp_path), out_path=str(out))
    content = out.read_text(encoding="utf-8")
    relevances = []
    for part in content.splitlines():
        if "relevance=" in part:
            val = float(part.split("relevance=")[1].split(")")[0])
            relevances.append(val)
    assert relevances and all(r > 0.5 for r in relevances)


@pytest.mark.asyncio
async def test_zip_and_rar(tmp_path):
    data_path = tmp_path / "data.txt"
    data_path.write_text("mars", encoding="utf-8")
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(data_path, arcname="data.txt")
    rar_path = tmp_path / "archive.rar"
    os.link(zip_path, rar_path)
    handler = FileHandler()
    zip_text = await handler.extract_async(str(zip_path))
    rar_text = await handler.extract_async(str(rar_path))
    assert "mars" in zip_text.lower()
    assert "mars" in rar_text.lower()


def test_markov_uses_apply_pulse(monkeypatch):
    calls = []

    def fake_apply(weights, pulse):
        calls.append((weights, pulse))
        return [1 / len(weights) for _ in weights]

    monkeypatch.setattr(cnp, "apply_pulse", fake_apply, raising=False)
    mm = cnp.MiniMarkov("mars starship chaos", n=1)
    mm.generate(5, start="mars")
    assert calls


@pytest.mark.asyncio
async def test_paraphrase_uses_dynamic_knowledge(monkeypatch):
    called = {}

    def fake_get(prompt):
        called["prompt"] = prompt
        return "dynamic"

    monkeypatch.setattr(cnp, "cg", None, raising=False)
    monkeypatch.setattr(cnp, "get_dynamic_knowledge", fake_get, raising=False)

    result = await cnp.paraphrase("hello world")
    assert "dynamic" in result
    assert called


@pytest.mark.asyncio
async def test_paraphrase_detects_language(monkeypatch):
    captured = {}

    def fake_get(prompt):
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(cnp, "cg", None, raising=False)
    monkeypatch.setattr(cnp, "get_dynamic_knowledge", fake_get, raising=False)

    await cnp.paraphrase("привет мир")
    assert captured["prompt"].startswith("Кратко перескажи для детей: ")


@pytest.fixture
def file_server(monkeypatch):
    monkeypatch.setenv("MAX_FILE_SIZE", "10")

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)

    def add(name, attrs):
        mod = types.ModuleType(f"utils.{name}")
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, f"utils.{name}", mod)
        setattr(utils_pkg, name, mod)

    add("dayandnight", {"day_and_night_task": lambda *a, **k: None})
    add("mirror", {"mirror_task": lambda *a, **k: None})
    add("prompt", {"get_chaos_response": lambda *a, **k: None, "build_system_prompt": lambda **k: ""})
    add("repo_monitor", {"monitor_repository": lambda *a, **k: None})
    add("vision", {"analyze_image": lambda *a, **k: None})
    add("context_neural_processor", {"parse_and_store_file": lambda *a, **k: "ok"})
    add("vector_engine", {"VectorGrokkyEngine": type("VectorGrokkyEngine", (), {})})
    add("hybrid_engine", {"HybridGrokkyEngine": type("HybridGrokkyEngine", (), {})})
    add("grok_chat_manager", {"GrokChatManager": type("GrokChatManager", (), {})})
    add("memory_manager", {"ImprovedMemoryManager": type("ImprovedMemoryManager", (), {})})
    add("dynamic_weights", {"DynamicWeights": type("DynamicWeights", (), {"__init__": lambda self, *a, **k: None})})
    add(
        "rl_trainer",
        {
            "RLTrainer": type(
                "RLTrainer", (), {"__init__": lambda self, *a, **k: None, "train": lambda self: None}
            ),
            "log_feedback": lambda *a, **k: None,
        },
    )
    add("metrics", {"REQUEST_LATENCY": None, "record_tokens": lambda *a, **k: None})

    plugins_mod = types.ModuleType("utils.plugins")
    plugins_mod.__path__ = []
    plugins_mod.load_plugins = lambda: []
    monkeypatch.setitem(sys.modules, "utils.plugins", plugins_mod)
    setattr(utils_pkg, "plugins", plugins_mod)
    coder_mod = types.ModuleType("utils.plugins.coder")
    coder_mod.interpret_code = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "utils.plugins.coder", coder_mod)
    mod42 = types.ModuleType("utils.plugins.42")
    mod42.handle = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "utils.plugins.42", mod42)
    plugins_mod.coder = coder_mod

    class Router:
        def __call__(self, *a, **k):
            def decorator(func):
                return func

            return decorator

        def register(self, *a, **k):
            pass

    class Dispatcher:
        def __init__(self, *a, **k):
            self.message = Router()
            self.callback_query = Router()

        async def feed_update(self, *a, **k):
            pass

    class Bot:
        def __init__(self, *a, **k):
            pass

        async def send_message(self, *a, **k):
            pass

        async def get_file(self, *a, **k):
            pass

    aiogram_pkg = types.ModuleType("aiogram")
    aiogram_pkg.__path__ = []
    aiogram_pkg.Bot = Bot
    aiogram_pkg.Dispatcher = Dispatcher
    monkeypatch.setitem(sys.modules, "aiogram", aiogram_pkg)
    monkeypatch.setitem(
        sys.modules,
        "aiogram.types",
        types.SimpleNamespace(
            Message=type("Message", (), {"reply": lambda self, *a, **k: None}),
            CallbackQuery=type("CallbackQuery", (), {"data": ""}),
            BotCommand=type("BotCommand", (), {"__init__": lambda self, *a, **k: None}),
            MenuButtonCommands=type("MenuButtonCommands", (), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiogram.enums",
        types.SimpleNamespace(ChatAction=type("ChatAction", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiogram.filters",
        types.SimpleNamespace(Command=type("Command", (), {"__init__": lambda self, *a, **k: None})),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiogram.exceptions",
        types.SimpleNamespace(TelegramAPIError=Exception),
    )

    import server
    importlib.reload(server)
    return server


@pytest.mark.asyncio
async def test_file_upload_too_large(file_server):
    from starlette.requests import Request
    from fastapi import UploadFile, HTTPException

    data = b"x" * 20
    upload = UploadFile(filename="big.txt", file=io.BytesIO(data))
    request = Request({"type": "http"})
    with pytest.raises(HTTPException) as exc:
        await file_server.handle_file_api(request, upload)
    assert exc.value.status_code == 400
    assert "file too large" in exc.value.detail


@pytest.mark.asyncio
async def test_file_upload_bad_extension(file_server):
    from starlette.requests import Request
    from fastapi import UploadFile, HTTPException

    upload = UploadFile(filename="bad.exe", file=io.BytesIO(b"ok"))
    request = Request({"type": "http"})
    with pytest.raises(HTTPException) as exc:
        await file_server.handle_file_api(request, upload)
    assert exc.value.status_code == 400
    assert "unsupported" in exc.value.detail

