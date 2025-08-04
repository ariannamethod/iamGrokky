import asyncio
import json
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.file_handling import FileHandler, create_repo_snapshot, _SEED_CORPUS

# logging helper

def log_result(name: str, status: str) -> None:
    log_dir = Path("logs/tests")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    entry = {"timestamp": datetime.now().isoformat(), "test": name, "status": status}
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@pytest.mark.asyncio
async def test_pdf_processing(tmp_path, monkeypatch):
    try:
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        class FakePage:
            def extract_text(self):
                return "Mars mission 2026"

        class FakeReader:
            def __init__(self, *args, **kwargs):
                self.pages = [FakePage()]

        monkeypatch.setattr("utils.file_handling.PdfReader", FakeReader)
        handler = FileHandler()
        text = await handler.extract_async(str(pdf_path))
        assert "Mars mission 2026" in text
        log_result("file_pdf_processing", "pass")
    except Exception:
        log_result("file_pdf_processing", "fail")
        raise


@pytest.mark.asyncio
async def test_txt_processing(tmp_path):
    try:
        txt_path = tmp_path / "doc.txt"
        txt_path.write_text("Starship chaos", encoding="utf-8")
        handler = FileHandler()
        text = await handler.extract_async(str(txt_path))
        assert "Starship chaos" in text
        log_result("file_txt_processing", "pass")
    except Exception:
        log_result("file_txt_processing", "fail")
        raise


@pytest.mark.asyncio
async def test_unsupported_file(tmp_path):
    try:
        bad_path = tmp_path / "data.xyz"
        bad_path.write_text("data")
        handler = FileHandler()
        result = await handler.extract_async(str(bad_path))
        assert "Unsupported file" in result
        log_file = Path("logs/failures") / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        assert log_file.exists() and "Unsupported file type" in log_file.read_text(encoding="utf-8")
        log_result("file_unsupported", "pass")
    except Exception:
        log_result("file_unsupported", "fail")
        raise


@pytest.mark.asyncio
async def test_batch_processing(tmp_path, monkeypatch):
    try:
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        txt = tmp_path / "a.txt"
        txt.write_text("hello", encoding="utf-8")
        img_path = tmp_path / "a.png"
        Image.new("RGB", (1, 1)).save(img_path)

        class FakePage:
            def extract_text(self):
                return "Mars mission 2026"

        class FakeReader:
            def __init__(self, *args, **kwargs):
                self.pages = [FakePage()]

        monkeypatch.setattr("utils.file_handling.PdfReader", FakeReader)
        handler = FileHandler()
        texts = await handler.extract_batch([str(pdf), str(txt), str(img_path)])
        assert len(texts) == 3 and all(isinstance(t, str) for t in texts)
        log_result("file_batch_processing", "pass")
    except Exception:
        log_result("file_batch_processing", "fail")
        raise


@pytest.mark.asyncio
async def test_repo_snapshot(tmp_path):
    try:
        f1 = tmp_path / "one.txt"
        f2 = tmp_path / "two.txt"
        f1.write_text(_SEED_CORPUS, encoding="utf-8")
        f2.write_text(_SEED_CORPUS, encoding="utf-8")
        out = tmp_path / "snap.md"
        await create_repo_snapshot(base_path=str(tmp_path), out_path=str(out))
        content = out.read_text(encoding="utf-8")
        relevances = [float(r) for r in re.findall(r"relevance=(\d+\.\d+)", content)]
        assert relevances and all(r > 0.5 for r in relevances)
        log_result("file_repo_snapshot", "pass")
    except Exception:
        log_result("file_repo_snapshot", "fail")
        raise


@pytest.mark.asyncio
async def test_archives(tmp_path):
    try:
        inner = tmp_path / "inner.txt"
        inner.write_text("Mars mission", encoding="utf-8")
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(inner, arcname="inner.txt")
        rar_path = tmp_path / "test.rar"
        rar_path.write_bytes(zip_path.read_bytes())
        handler = FileHandler()
        zip_text = await handler.extract_async(str(zip_path))
        rar_text = await handler.extract_async(str(rar_path))
        assert "Mars mission" in zip_text
        assert "Mars mission" in rar_text
        log_result("file_archives", "pass")
    except Exception:
        log_result("file_archives", "fail")
        raise
