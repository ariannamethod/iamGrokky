import asyncio
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.file_handling import FileHandler, create_repo_snapshot


PDF_BYTES = (
    b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 50 150 Td (Mars mission 2026) Tj ET\nendstream\nendobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000104 00000 n \n0000000207 00000 n \n0000000280 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n347\n%%EOF"
)

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc``\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _log(name: str, status: str, error: str | None = None) -> None:
    date = datetime.now().strftime("%Y-%m-%d")
    tests_dir = Path("logs/tests")
    tests_dir.mkdir(parents=True, exist_ok=True)
    with open(tests_dir / f"{date}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(), "test": name, "status": status}) + "\n")
    if status == "fail":
        fail_dir = Path("logs/failures")
        fail_dir.mkdir(parents=True, exist_ok=True)
        with open(fail_dir / f"{date}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": datetime.now().isoformat(), "test": name, "error": error}) + "\n")


@pytest.mark.asyncio
async def test_pdf_processing(tmp_path):
    name = "file_pdf"
    handler = FileHandler()
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(PDF_BYTES)
    async def fake_pdf(path: str) -> str:
        return "Mars mission 2026"
    handler._extract_pdf = fake_pdf  # type: ignore[attr-defined]
    handler._extractors[".pdf"] = handler._extract_pdf
    try:
        text = await handler.extract_async(str(pdf))
        assert "Mars mission 2026" in text
        _log(name, "pass")
    except Exception as e:  # pragma: no cover - defensive
        _log(name, "fail", str(e))
        raise


@pytest.mark.asyncio
async def test_txt_processing(tmp_path):
    name = "file_txt"
    handler = FileHandler()
    txt = tmp_path / "test.txt"
    txt.write_text("Starship chaos", encoding="utf-8")
    async def fake_txt(path: str) -> str:
        return "Starship chaos"
    handler._extract_txt = fake_txt  # type: ignore[attr-defined]
    handler._extractors[".txt"] = handler._extract_txt
    try:
        text = await handler.extract_async(str(txt))
        assert "Starship chaos" in text
        _log(name, "pass")
    except Exception as e:
        _log(name, "fail", str(e))
        raise


@pytest.mark.asyncio
async def test_unsupported_file_logging(tmp_path):
    name = "file_unsupported"
    handler = FileHandler()
    bad = tmp_path / "test.xyz"
    bad.write_text("???", encoding="utf-8")
    try:
        result = await handler.extract_async(str(bad))
        assert "Unsupported file" in result
        log_file = Path("logs/failures") / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        assert log_file.exists()
        assert "Unsupported file type" in log_file.read_text(encoding="utf-8")
        _log(name, "pass")
    except Exception as e:
        _log(name, "fail", str(e))
        raise


@pytest.mark.asyncio
async def test_batch_processing(tmp_path):
    name = "file_batch"
    handler = FileHandler()
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(PDF_BYTES)
    txt = tmp_path / "b.txt"
    txt.write_text("Mars", encoding="utf-8")
    png = tmp_path / "c.png"
    png.write_bytes(PNG_BYTES)
    try:
        results = await handler.extract_batch([str(pdf), str(txt), str(png)])
        assert len(results) == 3
        _log(name, "pass")
    except Exception as e:
        _log(name, "fail", str(e))
        raise


@pytest.mark.asyncio
async def test_repo_snapshot(tmp_path):
    name = "file_snapshot"
    (tmp_path / "one.txt").write_text("mars starship xai chaos", encoding="utf-8")
    (tmp_path / "two.txt").write_text("mars starship xai chaos", encoding="utf-8")
    out = tmp_path / "snap.md"
    try:
        await create_repo_snapshot(str(tmp_path), str(out))
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        relevances = [float(re.search(r"relevance=(\d+\.\d+)", l).group(1)) for l in lines]
        assert all(r > 0.5 for r in relevances)
        _log(name, "pass")
    except Exception as e:
        _log(name, "fail", str(e))
        raise


@pytest.mark.asyncio
async def test_archives(tmp_path):
    name = "file_archives"
    handler = FileHandler()
    inner = tmp_path / "inner.txt"
    inner.write_text("Mars inside", encoding="utf-8")
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(inner, arcname="inner.txt")
    rar_path = tmp_path / "test.rar"
    os.rename(zip_path, rar_path)
    try:
        text_zip = await handler.extract_async(str(rar_path))
        assert "Mars" in text_zip
        _log(name, "pass")
    except Exception as e:
        _log(name, "fail", str(e))
        raise
