import os
import zipfile
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image
from reportlab.pdfgen import canvas

from utils.file_handling import FileHandler, compute_relevance, create_repo_snapshot


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
