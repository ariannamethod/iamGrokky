"""Simple wrapper for OpenAI audio transcription."""
from __future__ import annotations

import os
import tempfile
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio(audio_bytes: bytes, model: str = "whisper-1") -> str:
    """Return text transcription for given audio bytes."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            with open(tmp.name, "rb") as f:
                resp = client.audio.transcriptions.create(model=model, file=f)
        return resp.text.strip()
    except Exception as exc:  # pragma: no cover - network
        return f"Transcription error: {exc}"


__all__ = ["transcribe_audio"]
