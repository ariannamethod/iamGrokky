"""Simple wrapper around OpenAI audio transcription API."""
from __future__ import annotations

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_audio(audio_path: str) -> str:
    """Return a short transcription for the provided audio file."""
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        text = getattr(response, "text", "")
        return text.strip() if isinstance(text, str) else ""
    except Exception as exc:  # pragma: no cover - network or file errors
        return f"Audio error: {exc}"


__all__ = ["analyze_audio"]
