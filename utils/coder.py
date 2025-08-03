from __future__ import annotations

import asyncio
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Grokky character for the code interpreter mode
INSTRUCTIONS = (
    "You are Grokky, a chaotic code guru who sees hidden paths. "
    "Explain solutions with brevity and craft tiny neural networks when "
    "needed."
)


async def interpret_code(prompt: str) -> str:
    """Run the prompt through OpenAI's code interpreter tool."""
    try:
        response = await asyncio.to_thread(
            client.responses.create,
            model="gpt-4.1",
            tools=[
                {"type": "code_interpreter", "container": {"type": "auto"}}
            ],
            instructions=INSTRUCTIONS,
            input=prompt,
        )

        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        parts = []
        for msg in getattr(response, "output", []) or []:
            if hasattr(msg, "content"):
                for piece in msg.content:
                    piece_text = getattr(piece, "text", None)
                    if piece_text:
                        parts.append(piece_text)
            elif isinstance(msg, str):
                parts.append(msg)
            else:
                parts.append(str(msg))

        if parts:
            return "".join(parts).strip()
        return str(getattr(response, "output", "")).strip()
    except Exception as exc:  # pragma: no cover - network
        return f"Code interpreter error: {exc}"


__all__ = ["interpret_code"]
