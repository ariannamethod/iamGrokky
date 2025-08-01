from __future__ import annotations

import os
from openai import OpenAI


def run_coder(query: str, *, model: str | None = None, instructions: str | None = None) -> str:
    """Execute a code-interpreter request via OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Coder error: OPENAI_API_KEY is missing."
    client = OpenAI(api_key=api_key)

    model = model or os.getenv("CODER_MODEL", "gpt-4.1")
    instructions = instructions or (
        "You are Grokky, the code sage. Explore unusual paths and craft mini neural nets."
    )

    try:
        response = client.responses.create(
            model=model,
            tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
            instructions=instructions,
            input=query,
        )
        return response.output
    except Exception as exc:  # pragma: no cover - network
        return f"Coder error: {exc}"


__all__ = ["run_coder"]
