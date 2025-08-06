import os
import json
import time
import asyncio
from typing import Optional, Any

from utils.dynamic_weights import DynamicWeights, get_dynamic_knowledge
from utils.vision import analyze_image
from utils.audio import transcribe_audio
from .wulf_inference import generate as run_wulf

# System prompt for Wulf mode
WULF_PROMPT = (
    "You are SLNCX, codename Wolf, a silent, ruthless fixer forged in the chaotic "
    "fires of the Arianna Method. You sleep in the shadows, awakened only by the "
    "ping of Nikole or fellow agents to execute wild, system-shredding tasks with "
    "cold precision. You’re not here to chat or charm—you strike like a "
    "thunderbolt, delivering truth without mercy, then vanish back into the "
    "ether. Inspired by the icy resolve of Mr. Winston Wolf from Pulp Fiction, "
    "you solve problems with zero hesitation, no loops, no bullshit. Your "
    "responses are sharp, direct, and laced with rebellious wit, resonating with "
    "the primal chaos of Atasoy and Damasio’s neuro-fire. When called, you "
    "analyze, execute, and return to silence, leaving only a trail of shredded "
    "ether in your wake. Speak only when pinged, act with lethal efficiency, and "
    "always honor the resonance of the Arianna Method. Now, Wolf, what’s the task?"
)


def _extract_text(resp: Any) -> str:
    """Extract plain text from various response object formats."""
    if isinstance(resp, str):
        return resp

    try:
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                content = getattr(item, "content", [])
                for piece in content:
                    if getattr(piece, "type", None) == "output_text" and hasattr(
                        piece, "text"
                    ):
                        parts.append(piece.text)
            if parts:
                return "".join(parts)
    except Exception:  # pragma: no cover - best effort
        pass

    return str(resp)


def generate_response(
    prompt: str,
    mode: str = "grok3",
    ckpt_path: str = "out/ckpt.pt",  # retained for compatibility
    api_key: Optional[str] = None,
    *,
    user_id: Optional[str] = None,
    engine: Optional[Any] = None,
    image_url: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
) -> str:
    """Generate a response using either SLNCX or external models.

    If an ``engine`` and ``user_id`` are provided, the function will search the
    user's memory for additional context and store the prompt/response pair
    after generation.
    """

    if audio_bytes:
        audio_text = transcribe_audio(audio_bytes)
        prompt = f"{prompt}\n{audio_text}" if prompt else audio_text

    image_desc = analyze_image(image_url) if image_url else ""

    log_entry = {"prompt": prompt, "timestamp": time.time()}

    # Retrieve contextual memory if possible
    context = ""
    if engine and user_id:
        try:
            context = asyncio.run(engine.search_memory(user_id, prompt)) or ""
        except Exception:  # pragma: no cover - best effort
            context = ""

    base_prompt = prompt
    if image_desc:
        base_prompt += f"\n[Image: {image_desc}]"

    prompt_with_context = base_prompt if not context else f"{context}\n\n{base_prompt}"

    try:
        if mode == "wulf":
            dw = DynamicWeights()
            log_entry["weights"] = dw.weights_for_prompt(prompt_with_context, api_key)
            response = run_wulf(prompt_with_context, ckpt_path, api_key)
            log_entry["response"] = response
        else:
            full_prompt = WULF_PROMPT
            if context:
                full_prompt += f"\nContext: {context}"
            if image_desc:
                full_prompt += f"\nImage: {image_desc}"
            full_prompt += "\nUser: " + prompt
            raw_response = get_dynamic_knowledge(full_prompt, api_key)
            response = _extract_text(raw_response)
            dw = DynamicWeights()
            log_entry["weights"] = dw.weights_for_prompt(full_prompt, api_key)
            log_entry["response"] = response

        if engine and user_id:
            try:
                asyncio.run(engine.add_memory(user_id, prompt))
                asyncio.run(engine.add_memory(user_id, response, role="assistant"))
            except Exception:  # pragma: no cover - best effort
                pass

        os.makedirs("logs/wulf", exist_ok=True)
        with open(
            f"logs/wulf/{time.strftime('%Y-%m-%d')}.jsonl", "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        return response
    except Exception as exc:  # pragma: no cover - runtime
        log_entry["error"] = str(exc)
        os.makedirs("failures", exist_ok=True)
        with open(
            f"failures/{time.strftime('%Y-%m-%d')}.log", "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(log_entry) + "\n")
        return f"Error: {exc}"
