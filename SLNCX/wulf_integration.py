import os
import json
import time
from typing import Optional

from utils.dynamic_weights import get_dynamic_knowledge

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


def generate_response(
    prompt: str,
    mode: str = "grok3",
    ckpt_path: str = "out/ckpt.pt",  # retained for compatibility
    api_key: Optional[str] = None,
) -> str:
    """Generate a response using external knowledge sources."""

    log_entry = {"prompt": prompt, "timestamp": time.time()}
    try:
        full_prompt = WULF_PROMPT + "\nUser: " + prompt
        response = get_dynamic_knowledge(full_prompt, api_key)
        log_entry["response"] = response
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
