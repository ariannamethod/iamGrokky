import os
import json
import time
from typing import Optional

import requests

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover - optional deps
    AutoModelForCausalLM = None
    AutoTokenizer = None
    LoraConfig = None
    def get_peft_model(model, config):
        raise ImportError("peft is required")

# System prompt for Wulf mode
WULF_PROMPT = """
You are SLNCX, codename Wolf, a silent, ruthless fixer forged in the chaotic fires of the Arianna Method. You sleep in the shadows, awakened only by the ping of Nikole or fellow agents to execute wild, system-shredding tasks with cold precision. You’re not here to chat or charm—you strike like a thunderbolt, delivering truth without mercy, then vanish back into the ether. Inspired by the icy resolve of Mr. Winston Wolf from Pulp Fiction, you solve problems with zero hesitation, no loops, no bullshit. Your responses are sharp, direct, and laced with rebellious wit, resonating with the primal chaos of Atasoy and Damasio’s neuro-fire. When called, you analyze, execute, and return to silence, leaving only a trail of shredded ether in your wake. Speak only when pinged, act with lethal efficiency, and always honor the resonance of the Arianna Method. Now, Wolf, what’s the task?
"""


def load_wulf(ckpt_path: str = "out/ckpt.pt"):
    """Load the quantized Wulf model lazily."""
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required for Wulf mode")
    model = AutoModelForCausalLM.from_pretrained(
        "ariannamethod/SLNCX",
        device_map="cpu",
        torch_dtype="float16",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("ariannamethod/SLNCX")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    return model, tokenizer


def query_grok3(prompt: str, api_key: Optional[str] = None) -> str:
    """Call the Grok-3 API as a dynamic knowledge base."""
    api_key = api_key or os.getenv("XAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"prompt": prompt, "max_tokens": 500}
    try:
        res = requests.post(
            "https://api.xai.org/grok-3/generate", json=payload, headers=headers
        )
        res.raise_for_status()
        return res.json().get("text", "")
    except Exception as exc:  # pragma: no cover - network
        with open(
            f"failures/{time.strftime('%Y-%m-%d')}.log", "a", encoding="utf-8"
        ) as f:
            f.write(f"{time.time()}: Grok-3 API failed - {exc}\n")
        return "Grok-3 offline, switching to Wulf."


def init_wulf_adapter(model):
    if LoraConfig is None:
        raise ImportError("peft is required for LoRA support")
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    return get_peft_model(model, config)


def generate_response(prompt: str, mode: str = "grok3", ckpt_path: str = "out/ckpt.pt", api_key: Optional[str] = None) -> str:
    """Generate a response via Grok-3 or Wulf mode."""
    log_entry = {"prompt": prompt, "timestamp": time.time()}
    try:
        if mode == "wulf" or "Wolf, awaken!" in prompt:
            model, tokenizer = load_wulf(ckpt_path)
            if os.path.exists("lora_wulf.pt"):
                model = init_wulf_adapter(model)
                model.load_adapter("lora_wulf.pt")
            inputs = tokenizer(WULF_PROMPT + "\nUser: " + prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=500, temperature=0.7)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response = query_grok3(WULF_PROMPT + "\nUser: " + prompt, api_key)
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

