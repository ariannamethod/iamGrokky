"""Lightweight Grok-1 inference utilities.

This module loads a quantised checkpoint produced by the training utilities
and performs token-by-token generation.  It purposely keeps the implementation
minimal so that the tiny test checkpoint can be executed quickly on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - torch is optional at runtime
    import torch
except ModuleNotFoundError:  # pragma: no cover - fallback to dynamic weights
    torch = None

from .nanogpt_model import GPT, GPTConfig
from utils.dynamic_weights import DynamicWeights


def _dequantize(tensor_or_dict: object) -> torch.Tensor:
    """Return a de-quantised tensor.

    The quantisation format mirrors the output of :mod:`SLNCX.quantize` where
    each tensor is stored as an ``int8`` weight matrix and a corresponding
    floating point scale.  If ``tensor_or_dict`` is already a tensor it is
    returned as-is.
    """

    if isinstance(tensor_or_dict, dict) and "weight" in tensor_or_dict and "scale" in tensor_or_dict:
        weight = tensor_or_dict["weight"].to(torch.float32)
        scale = tensor_or_dict["scale"].to(torch.float32)
        return weight * scale
    return torch.as_tensor(tensor_or_dict)


def _load_model(ckpt_path: str) -> GPT:
    """Load a quantised Grok-1 checkpoint."""

    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model inference")

    ckpt: Dict[str, object] = torch.load(Path(ckpt_path), map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    state_dict = {k: _dequantize(v) for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _encode(text: str) -> list[int]:
    """Very small character-level encoding used for the tests."""

    return [ord(c) for c in text]


def _decode(tokens: list[int]) -> str:
    return "".join(chr(t) for t in tokens)


def generate(
    prompt: str,
    ckpt_path: str = "out/ckpt.pt",
    api_key: Optional[str] = None,
    *,
    seed: Optional[int] = None,
    max_new_tokens: int = 20,
) -> str:
    """Generate text from ``prompt`` using a quantised model.

    ``DynamicWeights`` are consulted to derive a ``pulse`` value that modulates
    the sampling temperature.  ``seed`` controls the RNG used for sampling to
    allow deterministic outputs in tests.
    """

    if torch is None:
        dw = DynamicWeights()
        return dw.generate_response(prompt, api_key)

    try:
        model = _load_model(ckpt_path)
    except (FileNotFoundError, ModuleNotFoundError):
        dw = DynamicWeights()
        return dw.generate_response(prompt, api_key)

    dw = DynamicWeights()
    pulse = dw.pulse_from_prompt(prompt, api_key)
    temperature = 0.8 + 0.2 * pulse

    device = next(model.parameters()).device
    idx = torch.tensor([_encode(prompt)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    out_tokens: list[int] = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(idx[:, -model.config.block_size :])
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat((idx, next_id), dim=1)
        out_tokens.append(int(next_id))

    return _decode(out_tokens)

