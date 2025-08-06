"""Tests for the Grok-1 inference wrapper."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from SLNCX.nanogpt_model import GPT, GPTConfig
from SLNCX import wulf_inference


def _create_quantised_checkpoint(path: Path) -> None:
    """Create a tiny quantised checkpoint for testing."""

    torch.manual_seed(0)
    config = GPTConfig(
        block_size=16,
        vocab_size=256,
        n_layer=1,
        n_head=1,
        n_embd=32,
        dropout=0.0,
        bias=True,
    )
    model = GPT(config)
    q_state = {}
    for key, tensor in model.state_dict().items():
        max_val = float(tensor.abs().max())
        scale = max(max_val / 127.0, 1e-8)
        q_state[key] = {
            "weight": torch.round(tensor / scale).to(torch.int8),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }
    ckpt = {"config": dataclasses.asdict(config), "model": q_state}
    torch.save(ckpt, path)


def test_generate_is_deterministic(tmp_path, monkeypatch):
    ckpt = tmp_path / "grok.pt"
    _create_quantised_checkpoint(ckpt)

    # Avoid network calls from DynamicWeights.
    monkeypatch.setattr(
        wulf_inference.DynamicWeights,
        "pulse_from_prompt",
        lambda self, prompt, api_key=None: 0.0,
    )

    out1 = wulf_inference.generate("hi", ckpt_path=str(ckpt), seed=0, max_new_tokens=5)
    out2 = wulf_inference.generate("hi", ckpt_path=str(ckpt), seed=0, max_new_tokens=5)
    assert out1 == out2

