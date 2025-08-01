import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.coder import run_coder


def test_run_coder_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = run_coder("print(1+1)")
    assert out.startswith("Coder error")
