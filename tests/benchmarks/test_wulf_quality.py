"""Benchmark tests for Wulf quality metrics."""
from scripts import evaluate_wulf

TEST_DATA = [
    {"prompt": "Say hello", "reference": "hello"},
    {"prompt": "2 plus 2", "reference": "4"},
]

BASELINE = {"bleu": 0.9, "rouge_l": 0.9}

FAKE_OUTPUTS = {item["prompt"]: item["reference"] for item in TEST_DATA}


def test_wulf_quality(monkeypatch, tmp_path):
    def fake_generate(prompt, mode="wulf"):
        return FAKE_OUTPUTS[prompt]

    monkeypatch.setattr(evaluate_wulf.wulf_integration, "generate_response", fake_generate)
    metrics_file = tmp_path / "metrics.json"
    metrics = evaluate_wulf.run_evaluation(TEST_DATA, metrics_file)
    assert metrics["bleu"] >= BASELINE["bleu"]
    assert metrics["rouge_l"] >= BASELINE["rouge_l"]
