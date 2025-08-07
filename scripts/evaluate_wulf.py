"""Evaluate Wulf model quality on a small prompt set.

The script runs the Wulf generator on a predefined dataset and computes
simple BLEU and ROUGE-L metrics. Results are saved to disk to enable
regression comparisons on subsequent runs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Dict, List

from SLNCX import wulf_integration


def _bleu1(reference: str, hypothesis: str) -> float:
    """Compute a unigram BLEU score."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not hyp_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    overlap = sum(min(count, ref_counts[tok]) for tok, count in hyp_counts.items())
    return overlap / len(hyp_tokens)


def _lcs(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def _rouge_l(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        return 0.0
    lcs = _lcs(ref_tokens, hyp_tokens)
    return lcs / len(ref_tokens)


def run_evaluation(test_data: Iterable[Dict[str, str]], metrics_path: str | None = None) -> Dict[str, float]:
    """Run evaluation and optionally store metrics."""
    bleu_scores: List[float] = []
    rouge_scores: List[float] = []

    for item in test_data:
        prompt = item["prompt"]
        reference = item["reference"]
        generated = wulf_integration.generate_response(prompt, mode="wulf")
        bleu_scores.append(_bleu1(reference, generated))
        rouge_scores.append(_rouge_l(reference, generated))

    metrics = {
        "bleu": sum(bleu_scores) / len(bleu_scores),
        "rouge_l": sum(rouge_scores) / len(rouge_scores),
    }

    if metrics_path:
        path = Path(metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                previous = json.load(f)
            for k, old in previous.items():
                new = metrics.get(k, 0.0)
                if new < old:
                    print(f"Warning: {k} decreased from {old:.4f} to {new:.4f}")
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Wulf model quality")
    parser.add_argument(
        "--metrics-path",
        default="tests/benchmarks/wulf_metrics.json",
        help="Location to store computed metrics",
    )
    args = parser.parse_args()

    # Import test dataset lazily to avoid circular imports during testing
    from tests.benchmarks.test_wulf_quality import TEST_DATA

    run_evaluation(TEST_DATA, metrics_path=args.metrics_path)


if __name__ == "__main__":
    main()
