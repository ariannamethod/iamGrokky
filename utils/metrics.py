"""Prometheus metrics helpers for Grokky."""

from prometheus_client import Counter, Histogram

# Histograms for measuring request latency per endpoint.
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Time spent handling request", ["endpoint"]
)

# Counter for total tokens consumed per model.
TOKENS = Counter("tokens_total", "Number of tokens processed", ["model"])

# Counter for different error types.
ERRORS = Counter("errors_total", "Total errors", ["type"])

# Counter for memory cache hits.
MEMORY_HITS = Counter("memory_hits_total", "Memory hits")


def record_tokens(model: str, count: int) -> None:
    """Record ``count`` tokens used by ``model``."""
    TOKENS.labels(model=model).inc(count)


def record_error(err_type: str) -> None:
    """Increment error counter for ``err_type``."""
    ERRORS.labels(type=err_type).inc()


def record_memory_hit() -> None:
    """Increment memory hit counter."""
    MEMORY_HITS.inc()
