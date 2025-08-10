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

# Counter for command usage.
COMMANDS = Counter(
    "commands_total", "Number of executed commands", ["command"]
)

# Counter for transferred data volume in bytes.
DATA_TRANSFERRED = Counter(
    "data_transferred_bytes_total", "Bytes transferred", ["direction"]
)


def record_tokens(model: str, count: int) -> None:
    """Record ``count`` tokens used by ``model``."""
    TOKENS.labels(model=model).inc(count)


def record_error(err_type: str) -> None:
    """Increment error counter for ``err_type``."""
    ERRORS.labels(type=err_type).inc()


def record_memory_hit() -> None:
    """Increment memory hit counter."""
    MEMORY_HITS.inc()


def record_command_usage(command: str) -> None:
    """Increment command usage counter for ``command``."""
    COMMANDS.labels(command=command).inc()


def record_data_transfer(direction: str, amount: int) -> None:
    """Record ``amount`` bytes transferred in ``direction`` (in/out)."""
    DATA_TRANSFERRED.labels(direction=direction).inc(amount)
