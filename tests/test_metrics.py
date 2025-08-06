from utils.metrics import (
    TOKENS,
    ERRORS,
    MEMORY_HITS,
    record_tokens,
    record_error,
    record_memory_hit,
)


def test_token_counter_increments():
    counter = TOKENS.labels(model="test")
    before = counter._value.get()
    record_tokens("test", 3)
    assert counter._value.get() == before + 3


def test_error_counter_increments():
    counter = ERRORS.labels(type="timeout")
    before = counter._value.get()
    record_error("timeout")
    assert counter._value.get() == before + 1


def test_memory_hit_counter_increments():
    before = MEMORY_HITS._value.get()
    record_memory_hit()
    assert MEMORY_HITS._value.get() == before + 1
