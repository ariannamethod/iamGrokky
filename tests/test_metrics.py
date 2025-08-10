from utils.metrics import (
    TOKENS,
    ERRORS,
    MEMORY_HITS,
    COMMANDS,
    DATA_TRANSFERRED,
    record_tokens,
    record_error,
    record_memory_hit,
    record_command_usage,
    record_data_transfer,
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


def test_command_usage_counter_increments():
    counter = COMMANDS.labels(command="test")
    before = counter._value.get()
    record_command_usage("test")
    assert counter._value.get() == before + 1


def test_data_transfer_counter_increments():
    counter_in = DATA_TRANSFERRED.labels(direction="in")
    before_in = counter_in._value.get()
    record_data_transfer("in", 10)
    assert counter_in._value.get() == before_in + 10
