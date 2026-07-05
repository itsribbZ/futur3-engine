"""A1.4 BackoffQueue test suite — 3-constraint rate-limit enforcement.

Test discipline:
- Deterministic via injected FakeMonotonic + fake sleep (zero wall-clock burn).
- Each of 3 constraints (global + per-contract + concurrent) independently tested.
- Boundary tests on limit edges (exactly N, N+1).
- Exception safety: concurrent counter decrements on raise.
- Timeout behavior + error message detail.

References:
- futur3/data/sources/backoff_queue.py (implementation)
- internal design notes (IBKR rate-limit spec source)
"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import timedelta

import pytest

from futur3.data.sources.backoff_queue import (
    DEFAULT_CONCURRENT_LIMIT,
    DEFAULT_GLOBAL_LIMIT,
    DEFAULT_PER_CONTRACT_LIMIT,
    BackoffQueue,
    BackoffQueueConfig,
    BackoffQueueTimeout,
)

# ============================================================================
# Test fakes
# ============================================================================


class FakeMonotonic:
    """Deterministic monotonic clock — `advance()` ticks virtual time."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def monotonic(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"FakeMonotonic.advance() requires non-negative; got {seconds}")
        self._t += seconds


def fake_sleep_factory(clock: FakeMonotonic) -> object:
    """Factory: returns a sleep_fn that advances the FakeMonotonic instead of blocking."""

    def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    return fake_sleep


def make_queue(
    clock: FakeMonotonic,
    *,
    global_limit: int = DEFAULT_GLOBAL_LIMIT,
    per_contract_limit: int = DEFAULT_PER_CONTRACT_LIMIT,
    concurrent_limit: int = DEFAULT_CONCURRENT_LIMIT,
    global_window: timedelta = timedelta(minutes=10),
    per_contract_window: timedelta = timedelta(seconds=2),
    acquire_timeout_seconds: float = 60.0,
) -> BackoffQueue:
    """Helper: build a BackoffQueue wired to the given fake clock."""
    cfg = BackoffQueueConfig(
        global_limit=global_limit,
        per_contract_limit=per_contract_limit,
        concurrent_limit=concurrent_limit,
        global_window=global_window,
        per_contract_window=per_contract_window,
        acquire_timeout_seconds=acquire_timeout_seconds,
    )
    return BackoffQueue(config=cfg, clock=clock, sleep_fn=fake_sleep_factory(clock))


# ============================================================================
# TestBackoffQueue_Imports — module imports + internal notes defaults
# ============================================================================


class TestBackoffQueue_Imports:
    def test_defaults_match_documented_caps(self) -> None:
        """internal notes: global=60/10min, per-contract=5/2s (≥6 forbidden), concurrent=50."""
        assert DEFAULT_GLOBAL_LIMIT == 60
        assert DEFAULT_PER_CONTRACT_LIMIT == 5  # "≥6 forbidden" → max 5 allowed
        assert DEFAULT_CONCURRENT_LIMIT == 50

    def test_default_config_constructs(self) -> None:
        cfg = BackoffQueueConfig()
        assert cfg.global_limit == 60
        assert cfg.per_contract_limit == 5
        assert cfg.concurrent_limit == 50
        assert cfg.global_window == timedelta(minutes=10)
        assert cfg.per_contract_window == timedelta(seconds=2)

    def test_queue_constructs_with_defaults(self) -> None:
        q = BackoffQueue()
        assert q.concurrent_count == 0


# ============================================================================
# TestBackoffQueue_ConfigValidation — invalid config rejected
# ============================================================================


class TestBackoffQueue_ConfigValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"global_limit": 0}, "global_limit"),
            ({"global_limit": -1}, "global_limit"),
            ({"per_contract_limit": 0}, "per_contract_limit"),
            ({"concurrent_limit": 0}, "concurrent_limit"),
            ({"global_window": timedelta(0)}, "global_window"),
            ({"per_contract_window": timedelta(0)}, "per_contract_window"),
            ({"acquire_timeout_seconds": 0.0}, "acquire_timeout_seconds"),
            ({"acquire_timeout_seconds": -1.0}, "acquire_timeout_seconds"),
        ],
    )
    def test_invalid_config_raises(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            BackoffQueueConfig(**kwargs)  # type: ignore[arg-type]


# ============================================================================
# TestBackoffQueue_BasicAcquire — single + multiple acquires update counters
# ============================================================================


class TestBackoffQueue_BasicAcquire:
    def test_acquire_increments_concurrent(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock)
        with q.acquire("ESM26@CME@TRADES"):
            assert q.concurrent_count == 1
        assert q.concurrent_count == 0

    def test_acquire_increments_global_count(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock)
        assert q.global_count() == 0
        with q.acquire("ESM26@CME@TRADES"):
            pass
        assert q.global_count() == 1

    def test_acquire_increments_per_contract_count(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock)
        with q.acquire("ESM26@CME@TRADES"):
            pass
        assert q.per_contract_count("ESM26@CME@TRADES") == 1
        # Other contract is untouched
        assert q.per_contract_count("NQM26@CME@TRADES") == 0

    def test_consecutive_acquires_accumulate(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock)
        for _ in range(5):
            with q.acquire("ESM26@CME@TRADES"):
                pass
            clock.advance(0.5)  # 0.5s between requests
        assert q.global_count() == 5
        assert q.concurrent_count == 0


# ============================================================================
# TestBackoffQueue_GlobalRolling — 60-in-10-min cap (internal notes)
# ============================================================================


class TestBackoffQueue_GlobalRolling:
    def test_60_distinct_contracts_then_61st_blocks(self) -> None:
        """60 distinct-contract acquires fit globally; 61st must wait for global window expiry.

        Uses distinct contracts to isolate the GLOBAL constraint from the per-contract one.
        With acquire_timeout=2000s (> 600s window), the 61st acquire eventually proceeds.
        """
        clock = FakeMonotonic()
        q = make_queue(
            clock,
            global_window=timedelta(seconds=600),
            acquire_timeout_seconds=2000.0,
        )
        # Pre-fill 60 requests at time 0 across distinct contracts
        for i in range(60):
            with q.acquire(f"contract_{i}"):
                pass

        # 61st acquire on a new contract: global=60 at limit → blocks until first ages out
        with q.acquire("contract_NEW"):
            # After acquire returns, clock should have advanced past 600s
            assert clock.monotonic() >= 600.0

    def test_global_count_decreases_after_window_expiry(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock, global_window=timedelta(seconds=10))
        for _ in range(3):
            with q.acquire("ESM26@CME@TRADES"):
                pass
        assert q.global_count() == 3
        # Advance past window
        clock.advance(11)
        # Pruning happens on next inspection
        assert q.global_count() == 0

    def test_global_rolling_window_not_fixed(self) -> None:
        """Old requests fall off rolling — newer ones stay."""
        clock = FakeMonotonic()
        q = make_queue(clock, global_window=timedelta(seconds=10))
        with q.acquire("ESM26@CME@TRADES"):
            pass
        clock.advance(5)
        with q.acquire("NQM26@CME@TRADES"):
            pass
        # At t=5: both still in window
        assert q.global_count() == 2
        # Advance past 10s from FIRST — first should age out
        clock.advance(6)  # now at t=11
        # The first acquire at t=0 is now 11s old; window is 10s → expired
        assert q.global_count() == 1


# ============================================================================
# TestBackoffQueue_PerContract — 6-in-2s same-contract cap (internal notes)
# ============================================================================


class TestBackoffQueue_PerContract:
    def test_5_same_contract_then_6th_blocks(self) -> None:
        """Per internal notes: '≥6 forbidden' → 5 same-contract OK, 6th must wait window."""
        clock = FakeMonotonic()
        q = make_queue(clock, per_contract_window=timedelta(seconds=2))
        # 5 requests at t=0 for same contract — all fit
        for _ in range(5):
            with q.acquire("ESM26@CME@TRADES"):
                pass
        # 6th: at limit, must wait for window expiry
        with q.acquire("ESM26@CME@TRADES"):
            assert clock.monotonic() >= 2.0

    def test_per_contract_isolation(self) -> None:
        """Saturating one contract doesn't starve another."""
        clock = FakeMonotonic()
        q = make_queue(clock, per_contract_window=timedelta(seconds=2))
        for _ in range(5):
            with q.acquire("ESM26@CME@TRADES"):
                pass
        # ES is at limit. NQ should still be allowed without sleep.
        before = clock.monotonic()
        with q.acquire("NQM26@CME@TRADES"):
            pass
        after = clock.monotonic()
        assert after == before  # zero sleep occurred

    def test_per_contract_count_decreases_after_window_expiry(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock, per_contract_window=timedelta(seconds=2))
        for _ in range(3):
            with q.acquire("ESM26@CME@TRADES"):
                pass
        assert q.per_contract_count("ESM26@CME@TRADES") == 3
        clock.advance(3)
        assert q.per_contract_count("ESM26@CME@TRADES") == 0


# ============================================================================
# TestBackoffQueue_Concurrent — 50-in-flight cap
# ============================================================================


class TestBackoffQueue_Concurrent:
    def test_concurrent_counter_increments_on_acquire(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock)
        with q.acquire("ESM26@CME@TRADES"):
            with q.acquire("NQM26@CME@TRADES"):
                assert q.concurrent_count == 2
            assert q.concurrent_count == 1
        assert q.concurrent_count == 0

    def test_concurrent_limit_holds_multiple_open(self) -> None:
        """N concurrent acquires keep the counter at N until each releases."""
        clock = FakeMonotonic()
        q = make_queue(clock, concurrent_limit=3, per_contract_limit=100, global_limit=1000)
        with ExitStack() as stack:
            for i in range(3):
                stack.enter_context(q.acquire(f"contract_{i}"))
            assert q.concurrent_count == 3
        assert q.concurrent_count == 0

    def test_concurrent_at_limit_blocks_then_times_out(self) -> None:
        """When at concurrent_limit, the next acquire spins until timeout fires."""
        clock = FakeMonotonic()
        q = make_queue(
            clock,
            concurrent_limit=2,
            per_contract_limit=100,
            global_limit=1000,
            acquire_timeout_seconds=1.0,
        )
        with ExitStack() as stack:
            for i in range(2):
                stack.enter_context(q.acquire(f"contract_{i}"))
            assert q.concurrent_count == 2
            # 3rd attempt: nothing releases (single-threaded), so we hit timeout
            with pytest.raises(BackoffQueueTimeout), q.acquire("contract_X"):
                pass


# ============================================================================
# TestBackoffQueue_ExceptionSafety — concurrent decrements on raise
# ============================================================================


class TestBackoffQueue_ExceptionSafety:
    def test_exception_in_block_releases_concurrent(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock)
        with pytest.raises(ValueError, match="test boom"), q.acquire("ESM26@CME@TRADES"):
            assert q.concurrent_count == 1
            raise ValueError("test boom")
        # After exception, concurrent must be back to 0
        assert q.concurrent_count == 0

    def test_global_log_records_even_if_block_raises(self) -> None:
        """The request DID happen; rate limit must count it."""
        clock = FakeMonotonic()
        q = make_queue(clock)
        with pytest.raises(ValueError), q.acquire("ESM26@CME@TRADES"):
            raise ValueError("test boom")
        assert q.global_count() == 1


# ============================================================================
# TestBackoffQueue_Timeout — raises BackoffQueueTimeout on starve
# ============================================================================


class TestBackoffQueue_Timeout:
    def test_timeout_raises_when_window_far_exceeds_timeout(self) -> None:
        """Configure timeout LESS than wait-time → must raise BackoffQueueTimeout."""
        clock = FakeMonotonic()
        q = make_queue(
            clock,
            global_limit=2,
            global_window=timedelta(seconds=600),  # 10 min
            acquire_timeout_seconds=1.0,  # 1s timeout
        )
        # Fill to limit
        for _ in range(2):
            with q.acquire("ESM26@CME@TRADES"):
                pass
        # 3rd would need to wait ~600s; timeout is 1s → raise
        with (
            pytest.raises(BackoffQueueTimeout, match="could not acquire slot"),
            q.acquire("ESM26@CME@TRADES"),
        ):
            pass

    def test_timeout_message_includes_diagnostics(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(
            clock,
            global_limit=1,
            global_window=timedelta(seconds=1000),
            acquire_timeout_seconds=0.5,
        )
        with q.acquire("ESM26@CME@TRADES"):
            pass
        with pytest.raises(BackoffQueueTimeout) as exc_info, q.acquire("ESM26@CME@TRADES"):
            pass
        message = str(exc_info.value)
        # Diagnostic content
        assert "global_count" in message
        assert "per_contract_count" in message
        assert "concurrent" in message


# ============================================================================
# TestBackoffQueue_Validation — input validation
# ============================================================================


class TestBackoffQueue_Validation:
    def test_empty_contract_key_raises(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock)
        with pytest.raises(ValueError, match="non-empty contract_key"), q.acquire(""):
            pass

    def test_fake_clock_advance_negative_raises(self) -> None:
        clock = FakeMonotonic()
        with pytest.raises(ValueError, match="non-negative"):
            clock.advance(-1)


# ============================================================================
# TestBackoffQueue_FIFOOrder — pruning maintains FIFO semantics
# ============================================================================


class TestBackoffQueue_FIFOOrder:
    def test_global_log_is_chronological(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(clock, global_window=timedelta(seconds=100))
        with q.acquire("ESM26@CME@TRADES"):
            pass
        clock.advance(10)
        with q.acquire("NQM26@CME@TRADES"):
            pass
        clock.advance(10)
        with q.acquire("CLN26@NYMEX@TRADES"):
            pass
        # All within 100s window
        assert q.global_count() == 3
        # Advance past first entry's window expiry only
        clock.advance(85)  # total elapsed = 105; first was at 0 → expired (>100s)
        # Second at 10s, third at 20s — both still within 100s window from 105s
        assert q.global_count() == 2

    def test_per_contract_pruning_doesnt_affect_global(self) -> None:
        clock = FakeMonotonic()
        q = make_queue(
            clock,
            per_contract_window=timedelta(seconds=2),
            global_window=timedelta(seconds=600),
        )
        with q.acquire("ESM26@CME@TRADES"):
            pass
        clock.advance(3)  # past per-contract window
        # per-contract pruned, global still counts
        assert q.per_contract_count("ESM26@CME@TRADES") == 0
        assert q.global_count() == 1
