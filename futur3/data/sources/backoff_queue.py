"""BackoffQueue — three-constraint rate limiter for IBKR historical-data requests.

Enforces IBKR's documented rate caps per internal notes in a single context-managed gate:

1. **Global rolling**: ≤60 requests in any 10-minute window.
2. **Per-contract**: ≤6 requests in any 2-second window for the same contract-key.
3. **Concurrent**: ≤50 in-flight requests at any instant.

A single `acquire(contract_key)` blocks until ALL three constraints permit another
request. On exit (or exception), the in-flight counter is decremented automatically.

## Design choices

- **Sync-only** (by design).
  ib_async ships sync wrappers in 2.0+; futur3 stays SYNC for Phase A1-B1 because async
  adds 3+ bug classes (event-loop bugs, partial-result races, exception-swallowing).
- **Injectable clock + sleep_fn** so deterministic tests advance virtual time without
  burning wall-clock. Production passes `time.monotonic()` + `time.sleep`.
- **No threading**: futur3 Phase A1 is single-threaded. The counter is a plain int.
  If multi-thread access is ever introduced, swap counter for `threading.Semaphore`.
- **Fail-fast on bad input**: empty contract_key → ValueError; negative timeout → ValueError.

## Caller pattern

```python
queue = BackoffQueue()  # default = monotonic clock + time.sleep
with queue.acquire("ESM26@CME@TRADES"):
    bars = ib_client.req_historical_data(...)
```

Per internal notes "BID_ASK counts double": callers using `whatToShow=BID_ASK` should
record TWO acquires (or use `contract_key` that distinguishes BID_ASK to nudge the
6-in-2s constraint earlier).

## Invariants

- After every `acquire(...) ... release` cycle: `concurrent_count >= 0` (never negative).
- Global request log monotonically extends; pruning removes only entries past window.
- Per-contract logs are independent; one busy contract doesn't starve others.

## Reference

- internal design notes — IBKR rate-limit primary source
- the data-layer design — sync-over-async rationale
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol, runtime_checkable

from futur3.data.source import DataSourceError

logger = logging.getLogger(__name__)

# Per internal notes — IBKR historical-data rate caps
DEFAULT_GLOBAL_LIMIT: Final[int] = 60  # IBKR: "max 60 requests in any 10-min window"
DEFAULT_GLOBAL_WINDOW: Final[timedelta] = timedelta(minutes=10)
# IBKR rule: "cannot make ≥6 requests within 2 seconds" → ≤5 actually allowed. Limit=5
# matches internal notes "per-contract 400ms throttle" recommendation (2s / 400ms = 5/window).
DEFAULT_PER_CONTRACT_LIMIT: Final[int] = 5
DEFAULT_PER_CONTRACT_WINDOW: Final[timedelta] = timedelta(seconds=2)
DEFAULT_CONCURRENT_LIMIT: Final[int] = 50  # IBKR: max 50 simultaneous historical requests
DEFAULT_ACQUIRE_TIMEOUT_SECONDS: Final[float] = 60.0


class BackoffQueueTimeout(DataSourceError):
    """`acquire` could not obtain a slot within the configured timeout.

    Indicates upstream load exceeds the queue's capacity within the timeout window.
    Caller should treat this as a soft-failure and retry with backoff at a higher
    layer (e.g., `IBKRHistoricalDataSource` retries on next backtest pass).
    """


@runtime_checkable
class MonotonicClock(Protocol):
    """Monotonic clock interface — injectable for deterministic tests.

    Returns elapsed seconds since an arbitrary epoch. Must be monotonically
    non-decreasing across calls within a process lifetime.
    """

    def monotonic(self) -> float: ...


@runtime_checkable
class SleepFn(Protocol):
    """Sleep callable — injectable for deterministic tests.

    Production: `time.sleep`. Tests: a no-op that advances the FakeClock instead.
    """

    def __call__(self, seconds: float) -> None: ...


class _SystemMonotonic:
    """Default monotonic clock (wraps `time.monotonic`)."""

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class BackoffQueueConfig:
    """Tunable rate limits per BackoffQueue instance.

    Defaults match internal notes IBKR caps. Other sources (Tradovate, crypto venues)
    can construct with different limits.
    """

    global_limit: int = DEFAULT_GLOBAL_LIMIT
    global_window: timedelta = DEFAULT_GLOBAL_WINDOW
    per_contract_limit: int = DEFAULT_PER_CONTRACT_LIMIT
    per_contract_window: timedelta = DEFAULT_PER_CONTRACT_WINDOW
    concurrent_limit: int = DEFAULT_CONCURRENT_LIMIT
    acquire_timeout_seconds: float = DEFAULT_ACQUIRE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.global_limit <= 0:
            raise ValueError(f"global_limit must be > 0; got {self.global_limit}")
        if self.per_contract_limit <= 0:
            raise ValueError(f"per_contract_limit must be > 0; got {self.per_contract_limit}")
        if self.concurrent_limit <= 0:
            raise ValueError(f"concurrent_limit must be > 0; got {self.concurrent_limit}")
        if self.global_window.total_seconds() <= 0:
            raise ValueError(f"global_window must be > 0; got {self.global_window}")
        if self.per_contract_window.total_seconds() <= 0:
            raise ValueError(f"per_contract_window must be > 0; got {self.per_contract_window}")
        if self.acquire_timeout_seconds <= 0:
            raise ValueError(
                f"acquire_timeout_seconds must be > 0; got {self.acquire_timeout_seconds}"
            )


class BackoffQueue:
    """Three-constraint rate-limit gate (global + per-contract + concurrent)."""

    def __init__(
        self,
        config: BackoffQueueConfig | None = None,
        clock: MonotonicClock | None = None,
        sleep_fn: SleepFn | None = None,
    ) -> None:
        self._config: BackoffQueueConfig = config or BackoffQueueConfig()
        self._clock: MonotonicClock = clock or _SystemMonotonic()
        # mypy: `time.sleep` accepts `float | SupportsIndex` which is a wider type than
        # SleepFn's `float`; assignment is type-safe in practice (every float is acceptable).
        self._sleep_fn: SleepFn = sleep_fn or time.sleep  # type: ignore[assignment]

        # Rolling timestamps (monotonic seconds) of completed acquires
        self._global_log: deque[float] = deque()
        self._per_contract_log: defaultdict[str, deque[float]] = defaultdict(deque)
        self._concurrent_count: int = 0

    @contextmanager
    def acquire(self, contract_key: str) -> Iterator[None]:
        """Block until all 3 rate-limit constraints permit, then record + yield.

        Args:
            contract_key: Unique identifier for the per-contract bucket. Typical
                format: `"<symbol>@<exchange>@<what_to_show>"` (e.g., `"ESM26@CME@TRADES"`).
                Empty string raises ValueError.

        Yields after slot reserved. On `__exit__`, decrements `concurrent_count`
        (even if the wrapped block raises).

        Raises:
            ValueError: empty contract_key
            BackoffQueueTimeout: could not acquire within
                `config.acquire_timeout_seconds`
        """
        if not contract_key:
            raise ValueError("BackoffQueue.acquire requires non-empty contract_key")

        self._wait_for_slot(contract_key)
        self._record_acquire(contract_key)
        try:
            yield
        finally:
            self._release()

    # ------------------------------------------------------------------------
    # Inspection (for tests + telemetry)
    # ------------------------------------------------------------------------

    @property
    def concurrent_count(self) -> int:
        """Current in-flight request count."""
        return self._concurrent_count

    def global_count(self) -> int:
        """Count of requests in current global window (after pruning)."""
        self._prune_global()
        return len(self._global_log)

    def per_contract_count(self, contract_key: str) -> int:
        """Count of requests for `contract_key` in current per-contract window."""
        self._prune_per_contract(contract_key)
        return len(self._per_contract_log[contract_key])

    # ------------------------------------------------------------------------
    # Internal blocking logic
    # ------------------------------------------------------------------------

    def _wait_for_slot(self, contract_key: str) -> None:
        """Block until all 3 constraints allow another request OR timeout expires."""
        deadline = self._clock.monotonic() + self._config.acquire_timeout_seconds

        while True:
            self._prune_global()
            self._prune_per_contract(contract_key)

            # Earliest-time-to-unblock across the 3 constraints
            sleep_secs = self._compute_required_sleep(contract_key)
            if sleep_secs <= 0:
                return  # all 3 constraints satisfied

            # Bail on timeout
            now = self._clock.monotonic()
            if now + sleep_secs > deadline:
                raise BackoffQueueTimeout(
                    f"BackoffQueue could not acquire slot for {contract_key!r} within "
                    f"{self._config.acquire_timeout_seconds}s timeout "
                    f"(would require additional {sleep_secs:.2f}s; "
                    f"global_count={len(self._global_log)} / {self._config.global_limit}, "
                    f"per_contract_count={len(self._per_contract_log[contract_key])} / "
                    f"{self._config.per_contract_limit}, "
                    f"concurrent={self._concurrent_count} / {self._config.concurrent_limit})"
                )

            self._sleep_fn(sleep_secs)

    def _compute_required_sleep(self, contract_key: str) -> float:
        """Return seconds to sleep before all 3 constraints clear. 0 if all satisfied."""
        now = self._clock.monotonic()
        candidates: list[float] = []

        # Constraint 1: global rolling window
        if len(self._global_log) >= self._config.global_limit:
            oldest = self._global_log[0]
            sleep_to_expire = (oldest + self._config.global_window.total_seconds()) - now
            candidates.append(max(sleep_to_expire, 0.0))

        # Constraint 2: per-contract rolling window
        contract_log = self._per_contract_log[contract_key]
        if len(contract_log) >= self._config.per_contract_limit:
            oldest = contract_log[0]
            sleep_to_expire = (oldest + self._config.per_contract_window.total_seconds()) - now
            candidates.append(max(sleep_to_expire, 0.0))

        # Constraint 3: concurrent in-flight
        if self._concurrent_count >= self._config.concurrent_limit:
            # No fixed expiry — caller-driven. Wake at a small interval to recheck.
            candidates.append(0.05)  # 50ms re-poll

        if not candidates:
            return 0.0
        return max(candidates)

    def _record_acquire(self, contract_key: str) -> None:
        """Reserve the slot — log timestamp + increment concurrent."""
        now = self._clock.monotonic()
        self._global_log.append(now)
        self._per_contract_log[contract_key].append(now)
        self._concurrent_count += 1

    def _release(self) -> None:
        """Decrement concurrent counter. Called via context-manager exit."""
        if self._concurrent_count <= 0:
            logger.warning("BackoffQueue release called when concurrent_count <= 0 — bug")
            return
        self._concurrent_count -= 1

    # ------------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------------

    def _prune_global(self) -> None:
        """Drop global-log entries older than the rolling window."""
        cutoff = self._clock.monotonic() - self._config.global_window.total_seconds()
        while self._global_log and self._global_log[0] < cutoff:
            self._global_log.popleft()

    def _prune_per_contract(self, contract_key: str) -> None:
        """Drop per-contract entries older than the rolling window."""
        cutoff = self._clock.monotonic() - self._config.per_contract_window.total_seconds()
        log = self._per_contract_log[contract_key]
        while log and log[0] < cutoff:
            log.popleft()
