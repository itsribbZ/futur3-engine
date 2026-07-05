"""pytest fixtures for futur3 tests.

Phase A1 baseline: import-validation + smoke tests + Hypothesis property tests +
shared fixtures for fixture-based DataSource testing (HTTPClient injection pattern).

Test discipline:
- ALL tests in default `pytest` run hit ZERO live endpoints.
- Live-network smoke is gated behind `@pytest.mark.integration`.
- Fixture HTMLs are synthetic-but-realistic; never real CME quotes.
- Fixture-based HTTPClient pattern makes the entire parse + archive surface testable.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# ============================================================================
# Test-time HTTPClient (fixture loader; zero network)
# ============================================================================


class FixtureHTTPClient:
    """HTTPClient implementation backed by on-disk fixture HTML files.

    Maps URL paths → fixture filenames; loads bytes; computes SHA256.

    Used in fixture-based tests as a substitute for `_DefaultCMEHTTPClient`,
    which would hit the live network with curl_cffi.
    """

    def __init__(
        self,
        url_to_fixture: dict[str, str],
        fixtures_dir: Path,
    ) -> None:
        """
        Args:
            url_to_fixture: Mapping of URL (or URL substring) → fixture filename
                (relative to fixtures_dir). e.g.:
                {"e-mini-sandp500.settlements.html": "es_jun26_preliminary.html"}
            fixtures_dir: Directory containing fixture HTML files.
        """
        self._url_to_fixture = url_to_fixture
        self._fixtures_dir = fixtures_dir

    def fetch(self, url: str) -> tuple[bytes, str]:
        """Return (fixture_bytes, sha256_hex) for the matching URL."""
        for url_match, filename in self._url_to_fixture.items():
            if url_match in url:
                path = self._fixtures_dir / filename
                if not path.exists():
                    raise FileNotFoundError(
                        f"FixtureHTTPClient: missing fixture file {path} "
                        f"for URL pattern {url_match!r}"
                    )
                content = path.read_bytes()
                sha = hashlib.sha256(content).hexdigest()
                return content, sha
        raise KeyError(
            f"FixtureHTTPClient: no fixture mapping for URL {url!r}. "
            f"Configured patterns: {sorted(self._url_to_fixture)}"
        )

    def healthcheck(self, url: str) -> bool:
        """Always returns True for fixture tests."""
        return True


# ============================================================================
# Test-time Clock (deterministic UTC datetime)
# ============================================================================


class FakeClock:
    """ClockProtocol implementation returning a fixed datetime.

    Tests asserting on `as_of_iso` content can rely on this being deterministic.
    """

    def __init__(self, fixed_time: datetime) -> None:
        if fixed_time.tzinfo is None:
            raise ValueError(f"FakeClock requires TZ-aware datetime; got naive {fixed_time!r}")
        self._fixed_time = fixed_time

    def now_utc(self) -> datetime:
        return self._fixed_time

    def advance(self, *, seconds: float = 0.0) -> None:
        """Tick the clock forward (for tests of preliminary→final transitions etc.)."""
        self._fixed_time = self._fixed_time + timedelta(seconds=seconds)


# ============================================================================
# Shared pytest fixtures
# ============================================================================


@pytest.fixture
def cme_fixtures_dir() -> Path:
    """Path to the on-disk CME EOD fixture HTMLs."""
    return Path(__file__).parent / "fixtures" / "cme_eod"


@pytest.fixture
def fake_clock_s7() -> FakeClock:
    """Fixed clock at the fixture reference time (2026-05-21T22:00:00Z; 17:00 CT post-publish)."""
    return FakeClock(datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC))


@pytest.fixture
def cme_archive_tmp(tmp_path: Path) -> Path:
    """Temporary archive root for Parquet write tests. tmp_path is per-test isolated."""
    archive = tmp_path / "cme_eod_archive"
    archive.mkdir()
    return archive


@pytest.fixture
def es_only_http_client(cme_fixtures_dir: Path) -> FixtureHTTPClient:
    """HTTPClient that serves ES preliminary fixture for any ES URL."""
    return FixtureHTTPClient(
        url_to_fixture={"e-mini-sandp500.settlements.html": "es_jun26_preliminary.html"},
        fixtures_dir=cme_fixtures_dir,
    )


@pytest.fixture
def all_contracts_http_client(cme_fixtures_dir: Path) -> FixtureHTTPClient:
    """HTTPClient that serves the canonical fixture for each contract root.

    Micro contracts (MES/MNQ/MCL/MGC) reuse their parent's fixture per A1.3.2 design
    (schema identical; only URL routing differs).
    """
    return FixtureHTTPClient(
        url_to_fixture={
            "e-mini-sandp500.settlements.html": "es_jun26_preliminary.html",
            "e-mini-nasdaq-100.settlements.html": "nq_jun26_preliminary.html",
            "light-sweet-crude.settlements.html": "cl_jul26_preliminary.html",
            "gold.settlements.html": "gc_aug26_preliminary.html",
            "micro-bitcoin.settlements.html": "mbt_jun26_preliminary.html",
            "micro-ether.settlements.html": "met_jun26_preliminary.html",
            # Micro variants — reuse parent fixtures (same schema, same prices)
            "micro-e-mini-sandp-500.settlements.html": "es_jun26_preliminary.html",
            "micro-e-mini-nasdaq-100.settlements.html": "nq_jun26_preliminary.html",
            "micro-crude-oil.settlements.html": "cl_jul26_preliminary.html",
            "micro-gold.settlements.html": "gc_aug26_preliminary.html",
        },
        fixtures_dir=cme_fixtures_dir,
    )
