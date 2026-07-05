"""A1.8 MultiSourceVerifier test suite — fixture-based 3-source coverage.

Test discipline:
- Bit-repro anchor: same inputs → byte-equal `verifier_run_hash`.
- Source-order independence: passing sources in any order → same hash.
- Tier filter respected (T2+ only when policy.require_tier_t2_or_better).
- VerifiedSettle / VerifiedBar / DisagreementEvent / IncompleteBar
  emitted per spec §1 + §2.5.

References:
- futur3/data/verifier.py (implementation)
- futur3/data/verifier_policies.py (POLICY_PHASE_A1_DEFAULT)
- the verifier spec
- the verifier design
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import ClassVar

import pytest

from futur3.data.source import (
    BarsNotSupported,
    DataSource,
    DataSourceError,
    SettlesNotSupported,
    TicksNotSupported,
)
from futur3.data.types import (
    BarResolution,
    ContractSymbol,
    RawBar,
    RawTick,
    Settle,
    SettleState,
    SourceTier,
)
from futur3.data.verifier import (
    FIELD_OHLC,
    FIELD_SETTLE,
    DisagreementEvent,
    IncompleteBar,
    MultiSourceVerifier,
    VerifiedBar,
    VerifiedSettle,
    VerifierPolicy,
    _check_bar_consensus,
    _compute_verifier_run_hash,
    _source_provenance_hash,
    _tolerance_signature,
)
from futur3.data.verifier_policies import POLICY_PHASE_A1_DEFAULT

# ============================================================================
# Mock DataSources
# ============================================================================


@dataclass
class MockSettleSource(DataSource):
    """Returns a fixed Settle (or None) from latest_settle; raises configurable."""

    SOURCE_ID: ClassVar[str] = "mock"

    fixed_source_id: str = "mock_a"
    fixed_tier: SourceTier = SourceTier.T2_EXCHANGE
    settle_to_return: Settle | None = None
    raise_settles_not_supported: bool = False
    raise_data_source_error: bool = False
    bars_to_return: list[RawBar] = field(default_factory=list)
    raise_bars_not_supported: bool = False

    @property
    def source_id(self) -> str:
        return self.fixed_source_id

    @property
    def tier(self) -> SourceTier:
        return self.fixed_tier

    def get_bars(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
        resolution: BarResolution,
    ) -> Iterable[RawBar]:
        if self.raise_bars_not_supported:
            raise BarsNotSupported(f"{self.source_id}: configured raise")
        return [
            b
            for b in self.bars_to_return
            if ts_start <= b.ts < ts_end and b.resolution == resolution
        ]

    def get_ticks(
        self,
        contract: ContractSymbol,
        ts_start: datetime,
        ts_end: datetime,
    ) -> Iterable[RawTick]:
        raise TicksNotSupported(f"{self.source_id}: mock")

    def latest_settle(
        self,
        contract: ContractSymbol,
        as_of: datetime,
    ) -> Settle | None:
        if self.raise_settles_not_supported:
            raise SettlesNotSupported(f"{self.source_id}: configured raise")
        if self.raise_data_source_error:
            raise DataSourceError(f"{self.source_id}: configured raise")
        return self.settle_to_return


# ============================================================================
# Helpers
# ============================================================================


def _make_settle(
    *,
    contract: str = "ESM26",
    settle: str,
    settle_state: SettleState = "final",
    source_id: str,
    as_of_iso: datetime | None = None,
    content_bytes_sha: str | None = None,
    cme_month_code: str = "M",
) -> Settle:
    sha = content_bytes_sha or (source_id.encode().hex() * 16)[:64]
    return Settle(
        contract=ContractSymbol(contract),
        as_of_date=date(2026, 5, 21),
        settle=Decimal(settle),
        settle_state=settle_state,
        open=Decimal("5252.25"),
        high=Decimal("5263.50"),
        low=Decimal("5248.75"),
        last=Decimal(settle),
        change=Decimal("8.00"),
        volume_est=1_280_000,
        oi_prior=1_500_000,
        source_id=source_id,
        as_of_iso=as_of_iso or datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        content_bytes_sha=sha,
        cme_month_code=cme_month_code,
    )


def _make_bar(
    *,
    ts: datetime,
    close: str,
    source_id: str,
    content_bytes_sha: str | None = None,
    resolution: BarResolution = BarResolution.MIN_1,
) -> RawBar:
    sha = content_bytes_sha or (source_id.encode().hex() * 16)[:64]
    return RawBar(
        contract=ContractSymbol("ESM26"),
        ts=ts,
        resolution=resolution,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=1_000,
        oi=None,
        source_id=source_id,
        as_of_iso=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        content_bytes_sha=sha,
    )


def _settle_policy() -> VerifierPolicy:
    """Realistic settle policy: fail_closed, quorum=2, TIGHT, T2+ required."""
    return VerifierPolicy(
        policy_id="settle_exact_v1",
        kind="fail_closed",
        min_quorum=2,
        tolerance_band="TIGHT",
        timeout_seconds=60,
        require_tier_t2_or_better=True,
    )


def _ohlc_policy(*, min_quorum: int = 2) -> VerifierPolicy:
    """OHLC test policy: majority_vote, configurable quorum (2 for shell tests)."""
    return VerifierPolicy(
        policy_id="ohlc_test_v1",
        kind="majority_vote",
        min_quorum=min_quorum,
        tolerance_band="NORMAL",
        timeout_seconds=60,
        require_tier_t2_or_better=True,
    )


# ============================================================================
# TestA1_8_Imports — module + ABC sanity
# ============================================================================


class TestA1_8_Imports:
    def test_policy_phase_a1_default_assembled(self) -> None:
        p = POLICY_PHASE_A1_DEFAULT
        assert p.policy_id == "phase_a1_default_v1"
        assert p.kind == "fail_closed"
        assert p.min_quorum == 3
        # Per-field overrides present
        assert "settle" in p.per_field_policy
        assert "tick" in p.per_field_policy
        assert "ohlc" in p.per_field_policy
        assert "volume" in p.per_field_policy
        assert "oi" in p.per_field_policy

    def test_settle_policy_override_tight_quorum2(self) -> None:
        settle = POLICY_PHASE_A1_DEFAULT.resolve_for_field("settle")
        assert settle.policy_id == "settle_exact_v1"
        assert settle.kind == "fail_closed"
        assert settle.min_quorum == 2
        assert settle.tolerance_band == "TIGHT"

    def test_resolve_for_field_falls_back_to_umbrella(self) -> None:
        unknown = POLICY_PHASE_A1_DEFAULT.resolve_for_field("unknown_field")
        # umbrella is itself
        assert unknown.policy_id == POLICY_PHASE_A1_DEFAULT.policy_id

    def test_verifier_repr_lists_sources_and_policy(self) -> None:
        src_a = MockSettleSource(fixed_source_id="alpha")
        src_b = MockSettleSource(fixed_source_id="beta")
        v = MultiSourceVerifier([src_a, src_b], _settle_policy())
        rep = repr(v)
        assert "alpha" in rep and "beta" in rep
        assert "settle_exact_v1" in rep

    def test_verifier_sorts_sources_by_id(self) -> None:
        src_b = MockSettleSource(fixed_source_id="beta")
        src_a = MockSettleSource(fixed_source_id="alpha")
        v = MultiSourceVerifier([src_b, src_a], _settle_policy())
        assert v.source_ids == ("alpha", "beta")

    def test_empty_sources_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one DataSource"):
            MultiSourceVerifier([], _settle_policy())


# ============================================================================
# TestA1_8_VerifySettle — primary cross-source path
# ============================================================================


class TestA1_8_VerifySettle:
    def test_consensus_emits_verified_settle(self) -> None:
        s_a = _make_settle(settle="5260.00", source_id="cme_public_settlements")
        s_b = _make_settle(settle="5260.00", source_id="ibkr_tws_historical")
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            fixed_tier=SourceTier.T2_EXCHANGE,
            settle_to_return=s_a,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            settle_to_return=s_b,
        )
        v = MultiSourceVerifier([src_a, src_b], _settle_policy())
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert isinstance(result, VerifiedSettle)
        assert result.settle == Decimal("5260.00")
        assert result.n_sources_agreed == 2
        assert result.tier_used == SourceTier.T2_EXCHANGE  # min IntEnum
        assert len(result.verifier_run_hash) == 64
        assert len(result.source_provenance_hashes) == 2

    def test_disagreement_emits_event(self) -> None:
        s_a = _make_settle(settle="5260.00", source_id="cme_public_settlements")
        s_b = _make_settle(settle="5260.25", source_id="ibkr_tws_historical")
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            settle_to_return=s_a,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            settle_to_return=s_b,
        )
        v = MultiSourceVerifier([src_a, src_b], _settle_policy())
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert isinstance(result, DisagreementEvent)
        assert result.field == "settle"
        assert result.quorum_required == 2
        assert result.quorum_achieved == 2
        assert "5260.00" in result.values
        assert "5260.25" in result.values
        assert result.diff_magnitude_max == "0.25"

    def test_only_one_source_below_quorum_emits_incomplete(self) -> None:
        s_a = _make_settle(settle="5260.00", source_id="cme_public_settlements")
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            settle_to_return=s_a,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            settle_to_return=None,  # non-responding
        )
        v = MultiSourceVerifier([src_a, src_b], _settle_policy())
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert isinstance(result, IncompleteBar)
        assert result.n_sources_responded == 1
        assert result.n_sources_required == 2
        assert "ibkr_tws_historical" in result.sources_pending

    def test_settles_not_supported_treated_as_non_responding(self) -> None:
        s_a = _make_settle(settle="5260.00", source_id="cme_public_settlements")
        s_b = _make_settle(settle="5260.00", source_id="ibkr_tws_historical")
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            settle_to_return=s_a,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            settle_to_return=s_b,
        )
        src_c = MockSettleSource(
            fixed_source_id="crypto_no_settle",
            fixed_tier=SourceTier.T2_EXCHANGE,
            raise_settles_not_supported=True,
        )
        v = MultiSourceVerifier([src_a, src_b, src_c], _settle_policy())
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert isinstance(result, VerifiedSettle)
        assert result.n_sources_agreed == 2

    def test_data_source_error_treated_as_non_responding(self) -> None:
        s_a = _make_settle(settle="5260.00", source_id="cme_public_settlements")
        s_b = _make_settle(settle="5260.00", source_id="ibkr_tws_historical")
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            settle_to_return=s_a,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            settle_to_return=s_b,
        )
        src_bad = MockSettleSource(
            fixed_source_id="broken_source",
            fixed_tier=SourceTier.T2_EXCHANGE,
            raise_data_source_error=True,
        )
        v = MultiSourceVerifier([src_a, src_b, src_bad], _settle_policy())
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert isinstance(result, VerifiedSettle)
        assert result.n_sources_agreed == 2

    def test_tier_filter_excludes_t4_derived_by_default(self) -> None:
        s_a = _make_settle(settle="5260.00", source_id="cme_public_settlements")
        s_b = _make_settle(settle="5260.00", source_id="ibkr_tws_historical")
        s_replay = _make_settle(settle="5260.99", source_id="replay")  # would disagree IF counted
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            settle_to_return=s_a,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            settle_to_return=s_b,
        )
        src_replay = MockSettleSource(
            fixed_source_id="replay",
            fixed_tier=SourceTier.T4_DERIVED,
            settle_to_return=s_replay,
        )
        v = MultiSourceVerifier([src_a, src_b, src_replay], _settle_policy())
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        # T4 excluded → CME+IBKR agree → VerifiedSettle, NOT DisagreementEvent
        assert isinstance(result, VerifiedSettle)
        assert result.n_sources_agreed == 2

    def test_tier_filter_off_allows_t4_to_participate(self) -> None:
        s_a = _make_settle(settle="5260.00", source_id="cme_public_settlements")
        s_replay = _make_settle(settle="5260.00", source_id="replay")
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            settle_to_return=s_a,
        )
        src_replay = MockSettleSource(
            fixed_source_id="replay",
            fixed_tier=SourceTier.T4_DERIVED,
            settle_to_return=s_replay,
        )
        relaxed_policy = VerifierPolicy(
            policy_id="relaxed_v1",
            kind="fail_closed",
            min_quorum=2,
            tolerance_band="TIGHT",
            timeout_seconds=60,
            require_tier_t2_or_better=False,
        )
        v = MultiSourceVerifier([src_a, src_replay], relaxed_policy)
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert isinstance(result, VerifiedSettle)

    def test_settle_state_final_wins_over_preliminary(self) -> None:
        s_prelim = _make_settle(
            settle="5260.00",
            settle_state="preliminary",
            source_id="cme_public_settlements",
        )
        s_final = _make_settle(
            settle="5260.00",
            settle_state="final",
            source_id="ibkr_tws_historical",
        )
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            settle_to_return=s_prelim,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            settle_to_return=s_final,
        )
        v = MultiSourceVerifier([src_a, src_b], _settle_policy())
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert isinstance(result, VerifiedSettle)
        assert result.settle_state == "final"

    def test_tier_used_lowest_intenum_among_agreed(self) -> None:
        s_a = _make_settle(settle="5260.00", source_id="cme_public_settlements")
        s_b = _make_settle(settle="5260.00", source_id="ibkr_tws_historical")
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            fixed_tier=SourceTier.T2_EXCHANGE,  # 2
            settle_to_return=s_a,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,  # 3
            settle_to_return=s_b,
        )
        v = MultiSourceVerifier([src_a, src_b], _settle_policy())
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert isinstance(result, VerifiedSettle)
        assert result.tier_used == SourceTier.T2_EXCHANGE


# ============================================================================
# TestA1_8_DeterminismAnchor — verifier_run_hash bit-reproducibility anchor
# ============================================================================


class TestA1_8_DeterminismAnchor:
    def _setup(self) -> tuple[MockSettleSource, MockSettleSource]:
        s_a = _make_settle(settle="5260.00", source_id="cme_public_settlements")
        s_b = _make_settle(settle="5260.00", source_id="ibkr_tws_historical")
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            settle_to_return=s_a,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            settle_to_return=s_b,
        )
        return src_a, src_b

    def test_same_inputs_same_run_hash(self) -> None:
        src_a, src_b = self._setup()
        v1 = MultiSourceVerifier([src_a, src_b], _settle_policy())
        v2 = MultiSourceVerifier([src_a, src_b], _settle_policy())
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        r1 = v1.verify_settle(ContractSymbol("ESM26"), as_of)
        r2 = v2.verify_settle(ContractSymbol("ESM26"), as_of)
        assert isinstance(r1, VerifiedSettle)
        assert isinstance(r2, VerifiedSettle)
        assert r1.verifier_run_hash == r2.verifier_run_hash

    def test_source_order_independence(self) -> None:
        """Verifier sorts sources by source_id internally → input order doesn't change hash."""
        src_a, src_b = self._setup()
        v1 = MultiSourceVerifier([src_a, src_b], _settle_policy())
        v2 = MultiSourceVerifier([src_b, src_a], _settle_policy())  # reversed
        as_of = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        r1 = v1.verify_settle(ContractSymbol("ESM26"), as_of)
        r2 = v2.verify_settle(ContractSymbol("ESM26"), as_of)
        assert isinstance(r1, VerifiedSettle)
        assert isinstance(r2, VerifiedSettle)
        assert r1.verifier_run_hash == r2.verifier_run_hash

    def test_provenance_hashes_are_sha256_hex(self) -> None:
        src_a, src_b = self._setup()
        v = MultiSourceVerifier([src_a, src_b], _settle_policy())
        result = v.verify_settle(
            ContractSymbol("ESM26"),
            datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
        )
        assert isinstance(result, VerifiedSettle)
        for h in result.source_provenance_hashes:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_tolerance_signature_stable(self) -> None:
        """Same policy fields → same tolerance signature."""
        policy_a = _settle_policy()
        policy_b = _settle_policy()
        assert _tolerance_signature(policy_a) == _tolerance_signature(policy_b)
        assert len(_tolerance_signature(policy_a)) == 64

    def test_source_provenance_hash_deterministic(self) -> None:
        ts = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        sha = "a" * 64
        h1 = _source_provenance_hash("cme_public_settlements", ts, sha)
        h2 = _source_provenance_hash("cme_public_settlements", ts, sha)
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_verifier_run_hash_sort_independent(self) -> None:
        policy = _settle_policy()
        prov = ("aaa" * 21 + "a", "bbb" * 21 + "b")  # both 64-hex
        bar_id = "ESM26||2026-05-21||settle"
        h1 = _compute_verifier_run_hash(
            resolved_policy=policy,
            source_provenance_hashes=prov,
            bar_identifier=bar_id,
        )
        h2 = _compute_verifier_run_hash(
            resolved_policy=policy,
            source_provenance_hashes=(prov[1], prov[0]),  # reversed
            bar_identifier=bar_id,
        )
        assert h1 == h2


# ============================================================================
# TestA1_8_VerifyBar — secondary path
# ============================================================================


class TestA1_8_VerifyBar:
    def test_bar_consensus_emits_verified_bar(self) -> None:
        bar_ts = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        bar_a = _make_bar(ts=bar_ts, close="5260.00", source_id="ibkr_tws_historical")
        bar_b = _make_bar(ts=bar_ts, close="5260.00", source_id="alt_source")
        src_a = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            bars_to_return=[bar_a],
        )
        src_b = MockSettleSource(
            fixed_source_id="alt_source",
            fixed_tier=SourceTier.T2_EXCHANGE,
            bars_to_return=[bar_b],
        )
        v = MultiSourceVerifier([src_a, src_b], _ohlc_policy(min_quorum=2))
        result = v.verify_bar(ContractSymbol("ESM26"), bar_ts, BarResolution.MIN_1)
        assert isinstance(result, VerifiedBar)
        assert result.close == Decimal("5260.00")
        assert result.n_sources_agreed == 2
        assert result.prev_bar_hash is None  # first bar in chain

    def test_bar_chain_prev_hash_set_on_second(self) -> None:
        bar_ts_1 = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        bar_ts_2 = datetime(2026, 5, 21, 22, 1, 0, tzinfo=UTC)
        src_a = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            bars_to_return=[
                _make_bar(ts=bar_ts_1, close="5260.00", source_id="ibkr_tws_historical"),
                _make_bar(ts=bar_ts_2, close="5261.00", source_id="ibkr_tws_historical"),
            ],
        )
        src_b = MockSettleSource(
            fixed_source_id="alt_source",
            fixed_tier=SourceTier.T2_EXCHANGE,
            bars_to_return=[
                _make_bar(ts=bar_ts_1, close="5260.00", source_id="alt_source"),
                _make_bar(ts=bar_ts_2, close="5261.00", source_id="alt_source"),
            ],
        )
        v = MultiSourceVerifier([src_a, src_b], _ohlc_policy(min_quorum=2))
        r1 = v.verify_bar(ContractSymbol("ESM26"), bar_ts_1, BarResolution.MIN_1)
        r2 = v.verify_bar(ContractSymbol("ESM26"), bar_ts_2, BarResolution.MIN_1)
        assert isinstance(r1, VerifiedBar)
        assert isinstance(r2, VerifiedBar)
        assert r1.prev_bar_hash is None
        assert r2.prev_bar_hash == r1.verifier_run_hash

    def test_bar_disagreement_emits_event(self) -> None:
        bar_ts = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        src_a = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            bars_to_return=[_make_bar(ts=bar_ts, close="5260.00", source_id="ibkr_tws_historical")],
        )
        src_b = MockSettleSource(
            fixed_source_id="alt_source",
            fixed_tier=SourceTier.T2_EXCHANGE,
            bars_to_return=[_make_bar(ts=bar_ts, close="5260.50", source_id="alt_source")],
        )
        v = MultiSourceVerifier([src_a, src_b], _ohlc_policy(min_quorum=2))
        result = v.verify_bar(ContractSymbol("ESM26"), bar_ts, BarResolution.MIN_1)
        assert isinstance(result, DisagreementEvent)
        assert result.field == "close"
        assert "5260.00" in result.values
        assert "5260.50" in result.values
        assert result.diff_magnitude_max == "0.50"

    def test_bar_incomplete_when_only_one_source_returns(self) -> None:
        bar_ts = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        src_a = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            bars_to_return=[_make_bar(ts=bar_ts, close="5260.00", source_id="ibkr_tws_historical")],
        )
        src_b = MockSettleSource(
            fixed_source_id="alt_source",
            fixed_tier=SourceTier.T2_EXCHANGE,
            bars_to_return=[],  # empty
        )
        v = MultiSourceVerifier([src_a, src_b], _ohlc_policy(min_quorum=2))
        result = v.verify_bar(ContractSymbol("ESM26"), bar_ts, BarResolution.MIN_1)
        assert isinstance(result, IncompleteBar)
        assert result.n_sources_responded == 1
        assert "alt_source" in result.sources_pending

    def test_bars_not_supported_treated_as_non_responding(self) -> None:
        bar_ts = datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC)
        src_a = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            bars_to_return=[_make_bar(ts=bar_ts, close="5260.00", source_id="ibkr_tws_historical")],
        )
        src_b = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            fixed_tier=SourceTier.T2_EXCHANGE,
            raise_bars_not_supported=True,
        )
        src_c = MockSettleSource(
            fixed_source_id="alt_source",
            fixed_tier=SourceTier.T2_EXCHANGE,
            bars_to_return=[_make_bar(ts=bar_ts, close="5260.00", source_id="alt_source")],
        )
        v = MultiSourceVerifier([src_a, src_b, src_c], _ohlc_policy(min_quorum=2))
        result = v.verify_bar(ContractSymbol("ESM26"), bar_ts, BarResolution.MIN_1)
        assert isinstance(result, VerifiedBar)
        assert result.n_sources_agreed == 2

    def test_naive_bar_ts_raises(self) -> None:
        src = MockSettleSource(fixed_source_id="x")
        v = MultiSourceVerifier([src], _ohlc_policy(min_quorum=1))
        with pytest.raises(ValueError, match="TZ-aware"):
            v.verify_bar(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0),  # naive
                BarResolution.MIN_1,
            )

    def test_unsupported_resolution_raises(self) -> None:
        src = MockSettleSource(fixed_source_id="x")
        v = MultiSourceVerifier([src], _ohlc_policy(min_quorum=1))
        with pytest.raises(ValueError, match="not supported"):
            v.verify_bar(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                BarResolution.SETTLE,
            )


# ============================================================================
# TestA1_8_DataclassValidation — defensive __post_init__ checks
# ============================================================================


class TestA1_8_DataclassValidation:
    def test_verified_settle_bad_run_hash_length_raises(self) -> None:
        with pytest.raises(ValueError, match="hex-SHA256"):
            VerifiedSettle(
                contract=ContractSymbol("ESM26"),
                as_of_date=date(2026, 5, 21),
                settle=Decimal("5260.00"),
                settle_state="final",
                n_sources_agreed=2,
                source_provenance_hashes=("a" * 64,),
                verifier_run_hash="too_short",  # invalid
                tier_used=SourceTier.T2_EXCHANGE,
                policy_id="x",
            )

    def test_verified_settle_bad_provenance_length_raises(self) -> None:
        with pytest.raises(ValueError, match="hex-SHA256"):
            VerifiedSettle(
                contract=ContractSymbol("ESM26"),
                as_of_date=date(2026, 5, 21),
                settle=Decimal("5260.00"),
                settle_state="final",
                n_sources_agreed=1,
                source_provenance_hashes=("short",),  # invalid
                verifier_run_hash="a" * 64,
                tier_used=SourceTier.T2_EXCHANGE,
                policy_id="x",
            )

    def test_verified_bar_naive_ts_raises(self) -> None:
        with pytest.raises(ValueError, match="TZ-aware"):
            VerifiedBar(
                contract=ContractSymbol("ESM26"),
                ts=datetime(2026, 5, 21, 22, 0, 0),  # naive
                resolution=BarResolution.MIN_1,
                open=Decimal("0"),
                high=Decimal("0"),
                low=Decimal("0"),
                close=Decimal("0"),
                volume=0,
                n_sources_agreed=1,
                source_provenance_hashes=("a" * 64,),
                verifier_run_hash="a" * 64,
                prev_bar_hash=None,
                tier_used=SourceTier.T2_EXCHANGE,
                policy_id="x",
            )

    def test_disagreement_event_sources_values_len_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="!="):
            DisagreementEvent(
                contract=ContractSymbol("ESM26"),
                bar_ts=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                resolution=BarResolution.SETTLE,
                field="settle",
                sources=("a", "b"),
                values=("5260.00",),  # mismatched len
                diff_magnitude_max="0.0",
                tolerance_used="TIGHT",
                policy_kind="fail_closed",
                quorum_required=2,
                quorum_achieved=2,
                detected_at=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                policy_id="x",
            )

    def test_disagreement_event_naive_detected_at_raises(self) -> None:
        with pytest.raises(ValueError, match="TZ-aware"):
            DisagreementEvent(
                contract=ContractSymbol("ESM26"),
                bar_ts=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                resolution=BarResolution.SETTLE,
                field="settle",
                sources=("a",),
                values=("5260.00",),
                diff_magnitude_max="0.0",
                tolerance_used="TIGHT",
                policy_kind="fail_closed",
                quorum_required=2,
                quorum_achieved=1,
                detected_at=datetime(2026, 5, 21, 22, 0, 0),  # naive
                policy_id="x",
            )

    def test_incomplete_bar_naive_detected_at_raises(self) -> None:
        with pytest.raises(ValueError, match="TZ-aware"):
            IncompleteBar(
                contract=ContractSymbol("ESM26"),
                bar_ts=datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
                resolution=BarResolution.SETTLE,
                n_sources_responded=1,
                n_sources_required=2,
                sources_pending=("x",),
                detected_at=datetime(2026, 5, 21, 22, 0, 0),  # naive
                policy_id="x",
            )


# ============================================================================
# TestA1_8_FieldConstants — labels used by per-field policy lookup
# ============================================================================


class TestA1_8_FieldConstants:
    def test_field_settle_constant(self) -> None:
        assert FIELD_SETTLE == "settle"

    def test_field_ohlc_constant(self) -> None:
        assert FIELD_OHLC == "ohlc"


# ============================================================================
# TestVerifierCorrective — corrective fixes from an internal audit
# ============================================================================


class TestVerifierCorrective:
    """Corrective batch — fail-loud fixes surfaced by an internal audit.

    C1: `_check_bar_consensus` empty-list IndexError → AssertionError (verifier.py:416)
    M4: `verify_settle` None as_of_iso fallback → DataSourceError (verifier.py:586)
    """

    def test_c1_check_bar_consensus_empty_raises_assertion_error(self) -> None:
        """C1: empty bars list must raise AssertionError, not crash with IndexError.

        Defensive guard for caller invariant: verify_bar guarantees min_quorum>=1.
        If we reach _check_bar_consensus with [] it's a programming error;
        raise explicitly instead of crashing on `bars[0]`.
        """
        with pytest.raises(AssertionError, match="empty bars"):
            _check_bar_consensus([], _ohlc_policy())

    def test_m4_verify_settle_with_none_as_of_iso_raises_data_source_error(self) -> None:
        """M4: settle.as_of_iso=None must raise DataSourceError (not fall back to caller as_of).

        Fail-loud: hash determinism requires as_of_iso. Silent fallback to caller as_of
        would let two verify_settle calls with identical settle data produce
        different verifier_run_hash values, breaking bit-reproducibility.
        """
        # Construct Settle directly (bypassing _make_settle's `or` fallback) so
        # as_of_iso=None is preserved on the way to the verifier.
        bad_settle_a = Settle(
            contract=ContractSymbol("ESM26"),
            as_of_date=date(2026, 5, 21),
            settle=Decimal("5260.00"),
            settle_state="final",
            open=Decimal("5252.25"),
            high=Decimal("5263.50"),
            low=Decimal("5248.75"),
            last=Decimal("5260.00"),
            change=Decimal("8.00"),
            volume_est=1_280_000,
            oi_prior=1_500_000,
            source_id="cme_public_settlements",
            as_of_iso=None,  # ← intentionally None
            content_bytes_sha=("aa" * 32),
            cme_month_code="M",
        )
        bad_settle_b = Settle(
            contract=ContractSymbol("ESM26"),
            as_of_date=date(2026, 5, 21),
            settle=Decimal("5260.00"),
            settle_state="final",
            open=Decimal("5252.25"),
            high=Decimal("5263.50"),
            low=Decimal("5248.75"),
            last=Decimal("5260.00"),
            change=Decimal("8.00"),
            volume_est=1_280_000,
            oi_prior=1_500_000,
            source_id="ibkr_tws_historical",
            as_of_iso=None,  # ← intentionally None
            content_bytes_sha=("bb" * 32),
            cme_month_code="M",
        )
        src_a = MockSettleSource(
            fixed_source_id="cme_public_settlements",
            fixed_tier=SourceTier.T2_EXCHANGE,
            settle_to_return=bad_settle_a,
        )
        src_b = MockSettleSource(
            fixed_source_id="ibkr_tws_historical",
            fixed_tier=SourceTier.T2_BROKER,
            settle_to_return=bad_settle_b,
        )
        v = MultiSourceVerifier([src_a, src_b], _settle_policy())
        with pytest.raises(DataSourceError, match="as_of_iso is None"):
            v.verify_settle(
                ContractSymbol("ESM26"),
                datetime(2026, 5, 21, 22, 0, 0, tzinfo=UTC),
            )
