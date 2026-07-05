"""COT data-axis types test suite (Ship 1: cot_types).

Test discipline:
- Dataclass validation (frozen, count >= 0, tz-aware boundary, sha length, total_traders).
- Normalized net properties (signed spec_net / comm_net).
- The 6-contract COT_CONTRACT_SPECS HYPOTHESIS mapping (structure, flavors, crypto_thin gate).
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from futur3.data.cot_types import (
    COT_CONTRACT_SPECS,
    COTContractSpec,
    COTReport,
    COTReportFlavor,
)
from futur3.data.types import content_sha256

ET = ZoneInfo("America/New_York")
_SHA = content_sha256(b"cot-fixture")


def _valid_report(
    *,
    cftc_contract_market_code: str = "067651",
    value_known_at_iso: datetime | None = None,
    as_of_iso: datetime | None = None,
    open_interest_all: int = 500_000,
    spec_long: int = 120_000,
    spec_short: int = 80_000,
    comm_long: int = 200_000,
    comm_short: int = 240_000,
    content_bytes_sha: str = _SHA,
    total_traders: int | None = 350,
) -> COTReport:
    return COTReport(
        cftc_contract_market_code=cftc_contract_market_code,
        flavor=COTReportFlavor.DISAGGREGATED,
        report_date=date(2026, 5, 19),  # a Tuesday
        value_known_at_iso=value_known_at_iso or datetime(2026, 5, 22, 15, 30, tzinfo=ET),
        open_interest_all=open_interest_all,
        spec_long=spec_long,
        spec_short=spec_short,
        comm_long=comm_long,
        comm_short=comm_short,
        source_id="cftc_socrata",
        as_of_iso=as_of_iso or datetime(2026, 5, 22, 20, 0, tzinfo=UTC),
        content_bytes_sha=content_bytes_sha,
        total_traders=total_traders,
    )


class TestCOTReportValidation:
    def test_valid_report_constructs(self) -> None:
        report = _valid_report()
        assert report.flavor is COTReportFlavor.DISAGGREGATED
        assert report.cftc_contract_market_code == "067651"

    def test_naive_value_known_at_raises(self) -> None:
        with pytest.raises(ValueError, match="value_known_at_iso"):
            _valid_report(value_known_at_iso=datetime(2026, 5, 22, 15, 30))

    def test_naive_as_of_raises(self) -> None:
        with pytest.raises(ValueError, match="as_of_iso"):
            _valid_report(as_of_iso=datetime(2026, 5, 22, 20, 0))

    def test_empty_market_code_raises(self) -> None:
        with pytest.raises(ValueError, match="cftc_contract_market_code"):
            _valid_report(cftc_contract_market_code="")

    @pytest.mark.parametrize(
        "field",
        ["open_interest_all", "spec_long", "spec_short", "comm_long", "comm_short"],
    )
    def test_negative_count_raises(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            _valid_report(**{field: -1})

    def test_total_traders_none_is_valid(self) -> None:
        assert _valid_report(total_traders=None).total_traders is None

    def test_total_traders_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="total_traders"):
            _valid_report(total_traders=-1)

    def test_bad_sha_length_raises(self) -> None:
        with pytest.raises(ValueError, match="content_bytes_sha"):
            _valid_report(content_bytes_sha="deadbeef")

    def test_frozen(self) -> None:
        report = _valid_report()
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.spec_long = 5  # type: ignore[misc]


class TestCOTNetProperties:
    def test_spec_net_positive(self) -> None:
        assert _valid_report(spec_long=120_000, spec_short=80_000).spec_net == 40_000

    def test_spec_net_negative(self) -> None:
        assert _valid_report(spec_long=50_000, spec_short=90_000).spec_net == -40_000

    def test_comm_net_signed(self) -> None:
        # comm is structurally the other side of spec in a hedger-vs-spec market.
        assert _valid_report(comm_long=200_000, comm_short=240_000).comm_net == -40_000


class TestCOTReportFlavor:
    def test_flavor_values(self) -> None:
        assert COTReportFlavor.TFF.value == "tff"
        assert COTReportFlavor.DISAGGREGATED.value == "disaggregated"
        assert COTReportFlavor.LEGACY.value == "legacy"
        assert COTReportFlavor.SUPPLEMENTAL.value == "supplemental"


class TestCOTContractSpecs:
    def test_all_six_roots_present(self) -> None:
        assert set(COT_CONTRACT_SPECS) == {"ES", "NQ", "CL", "GC", "MBT", "MET"}

    def test_each_spec_self_consistent(self) -> None:
        for root, spec in COT_CONTRACT_SPECS.items():
            assert spec.root == root
            assert spec.cftc_contract_market_code  # non-empty
            assert isinstance(spec.flavor, COTReportFlavor)

    def test_physical_commodities_are_disaggregated(self) -> None:
        assert COT_CONTRACT_SPECS["CL"].flavor is COTReportFlavor.DISAGGREGATED
        assert COT_CONTRACT_SPECS["GC"].flavor is COTReportFlavor.DISAGGREGATED

    def test_financials_are_tff(self) -> None:
        for root in ("ES", "NQ", "MBT", "MET"):
            assert COT_CONTRACT_SPECS[root].flavor is COTReportFlavor.TFF

    def test_crypto_thin_flagged_only_on_crypto(self) -> None:
        assert COT_CONTRACT_SPECS["MBT"].crypto_thin
        assert COT_CONTRACT_SPECS["MET"].crypto_thin
        assert not COT_CONTRACT_SPECS["ES"].crypto_thin
        assert not COT_CONTRACT_SPECS["CL"].crypto_thin

    def test_spec_empty_root_raises(self) -> None:
        with pytest.raises(ValueError, match="root"):
            COTContractSpec("", COTReportFlavor.TFF, "13874A")

    def test_spec_empty_code_raises(self) -> None:
        with pytest.raises(ValueError, match="cftc_contract_market_code"):
            COTContractSpec("ES", COTReportFlavor.TFF, "")
