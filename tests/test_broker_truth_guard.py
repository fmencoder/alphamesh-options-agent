"""Broker truth as a hard gate on new exposure.

On 2026-08-31 NVDA held an open bull call spread and a bear put order was
still allowed through: the journal and the Alpaca account had diverged, and
every duplicate gate consulted only the journal. New exposure now requires
BOTH the journal AND the account to permit it. Where they disagree the broker
wins for blocking, because the account is what actually holds the risk.
"""

from __future__ import annotations

import pytest

from alphamesh.alpaca.execution import SimulatedBroker
from alphamesh.alpaca.options import occ_underlying
from alphamesh.models.domain import ReasonCode, Strategy
from alphamesh.risk.governor import RiskGovernor
from tests.conftest import NOW, make_account, make_decision, make_portfolio
from tests.test_risk_governor import spread_with


@pytest.fixture
def governor(config):  # type: ignore[no-untyped-def]
    return RiskGovernor(config.risk, paper_confirmed=True)


@pytest.fixture
def single_symbol_config(config):  # type: ignore[no-untyped-def]
    return config.model_copy(
        update={"universe": config.universe.model_copy(update={"symbols": ["SPY"]})}
    )


def approve(governor, portfolio, symbol="NVDA", strategy=Strategy.BULL_CALL_SPREAD):  # type: ignore[no-untyped-def]
    return governor.approve(
        make_decision(symbol=symbol, strategy=strategy),
        spread_with(400, symbol=symbol),
        portfolio,
        NOW,
        f"client-order-{symbol}",
    )


class TestBrokerPositionBlocks:
    def test_broker_position_with_empty_journal_blocks(self, governor) -> None:  # type: ignore[no-untyped-def]
        """Requirement 1 — the exact NVDA divergence."""
        portfolio = make_portfolio(broker_position_symbols=frozenset({"NVDA"}))
        assert portfolio.open_positions == ()          # journal sees nothing
        assert portfolio.working_order_symbols == frozenset()

        result = approve(governor, portfolio)
        assert not result.approved
        assert ReasonCode.BROKER_OPEN_POSITION in result.reason_codes

    def test_broker_working_order_with_empty_journal_blocks(self, governor) -> None:  # type: ignore[no-untyped-def]
        """Requirement 2."""
        portfolio = make_portfolio(broker_working_symbols=frozenset({"NVDA"}))
        result = approve(governor, portfolio)
        assert not result.approved
        assert ReasonCode.BROKER_WORKING_ORDER in result.reason_codes


class TestDirectionIsIrrelevant:
    def test_opposite_direction_is_still_blocked(self, governor) -> None:  # type: ignore[no-untyped-def]
        """Requirement 3 — a bear put against an open bull call is still a
        second position on the same underlying."""
        portfolio = make_portfolio(broker_position_symbols=frozenset({"NVDA"}))
        result = approve(governor, portfolio, strategy=Strategy.BEAR_PUT_SPREAD)
        assert not result.approved
        assert ReasonCode.BROKER_OPEN_POSITION in result.reason_codes

    @pytest.mark.parametrize(
        "strategy", [Strategy.BULL_CALL_SPREAD, Strategy.BEAR_PUT_SPREAD]
    )
    def test_neither_direction_can_slip_past(self, governor, strategy) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(broker_working_symbols=frozenset({"NVDA"}))
        assert not approve(governor, portfolio, strategy=strategy).approved


class TestUnrelatedUnderlyingsStillTrade:
    def test_different_underlying_is_allowed(self, governor) -> None:  # type: ignore[no-untyped-def]
        """Requirement 4 — the guard must not become a global halt."""
        portfolio = make_portfolio(
            broker_position_symbols=frozenset({"NVDA"}),
            broker_working_symbols=frozenset({"AMD"}),
        )
        result = approve(governor, portfolio, symbol="SPY")
        assert result.approved, result.reason_codes


class TestAmbiguousStateFailsClosed:
    def test_unreadable_broker_refuses_new_exposure(self, governor) -> None:  # type: ignore[no-untyped-def]
        """An unreadable account is exactly the blind spot this closes."""
        portfolio = make_portfolio(broker_truth_available=False)
        result = approve(governor, portfolio, symbol="SPY")
        assert not result.approved
        assert ReasonCode.BROKER_OPEN_POSITION in result.reason_codes


class TestMismatchIsLogged:
    def test_state_mismatch_is_logged_and_journalled(self, single_symbol_config, caplog) -> None:  # type: ignore[no-untyped-def]
        """Requirement 5."""
        from tests.test_orchestrator import build

        class DivergentBroker(SimulatedBroker):
            def working_order_symbols(self):  # type: ignore[no-untyped-def]
                return frozenset({"NVDA"})

        orch, journal, _ = build(single_symbol_config, broker=DivergentBroker(make_account()))
        with caplog.at_level("WARNING"):
            state = orch.portfolio_state(NOW)

        assert state.broker_working_symbols == frozenset({"NVDA"})
        assert "exposure_state_mismatch" in caplog.text
        assert "BROKER_WINS_FOR_BLOCKING" in caplog.text
        events = [
            e
            for e in journal.recent_events()
            if e["event_type"] == "exposure_state_mismatch"
        ]
        assert events, "the mismatch must be journalled, not only logged"


class TestRestartCannotDuplicateExposure:
    def test_a_wiped_journal_cannot_reopen_the_same_symbol(self, governor) -> None:  # type: ignore[no-untyped-def]
        """Requirement 6 — the restart scenario, stated as state not history.

        A fresh journal knows nothing; only the broker remembers. That must be
        enough to refuse a second position on the same underlying.
        """
        after_restart = make_portfolio(
            open_positions=(),
            working_order_symbols=frozenset(),
            broker_position_symbols=frozenset({"NVDA"}),
            broker_working_symbols=frozenset({"AMD"}),
        )
        assert not approve(governor, after_restart, symbol="NVDA").approved
        assert not approve(governor, after_restart, symbol="AMD").approved
        assert approve(governor, after_restart, symbol="SPY").approved


class TestOccUnderlying:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("NVDA260902C00220000", "NVDA"),
            ("AMD260902P00470000", "AMD"),
            ("SPY260904C00765000", "SPY"),
            ("AAPL260902P00315000", "AAPL"),
        ],
    )
    def test_roots_are_extracted(self, symbol: str, expected: str) -> None:
        assert occ_underlying(symbol) == expected

    def test_garbage_is_rejected_rather_than_guessed(self) -> None:
        assert occ_underlying("NOTANOPTION") is None
