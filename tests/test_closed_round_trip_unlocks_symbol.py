"""A closed round trip must give the symbol back.

Production evidence, 2026-09-03: every cycle scanned 8 symbols, the quant gate
passed 610 of them, the AI council returned 420 tradable, 409 spreads were
built and ranked -- and the Risk Governor rejected every single one with
DUPLICATE_ORDER. Zero orders were submitted all session. The account was flat:
no positions, no working orders. The journal, however, reported a working order
on all eight universe symbols.

The cause was the definition of "working order", not the governor. Opening a
position moves its ENTRY order to MONITORING; closing the position writes the
outcome and retires the EXIT order, and nothing ever retires the entry order.
``open_orders()`` returns everything that is not CLOSED/REJECTED/FAILED, so
that MONITORING entry row was still reported as a live working order long after
the position behind it had been closed and settled. One completed round trip
therefore locked its symbol out of trading permanently.
"""

from __future__ import annotations

import pytest

from alphamesh.alpaca.client import AlpacaStack
from alphamesh.alpaca.execution import SimulatedBroker
from alphamesh.alpaca.options import CaptureOptionChain
from alphamesh.execution.order_builder import build_order_intent
from alphamesh.models.domain import (
    WORKING_ORDER_STATES,
    ReasonCode,
    RiskDecision,
    TradeState,
)
from alphamesh.orchestrator import Orchestrator
from alphamesh.persistence.journal import Journal
from alphamesh.risk.governor import RiskGovernor
from alphamesh.safety import GuardResult
from tests.conftest import CAPTURE_DIR, NOW, make_account, make_decision
from tests.test_orchestrator import TrendingMarketData, bull_script
from tests.test_risk_governor import spread_with

SYMBOL = "QQQ"


def build_orchestrator(config):  # type: ignore[no-untyped-def]
    """An orchestrator whose broker is flat, exactly as production was."""
    broker = SimulatedBroker(make_account())
    stack = AlpacaStack(
        guard=GuardResult(paper=True, detail="test", checks=("ALPACA_PAPER=true",)),
        market_data=TrendingMarketData(),
        option_chain=CaptureOptionChain(CAPTURE_DIR),
        broker=broker,
        live_broker=False,
    )
    journal = Journal(":memory:")
    return Orchestrator(config, stack, journal, bull_script()), journal


def seed_entry_order(journal: Journal, state: TradeState, symbol: str = SYMBOL) -> str:
    """Put one entry order in the journal in the given state, nothing at the broker."""
    intent = build_order_intent(
        make_decision(symbol=symbol),
        spread_with(400, symbol=symbol),
        RiskDecision(
            approved=True,
            quantity=1,
            max_loss_cents=40_000,
            max_profit_cents=10_000,
            reason_codes=(),
            detail="test",
            checks_run=("test",),
        ),
        NOW,
    )
    assert journal.reserve_order(intent)
    if state is not TradeState.CONSTRUCTED:
        journal.set_order_state(intent.client_order_id, state, "seeded for test")
    return intent.client_order_id


class TestSettledEntryOrdersReleaseTheSymbol:
    """The exact production state: a settled entry order, a flat account."""

    @pytest.mark.parametrize("state", [TradeState.MONITORING, TradeState.FILLED])
    def test_settled_entry_order_is_not_a_working_order(self, config, state) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal = build_orchestrator(config)
        cid = seed_entry_order(journal, state)

        portfolio = orchestrator.portfolio_state(NOW)

        # The row is still on the books -- recovery and the audit trail need it.
        assert cid in {o["client_order_id"] for o in journal.open_orders()}
        # It is not exposure in waiting, so it must not hold the duplicate lock.
        assert portfolio.working_order_symbols == frozenset()
        assert not portfolio.has_working_order_for(SYMBOL)

    @pytest.mark.parametrize("state", [TradeState.MONITORING, TradeState.FILLED])
    def test_governor_approves_the_symbol_again(self, config, state) -> None:  # type: ignore[no-untyped-def]
        """The zero-order session, end to end: flat account, settled journal row,
        a qualified candidate must reach RISK_APPROVED."""
        orchestrator, journal = build_orchestrator(config)
        seed_entry_order(journal, state)
        governor = RiskGovernor(config.risk, paper_confirmed=True)

        result = governor.approve(
            make_decision(symbol=SYMBOL),
            spread_with(400, symbol=SYMBOL),
            orchestrator.portfolio_state(NOW),
            NOW,
            f"client-order-{SYMBOL}-next",
        )

        assert ReasonCode.DUPLICATE_ORDER not in result.reason_codes
        assert result.approved, result.reason_codes


class TestLiveOrdersStillLockTheSymbol:
    """Nothing above may weaken duplicate protection while an order is real."""

    @pytest.mark.parametrize(
        "state",
        [
            TradeState.CONSTRUCTED,
            TradeState.SUBMITTED,
            TradeState.PARTIALLY_FILLED,
            TradeState.EXIT_REQUESTED,
        ],
    )
    def test_live_order_still_blocks(self, config, state) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal = build_orchestrator(config)
        seed_entry_order(journal, state)

        portfolio = orchestrator.portfolio_state(NOW)
        assert portfolio.has_working_order_for(SYMBOL)

        governor = RiskGovernor(config.risk, paper_confirmed=True)
        result = governor.approve(
            make_decision(symbol=SYMBOL),
            spread_with(400, symbol=SYMBOL),
            portfolio,
            NOW,
            f"client-order-{SYMBOL}-next",
        )
        assert not result.approved
        assert ReasonCode.DUPLICATE_ORDER in result.reason_codes

    def test_unreadable_state_fails_closed(self, config) -> None:  # type: ignore[no-untyped-def]
        """A state we cannot parse must block, not wave the symbol through."""
        orchestrator, journal = build_orchestrator(config)
        cid = seed_entry_order(journal, TradeState.SUBMITTED)
        with journal.transaction() as conn:
            conn.execute(
                "UPDATE orders SET state = ? WHERE client_order_id = ?",
                ("NOT_A_STATE", cid),
            )

        assert orchestrator.portfolio_state(NOW).has_working_order_for(SYMBOL)


class TestWorkingStateSet:
    def test_settled_states_are_excluded(self) -> None:
        assert TradeState.MONITORING not in WORKING_ORDER_STATES
        assert TradeState.FILLED not in WORKING_ORDER_STATES

    def test_terminal_states_are_excluded(self) -> None:
        for state in (TradeState.CLOSED, TradeState.REJECTED, TradeState.FAILED):
            assert state not in WORKING_ORDER_STATES
