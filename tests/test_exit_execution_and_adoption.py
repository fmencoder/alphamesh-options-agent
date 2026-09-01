"""Regression tests for the phantom-exit defect and broker position adoption.

The defect these pin down: ``_exit_position`` wrote a CLOSED position and a
realised P&L computed from a *mark*, without ever sending a closing order.
``close_spread`` existed and was tested, but nothing called it. The account
therefore kept nine spreads the journal believed were closed, no exit rule ran
against them, and the broker-truth entry guard -- working exactly as designed --
blocked every underlying in the universe because the positions never went away.

Each test below names the property it protects. None of them assert on the
shape of the fix; they assert on what the broker was asked to do and on where
realised money came from.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from alphamesh.alpaca.client import AlpacaStack
from alphamesh.alpaca.execution import SimulatedBroker
from alphamesh.alpaca.options import CaptureOptionChain
from alphamesh.alpaca.types import (
    BrokerOrderLeg,
    BrokerOrderSummary,
    BrokerPosition,
)
from alphamesh.execution.adoption import (
    ENTRY_BASIS_ORDER_FILL,
    ENTRY_BASIS_POSITION_COST,
    reconstruct_spreads,
)
from alphamesh.intelligence.reasoning import NullProvider
from alphamesh.models.domain import (
    OPTION_MULTIPLIER,
    ReasonCode,
    Regime,
    Strategy,
    TradeState,
)
from alphamesh.orchestrator import Orchestrator
from alphamesh.persistence.journal import (
    ORDER_KIND_ENTRY,
    ORDER_KIND_EXIT,
    PHANTOM_CLOSE_NOTE,
    Journal,
)
from alphamesh.risk.governor import RiskGovernor
from alphamesh.safety import GuardResult
from tests.conftest import CAPTURE_DIR, NOW, make_account, make_portfolio
from tests.test_orchestrator import TrendingMarketData, bull_script

# Real symbols from the captured Alpaca chain, so marks and closing orders run
# against contracts that actually exist rather than invented ones.
SPY_LONG = "SPY260903C00762000"
SPY_SHORT = "SPY260903C00767000"


def build(
    config,  # type: ignore[no-untyped-def]
    broker=None,  # type: ignore[no-untyped-def]
    journal=None,  # type: ignore[no-untyped-def]
    market=None,  # type: ignore[no-untyped-def]
    provider=None,  # type: ignore[no-untyped-def]
):
    stack = AlpacaStack(
        guard=GuardResult(paper=True, detail="test", checks=("ALPACA_PAPER=true",)),
        market_data=market or TrendingMarketData(),
        option_chain=CaptureOptionChain(CAPTURE_DIR),
        broker=broker or SimulatedBroker(make_account()),
        live_broker=False,
    )
    j = journal or Journal(":memory:")
    return Orchestrator(config, stack, j, provider or bull_script()), j, stack


@pytest.fixture
def spy_config(config):  # type: ignore[no-untyped-def]
    return config.model_copy(
        update={"universe": config.universe.model_copy(update={"symbols": ["SPY"]})}
    )


def option_leg(symbol: str, quantity: int, avg_price: float) -> BrokerPosition:
    return BrokerPosition(
        symbol=symbol,
        quantity=quantity,
        avg_entry_price=avg_price,
        market_value=avg_price * OPTION_MULTIPLIER * quantity,
        unrealized_pl=0.0,
    )


def spy_bull_call_legs(quantity: int = 1) -> list[BrokerPosition]:
    """One long 762 call against one short 767 call: a $5-wide bull call spread."""
    return [
        option_leg(SPY_LONG, quantity, 8.99),
        option_leg(SPY_SHORT, -quantity, 5.06),
    ]


def write_phantom_close(
    journal, quantity: int = 1, realized_pnl_cents: int = -12_600
):  # type: ignore[no-untyped-def]
    """Write the exact rows the defect produced: CLOSED with no closing order.

    A position and an outcome exist, the position reads CLOSED, and no exit
    order was ever raised -- which is what makes the money on it invented.
    """
    from datetime import date

    from alphamesh.models.domain import ExitReason, PositionRecord, TradeOutcome

    position_id = "phantom00000001"
    journal.record_position(
        PositionRecord(
            position_id=position_id,
            decision_id="phantomdecision",
            client_order_id="alphamesh-SPY-BCS-phantom01",
            symbol="SPY",
            strategy=Strategy.BULL_CALL_SPREAD,
            quantity=quantity,
            entry_debit_cents=393 * OPTION_MULTIPLIER * quantity,
            max_loss_cents=393 * OPTION_MULTIPLIER * quantity,
            max_profit_cents=107 * OPTION_MULTIPLIER * quantity,
            opened_at=NOW,
            expiration=date(2026, 9, 3),
            long_symbol=SPY_LONG,
            short_symbol=SPY_SHORT,
            state=TradeState.MONITORING,
        )
    )
    journal.record_outcome(
        TradeOutcome(
            position_id=position_id,
            decision_id="phantomdecision",
            symbol="SPY",
            strategy=Strategy.BULL_CALL_SPREAD,
            regime=Regime.BULLISH_TREND,
            confidence=0.0,
            quantity=quantity,
            entry_debit_cents=393 * OPTION_MULTIPLIER * quantity,
            exit_value_cents=393 * OPTION_MULTIPLIER * quantity + realized_pnl_cents,
            realized_pnl_cents=realized_pnl_cents,
            return_on_defined_risk=0.0,
            holding_minutes=42.0,
            exit_reason=ExitReason.END_OF_DAY,
            opened_at=NOW,
            closed_at=NOW + timedelta(hours=1),
        )
    )
    return position_id


def open_one_position(orchestrator, journal, now=NOW):  # type: ignore[no-untyped-def]
    """Drive a real entry through the orchestrator and return the position."""
    orchestrator.startup()
    orchestrator.run_cycle(now=now)
    positions = journal.open_positions()
    assert len(positions) == 1, "the entry path did not open a position"
    return positions[0]


def flatten_window(now=NOW):  # type: ignore[no-untyped-def]
    """A time inside the end-of-day flatten window, so an exit is due."""
    return now + timedelta(hours=3, minutes=50)


# --------------------------------------------------------------------------- #
# 1-4: the exit actually reaches the broker, and money comes from a real fill
# --------------------------------------------------------------------------- #
class TestExitsReachTheBroker:
    def test_an_exit_signal_sends_a_closing_order(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 1. The defect: an exit decision closed the journal only."""
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        position = open_one_position(orchestrator, journal)
        assert broker.close_payloads == [], "nothing should be closed yet"

        orchestrator.run_cycle(now=flatten_window())

        assert len(broker.close_payloads) == 1, (
            "an exit was decided but no closing order reached the broker"
        )
        closing = broker.close_payloads[0]
        assert sorted(closing["legs"]) == sorted(
            [position.long_symbol, position.short_symbol]
        )
        assert closing["quantity"] == position.quantity
        assert closing["limit_price_cents"] > 0
        journal.close()

    def test_closing_order_carries_to_close_intents_on_both_legs(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 1, continued: the mirror order closes rather than re-opens."""
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        open_one_position(orchestrator, journal)
        orchestrator.run_cycle(now=flatten_window())

        exit_id = broker.close_payloads[0]["client_order_id"]
        intents = {leg.position_intent for leg in broker.leg_details[exit_id]}
        assert intents == {"sell_to_close", "buy_to_close"}
        journal.close()

    def test_exit_requested_does_not_close_before_the_broker_confirms(
        self, spy_config
    ) -> None:
        """Property 2. A submitted close is a request, never an outcome."""
        # fill=False: the closing order sits on the wire unfilled.
        broker = SimulatedBroker(make_account(), fill=True)
        orchestrator, journal, _ = build(spy_config, broker=broker)
        position = open_one_position(orchestrator, journal)
        broker.fill = False

        report = orchestrator.run_cycle(now=flatten_window())

        assert report.exit_orders_submitted, "no closing order was raised"
        assert report.exits_taken == [], "an unfilled close must not count as an exit"
        still_open = journal.get_position(position.position_id)
        assert still_open is not None
        assert still_open.state is TradeState.EXIT_REQUESTED
        assert still_open.state is not TradeState.CLOSED
        assert journal.outcomes() == [], "no outcome may exist without a closing fill"
        journal.close()

    def test_no_realised_pnl_is_invented_from_a_mark(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 3. The exact defect: mark - entry_debit booked as realised."""
        broker = SimulatedBroker(make_account(), fill=True)
        orchestrator, journal, _ = build(spy_config, broker=broker)
        open_one_position(orchestrator, journal)
        broker.fill = False

        orchestrator.run_cycle(now=flatten_window())

        assert journal.outcomes() == []
        assert journal.realized_pnl_cents() == 0, (
            "realised P&L appeared without a closing fill behind it"
        )
        journal.close()

    def test_realised_pnl_comes_from_the_closing_fill(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 4. Entry fill against exit fill, and nothing else."""
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        position = open_one_position(orchestrator, journal)

        orchestrator.run_cycle(now=flatten_window())

        outcomes = journal.outcomes()
        assert len(outcomes) == 1
        outcome = outcomes[0]

        # The simulator fills a close at the submitted limit, so the exit value
        # must be exactly that price times the multiplier and the quantity.
        exit_id = broker.close_payloads[0]["client_order_id"]
        fill_cents = broker.orders[exit_id].filled_avg_price_cents
        expected_value = fill_cents * OPTION_MULTIPLIER * position.quantity
        assert outcome["exit_value_cents"] == expected_value
        assert outcome["realized_pnl_cents"] == (
            expected_value - position.entry_debit_cents
        )
        assert outcome["reconciliation_note"] is None
        journal.close()


# --------------------------------------------------------------------------- #
# 5-8: adopting what the broker holds and the journal forgot
# --------------------------------------------------------------------------- #
class TestAdoption:
    def test_a_broker_position_missing_from_the_journal_is_adopted(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 5. The production state: broker holds it, journal does not."""
        broker = SimulatedBroker(make_account())
        broker.open_positions = spy_bull_call_legs(2)
        orchestrator, journal, _ = build(spy_config, broker=broker)
        assert journal.open_positions() == []

        summary = orchestrator.adopt_broker_positions(NOW)

        assert summary.adopted == 1
        positions = journal.open_positions()
        assert len(positions) == 1
        adopted = positions[0]
        assert adopted.symbol == "SPY"
        assert adopted.strategy is Strategy.BULL_CALL_SPREAD
        assert adopted.quantity == 2
        assert adopted.long_symbol == SPY_LONG
        assert adopted.short_symbol == SPY_SHORT
        assert adopted.state is TradeState.MONITORING
        assert adopted.max_loss_cents > 0
        journal.close()

    def test_an_adopted_position_is_then_managed_and_exited(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 5, the point of it: adoption restores exit management."""
        broker = SimulatedBroker(make_account())
        broker.open_positions = spy_bull_call_legs(1)
        orchestrator, journal, _ = build(spy_config, broker=broker)
        orchestrator.startup()

        orchestrator.run_cycle(now=flatten_window())

        assert broker.close_payloads, (
            "an adopted position was never exited, so adoption bought nothing"
        )
        journal.close()

    def test_a_phantom_closed_row_does_not_hide_a_live_broker_position(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 6. The journal saying CLOSED must not veto adoption.

        The phantom is written the way the defect wrote it: a CLOSED position
        and a booked outcome with NO closing order anywhere behind them, while
        the broker still holds both legs.
        """
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        phantom_id = write_phantom_close(journal, quantity=1)
        assert journal.open_positions() == [], "the phantom must look closed"

        broker.open_positions = spy_bull_call_legs(1)
        summary = orchestrator.adopt_broker_positions(NOW)

        assert summary.adopted == 1
        adopted = journal.open_positions()
        assert len(adopted) == 1
        assert adopted[0].position_id != phantom_id
        assert adopted[0].long_symbol == SPY_LONG

        # The phantom row is annotated, never deleted.
        notes = {o["position_id"]: o["reconciliation_note"] for o in journal.outcomes()}
        assert notes[phantom_id] == PHANTOM_CLOSE_NOTE
        events = [e["event_type"] for e in journal.recent_events(200)]
        assert "phantom_close_reconciled" in events
        journal.close()

    def test_an_annotated_phantom_close_is_excluded_from_realised_pnl(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 6, continued: invented money leaves the totals."""
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        write_phantom_close(journal, quantity=1, realized_pnl_cents=-12_600)
        assert journal.realized_pnl_cents() == -12_600, (
            "the phantom should count until it is proven phantom"
        )

        broker.open_positions = spy_bull_call_legs(1)
        orchestrator.adopt_broker_positions(NOW)

        assert journal.realized_pnl_cents() == 0, (
            "invented realised money is still being reported"
        )
        # The audit row survives; only its contribution to the total is removed.
        assert len(journal.outcomes()) == 1
        journal.close()

    def test_a_genuine_close_is_never_annotated_as_phantom(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 6, the other direction: a real result must not be erased.

        The same two contracts can legitimately be traded twice. A close backed
        by a filled exit order is real money, and re-opening the spread later
        must not retroactively strike it from the record.
        """
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        position = open_one_position(orchestrator, journal)
        orchestrator.run_cycle(now=flatten_window())
        outcomes = journal.outcomes()
        assert len(outcomes) == 1, "expected one genuine, broker-filled close"
        real_pnl = journal.realized_pnl_cents()

        # The identical spread is open at the broker again.
        broker.open_positions = [
            option_leg(position.long_symbol, position.quantity, 8.99),
            option_leg(position.short_symbol, -position.quantity, 5.06),
        ]
        orchestrator.adopt_broker_positions(NOW)

        assert journal.outcomes()[0]["reconciliation_note"] is None
        assert journal.realized_pnl_cents() == real_pnl
        journal.close()

    @pytest.mark.parametrize(
        ("long_symbol", "short_symbol", "expected"),
        [
            # The nine live production shapes, generically: both strategies,
            # several underlyings, several expirations and widths.
            ("SPY260903C00762000", "SPY260903C00767000", Strategy.BULL_CALL_SPREAD),
            ("QQQ260902C00716000", "QQQ260902C00721000", Strategy.BULL_CALL_SPREAD),
            ("IWM260902C00293000", "IWM260902C00295000", Strategy.BULL_CALL_SPREAD),
            ("NVDA260902C00220000", "NVDA260902C00222500", Strategy.BULL_CALL_SPREAD),
            ("AMD260904C00457500", "AMD260904C00472500", Strategy.BULL_CALL_SPREAD),
            ("AAPL260902P00315000", "AAPL260902P00310000", Strategy.BEAR_PUT_SPREAD),
            ("TSLA260902P00367500", "TSLA260902P00357500", Strategy.BEAR_PUT_SPREAD),
            ("NVDA260902P00220000", "NVDA260902P00217500", Strategy.BEAR_PUT_SPREAD),
            ("DIA260904P00530000", "DIA260904P00526000", Strategy.BEAR_PUT_SPREAD),
        ],
    )
    def test_every_orphan_shape_reconstructs(
        self, long_symbol: str, short_symbol: str, expected: Strategy
    ) -> None:
        """Property 7. Reconstruction is generic, not special-cased per symbol."""
        result = reconstruct_spreads(
            [option_leg(long_symbol, 2, 3.15), option_leg(short_symbol, -2, 1.39)]
        )
        assert result.ambiguous == []
        assert len(result.spreads) == 1
        spread = result.spreads[0]
        assert spread.strategy is expected
        assert spread.long_symbol == long_symbol
        assert spread.short_symbol == short_symbol
        assert spread.quantity == 2
        # A debit spread's defined risk is the premium paid, in cents.
        assert spread.max_loss_cents == (315 - 139) * OPTION_MULTIPLIER * 2
        assert spread.entry_basis == ENTRY_BASIS_POSITION_COST

    def test_the_entry_order_is_preferred_over_per_leg_cost_basis(self) -> None:
        """Property 7, continued: the originating order is authoritative."""
        entry = BrokerOrderSummary(
            client_order_id="alphamesh-SPY-BCS-abc123",
            status="filled",
            filled_quantity=1,
            filled_avg_price_cents=400,
            legs=(
                BrokerOrderLeg(symbol=SPY_LONG, side="buy", position_intent="buy_to_open"),
                BrokerOrderLeg(
                    symbol=SPY_SHORT, side="sell", position_intent="sell_to_open"
                ),
            ),
        )
        result = reconstruct_spreads(spy_bull_call_legs(1), [entry])

        assert len(result.spreads) == 1
        spread = result.spreads[0]
        assert spread.entry_basis == ENTRY_BASIS_ORDER_FILL
        assert spread.entry_debit_cents == 400 * OPTION_MULTIPLIER
        assert spread.client_order_id == "alphamesh-SPY-BCS-abc123"

    def test_a_closing_order_is_never_mistaken_for_the_entry(self) -> None:
        """Property 7, continued: intent decides, not the leg set."""
        closing = BrokerOrderSummary(
            client_order_id="alphamesh-SPY-BCSX-dead99",
            status="filled",
            filled_quantity=1,
            filled_avg_price_cents=450,
            legs=(
                BrokerOrderLeg(
                    symbol=SPY_LONG, side="sell", position_intent="sell_to_close"
                ),
                BrokerOrderLeg(
                    symbol=SPY_SHORT, side="buy", position_intent="buy_to_close"
                ),
            ),
        )
        result = reconstruct_spreads(spy_bull_call_legs(1), [closing])

        assert len(result.spreads) == 1
        assert result.spreads[0].entry_basis == ENTRY_BASIS_POSITION_COST
        assert result.spreads[0].client_order_id is None

    @pytest.mark.parametrize(
        ("legs", "why"),
        [
            ([option_leg(SPY_LONG, 1, 8.99)], "a lone long leg is not a spread"),
            (
                [option_leg(SPY_SHORT, -1, 5.06)],
                "a lone short leg would be naked if half-closed",
            ),
            (
                [option_leg(SPY_LONG, 1, 8.99), option_leg(SPY_SHORT, -2, 5.06)],
                "mismatched leg sizes are not one vertical",
            ),
            (
                [option_leg(SPY_LONG, 1, 8.99), option_leg(SPY_SHORT, 1, 5.06)],
                "two longs are not a vertical",
            ),
            (
                # Long the further-out-of-the-money call: a credit spread, which
                # this agent never opens and must not try to manage.
                [option_leg(SPY_SHORT, 1, 5.06), option_leg(SPY_LONG, -1, 8.99)],
                "a credit structure is not a supported debit vertical",
            ),
        ],
    )
    def test_ambiguous_legs_fail_closed(self, legs, why: str) -> None:  # type: ignore[no-untyped-def]
        """Property 8. Never guess a pairing; a wrong guess strands a naked leg."""
        result = reconstruct_spreads(legs)

        assert result.spreads == [], why
        assert result.ambiguous, why
        assert result.ambiguous_symbols == frozenset({"SPY"})

    def test_an_ambiguous_broker_position_blocks_new_exposure(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 8, continued: uncounted risk is a reason to stop, not shrug."""
        broker = SimulatedBroker(make_account())
        broker.open_positions = [option_leg(SPY_SHORT, -1, 5.06)]  # a naked short leg
        orchestrator, journal, _ = build(spy_config, broker=broker)

        summary = orchestrator.adopt_broker_positions(NOW)
        assert summary.adopted == 0
        assert summary.ambiguous == 1

        portfolio = orchestrator.portfolio_state(NOW)
        assert not portfolio.exposure_fully_accounted
        journal.close()

    def test_adoption_is_idempotent(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """A second pass must not create a second position over the same legs."""
        broker = SimulatedBroker(make_account())
        broker.open_positions = spy_bull_call_legs(1)
        orchestrator, journal, _ = build(spy_config, broker=broker)

        assert orchestrator.adopt_broker_positions(NOW).adopted == 1
        assert orchestrator.adopt_broker_positions(NOW).adopted == 0
        assert len(journal.open_positions()) == 1
        journal.close()


# --------------------------------------------------------------------------- #
# 9-11: recovered exposure counts, and exits are never blocked by caps
# --------------------------------------------------------------------------- #
class TestRecoveredExposureCountsAsRisk:
    def test_adopted_risk_counts_against_the_portfolio_cap(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 9."""
        broker = SimulatedBroker(make_account())
        broker.open_positions = spy_bull_call_legs(2)
        orchestrator, journal, _ = build(spy_config, broker=broker)

        before = orchestrator.portfolio_state(NOW).total_defined_risk_cents
        assert before == 0
        orchestrator.adopt_broker_positions(NOW)
        after = orchestrator.portfolio_state(NOW)

        adopted = journal.open_positions()[0]
        assert after.total_defined_risk_cents == adopted.max_loss_cents > 0
        assert after.open_position_count == 1
        journal.close()

    def test_adopted_risk_counts_against_the_correlation_group_cap(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 10."""
        broker = SimulatedBroker(make_account())
        broker.open_positions = spy_bull_call_legs(2)
        orchestrator, journal, _ = build(spy_config, broker=broker)
        orchestrator.adopt_broker_positions(NOW)

        portfolio = orchestrator.portfolio_state(NOW)
        limits = spy_config.risk
        group = limits.group_for("SPY")
        assert group is not None, "SPY must belong to a correlation group"
        assert portfolio.positions_in_group(limits, group), (
            "an adopted position is missing from its correlation group"
        )
        assert portfolio.defined_risk_in_group_cents(limits, group) > 0
        journal.close()

    def test_an_over_cap_portfolio_blocks_new_exposure(self, spy_config, now) -> None:  # type: ignore[no-untyped-def]
        """Property 11, first half."""
        from alphamesh.strategies.bull_call import build_bull_call_spread
        from tests.conftest import make_decision

        governor = RiskGovernor(spy_config.risk, paper_confirmed=True)
        chain = CaptureOptionChain(CAPTURE_DIR).chain(
            "SPY", __import__("alphamesh.models.domain", fromlist=["x"]).OptionType.CALL,
            now.date(), spy_config.strategies.min_dte, spy_config.strategies.max_dte,
        )
        selection = build_bull_call_spread(
            "SPY", chain, spy_config.strategies, spy_config.risk, now, as_of_date=now.date()
        )
        assert selection.spread is not None

        over_cap = make_portfolio(
            # Already past the portfolio defined-risk cap.
            open_positions=(),
            realized_pnl_today_cents=0,
        )
        # Push total defined risk beyond the cap via a real position record.
        cap_cents = round(spy_config.risk.max_portfolio_defined_risk * 100)
        from alphamesh.models.domain import PositionRecord

        heavy = PositionRecord(
            position_id="over0001",
            decision_id="over0001",
            client_order_id="over0001",
            symbol="AAPL",
            strategy=Strategy.BULL_CALL_SPREAD,
            quantity=1,
            entry_debit_cents=cap_cents + 10_000,
            max_loss_cents=cap_cents + 10_000,
            max_profit_cents=1,
            opened_at=now,
            expiration=selection.spread.expiration,
            long_symbol="AAPL260902C00310000",
            short_symbol="AAPL260902C00315000",
        )
        over_cap = make_portfolio(open_positions=(heavy,))

        verdict = governor.approve(
            make_decision(symbol="SPY"),
            selection.spread,
            over_cap,
            now,
            client_order_id="alphamesh-SPY-BCS-newrisk",
        )
        assert not verdict.approved
        assert ReasonCode.MAX_PORTFOLIO_RISK in verdict.reason_codes

    def test_an_over_cap_portfolio_still_permits_a_risk_reducing_exit(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 11, second half. Closing must never be gated on headroom.

        Structural, not incidental: the exit path does not consult the governor
        at all, so no cap can refuse a close. This drives a real over-cap
        portfolio through a real exit to prove it.
        """
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        position = open_one_position(orchestrator, journal)

        # Make the portfolio unambiguously over every aggregate cap.
        original = orchestrator.portfolio_state
        cap_cents = round(spy_config.risk.max_portfolio_defined_risk * 100)

        def over_cap(now=None):  # type: ignore[no-untyped-def]
            state = original(now)
            inflated = position.model_copy(
                update={"max_loss_cents": cap_cents + 50_000}
            )
            return type(state)(
                **{
                    **{
                        f.name: getattr(state, f.name)
                        for f in state.__dataclass_fields__.values()
                    },
                    "open_positions": (inflated,),
                }
            )

        orchestrator.portfolio_state = over_cap  # type: ignore[assignment]
        report = orchestrator.run_cycle(now=flatten_window())

        assert broker.close_payloads, (
            "an over-cap portfolio blocked a risk-reducing exit"
        )
        assert report.exits_taken, "the close did not complete"
        journal.close()


# --------------------------------------------------------------------------- #
# 12-14: market hours, partial fills, restart
# --------------------------------------------------------------------------- #
class TestExitExecutionSafety:
    def test_recovery_after_the_close_submits_zero_closing_orders(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 12. Discovery is not execution."""
        broker = SimulatedBroker(make_account())
        # Every one of these is past its expiration, so an exit is unambiguously
        # due -- and must still not be sent into a closed market.
        broker.open_positions = spy_bull_call_legs(1)
        closed = TrendingMarketData(is_open=False)
        orchestrator, journal, _ = build(spy_config, broker=broker, market=closed)

        orchestrator.startup()
        report = orchestrator.run_cycle(now=flatten_window())

        assert not report.market_open
        assert broker.close_payloads == [], (
            "a closing order was queued against a closed book"
        )
        assert journal.open_positions(), "the position should be journalled and held"
        journal.close()

    def test_adoption_outside_market_hours_still_journals(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 12, continued: the point of running adoption overnight."""
        broker = SimulatedBroker(make_account())
        broker.open_positions = spy_bull_call_legs(1)
        closed = TrendingMarketData(is_open=False)
        orchestrator, journal, _ = build(spy_config, broker=broker, market=closed)

        orchestrator.startup()

        assert len(journal.open_positions()) == 1
        assert broker.close_payloads == []
        journal.close()

    def test_a_partial_closing_fill_never_closes_the_position(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 13. The unclosed balance is still real exposure."""
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        position = open_one_position(orchestrator, journal)
        if position.quantity < 2:
            pytest.skip("a partial fill needs a multi-spread position")

        # The close must go out UNFILLED, or there is no partial to test.
        broker.fill = False
        orchestrator.run_cycle(now=flatten_window())
        exit_id = broker.close_payloads[0]["client_order_id"]
        assert journal.get_position(position.position_id).state is (
            TradeState.EXIT_REQUESTED
        )

        # The broker now reports some, but not all, of the spreads closed.
        broker.orders[exit_id] = broker.orders[exit_id].model_copy(
            update={
                "status": "partially_filled",
                "raw_status": "partially_filled",
                "filled_quantity": position.quantity - 1,
                "filled_avg_price_cents": 300,
            }
        )

        report = orchestrator.run_cycle(now=flatten_window() + timedelta(minutes=1))

        still_open = journal.get_position(position.position_id)
        assert still_open is not None
        assert still_open.state is not TradeState.CLOSED
        assert report.exits_taken == []
        assert not any(
            o["position_id"] == position.position_id for o in journal.outcomes()
        ), "a partial close booked an outcome"
        assert journal.realized_pnl_cents() == 0
        journal.close()

    def test_a_partially_filled_exit_is_never_cancelled_by_the_sweep(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 13, continued: cancelling a partial strands the balance."""
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        position = open_one_position(orchestrator, journal)
        if position.quantity < 2:
            pytest.skip("a partial fill needs a multi-spread position")

        broker.fill = False
        orchestrator.run_cycle(now=flatten_window())
        exit_id = broker.close_payloads[0]["client_order_id"]
        broker.orders[exit_id] = broker.orders[exit_id].model_copy(
            update={
                "status": "partially_filled",
                "raw_status": "partially_filled",
                "filled_quantity": 1,
            }
        )

        # Well past the exit TTL: the sweep runs and must still leave it alone.
        long_after = flatten_window() + timedelta(
            seconds=spy_config.settings.exit_order_ttl_seconds + 60
        )
        report = orchestrator.run_cycle(now=long_after)

        assert report.exit_orders_repriced == 0, (
            "the sweep retired an exit that had already partly filled"
        )
        assert broker.orders[exit_id].status == "partially_filled"
        assert len(broker.close_payloads) == 1, "a duplicate close was raised"
        journal.close()

    def test_a_restart_mid_exit_does_not_send_a_second_closing_order(
        self, spy_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Property 14. Two closes on one spread would go net short."""
        broker = SimulatedBroker(make_account())
        journal = Journal(":memory:")
        orchestrator, _, _ = build(spy_config, broker=broker, journal=journal)
        open_one_position(orchestrator, journal)
        broker.fill = False
        orchestrator.run_cycle(now=flatten_window())
        assert len(broker.close_payloads) == 1

        # Restart: a brand new orchestrator over the same journal and broker.
        restarted, _, _ = build(spy_config, broker=broker, journal=journal)
        restarted.startup()
        restarted.run_cycle(now=flatten_window() + timedelta(seconds=30))

        assert len(broker.close_payloads) == 1, (
            "a restart raised a duplicate closing order"
        )
        journal.close()


# --------------------------------------------------------------------------- #
# 15-16: the guards that were already there stay there
# --------------------------------------------------------------------------- #
class TestExistingGuardsIntact:
    def test_the_broker_truth_entry_guard_still_blocks(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 15. This fix must not reopen the hole 854821f closed."""
        governor = RiskGovernor(spy_config.risk, paper_confirmed=True)
        from alphamesh.models.domain import OptionType
        from alphamesh.strategies.bull_call import build_bull_call_spread
        from tests.conftest import make_decision

        chain = CaptureOptionChain(CAPTURE_DIR).chain(
            "SPY", OptionType.CALL, NOW.date(),
            spy_config.strategies.min_dte, spy_config.strategies.max_dte,
        )
        selection = build_bull_call_spread(
            "SPY", chain, spy_config.strategies, spy_config.risk, NOW, as_of_date=NOW.date()
        )
        assert selection.spread is not None

        held_at_broker = make_portfolio(broker_position_symbols=frozenset({"SPY"}))
        verdict = governor.approve(
            make_decision(symbol="SPY"),
            selection.spread,
            held_at_broker,
            NOW,
            client_order_id="alphamesh-SPY-BCS-guard01",
        )
        assert not verdict.approved
        assert ReasonCode.BROKER_OPEN_POSITION in verdict.reason_codes

    def test_the_entry_ttl_still_retires_a_stale_entry_order(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 16, first half."""
        broker = SimulatedBroker(make_account(), fill=False)
        orchestrator, journal, _ = build(spy_config, broker=broker)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)
        entries = journal.open_orders(kind=ORDER_KIND_ENTRY)
        assert entries, "no entry order to age"

        ttl = spy_config.settings.entry_order_ttl_seconds
        report = orchestrator.run_cycle(now=NOW + timedelta(seconds=ttl + 30))

        assert report.stale_orders_cancelled == 1
        journal.close()

    def test_the_entry_ttl_never_touches_an_exit_order(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Property 16, second half. Cancelling a close leaves exposure open.

        The exit sweep is held off deliberately (a very long exit TTL) so that
        the only sweep that could touch this order is the entry one. If the
        order survives, the two sweeps are genuinely separate rather than
        coincidentally ordered.
        """
        patient = spy_config.model_copy(
            update={
                "settings": spy_config.settings.model_copy(
                    update={"exit_order_ttl_seconds": 86_400}
                )
            }
        )
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(patient, broker=broker)
        open_one_position(orchestrator, journal)
        broker.fill = False
        orchestrator.run_cycle(now=flatten_window())

        exits = journal.open_orders(kind=ORDER_KIND_EXIT)
        assert len(exits) == 1
        exit_id = exits[0]["client_order_id"]

        entry_ttl = patient.settings.entry_order_ttl_seconds
        report = orchestrator.run_cycle(
            now=flatten_window() + timedelta(seconds=entry_ttl + 60)
        )

        assert report.stale_orders_cancelled == 0, (
            "the entry TTL sweep cancelled a closing order"
        )
        assert report.exit_orders_repriced == 0
        assert broker.orders[exit_id].status != "canceled"
        assert journal.open_orders(kind=ORDER_KIND_EXIT), (
            "the closing order was retired by the wrong sweep"
        )
        journal.close()

    def test_an_exit_order_is_not_reported_as_an_entry(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """The kind split is what keeps the two sweeps apart."""
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        open_one_position(orchestrator, journal)
        broker.fill = False
        orchestrator.run_cycle(now=flatten_window())

        entry_ids = {o["client_order_id"] for o in journal.open_orders(ORDER_KIND_ENTRY)}
        exit_ids = {o["client_order_id"] for o in journal.open_orders(ORDER_KIND_EXIT)}
        assert exit_ids
        assert entry_ids & exit_ids == set()
        journal.close()


class TestPreflightStillPlacesNoOrders:
    def test_the_zero_order_proxy_blocks_closing_orders(self) -> None:
        """Adoption reads through preflight; it must never be able to close."""
        from alphamesh.main import OrderSubmissionForbiddenError, _ZeroOrderBroker

        broker = _ZeroOrderBroker(SimulatedBroker(make_account()))
        assert broker.recent_orders() == []
        with pytest.raises(OrderSubmissionForbiddenError):
            broker.close_spread("anything", 100, "id")


class TestNullProviderStillWorks:
    def test_deterministic_fallback_path_is_untouched(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(spy_config, provider=NullProvider())
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)
        assert report.market_open
        journal.close()


class TestJournalMigration:
    """The production journal is a v1 file on a Railway volume.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing table's columns alone, so
    every column added after v1 has to be applied by hand. If that migration is
    wrong the agent cannot start, and it cannot start with the volume it has.
    """

    V1_SCHEMA = """
    CREATE TABLE orders (
        client_order_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL,
        symbol TEXT NOT NULL, strategy TEXT NOT NULL, quantity INTEGER NOT NULL,
        limit_price_cents INTEGER NOT NULL, max_loss_cents INTEGER NOT NULL,
        legs TEXT NOT NULL, state TEXT NOT NULL, broker_order_id TEXT,
        broker_status TEXT, filled_quantity INTEGER NOT NULL DEFAULT 0,
        filled_avg_price_cents INTEGER, created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL);
    CREATE TABLE positions (
        position_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL,
        client_order_id TEXT NOT NULL, symbol TEXT NOT NULL,
        strategy TEXT NOT NULL, quantity INTEGER NOT NULL,
        entry_debit_cents INTEGER NOT NULL, max_loss_cents INTEGER NOT NULL,
        max_profit_cents INTEGER NOT NULL, opened_at TEXT NOT NULL,
        expiration TEXT NOT NULL, long_symbol TEXT NOT NULL,
        short_symbol TEXT NOT NULL, state TEXT NOT NULL,
        mfe_cents INTEGER, mae_cents INTEGER);
    CREATE TABLE outcomes (
        position_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL,
        symbol TEXT NOT NULL, strategy TEXT NOT NULL, regime TEXT NOT NULL,
        confidence REAL NOT NULL, quantity INTEGER NOT NULL,
        entry_debit_cents INTEGER NOT NULL, exit_value_cents INTEGER NOT NULL,
        realized_pnl_cents INTEGER NOT NULL, return_on_defined_risk REAL NOT NULL,
        holding_minutes REAL NOT NULL, mfe_cents INTEGER, mae_cents INTEGER,
        exit_reason TEXT NOT NULL, opened_at TEXT NOT NULL,
        closed_at TEXT NOT NULL);
    """

    def _v1_database(self, tmp_path):  # type: ignore[no-untyped-def]
        import sqlite3

        path = tmp_path / "alphamesh.db"
        conn = sqlite3.connect(str(path))
        conn.executescript(self.V1_SCHEMA)
        conn.execute(
            "INSERT INTO outcomes VALUES "
            "('p1','d1','SPY','BULL_CALL_SPREAD','BULLISH_TREND',0.0,1,"
            "39300,26700,-12600,-0.32,42.0,NULL,NULL,'END_OF_DAY',"
            "'2026-08-31T16:00:00+00:00','2026-08-31T19:00:00+00:00')"
        )
        conn.commit()
        conn.close()
        return path

    def test_a_v1_journal_opens_and_gains_the_new_columns(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = self._v1_database(tmp_path)

        journal = Journal(path)

        columns = {
            table: {
                r["name"]
                for r in journal._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for table in ("orders", "positions", "outcomes")
        }
        assert {"kind", "position_id"} <= columns["orders"]
        assert {"origin", "entry_basis"} <= columns["positions"]
        assert "reconciliation_note" in columns["outcomes"]
        journal.close()

    def test_existing_rows_survive_the_migration(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Additive only. History is annotated, never rewritten or dropped."""
        path = self._v1_database(tmp_path)

        journal = Journal(path)

        outcomes = journal.outcomes()
        assert len(outcomes) == 1
        assert outcomes[0]["position_id"] == "p1"
        assert outcomes[0]["realized_pnl_cents"] == -12_600
        # Pre-existing rows are unannotated, so they still count until proven
        # phantom by a live broker position.
        assert outcomes[0]["reconciliation_note"] is None
        assert journal.realized_pnl_cents() == -12_600
        journal.close()

    def test_migrating_twice_is_a_no_op(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = self._v1_database(tmp_path)
        Journal(path).close()

        journal = Journal(path)

        assert len(journal.outcomes()) == 1
        journal.close()

    def test_a_migrated_journal_records_an_exit_order(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The new columns must actually be usable, not merely present."""
        from alphamesh.models.domain import PositionRecord

        path = self._v1_database(tmp_path)
        journal = Journal(path)
        position = PositionRecord(
            position_id="p2",
            decision_id="d2",
            client_order_id="alphamesh-SPY-BCS-migrate1",
            symbol="SPY",
            strategy=Strategy.BULL_CALL_SPREAD,
            quantity=1,
            entry_debit_cents=39_300,
            max_loss_cents=39_300,
            max_profit_cents=10_700,
            opened_at=NOW,
            expiration=NOW.date(),
            long_symbol=SPY_LONG,
            short_symbol=SPY_SHORT,
        )
        journal.record_position(position)

        from alphamesh.execution.order_builder import build_exit_intent

        chain = {
            c.symbol: c
            for c in CaptureOptionChain(CAPTURE_DIR).chain(
                "SPY",
                __import__("alphamesh.models.domain", fromlist=["x"]).OptionType.CALL,
                NOW.date(),
                0,
                30,
            )
        }
        intent = build_exit_intent(
            position, chain[SPY_LONG], chain[SPY_SHORT], 400, NOW
        )
        assert journal.reserve_order(intent, kind=ORDER_KIND_EXIT, position_id="p2")

        found = journal.exit_order_for("p2")
        assert found is not None
        assert found["kind"] == ORDER_KIND_EXIT
        assert journal.open_orders(kind=ORDER_KIND_ENTRY) == []
        journal.close()


class TestAmbiguousExitSubmission:
    """An exit that may or may not have reached the broker.

    Never resend blind, and never leave the position stranded either: the
    reservation is resolved against the broker, and only a reservation the
    broker has never heard of is retired.
    """

    def test_an_ambiguous_exit_is_not_resent(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        from alphamesh.alpaca.execution import AmbiguousSubmissionError

        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        position = open_one_position(orchestrator, journal)

        calls: list[str] = []

        def ambiguous(intent, limit_cents, client_order_id):  # type: ignore[no-untyped-def]
            calls.append(client_order_id)
            raise AmbiguousSubmissionError(client_order_id, "ReadTimeout")

        broker.close_spread = ambiguous  # type: ignore[assignment]
        orchestrator.run_cycle(now=flatten_window())

        assert len(calls) == 1, "an ambiguous close was resent"
        assert journal.get_position(position.position_id).state is not TradeState.CLOSED
        assert journal.outcomes() == []
        journal.close()

    def test_a_reservation_the_broker_never_saw_is_retired(self, spy_config) -> None:  # type: ignore[no-untyped-def]
        """Otherwise the position sits in EXIT_REQUESTED with nothing able to act."""
        from alphamesh.alpaca.execution import AmbiguousSubmissionError

        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(spy_config, broker=broker)
        position = open_one_position(orchestrator, journal)

        real_close = broker.close_spread

        def ambiguous(intent, limit_cents, client_order_id):  # type: ignore[no-untyped-def]
            raise AmbiguousSubmissionError(client_order_id, "ReadTimeout")

        broker.close_spread = ambiguous  # type: ignore[assignment]
        orchestrator.run_cycle(now=flatten_window())
        assert journal.get_position(position.position_id).state is (
            TradeState.EXIT_REQUESTED
        )

        # Next cycle: the broker confirms it never saw the order, so the stuck
        # reservation is retired and a fresh close can be raised.
        broker.close_spread = real_close  # type: ignore[assignment]
        report = orchestrator.run_cycle(now=flatten_window() + timedelta(minutes=1))

        assert broker.close_payloads, "the position never got a second chance to exit"
        assert report.exits_taken, "the retried close did not complete"
        assert journal.get_position(position.position_id).state is TradeState.CLOSED
        journal.close()


class TestAdoptionSummaryPayload:
    def test_the_ambiguous_count_survives_serialisation(self) -> None:
        """The count and the list share a name; nesting keeps both readable."""
        from alphamesh.execution.adoption import AdoptionSummary

        result = reconstruct_spreads([option_leg(SPY_SHORT, -1, 5.06)])
        summary = AdoptionSummary(
            adopted=0, ambiguous=len(result.ambiguous), detail=result.as_dict()
        )
        payload = summary.as_dict()

        assert payload["ambiguous"] == 1, "the count was overwritten by the list"
        assert payload["adopted"] == 0
        assert isinstance(payload["detail"], dict)
        assert len(payload["detail"]["ambiguous"]) == 1
