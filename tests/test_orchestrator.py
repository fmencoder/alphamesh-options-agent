"""End-to-end lifecycle tests.

These drive the real orchestrator over the real captured Alpaca option chain.
Fills come from the in-process simulator, so nothing here touches a network.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from alphamesh.alpaca.client import AlpacaStack
from alphamesh.alpaca.execution import AmbiguousSubmissionError, SimulatedBroker
from alphamesh.alpaca.options import CaptureOptionChain
from alphamesh.alpaca.types import MarketClock
from alphamesh.intelligence.reasoning import NullProvider, ScriptedProvider
from alphamesh.models.domain import ExitReason, ReasonCode, Strategy, TradeState
from alphamesh.orchestrator import Orchestrator
from alphamesh.persistence.journal import Journal
from alphamesh.safety import GuardResult
from tests.conftest import CAPTURE_DIR, NOW, make_account, make_bars, make_snapshot


class TrendingMarketData:
    """Market data that produces a clean uptrend, so the quant gate opens.

    The bars are synthetic; the option chain the orchestrator pairs them with is
    the real captured Alpaca chain. That split is deliberate: the captured
    session is genuinely flat, so a trending tape has to be supplied to exercise
    the entry path at all.
    """

    def __init__(self, drift: float = 0.12, is_open: bool = True) -> None:
        self.drift = drift
        self.is_open = is_open

    def snapshot(self, symbol: str, lookback_minutes: int = 180):  # type: ignore[no-untyped-def]
        base = 769.0 if symbol == "SPY" else 716.0
        bars = make_bars(
            140,
            start_price=base - self.drift * 140,
            drift_per_bar=self.drift,
            start=NOW - timedelta(minutes=139),
        )
        return make_snapshot(symbol, bars)

    def clock(self) -> MarketClock:
        return MarketClock(
            timestamp=NOW, is_open=self.is_open, next_close=NOW + timedelta(hours=4)
        )


def bull_script(rounds: int = 6) -> ScriptedProvider:
    responses: list[dict] = []
    for _ in range(rounds):
        responses += [
            {"thesis": "Uptrend intact.", "key_points": ["higher highs"], "conviction": 0.9},
            {"thesis": "Weak downside.", "key_points": ["no follow-through"], "conviction": 0.1},
            {
                "strategy": "BULL_CALL_SPREAD",
                "confidence": 0.82,
                "bull_score": 0.9,
                "bear_score": 0.15,
                "rationale": "Clean trend with participation.",
            },
        ]
    return ScriptedProvider(responses)


def build(
    config,  # type: ignore[no-untyped-def]
    provider=None,  # type: ignore[no-untyped-def]
    market=None,  # type: ignore[no-untyped-def]
    broker=None,  # type: ignore[no-untyped-def]
    journal=None,  # type: ignore[no-untyped-def]
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
def single_symbol_config(config):  # type: ignore[no-untyped-def]
    """One symbol keeps the correlated-exposure gate out of the way."""
    return config.model_copy(
        update={"universe": config.universe.model_copy(update={"symbols": ["SPY"]})}
    )


class TestFullEntryLifecycle:
    def test_agent_opens_a_real_spread_with_no_human_involved(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _stack = build(single_symbol_config)
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)

        assert report.orders_submitted, f"no order placed: {report.rejections}"
        positions = journal.open_positions()
        assert len(positions) == 1

        position = positions[0]
        assert position.strategy is Strategy.BULL_CALL_SPREAD
        assert position.quantity >= 1
        # Real Alpaca contract symbols from the captured chain, not invented ones.
        chain = {
            c.symbol
            for c in CaptureOptionChain(CAPTURE_DIR).chain(
                "SPY", __import__("alphamesh.models.domain", fromlist=["x"]).OptionType.CALL,
                NOW.date(), 0, 30,
            )
        }
        assert position.long_symbol in chain
        assert position.short_symbol in chain
        journal.close()

    def test_defined_loss_never_exceeds_the_configured_cap(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single_symbol_config)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)
        for position in journal.open_positions():
            assert position.max_loss_cents <= int(
                single_symbol_config.risk.absolute_max_defined_loss * 100
            )
        journal.close()

    def test_submitted_order_is_a_multi_leg_limit_order(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        broker = SimulatedBroker(make_account())
        orchestrator, journal, _ = build(single_symbol_config, broker=broker)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)

        assert len(broker.submitted_payloads) == 1
        payload = broker.submitted_payloads[0]
        assert payload["order_class"] == "mleg"
        assert payload["type"] == "limit"
        assert len(payload["legs"]) == 2
        journal.close()

    def test_every_state_transition_is_journalled(self, single_symbol_config) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single_symbol_config)
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)
        cid = report.orders_submitted[0]
        states = [t["to_state"] for t in journal.transitions_for(cid)]
        assert states == ["SUBMITTED", "FILLED", "MONITORING"]
        journal.close()

    def test_the_full_decision_trail_is_reconstructable(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single_symbol_config)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)

        decision = journal.latest_decision()
        assert decision["quant_score"] > 0
        assert decision["regime"]
        assert decision["bull_argument"] and decision["bear_argument"]
        assert decision["judge_verdict"]
        assert decision["features"]

        events = {e["event_type"] for e in journal.recent_events(50)}
        assert {"contract_selection", "position_opened", "cycle_complete"} <= events
        journal.close()


class TestDuplicateProtectionEndToEnd:
    def test_a_second_cycle_does_not_open_a_second_position(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single_symbol_config)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)
        assert len(journal.open_positions()) == 1

        second = orchestrator.run_cycle(now=NOW + timedelta(minutes=1))
        assert second.orders_submitted == []
        assert len(journal.open_positions()) == 1
        codes = {c for _s, cs in second.rejections for c in cs}
        assert ReasonCode.DUPLICATE_ORDER in codes
        journal.close()

    def test_restart_does_not_duplicate_an_open_position(
        self, single_symbol_config, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """A fresh process over the same journal must not re-enter the trade."""
        path = tmp_path / "restart.db"
        broker = SimulatedBroker(make_account())

        first_journal = Journal(path)
        first, _, _ = build(single_symbol_config, broker=broker, journal=first_journal)
        first.startup()
        first.run_cycle(now=NOW)
        assert len(first_journal.open_positions()) == 1
        first_journal.close()

        second_journal = Journal(path)
        second, _, _ = build(single_symbol_config, broker=broker, journal=second_journal)
        recovery = second.startup()
        report = second.run_cycle(now=NOW + timedelta(minutes=2))

        assert recovery["orphaned"] == 0
        assert report.orders_submitted == []
        assert len(second_journal.open_positions()) == 1
        assert len(broker.submitted_payloads) == 1
        second_journal.close()


class TestAmbiguousSubmission:
    def test_ambiguous_submit_leaves_a_reservation_and_sends_nothing_else(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        broker = SimulatedBroker(
            make_account(),
            fail_next_submit_with=AmbiguousSubmissionError("cid", "ReadTimeout"),
        )
        orchestrator, journal, _ = build(single_symbol_config, broker=broker)
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)

        assert report.orders_submitted == []
        assert any("ambiguous" in e for e in report.errors)
        assert broker.submitted_payloads == []
        # The reservation survives so recovery can reconcile it.
        reserved = journal.open_orders()
        assert len(reserved) == 1
        assert reserved[0]["state"] == TradeState.CONSTRUCTED.value
        assert journal.open_positions() == []
        journal.close()

    def test_recovery_after_an_ambiguous_submit_retires_the_reservation(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        broker = SimulatedBroker(
            make_account(),
            fail_next_submit_with=AmbiguousSubmissionError("cid", "ReadTimeout"),
        )
        orchestrator, journal, _ = build(single_symbol_config, broker=broker)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)

        recovery = orchestrator.startup()
        assert recovery["orphaned"] == 1
        assert journal.open_orders() == []
        journal.close()


class TestExitLifecycle:
    def test_end_of_day_flattens_the_position_and_books_an_outcome(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single_symbol_config)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)
        assert len(journal.open_positions()) == 1

        # Wind the clock to inside the flatten window.
        orchestrator.stack.market_data = TrendingMarketData()
        late = NOW + timedelta(hours=3, minutes=50)
        orchestrator.run_cycle(now=late)

        assert journal.open_positions() == []
        outcomes = journal.outcomes()
        assert len(outcomes) == 1
        assert outcomes[0]["exit_reason"] == ExitReason.END_OF_DAY.value
        assert outcomes[0]["holding_minutes"] > 0
        journal.close()

    def test_circuit_breaker_forces_an_exit(self, single_symbol_config) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single_symbol_config)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)
        assert len(journal.open_positions()) == 1

        original = orchestrator.portfolio_state

        def tripped(now=None):  # type: ignore[no-untyped-def]
            return replace(original(now), realized_pnl_today_cents=-500_000)

        orchestrator.portfolio_state = tripped  # type: ignore[assignment]
        report = orchestrator.run_cycle(now=NOW + timedelta(minutes=5))

        assert report.circuit_breaker_tripped
        assert journal.open_positions() == []
        assert journal.outcomes()[0]["exit_reason"] == ExitReason.CIRCUIT_BREAKER.value
        journal.close()

    def test_outcome_records_the_full_pnl_picture(self, single_symbol_config) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single_symbol_config)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)
        orchestrator.run_cycle(now=NOW + timedelta(hours=3, minutes=50))

        outcome = journal.outcomes()[0]
        for field in (
            "entry_debit_cents",
            "exit_value_cents",
            "realized_pnl_cents",
            "return_on_defined_risk",
            "holding_minutes",
            "exit_reason",
        ):
            assert outcome[field] is not None
        journal.close()


class TestNoTradePath:
    def test_flat_tape_produces_no_trade_and_no_order(self, single_symbol_config) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _stack = build(
            single_symbol_config, market=TrendingMarketData(drift=0.0)
        )
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)

        assert report.orders_submitted == []
        assert all(d.strategy is Strategy.NO_TRADE for d in report.decisions)
        assert all(d.no_trade_reason for d in report.decisions)
        assert journal.open_positions() == []
        journal.close()

    def test_no_trade_is_still_fully_journalled(self, single_symbol_config) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(
            single_symbol_config, market=TrendingMarketData(drift=0.0)
        )
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)
        decision = journal.latest_decision()
        assert decision["strategy"] == "NO_TRADE"
        assert decision["no_trade_reason"]
        assert decision["features"]
        journal.close()

    def test_agent_survives_with_no_llm_at_all(self, single_symbol_config) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single_symbol_config, provider=NullProvider())
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)
        assert report.errors == []
        assert report.decisions
        journal.close()


class TestResilience:
    def test_market_data_outage_does_not_crash_the_cycle(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        class Broken(TrendingMarketData):
            def snapshot(self, symbol, lookback_minutes=180):  # type: ignore[no-untyped-def]
                raise RuntimeError("data feed down")

        orchestrator, journal, _ = build(single_symbol_config, market=Broken())
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)
        assert report.symbols_scanned == 0
        assert report.orders_submitted == []
        journal.close()

    def test_broker_outage_is_reported_not_raised(self, single_symbol_config) -> None:  # type: ignore[no-untyped-def]
        class Broken(SimulatedBroker):
            def account(self):  # type: ignore[no-untyped-def]
                raise RuntimeError("broker down")

        orchestrator, journal, _ = build(
            single_symbol_config, broker=Broken(make_account())
        )
        report = orchestrator.run_cycle(now=NOW)
        assert report.errors
        assert report.orders_submitted == []
        journal.close()


class TestCorrelatedExposureEndToEnd:
    def test_second_correlated_symbol_is_gated(self, config) -> None:  # type: ignore[no-untyped-def]
        """SPY and QQQ share a bucket, so the third correlated entry is refused."""
        narrow = config.model_copy(
            update={
                "risk": config.risk.model_copy(
                    update={"max_positions_per_correlation_group": 1}
                )
            }
        )
        orchestrator, journal, _ = build(narrow, provider=bull_script(rounds=8))
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)

        assert len(journal.open_positions()) == 1
        codes = {c for _s, cs in report.rejections for c in cs}
        assert ReasonCode.CORRELATED_EXPOSURE in codes
        journal.close()
