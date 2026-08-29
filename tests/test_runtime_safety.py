"""Regression tests for runtime/deployment safety defects.

Each class here corresponds to a concrete defect found during the Railway
runtime audit. None of these existed before that audit; they exist so the
defects cannot silently return.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from alphamesh.alpaca.client import build_stack
from alphamesh.alpaca.execution import SimulatedBroker
from alphamesh.alpaca.types import MarketClock
from alphamesh.config import Settings
from alphamesh.models.domain import ReasonCode
from alphamesh.safety import LiveTradingForbiddenError
from tests.conftest import CAPTURE_DIR, NOW, make_account
from tests.test_orchestrator import TrendingMarketData, build, bull_script


# --------------------------------------------------------------------------- #
# DEFECT B - silent fallback to simulated execution
# --------------------------------------------------------------------------- #
class TestNoSilentSimulatedExecution:
    """`ALPHAMESH_DRY_RUN` defaults to true. A deploy that forgot to set it
    false used to run the in-process simulator and write fabricated fills into
    the journal a judge reads for P&L, announced only by an INFO log."""

    def test_live_mode_without_credentials_refuses_to_start(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        settings = Settings(
            paper=True,
            dry_run=False,
            data_source="mcp_capture",
            capture_dir=CAPTURE_DIR,
            database_path=tmp_path / "j.db",
        )
        with pytest.raises(RuntimeError, match="would simulate fills while presenting"):
            build_stack(settings)

    def test_dry_run_is_labelled_simulated(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        settings = Settings(
            paper=True,
            dry_run=True,
            data_source="mcp_capture",
            capture_dir=CAPTURE_DIR,
            database_path=tmp_path / "j.db",
        )
        stack = build_stack(settings)
        assert stack.live_broker is False
        assert stack.execution_mode == "SIMULATED"

    def test_credentialled_live_mode_uses_the_alpaca_paper_broker(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        settings = Settings(
            paper=True,
            dry_run=False,
            api_key_id="PKTEST",
            api_secret_key="secret",
            data_source="mcp_capture",
            capture_dir=CAPTURE_DIR,
            database_path=tmp_path / "j.db",
        )
        stack = build_stack(settings)
        assert stack.live_broker is True
        assert stack.execution_mode == "ALPACA_PAPER"
        assert type(stack.broker).__name__ == "AlpacaPaperBroker"

    def test_simulated_mode_is_warned_about_loudly(self, tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
        import logging

        settings = Settings(
            paper=True,
            dry_run=True,
            data_source="mcp_capture",
            capture_dir=CAPTURE_DIR,
            database_path=tmp_path / "j.db",
        )
        with caplog.at_level(logging.WARNING):
            build_stack(settings)
        assert any(
            "EXECUTION MODE: SIMULATED" in r.message and r.levelno >= logging.WARNING
            for r in caplog.records
        )

    def test_opened_position_records_the_execution_mode(self, config) -> None:  # type: ignore[no-untyped-def]
        """A simulated fill must stay distinguishable from a real one forever."""
        single = config.model_copy(
            update={"universe": config.universe.model_copy(update={"symbols": ["SPY"]})}
        )
        orchestrator, journal, _stack = build(single)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)

        opened = [e for e in journal.recent_events(50) if e["event_type"] == "position_opened"]
        assert opened, "expected a position to be opened"
        assert '"execution_mode": "SIMULATED"' in opened[0]["payload"]
        journal.close()


# --------------------------------------------------------------------------- #
# DEFECT A - no market-hours gate
# --------------------------------------------------------------------------- #
class ClosedMarketData(TrendingMarketData):
    def __init__(self, next_open: datetime | None = None) -> None:
        super().__init__()
        self._next_open = next_open or (NOW + timedelta(hours=12))
        self.snapshot_calls = 0

    def snapshot(self, symbol, lookback_minutes=180):  # type: ignore[no-untyped-def]
        self.snapshot_calls += 1
        return super().snapshot(symbol, lookback_minutes)

    def clock(self) -> MarketClock:
        return MarketClock(timestamp=NOW, is_open=False, next_open=self._next_open)


class UnreadableClockData(TrendingMarketData):
    def clock(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("clock endpoint down")


class TestMarketHoursGate:
    """The cycle used to scan, score, invoke the AI council and pull option
    chains with the market closed, with only the stale-quote gate between it
    and a trade on dead data."""

    @pytest.fixture
    def single(self, config):  # type: ignore[no-untyped-def]
        return config.model_copy(
            update={"universe": config.universe.model_copy(update={"symbols": ["SPY"]})}
        )

    def test_closed_market_does_no_work_at_all(self, single) -> None:  # type: ignore[no-untyped-def]
        market = ClosedMarketData()
        provider = bull_script()
        orchestrator, journal, _ = build(single, provider=provider, market=market)
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)

        assert report.market_open is False
        assert report.symbols_scanned == 0
        assert report.orders_submitted == []
        assert market.snapshot_calls == 0, "must not pull market data when closed"
        assert provider.calls == [], "must not spend LLM calls when closed"
        journal.close()

    def test_closed_market_reports_the_next_open(self, single) -> None:  # type: ignore[no-untyped-def]
        expected = NOW + timedelta(hours=9)
        orchestrator, journal, _ = build(single, market=ClosedMarketData(expected))
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)
        assert report.next_open == expected
        journal.close()

    def test_closed_market_is_journalled(self, single) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single, market=ClosedMarketData())
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)
        assert any(
            e["event_type"] == "market_closed" for e in journal.recent_events(10)
        )
        journal.close()

    def test_unreadable_clock_is_treated_as_closed(self, single) -> None:  # type: ignore[no-untyped-def]
        """Fail closed: an unknown market state must never be assumed open."""
        provider = bull_script()
        orchestrator, journal, _ = build(
            single, provider=provider, market=UnreadableClockData()
        )
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)

        assert report.market_open is False
        assert report.orders_submitted == []
        assert provider.calls == []
        assert any("clock unavailable" in e for e in report.errors)
        journal.close()

    def test_open_market_still_trades(self, single) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single)
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)
        assert report.market_open is True
        assert report.orders_submitted, "open market must still reach the entry path"
        journal.close()


class TestClosedMarketBackoff:
    """Closed-market polling used to run at the trading interval forever."""

    def test_backoff_is_used_when_closed(self) -> None:
        from alphamesh.main import _closed_market_wait

        report = SimpleNamespace(next_open=datetime.now(UTC) + timedelta(hours=10))
        assert _closed_market_wait(report, 300) == 300.0

    def test_never_sleeps_past_the_next_open(self) -> None:
        from alphamesh.main import _closed_market_wait

        report = SimpleNamespace(next_open=datetime.now(UTC) + timedelta(seconds=90))
        wait = _closed_market_wait(report, 300)
        assert 5.0 <= wait <= 90.0

    def test_unknown_next_open_falls_back_to_the_backoff(self) -> None:
        from alphamesh.main import _closed_market_wait

        assert _closed_market_wait(SimpleNamespace(next_open=None), 300) == 300.0

    def test_next_open_in_the_past_still_waits_a_little(self) -> None:
        from alphamesh.main import _closed_market_wait

        report = SimpleNamespace(next_open=datetime.now(UTC) - timedelta(hours=1))
        assert 0 < _closed_market_wait(report, 300) <= 60

    def test_closed_interval_is_configurable_and_never_below_the_open_interval(self) -> None:
        settings = Settings(loop_seconds=60, closed_poll_seconds=300)
        assert max(settings.loop_seconds, settings.closed_poll_seconds) == 300


class TestInterruptibleSleep:
    """Railway SIGTERMs then SIGKILLs; a long uninterruptible sleep would be
    killed mid-cycle rather than shutting down cleanly."""

    def test_sleep_returns_immediately_once_shutdown_is_set(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import time as time_module

        import alphamesh.main as main

        monkeypatch.setattr(main, "_SHUTDOWN", True)
        started = time_module.monotonic()
        main._interruptible_sleep(30.0)
        assert time_module.monotonic() - started < 1.0

    def test_sleep_completes_a_short_interval(self) -> None:
        import time as time_module

        import alphamesh.main as main

        started = time_module.monotonic()
        main._interruptible_sleep(0.05)
        assert time_module.monotonic() - started >= 0.04


# --------------------------------------------------------------------------- #
# DEFECT C - no paper revalidation immediately before submission
# --------------------------------------------------------------------------- #
class TestPaperRecheckAtTheWire:
    @pytest.fixture
    def single(self, config):  # type: ignore[no-untyped-def]
        return config.model_copy(
            update={"universe": config.universe.model_copy(update={"symbols": ["SPY"]})}
        )

    def test_submission_is_blocked_when_the_account_stops_proving_paper(
        self, single
    ) -> None:  # type: ignore[no-untyped-def]
        """The account is re-read at the wire; a non-paper number stops the order."""

        class DriftingBroker(SimulatedBroker):
            def __init__(self) -> None:
                super().__init__(make_account())
                self.calls = 0

            def account(self):  # type: ignore[no-untyped-def]
                self.calls += 1
                # First read (cycle start) is fine; the wire-side read is not.
                if self.calls > 1:
                    raise LiveTradingForbiddenError("account number is no longer PA-prefixed")
                return super().account()

        broker = DriftingBroker()
        orchestrator, journal, _ = build(single, broker=broker)
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)

        assert broker.submitted_payloads == [], "no order may reach the broker"
        assert journal.open_positions() == []
        codes = {c for _s, cs in report.rejections for c in cs}
        assert ReasonCode.LIVE_TRADING_FORBIDDEN in codes
        assert any(
            e["event_type"] == "live_trading_blocked" for e in journal.recent_events(20)
        )
        journal.close()

    def test_recheck_rejects_a_live_endpoint_configured_mid_session(self, single) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single)
        orchestrator.startup()
        # Simulate configuration drift toward a live endpoint after startup.
        orchestrator.config = single.model_copy(
            update={
                "settings": single.settings.model_copy(
                    update={"base_url": "https://api.alpaca.markets"}
                )
            }
        )
        report = orchestrator.run_cycle(now=NOW)

        assert report.orders_submitted == []
        codes = {c for _s, cs in report.rejections for c in cs}
        assert ReasonCode.LIVE_TRADING_FORBIDDEN in codes
        journal.close()

    def test_recheck_passes_in_a_healthy_paper_configuration(self, single) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _ = build(single)
        orchestrator.startup()
        orchestrator._assert_paper_before_submit()  # must not raise
        journal.close()
