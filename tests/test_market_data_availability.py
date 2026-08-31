"""Closed-market market-data availability.

Regression cover for the defect that made the real Railway preflight fail:
the historical bar window was `now - N minutes`, which contains zero market
minutes on a weekend and falls short of the previous session at the opening
bell. These tests pin the fix and the failure classification around it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from alphamesh.alpaca.client import AlpacaStack
from alphamesh.alpaca.execution import SimulatedBroker
from alphamesh.alpaca.market_data import (
    BAR_REQUEST_LIMIT,
    LOOKBACK_CALENDAR_DAYS,
    AlpacaRestMarketData,
    MarketDataUnavailableError,
)
from alphamesh.alpaca.options import CaptureOptionChain
from alphamesh.alpaca.types import MarketClock
from alphamesh.config import Settings, load_config
from alphamesh.main import _describe_data_error, cmd_preflight
from alphamesh.models.domain import Bar
from alphamesh.safety import GuardResult
from tests.conftest import CAPTURE_DIR, make_account
from tests.test_preflight import parse_flags

# Friday 2026-08-28, the last completed session before the weekend.
LAST_SESSION_CLOSE = datetime(2026, 8, 28, 20, 0, tzinfo=UTC)

SATURDAY = datetime(2026, 8, 29, 22, 57, tzinfo=UTC)
SUNDAY = datetime(2026, 8, 30, 15, 0, tzinfo=UTC)
HOLIDAY = datetime(2026, 9, 7, 15, 0, tzinfo=UTC)  # Labor Day
OPEN_DAY = datetime(2026, 8, 31, 14, 0, tzinfo=UTC)


def session_bars(count: int = 180, end: datetime = LAST_SESSION_CLOSE) -> list[Bar]:
    """Real-shaped bars from the last completed session."""
    return [
        Bar(
            timestamp=end - timedelta(minutes=count - i),
            open=769.0,
            high=769.2,
            low=768.8,
            close=769.0 + i * 0.001,
            volume=50_000,
            vwap=769.0,
        )
        for i in range(count)
    ]


class FakeMarketData:
    """Provider whose latest-bar and historical behaviour can be steered per symbol."""

    def __init__(
        self,
        is_open: bool = False,
        bars: int = 180,
        latest_error: dict[str, Exception] | None = None,
        history_error: dict[str, Exception] | None = None,
        bars_by_symbol: dict[str, int] | None = None,
        feed: str = "iex",
    ) -> None:
        self.is_open = is_open
        self.bars = bars
        self.latest_error = latest_error or {}
        self.history_error = history_error or {}
        self.bars_by_symbol = bars_by_symbol or {}
        self.feed = feed

    def latest_bar(self, symbol: str) -> Bar:
        if symbol in self.latest_error:
            raise self.latest_error[symbol]
        return session_bars(1)[0]

    def snapshot(self, symbol: str, lookback_minutes: int = 180):  # type: ignore[no-untyped-def]
        from alphamesh.models.domain import MarketSnapshot

        if symbol in self.history_error:
            raise self.history_error[symbol]
        count = self.bars_by_symbol.get(symbol, self.bars)
        if count == 0:
            raise MarketDataUnavailableError(
                f"Alpaca returned no bars for {symbol} over the last "
                f"{LOOKBACK_CALENDAR_DAYS} calendar days on feed {self.feed}"
            )
        series = session_bars(count)
        return MarketSnapshot(
            symbol=symbol,
            as_of=series[-1].timestamp,
            last_price=series[-1].close,
            session_open=series[0].open,
            bars=tuple(series),
        )

    def clock(self) -> MarketClock:
        return MarketClock(
            timestamp=SATURDAY,
            is_open=self.is_open,
            next_open=datetime(2026, 8, 31, 13, 30, tzinfo=UTC),
        )


def run_preflight(monkeypatch, capsys, market, tmp_path):  # type: ignore[no-untyped-def]
    """Drive the real cmd_preflight with a steered market-data provider."""
    broker = SimulatedBroker(make_account())
    stack = AlpacaStack(
        guard=GuardResult(paper=True, detail="test", checks=("ALPACA_PAPER=true",)),
        market_data=market,
        option_chain=CaptureOptionChain(CAPTURE_DIR),
        broker=broker,
        live_broker=False,
    )
    monkeypatch.setattr("alphamesh.main.build_stack", lambda _s: stack)
    config = _captured_universe(
        load_config(
            settings=Settings(
                paper=True,
                dry_run=True,
                data_source="mcp_capture",
                capture_dir=CAPTURE_DIR,
                database_path=tmp_path / "pf.db",
            )
        )
    )
    code = cmd_preflight(config)
    out = capsys.readouterr().out
    return code, parse_flags(out), out, broker



def _captured_universe(config):  # type: ignore[no-untyped-def]
    """Pin the universe to the symbols the offline capture actually contains.

    The live universe is eight symbols; the MCP capture fixtures cover SPY and
    QQQ. Preflight correctly fails when asked for data that genuinely is not
    there, so these offline tests scope the universe to what was captured
    rather than relaxing the gate.
    """
    return config.model_copy(
        update={"universe": config.universe.model_copy(update={"symbols": ["SPY", "QQQ"]})}
    )


class TestClosedMarketPreflightPasses:
    """1-4: the gate must pass whenever real data is reachable, open or shut."""

    @pytest.mark.parametrize(
        "label", ["saturday", "sunday", "market_holiday", "ordinary_open_day"]
    )
    def test_preflight_passes_regardless_of_session_state(
        self, label, monkeypatch, capsys, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        is_open = label == "ordinary_open_day"
        code, flags, _out, _ = run_preflight(
            monkeypatch, capsys, FakeMarketData(is_open=is_open), tmp_path
        )
        assert flags["PREFLIGHT_MARKET_DATA"] == "PASS", label
        assert flags["PREFLIGHT_READY"] == "YES", label
        assert code == 0

    def test_closed_market_is_reported_not_treated_as_failure(
        self, monkeypatch, capsys, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        _code, flags, out, _ = run_preflight(
            monkeypatch, capsys, FakeMarketData(is_open=False), tmp_path
        )
        assert '"is_open": false' in out
        assert flags["PREFLIGHT_MARKET_DATA"] == "PASS"
        assert "MARKET_CLOSED from" in out  # the distinguishing note

    def test_latest_bar_from_the_previous_session_is_recorded(
        self, monkeypatch, capsys, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        _code, _flags, out, _ = run_preflight(
            monkeypatch, capsys, FakeMarketData(is_open=False), tmp_path
        )
        assert '"latest_bar_source": "alpaca_latest_bar"' in out
        assert '"latest_bar_timestamp": "2026-08-28' in out


class TestMarketDataFailuresStillFail:
    """5-9: the gate must not be softened by the closed-market fix."""

    def test_genuinely_no_data_fails(self, monkeypatch, capsys, tmp_path) -> None:  # type: ignore[no-untyped-def]
        code, flags, out, _ = run_preflight(
            monkeypatch, capsys, FakeMarketData(bars=0), tmp_path
        )
        assert flags["PREFLIGHT_MARKET_DATA"] == "FAIL"
        assert flags["PREFLIGHT_READY"] == "NO"
        assert code == 1
        assert "NO_DATA_RETURNED" in out

    def test_401_fails_with_an_auth_reason(self, monkeypatch, capsys, tmp_path) -> None:  # type: ignore[no-untyped-def]
        err = RuntimeError('GET /v2/stocks/bars -> 401: {"message": "unauthorized."}')
        market = FakeMarketData(
            latest_error={"SPY": err, "QQQ": err}, history_error={"SPY": err, "QQQ": err}
        )
        code, flags, out, _ = run_preflight(monkeypatch, capsys, market, tmp_path)
        assert flags["PREFLIGHT_MARKET_DATA"] == "FAIL"
        assert "AUTH_FAILED (401)" in out
        assert code == 1


    def test_403_entitlement_fails_with_a_precise_reason(
        self, monkeypatch, capsys, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        err = RuntimeError("403 Forbidden: subscription does not permit sip feed")
        market = FakeMarketData(
            latest_error={"SPY": err, "QQQ": err}, history_error={"SPY": err, "QQQ": err}
        )
        _code, flags, out, _ = run_preflight(monkeypatch, capsys, market, tmp_path)
        assert flags["PREFLIGHT_MARKET_DATA"] == "FAIL"
        assert "FEED_NOT_ENTITLED (403)" in out
        assert "AUTH_FAILED" not in out

    def test_one_symbol_succeeding_does_not_carry_the_other(
        self, monkeypatch, capsys, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """SPY healthy, QQQ starved -> the gate must still fail."""
        market = FakeMarketData(bars_by_symbol={"SPY": 180, "QQQ": 0})
        code, flags, out, _ = run_preflight(monkeypatch, capsys, market, tmp_path)
        assert flags["PREFLIGHT_MARKET_DATA"] == "FAIL"
        assert code == 1
        assert '"status": "OK"' in out  # SPY passed
        assert '"status": "MARKET_DATA_UNAVAILABLE"' in out  # QQQ did not

    def test_insufficient_depth_fails_even_with_a_latest_bar(
        self, monkeypatch, capsys, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """A reachable feed with too little history is not tradable."""
        _code, flags, out, _ = run_preflight(
            monkeypatch, capsys, FakeMarketData(bars=5), tmp_path
        )
        assert flags["PREFLIGHT_MARKET_DATA"] == "FAIL"
        assert '"sufficient": false' in out

    def test_latest_bar_failure_alone_fails_the_gate(
        self, monkeypatch, capsys, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        market = FakeMarketData(
            latest_error={"SPY": MarketDataUnavailableError("no latest bar for SPY")}
        )
        _code, flags, out, _ = run_preflight(monkeypatch, capsys, market, tmp_path)
        assert flags["PREFLIGHT_MARKET_DATA"] == "FAIL"
        assert "NO_DATA_RETURNED" in out


class TestStillZeroOrder:
    """10: none of the above may loosen the zero-order guarantee."""

    def test_preflight_submits_nothing_on_the_happy_path(
        self, monkeypatch, capsys, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        _code, _flags, out, broker = run_preflight(
            monkeypatch, capsys, FakeMarketData(), tmp_path
        )
        assert broker.submitted_payloads == []
        assert broker.orders == {}
        assert '"orders_submitted": 0' in out

    def test_preflight_submits_nothing_when_market_data_fails(
        self, monkeypatch, capsys, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        _code, _flags, _out, broker = run_preflight(
            monkeypatch, capsys, FakeMarketData(bars=0), tmp_path
        )
        assert broker.submitted_payloads == []


class TestErrorClassification:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ('401: {"message": "unauthorized."}', "AUTH_FAILED (401)"),
            ("Unauthorized", "AUTH_FAILED (401)"),
            ("403 Forbidden", "FEED_NOT_ENTITLED (403)"),
            ("subscription required for sip", "FEED_NOT_ENTITLED (403)"),
            ("Alpaca returned no bars for SPY", "NO_DATA_RETURNED"),
            ("Alpaca returned no latest bar for QQQ", "NO_DATA_RETURNED"),
        ],
    )
    def test_failures_are_classified_precisely(self, message, expected) -> None:  # type: ignore[no-untyped-def]
        assert _describe_data_error(RuntimeError(message)).startswith(expected)

    def test_unknown_failure_keeps_its_type(self) -> None:
        assert _describe_data_error(ValueError("weird")).startswith("ValueError")


class TestRestRequestWindow:
    """The actual defect: the request window must be calendar-day based."""

    def _capture_request(self, monkeypatch):  # type: ignore[no-untyped-def]
        provider = AlpacaRestMarketData("k", "s", feed="iex")
        captured: dict = {}

        def get_stock_bars(request):  # type: ignore[no-untyped-def]
            captured["request"] = request
            return SimpleNamespace(data={"SPY": []})

        monkeypatch.setattr(
            provider, "_stock_client", lambda: SimpleNamespace(get_stock_bars=get_stock_bars)
        )
        with pytest.raises(MarketDataUnavailableError):
            provider.snapshot("SPY", lookback_minutes=180)
        return captured["request"]

    def test_window_spans_calendar_days_not_minutes(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        request = self._capture_request(monkeypatch)
        span = request.end - request.start
        assert span >= timedelta(days=LOOKBACK_CALENDAR_DAYS)

    def test_window_is_wide_enough_to_cross_a_long_holiday_weekend(
        self, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Thu holiday + Fri + Sat + Sun is the worst realistic gap."""
        request = self._capture_request(monkeypatch)
        assert (request.end - request.start) >= timedelta(days=4)

    def test_old_minute_based_window_would_have_been_too_narrow(self) -> None:
        """Pins the original bug so it cannot silently return."""
        old_span = timedelta(minutes=180 * 3)
        assert old_span < timedelta(days=1)
        saturday_start = SATURDAY - old_span
        assert saturday_start.strftime("%A") == "Saturday"  # no market minutes at all

    def test_request_is_per_symbol_and_bounded(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        request = self._capture_request(monkeypatch)
        assert request.symbol_or_symbols == "SPY"  # never a shared multi-symbol limit
        assert request.limit == BAR_REQUEST_LIMIT

    def test_feed_is_not_hardcoded_to_a_paid_tier(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        request = self._capture_request(monkeypatch)
        assert request.feed.value == "iex"

    def test_error_message_names_the_window_and_feed(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestMarketData("k", "s", feed="iex")
        monkeypatch.setattr(
            provider,
            "_stock_client",
            lambda: SimpleNamespace(
                get_stock_bars=lambda r: SimpleNamespace(data={"SPY": []})
            ),
        )
        with pytest.raises(MarketDataUnavailableError, match="calendar days on feed iex"):
            provider.snapshot("SPY")


class TestRestLatestBar:
    def test_maps_the_latest_bar(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestMarketData("k", "s")
        raw = SimpleNamespace(
            timestamp=LAST_SESSION_CLOSE,
            open=769.6,
            high=769.8,
            low=769.1,
            close=769.34,
            volume=1_292_176,
            vwap=769.34,
        )
        monkeypatch.setattr(
            provider,
            "_stock_client",
            lambda: SimpleNamespace(get_stock_latest_bar=lambda r: {"SPY": raw}),
        )
        bar = provider.latest_bar("SPY")
        assert bar.close == 769.34
        assert bar.timestamp == LAST_SESSION_CLOSE

    def test_missing_symbol_raises(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestMarketData("k", "s")
        monkeypatch.setattr(
            provider,
            "_stock_client",
            lambda: SimpleNamespace(get_stock_latest_bar=lambda r: {}),
        )
        with pytest.raises(MarketDataUnavailableError, match="no latest bar"):
            provider.latest_bar("SPY")

    def test_uses_the_configured_feed(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestMarketData("k", "s", feed="iex")
        captured: dict = {}

        def get_latest(request):  # type: ignore[no-untyped-def]
            captured["request"] = request
            return {}

        monkeypatch.setattr(
            provider,
            "_stock_client",
            lambda: SimpleNamespace(get_stock_latest_bar=get_latest),
        )
        with pytest.raises(MarketDataUnavailableError):
            provider.latest_bar("SPY")
        assert captured["request"].feed.value == "iex"


class TestCaptureProviderParity:
    def test_capture_source_also_implements_latest_bar(self) -> None:
        from alphamesh.alpaca.market_data import CaptureMarketData

        source = CaptureMarketData(CAPTURE_DIR)
        assert source.latest_bar("SPY").close == pytest.approx(769.34)
