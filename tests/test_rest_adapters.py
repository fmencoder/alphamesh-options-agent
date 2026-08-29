"""Alpaca REST adapters.

The REST path could not be exercised over the network from the build
environment (outbound access to Alpaca is blocked there), so these tests cover
the part that is ours: request construction and the mapping from alpaca-py
objects onto AlphaMesh domain models. The clients themselves are stubbed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from alphamesh.alpaca.execution import AlpacaPaperBroker, BrokerError
from alphamesh.alpaca.market_data import AlpacaRestMarketData, MarketDataUnavailableError
from alphamesh.alpaca.options import AlpacaRestOptionChain
from alphamesh.models.domain import OptionType
from alphamesh.safety import LiveTradingForbiddenError


class FakeBar:
    def __init__(self, minute: int, close: float) -> None:
        self.timestamp = datetime(2026, 8, 28, 19, minute, tzinfo=UTC)
        self.open = close - 0.1
        self.high = close + 0.2
        self.low = close - 0.3
        self.close = close
        self.volume = 1000 + minute
        self.vwap = close


class TestRestMarketData:
    def test_maps_alpaca_bars_onto_the_domain_model(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestMarketData("key", "secret", feed="iex")
        bars = [FakeBar(m, 769.0 + m * 0.01) for m in range(5)]
        monkeypatch.setattr(
            provider,
            "_stock_client",
            lambda: SimpleNamespace(
                get_stock_bars=lambda request: SimpleNamespace(data={"SPY": bars})
            ),
        )
        snapshot = provider.snapshot("SPY", lookback_minutes=180)
        assert snapshot.symbol == "SPY"
        assert snapshot.bar_count == 5
        assert snapshot.last_price == pytest.approx(769.04)
        assert snapshot.bars[0].vwap == pytest.approx(769.0)

    def test_empty_response_is_an_explicit_failure(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestMarketData("key", "secret")
        monkeypatch.setattr(
            provider,
            "_stock_client",
            lambda: SimpleNamespace(
                get_stock_bars=lambda request: SimpleNamespace(data={})
            ),
        )
        with pytest.raises(MarketDataUnavailableError):
            provider.snapshot("SPY")

    def test_window_is_trimmed_to_the_lookback(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestMarketData("key", "secret")
        bars = [FakeBar(m % 60, 769.0) for m in range(200)]
        monkeypatch.setattr(
            provider,
            "_stock_client",
            lambda: SimpleNamespace(
                get_stock_bars=lambda request: SimpleNamespace(data={"SPY": bars})
            ),
        )
        assert provider.snapshot("SPY", lookback_minutes=50).bar_count == 50


class FakeSnapshot:
    def __init__(self, delta: float, bid: float, ask: float, volume: int = 500) -> None:
        self.latest_quote = SimpleNamespace(
            bid_price=bid,
            ask_price=ask,
            bid_size=10,
            ask_size=12,
            timestamp=datetime(2026, 8, 28, 19, 59, 59, tzinfo=UTC),
        )
        self.greeks = SimpleNamespace(delta=delta, gamma=0.04, theta=-0.3, vega=0.35)
        self.implied_volatility = 0.088
        self.daily_bar = SimpleNamespace(volume=volume)


class TestRestOptionChain:
    def test_maps_alpaca_snapshots_onto_contract_candidates(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestOptionChain("key", "secret")
        raw = {
            "SPY260903C00769000": FakeSnapshot(0.5391, 3.86, 3.94, 563),
            "SPY260903C00774000": FakeSnapshot(0.3037, 1.50, 1.60, 585),
        }
        monkeypatch.setattr(
            provider,
            "_option_client",
            lambda: SimpleNamespace(get_option_chain=lambda request: raw),
        )
        contracts = provider.chain("SPY", OptionType.CALL, date(2026, 8, 29), 2, 10)
        assert len(contracts) == 2
        by_symbol = {c.symbol: c for c in contracts}
        long_call = by_symbol["SPY260903C00769000"]
        assert long_call.strike == 769.0
        assert long_call.expiration == date(2026, 9, 3)
        assert long_call.greeks.delta == pytest.approx(0.5391)
        assert long_call.quote.bid == 3.86
        assert long_call.day_volume == 563

    def test_unparseable_symbols_are_dropped(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestOptionChain("key", "secret")
        monkeypatch.setattr(
            provider,
            "_option_client",
            lambda: SimpleNamespace(
                get_option_chain=lambda request: {"GARBAGE": FakeSnapshot(0.5, 1, 2)}
            ),
        )
        assert provider.chain("SPY", OptionType.CALL, date(2026, 8, 29), 2, 10) == []

    def test_request_is_built_for_the_right_dte_window(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        captured: dict = {}

        def capture(request):  # type: ignore[no-untyped-def]
            captured["request"] = request
            return {}

        provider = AlpacaRestOptionChain("key", "secret")
        monkeypatch.setattr(
            provider, "_option_client", lambda: SimpleNamespace(get_option_chain=capture)
        )
        provider.chain("SPY", OptionType.PUT, date(2026, 8, 29), 2, 10)
        request = captured["request"]
        assert request.underlying_symbol == "SPY"
        assert request.type.value == "put"
        assert request.expiration_date_gte == date(2026, 8, 31)
        assert request.expiration_date_lte == date(2026, 9, 8)


class TestPaperBrokerMapping:
    def test_order_response_maps_onto_an_execution_record(self) -> None:
        record = AlpacaPaperBroker._to_record(
            "alphamesh-SPY-BCS-abc",
            {
                "id": "e1b2c3",
                "status": "filled",
                "filled_qty": "2",
                "filled_avg_price": "2.38",
                "submitted_at": "2026-08-31T13:31:00Z",
                "updated_at": "2026-08-31T13:31:02Z",
            },
        )
        assert record.broker_order_id == "e1b2c3"
        assert record.filled_quantity == 2
        assert record.filled_avg_price_cents == 238
        assert record.submitted_at.year == 2026

    def test_unfilled_order_has_no_fill_price(self) -> None:
        record = AlpacaPaperBroker._to_record("cid", {"id": "x", "status": "new"})
        assert record.filled_quantity == 0
        assert record.filled_avg_price_cents is None

    def test_account_read_enforces_the_paper_prefix(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        broker = AlpacaPaperBroker("key", "secret")
        monkeypatch.setattr(
            broker, "_request", lambda *a, **k: {"account_number": "920000001"}
        )
        with pytest.raises(LiveTradingForbiddenError):
            broker.account()

    def test_account_read_accepts_a_paper_account(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        broker = AlpacaPaperBroker("key", "secret")
        monkeypatch.setattr(
            broker,
            "_request",
            lambda *a, **k: {
                "account_number": "PA0000EXAMPLE",
                "status": "ACTIVE",
                "equity": "100000",
                "last_equity": "100000",
                "cash": "100000",
                "buying_power": "200000",
                "options_buying_power": "100000",
                "options_trading_level": 3,
                "trading_blocked": False,
                "account_blocked": False,
            },
        )
        account = broker.account()
        assert account.is_tradeable
        assert account.options_trading_level == 3

    def test_missing_order_lookup_returns_none(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        broker = AlpacaPaperBroker("key", "secret")

        def boom(*a, **k):  # type: ignore[no-untyped-def]
            raise BrokerError("404")

        monkeypatch.setattr(broker, "_request", boom)
        assert broker.get_order_by_client_id("nope") is None

    def test_close_payload_flips_both_legs_to_closing_intents(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from alphamesh.execution.order_builder import build_order_intent
        from tests.conftest import NOW, make_decision
        from tests.test_execution import approved, make_spread

        broker = AlpacaPaperBroker("key", "secret")
        captured: dict = {}

        def capture(method, path, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs.get("json", {}))
            return {"id": "x", "status": "new"}

        monkeypatch.setattr(broker, "_request", capture)
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        broker.close_spread(intent, 300, "alphamesh-SPY-BCS-close")

        assert [leg["side"] for leg in captured["legs"]] == ["sell", "buy"]
        assert [leg["position_intent"] for leg in captured["legs"]] == [
            "sell_to_close",
            "buy_to_close",
        ]
        assert captured["limit_price"] == "3.00"
