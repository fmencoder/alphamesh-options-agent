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


def raw_snapshot(delta: float, bid: float, ask: float, volume: int = 500) -> dict:
    """A snapshot in Alpaca's wire shape.

    These are the verbatim JSON keys the options snapshot endpoint returns.
    The previous version of this fixture was a Python object carrying a
    ``daily_bar`` attribute, which alpaca-py's OptionsSnapshot model does not
    have -- the fake was more capable than the real SDK, so a mapping bug that
    zeroed day_volume in production passed these tests for free.
    """
    return {
        "latestQuote": {
            "bp": bid,
            "ap": ask,
            "bs": 10,
            "as": 12,
            "t": "2026-08-28T19:59:59.554748431Z",
        },
        "greeks": {"delta": delta, "gamma": 0.04, "theta": -0.3, "vega": 0.35, "rho": 0.05},
        "impliedVolatility": 0.088,
        "dailyBar": {"v": volume, "c": ask, "o": bid},
        "latestTrade": {"p": (bid + ask) / 2, "s": 4, "t": "2026-08-28T19:59:58Z"},
    }


class TestRestOptionChain:
    def test_maps_alpaca_snapshots_onto_contract_candidates(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestOptionChain("key", "secret")
        raw = {
            "SPY260903C00769000": raw_snapshot(0.5391, 3.86, 3.94, 563),
            "SPY260903C00774000": raw_snapshot(0.3037, 1.50, 1.60, 585),
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
                get_option_chain=lambda request: {"GARBAGE": raw_snapshot(0.5, 1, 2)}
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


class TestOptionChainDayVolumeRegression:
    """Regression cover for the live 2026-08-31 contract-selection failure.

    Every tradable decision that session died at contract selection with
    ILLIQUID_CONTRACT. Cause: the adapter read ``snap.daily_bar``, but
    alpaca-py's OptionsSnapshot model does not declare that field, so
    day_volume was always 0 and every contract failed
    min_contract_day_volume regardless of how liquid it really was.
    """

    def test_sdk_model_really_has_no_daily_bar(self) -> None:
        """Pin the SDK reality the old hand-rolled fake contradicted."""
        from alpaca.data.models.snapshots import OptionsSnapshot

        assert "daily_bar" not in OptionsSnapshot.model_fields
        wire = raw_snapshot(0.54, 4.34, 4.53, 2738)
        parsed = OptionsSnapshot("SPY260904C00765000", wire)
        # The volume is on the wire but the model drops it entirely.
        assert wire["dailyBar"]["v"] == 2738
        assert getattr(parsed, "daily_bar", None) is None

    def test_client_requests_raw_data(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        captured: dict = {}

        class FakeClient:
            def __init__(self, key, secret, raw_data=False):  # type: ignore[no-untyped-def]
                captured["raw_data"] = raw_data

        import alpaca.data.historical.option as option_module

        monkeypatch.setattr(
            option_module, "OptionHistoricalDataClient", FakeClient, raising=True
        )
        AlpacaRestOptionChain("key", "secret")._option_client()
        assert captured["raw_data"] is True

    def test_day_volume_survives_the_mapping(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        provider = AlpacaRestOptionChain("key", "secret")
        monkeypatch.setattr(
            provider,
            "_option_client",
            lambda: SimpleNamespace(
                get_option_chain=lambda request: {
                    "SPY260904C00765000": raw_snapshot(0.5438, 4.34, 4.53, 2738)
                }
            ),
        )
        contracts = provider.chain("SPY", OptionType.CALL, date(2026, 8, 31), 2, 10)
        assert contracts[0].day_volume == 2738

    def test_live_payload_now_yields_a_spread(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """The exact strikes that were rejected live must now select."""
        from alphamesh.config import load_config
        from alphamesh.models.domain import Strategy
        from alphamesh.strategies.contracts import select_vertical_spread

        provider = AlpacaRestOptionChain("key", "secret")
        monkeypatch.setattr(
            provider,
            "_option_client",
            lambda: SimpleNamespace(
                get_option_chain=lambda request: {
                    "SPY260904C00765000": raw_snapshot(0.5438, 4.34, 4.53, 2738),
                    "SPY260904C00770000": raw_snapshot(0.3331, 1.98, 1.99, 3026),
                }
            ),
        )
        chain = provider.chain("SPY", OptionType.CALL, date(2026, 8, 31), 2, 10)
        cfg = load_config()
        now = datetime(2026, 8, 28, 20, 0, tzinfo=UTC)
        result = select_vertical_spread(
            Strategy.BULL_CALL_SPREAD, "SPY", chain, cfg.strategies, cfg.risk,
            now, date(2026, 8, 31),
        )
        assert result.spread is not None, result.reason_codes
        assert result.spread.long_leg.contract.strike == 765.0
        assert result.spread.short_leg.contract.strike == 770.0

    def test_genuinely_illiquid_contracts_are_still_rejected(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """The gate was fixed, not weakened: real low volume must still fail."""
        from alphamesh.config import load_config
        from alphamesh.models.domain import ReasonCode, Strategy
        from alphamesh.strategies.contracts import select_vertical_spread

        provider = AlpacaRestOptionChain("key", "secret")
        monkeypatch.setattr(
            provider,
            "_option_client",
            lambda: SimpleNamespace(
                get_option_chain=lambda request: {
                    # Below min_contract_day_volume (25) on the wire.
                    "SPY260904C00765000": raw_snapshot(0.5438, 4.34, 4.53, 3),
                    "SPY260904C00770000": raw_snapshot(0.3331, 1.98, 1.99, 2),
                }
            ),
        )
        chain = provider.chain("SPY", OptionType.CALL, date(2026, 8, 31), 2, 10)
        assert [c.day_volume for c in chain] == [3, 2]
        cfg = load_config()
        result = select_vertical_spread(
            Strategy.BULL_CALL_SPREAD, "SPY", chain, cfg.strategies, cfg.risk,
            datetime(2026, 8, 28, 20, 0, tzinfo=UTC), date(2026, 8, 31),
        )
        assert result.spread is None
        assert ReasonCode.ILLIQUID_CONTRACT in result.reason_codes
