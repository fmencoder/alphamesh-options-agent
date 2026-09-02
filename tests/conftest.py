"""Shared fixtures.

No test in this suite makes a network call or requires an LLM key.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from alphamesh.alpaca.options import CaptureOptionChain, build_occ_symbol
from alphamesh.alpaca.types import AccountState
from alphamesh.config import AppConfig, Settings, load_config
from alphamesh.models.domain import (
    Bar,
    Direction,
    Greeks,
    MarketSnapshot,
    OptionContractCandidate,
    OptionQuote,
    OptionType,
    QuantSignal,
    Regime,
    RegimeAssessment,
    Strategy,
    TradeDecision,
)
from alphamesh.risk.portfolio import PortfolioState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = PROJECT_ROOT / "data" / "mcp_capture"

NOW = datetime(2026, 8, 28, 19, 59, 30, tzinfo=UTC)
TODAY = date(2026, 8, 29)


@pytest.fixture
def config() -> AppConfig:
    return load_config(
        settings=Settings(
            paper=True,
            data_source="mcp_capture",
            capture_dir=CAPTURE_DIR,
            database_path=Path(":memory:"),
            dry_run=True,
        )
    )


@pytest.fixture
def limits(config: AppConfig):  # type: ignore[no-untyped-def]
    return config.risk


@pytest.fixture
def strategies(config: AppConfig):  # type: ignore[no-untyped-def]
    return config.strategies


@pytest.fixture
def now() -> datetime:
    return NOW


class PinnedCaptureOptionChain:
    """A capture chain whose evaluation date is the capture's own market date.

    The committed option snapshots are a photograph of 2026-08-28, with
    expirations on 2026-09-03 and 2026-09-04. Production correctly measures DTE
    against the wall clock, so as real time advances those fixed expirations
    drift out of the configured 2-10 DTE window and eventually vanish from the
    chain entirely. Any test that drives the real preflight over this fixture
    therefore passes or fails depending on the calendar date it is run, which
    is not a property a test suite may have: it went red overnight on 2026-09-02
    with no code change, because 2026-09-03 became 1 DTE against a min_dte of 2.

    Substituting the fixture's own date makes a historical snapshot behave like
    the day it was taken. Nothing about production DTE handling changes -- the
    live chain is still measured against the live clock -- and no gate is
    relaxed: a symbol genuinely absent from the capture still fails, which is
    what the surrounding tests rely on.
    """

    def __init__(self, capture_dir: Path, as_of: date = TODAY) -> None:
        self._inner = CaptureOptionChain(capture_dir)
        self._as_of = as_of

    def chain(
        self,
        underlying: str,
        option_type: OptionType,
        as_of: date,
        min_dte: int,
        max_dte: int,
    ) -> list[OptionContractCandidate]:
        # The caller's as_of is deliberately discarded: it is the wall clock,
        # and the whole point is that this fixture predates it.
        return self._inner.chain(underlying, option_type, self._as_of, min_dte, max_dte)


@pytest.fixture
def pin_capture_chain_to_fixture_date(monkeypatch):  # type: ignore[no-untyped-def]
    """Make ``build_stack`` hand out a date-pinned capture chain.

    Used by the tests that drive the real ``cmd_preflight``, which builds its own
    stack internally and so cannot be given a provider directly.
    """
    monkeypatch.setattr(
        "alphamesh.alpaca.client.CaptureOptionChain", PinnedCaptureOptionChain
    )


@pytest.fixture
def capture_chain() -> CaptureOptionChain:
    return CaptureOptionChain(CAPTURE_DIR)


@pytest.fixture
def spy_calls(capture_chain: CaptureOptionChain) -> list[OptionContractCandidate]:
    return capture_chain.chain("SPY", OptionType.CALL, TODAY, 2, 10)


@pytest.fixture
def spy_puts(capture_chain: CaptureOptionChain) -> list[OptionContractCandidate]:
    return capture_chain.chain("SPY", OptionType.PUT, TODAY, 2, 10)


@pytest.fixture
def qqq_calls(capture_chain: CaptureOptionChain) -> list[OptionContractCandidate]:
    return capture_chain.chain("QQQ", OptionType.CALL, TODAY, 2, 10)


def make_contract(
    underlying: str = "SPY",
    strike: float = 770.0,
    option_type: OptionType = OptionType.CALL,
    delta: float | None = 0.55,
    bid: float = 3.00,
    ask: float = 3.10,
    bid_size: int = 50,
    ask_size: int = 50,
    day_volume: int = 500,
    expiration: date = date(2026, 9, 3),
    quote_at: datetime | None = None,
    with_quote: bool = True,
) -> OptionContractCandidate:
    """Build a synthetic contract for boundary tests.

    Synthetic contracts exist only to drive edge cases the captured chain does
    not contain (missing greeks, stale quotes, crossed markets). Anything that
    asserts real behaviour uses the captured Alpaca chain instead.
    """
    quote = None
    if with_quote:
        quote = OptionQuote(
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            quote_timestamp=quote_at or NOW,
        )
    return OptionContractCandidate(
        symbol=build_occ_symbol(underlying, expiration, option_type, strike),
        underlying=underlying,
        expiration=expiration,
        option_type=option_type,
        strike=strike,
        quote=quote,
        greeks=Greeks(delta=delta, gamma=0.04, theta=-0.3, vega=0.35, implied_volatility=0.09),
        day_volume=day_volume,
    )


def make_bars(
    count: int = 120,
    start_price: float = 500.0,
    drift_per_bar: float = 0.0,
    volume: float = 50_000.0,
    start: datetime = NOW - timedelta(minutes=120),
    wobble: float = 0.0,
) -> tuple[Bar, ...]:
    """Deterministic bar series with a controllable drift."""
    bars: list[Bar] = []
    price = start_price
    for i in range(count):
        offset = wobble if i % 2 == 0 else -wobble
        close = price + drift_per_bar + offset
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=i),
                open=price,
                high=max(price, close) + 0.02,
                low=min(price, close) - 0.02,
                close=close,
                volume=volume,
                vwap=(price + close) / 2,
            )
        )
        price = close
    return tuple(bars)


def make_snapshot(
    symbol: str = "SPY", bars: tuple[Bar, ...] | None = None
) -> MarketSnapshot:
    series = bars if bars is not None else make_bars()
    return MarketSnapshot(
        symbol=symbol,
        as_of=series[-1].timestamp,
        last_price=series[-1].close,
        session_open=series[0].open,
        bars=series,
    )


def make_account(
    equity: float = 100_000.0,
    options_buying_power: float = 100_000.0,
    options_level: int = 3,
    status: str = "ACTIVE",
    trading_blocked: bool = False,
    account_number: str = "PA3TESTACCT1",
) -> AccountState:
    return AccountState(
        account_number=account_number,
        status=status,
        equity=equity,
        last_equity=equity,
        cash=equity,
        buying_power=equity * 2,
        options_buying_power=options_buying_power,
        options_trading_level=options_level,
        trading_blocked=trading_blocked,
        account_blocked=False,
    )


def make_portfolio(**kwargs) -> PortfolioState:  # type: ignore[no-untyped-def]
    kwargs.setdefault("account", make_account())
    # A readable broker reporting no exposure is the normal case. The
    # production default stays False so an unreadable broker fails closed;
    # tests that care about that set it explicitly.
    kwargs.setdefault("broker_truth_available", True)
    return PortfolioState(**kwargs)


def make_decision(
    symbol: str = "SPY",
    strategy: Strategy = Strategy.BULL_CALL_SPREAD,
    confidence: float = 0.70,
    regime: Regime = Regime.BULLISH_TREND,
    direction: Direction = Direction.BULLISH,
    decision_id: str = "decision0001",
    quant_score: float = 0.70,
) -> TradeDecision:
    return TradeDecision(
        decision_id=decision_id,
        symbol=symbol,
        timestamp=NOW,
        regime=regime,
        direction=direction,
        strategy=strategy,
        confidence=confidence,
        bull_score=0.8 if direction is Direction.BULLISH else 0.2,
        bear_score=0.2 if direction is Direction.BULLISH else 0.8,
        quant_score=quant_score,
        reason_codes=(),
        no_trade_reason=None if strategy is not Strategy.NO_TRADE else "test",
        ai_provider="test",
    )


def make_signal(
    symbol: str = "SPY",
    quant_score: float = 0.70,
    bias: Direction = Direction.BULLISH,
    passes: bool = True,
    features: dict[str, float] | None = None,
) -> QuantSignal:
    return QuantSignal(
        symbol=symbol,
        as_of=NOW,
        features=features
        or {
            "ret_5m": 0.001,
            "ret_15m": 0.002,
            "trend_strength": 0.35,
            "vwap_deviation": 0.0006,
            "realized_vol": 0.14,
            "volume_acceleration": 1.4,
            "opening_range_position": 0.8,
            "distance_from_high": 0.0005,
            "distance_from_low": 0.003,
            "atr_pct": 0.0004,
        },
        quant_score=quant_score,
        directional_bias=bias,
        passes_gate=passes,
    )


def make_regime(
    symbol: str = "SPY",
    regime: Regime = Regime.BULLISH_TREND,
    direction: Direction = Direction.BULLISH,
    confidence: float = 0.8,
) -> RegimeAssessment:
    return RegimeAssessment(
        symbol=symbol,
        as_of=NOW,
        regime=regime,
        direction=direction,
        confidence=confidence,
        evidence={"trend_strength": 0.35},
        risk_flags=(),
    )
