"""Market-data access.

Two interchangeable implementations sit behind one protocol:

``AlpacaRestMarketData``
    The production path. Uses ``alpaca-py`` against Alpaca's market data API.

``CaptureMarketData``
    Replays real Alpaca payloads previously captured through the Alpaca MCP
    server. This exists because some deployment environments (including the
    build container this project was developed in) block outbound access to
    ``data.alpaca.markets``. The data is real; only the transport differs.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from alphamesh.alpaca.types import MarketClock
from alphamesh.models.domain import Bar, MarketSnapshot

LOOKBACK_CALENDAR_DAYS = 7
"""Calendar days of history requested to build a bar window.

Deliberately calendar-based, not minute-based. A ``now - N minutes`` window
contains zero market minutes over a weekend, and at the opening bell it falls
short of the previous session, so the agent would starve for the first hour of
every day. Seven days clears a three-day holiday weekend with room to spare.
"""


@runtime_checkable
class MarketDataProvider(Protocol):
    def snapshot(self, symbol: str, lookback_minutes: int = 180) -> MarketSnapshot: ...

    def latest_bar(self, symbol: str) -> Bar: ...

    def clock(self) -> MarketClock: ...


BAR_REQUEST_LIMIT = 10_000
"""Upper bound on bars per request. Requests are per-symbol, so a shared limit
can never satisfy one symbol while starving the other."""


class MarketDataUnavailableError(RuntimeError):
    """The provider could not supply usable data for a symbol."""


def _to_bar(raw: object) -> Bar:
    """Map an alpaca-py bar onto the domain model."""
    return Bar(
        timestamp=raw.timestamp,  # type: ignore[attr-defined]
        open=float(raw.open),  # type: ignore[attr-defined]
        high=float(raw.high),  # type: ignore[attr-defined]
        low=float(raw.low),  # type: ignore[attr-defined]
        close=float(raw.close),  # type: ignore[attr-defined]
        volume=float(raw.volume),  # type: ignore[attr-defined]
        vwap=float(raw.vwap) if getattr(raw, "vwap", None) is not None else None,  # type: ignore[attr-defined]
    )


# --------------------------------------------------------------------------- #
# Capture replay
# --------------------------------------------------------------------------- #
def load_bars_csv(path: Path) -> list[Bar]:
    """Load a captured 1-minute bar file."""
    bars: list[Bar] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            bars.append(
                Bar(
                    timestamp=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    vwap=float(row["vwap"]) if row.get("vwap") else None,
                )
            )
    bars.sort(key=lambda b: b.timestamp)
    return bars


class CaptureMarketData:
    """Replays bar windows captured from Alpaca via MCP.

    ``advance()`` walks a cursor forward one bar at a time so a dry run can
    step through a real session deterministically instead of seeing one frozen
    instant.
    """

    def __init__(self, capture_dir: Path, market_open: bool = True) -> None:
        self.capture_dir = Path(capture_dir)
        self.market_open = market_open
        self._bars: dict[str, list[Bar]] = {}
        self._cursor: dict[str, int] = {}

    def _load(self, symbol: str) -> list[Bar]:
        if symbol not in self._bars:
            matches = sorted(self.capture_dir.glob(f"bars_1min_{symbol}_*.csv"))
            if not matches:
                raise MarketDataUnavailableError(
                    f"no captured bars for {symbol} in {self.capture_dir}"
                )
            self._bars[symbol] = load_bars_csv(matches[-1])
        return self._bars[symbol]

    def available_symbols(self) -> list[str]:
        return sorted(
            p.name.split("_")[2] for p in self.capture_dir.glob("bars_1min_*_*.csv")
        )

    def advance(self, steps: int = 1) -> None:
        """Move every symbol's cursor forward, stopping at the end of capture."""
        for symbol, bars in self._bars.items():
            self._cursor[symbol] = min(
                self._cursor.get(symbol, len(bars)) + steps, len(bars)
            )

    def snapshot(self, symbol: str, lookback_minutes: int = 180) -> MarketSnapshot:
        bars = self._load(symbol)
        if not bars:
            raise MarketDataUnavailableError(f"captured bar file for {symbol} is empty")
        end = self._cursor.setdefault(symbol, len(bars))
        window = bars[max(0, end - lookback_minutes) : end]
        if not window:
            raise MarketDataUnavailableError(f"cursor for {symbol} yields an empty window")
        last = window[-1]
        return MarketSnapshot(
            symbol=symbol,
            as_of=last.timestamp,
            last_price=last.close,
            bid=None,
            ask=None,
            session_open=bars[0].open,
            prev_close=None,
            bars=tuple(window),
        )

    def latest_bar(self, symbol: str) -> Bar:
        bars = self._load(symbol)
        if not bars:
            raise MarketDataUnavailableError(f"captured bar file for {symbol} is empty")
        return bars[-1]

    def clock(self) -> MarketClock:
        return MarketClock(timestamp=datetime.now(UTC), is_open=self.market_open)


# --------------------------------------------------------------------------- #
# Alpaca REST
# --------------------------------------------------------------------------- #
class AlpacaRestMarketData:
    """Production market data via ``alpaca-py``.

    Imports are deferred so the rest of the system - and the whole test suite -
    runs without the SDK or any network access present.
    """

    def __init__(self, api_key: str, api_secret: str, feed: str = "iex") -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self.feed = feed
        self._client = None

    def _stock_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(self._api_key, self._api_secret)
        return self._client

    def latest_bar(self, symbol: str) -> Bar:
        """Most recent 1-minute bar Alpaca holds for the symbol.

        Unlike a historical range request this returns the last completed
        session's final bar when the market is shut, so it proves live REST
        market-data access on a weekend or holiday.
        """
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestBarRequest

        request = StockLatestBarRequest(
            symbol_or_symbols=symbol, feed=DataFeed(self.feed)
        )
        response = self._stock_client().get_stock_latest_bar(request)
        raw = response.get(symbol) if hasattr(response, "get") else None
        if raw is None:
            raise MarketDataUnavailableError(
                f"Alpaca returned no latest bar for {symbol} on feed {self.feed}"
            )
        return _to_bar(raw)

    def snapshot(self, symbol: str, lookback_minutes: int = 180) -> MarketSnapshot:
        from datetime import timedelta

        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        # Calendar-day window, not `now - N minutes`. See LOOKBACK_CALENDAR_DAYS.
        # The window is deliberately wide and then trimmed to the most recent
        # `lookback_minutes` bars, so behaviour mid-session is unchanged while a
        # weekend, holiday or opening bell no longer yields an empty result.
        end = datetime.now(UTC)
        start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed(self.feed),
            limit=BAR_REQUEST_LIMIT,
        )
        response = self._stock_client().get_stock_bars(request)
        raw = response.data.get(symbol, []) if hasattr(response, "data") else []
        bars = [_to_bar(b) for b in raw]
        bars = bars[-lookback_minutes:]
        if not bars:
            raise MarketDataUnavailableError(
                f"Alpaca returned no bars for {symbol} over the last "
                f"{LOOKBACK_CALENDAR_DAYS} calendar days on feed {self.feed}"
            )
        last = bars[-1]
        return MarketSnapshot(
            symbol=symbol,
            as_of=last.timestamp,
            last_price=last.close,
            session_open=bars[0].open,
            bars=tuple(bars),
        )

    def clock(self) -> MarketClock:
        from alpaca.trading.client import TradingClient

        client = TradingClient(self._api_key, self._api_secret, paper=True)
        clock = client.get_clock()
        # alpaca-py types this as ``Clock | dict``; the raw-dict branch only
        # occurs for a raw-data client, which we never construct.
        if isinstance(clock, dict):
            raise MarketDataUnavailableError("Alpaca returned an untyped clock payload")
        return MarketClock(
            timestamp=clock.timestamp,
            is_open=bool(clock.is_open),
            next_open=clock.next_open,
            next_close=clock.next_close,
        )


__all__ = [
    "BAR_REQUEST_LIMIT",
    "LOOKBACK_CALENDAR_DAYS",
    "AlpacaRestMarketData",
    "CaptureMarketData",
    "MarketDataProvider",
    "MarketDataUnavailableError",
    "load_bars_csv",
]
