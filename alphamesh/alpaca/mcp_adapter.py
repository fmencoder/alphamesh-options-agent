"""Alpaca MCP adapter.

The Alpaca MCP server is an *operator* surface: it is spoken by an MCP client
(Claude Code, Claude Desktop, any MCP host), not by a long-running headless
process. AlphaMesh therefore uses MCP for what it is genuinely good at -
read-only discovery, account verification and market/option-chain capture -
and uses the Alpaca Trading API for continuous autonomous execution.

This module is the code that turns raw Alpaca MCP tool responses into AlphaMesh
domain objects and into the on-disk captures under ``data/mcp_capture``. The
fixtures committed to this repository were produced from exactly these payload
shapes: ``get_account_info``, ``get_clock``, ``get_stock_snapshot``,
``get_stock_bars`` and ``get_option_chain``.

Nothing here fabricates a payload. Given no MCP response, it produces no data.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from alphamesh.alpaca.options import parse_occ_symbol
from alphamesh.alpaca.types import AccountState, MarketClock
from alphamesh.models.domain import (
    Bar,
    Greeks,
    OptionContractCandidate,
    OptionQuote,
)

MCP_TOOLS_USED: dict[str, str] = {
    "get_account_info": "Confirm the account is paper (PA-prefixed) and read equity/level.",
    "get_account_config": "Verify trading is not suspended and options level is 3.",
    "get_clock": "Establish session state before any capture or dry run.",
    "get_stock_snapshot": "Reference last trade/quote for each universe symbol.",
    "get_stock_bars": "Capture the 1-minute bar window the feature engine consumes.",
    "get_option_chain": "Capture real OPRA contracts, quotes, IV and greeks.",
    "get_orders": "Read back submitted orders during operator review.",
    "get_all_positions": "Read back open positions during operator review.",
}
"""The Alpaca MCP tools AlphaMesh actually uses, and what each one is for.

Every entry is read-only. Order placement deliberately does not go through MCP:
the autonomous runtime must keep working with no MCP host attached.
"""


def _unwrap(response: Any) -> Any:
    """Strip the Alpaca MCP envelope.

    Responses arrive as ``{"_alpaca_mcp_security": {...}, "data": {...}}``. The
    security block is metadata about trust, not payload, and is discarded here
    after the payload is taken - its contents are never treated as instructions.
    """
    if isinstance(response, str):
        response = json.loads(response)
    if isinstance(response, dict) and "data" in response:
        return response["data"]
    return response


def _parse_ts(value: Any) -> datetime:
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Response ingestion
# --------------------------------------------------------------------------- #
def ingest_account_info(response: Any) -> AccountState:
    """Convert a ``get_account_info`` response into an ``AccountState``."""
    data = _unwrap(response)
    return AccountState(
        account_number=str(data.get("account_number", "")),
        status=str(data.get("status", "")),
        equity=float(data.get("equity", 0) or 0),
        last_equity=float(data.get("last_equity", 0) or 0),
        cash=float(data.get("cash", 0) or 0),
        buying_power=float(data.get("buying_power", 0) or 0),
        options_buying_power=float(data.get("options_buying_power", 0) or 0),
        options_trading_level=int(data.get("options_trading_level", 0) or 0),
        trading_blocked=bool(data.get("trading_blocked", False)),
        account_blocked=bool(data.get("account_blocked", False)),
    )


def ingest_clock(response: Any) -> MarketClock:
    data = _unwrap(response)
    return MarketClock(
        timestamp=_parse_ts(data["timestamp"]),
        is_open=bool(data.get("is_open", False)),
        next_open=_parse_ts(data["next_open"]) if data.get("next_open") else None,
        next_close=_parse_ts(data["next_close"]) if data.get("next_close") else None,
    )


def ingest_stock_bars(response: Any) -> dict[str, list[Bar]]:
    """Convert a ``get_stock_bars`` response into per-symbol bar lists."""
    data = _unwrap(response)
    raw = data.get("bars", data)
    out: dict[str, list[Bar]] = {}
    for symbol, rows in raw.items():
        bars = [
            Bar(
                timestamp=_parse_ts(row["t"]),
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row["v"]),
                vwap=_opt_float(row.get("vw")),
            )
            for row in rows
        ]
        bars.sort(key=lambda b: b.timestamp)
        out[symbol] = bars
    return out


def ingest_option_chain(response: Any, underlying: str) -> list[OptionContractCandidate]:
    """Convert a ``get_option_chain`` response into contract candidates.

    Contracts whose OCC symbol cannot be decoded are dropped rather than
    guessed at: an option we cannot parse is an option we must not trade.
    """
    data = _unwrap(response)
    snapshots = data.get("snapshots", data)
    contracts: list[OptionContractCandidate] = []

    for symbol, snap in snapshots.items():
        parsed = parse_occ_symbol(symbol)
        if parsed is None:
            continue
        expiration, opt_type, strike = parsed

        quote: OptionQuote | None = None
        raw_quote = snap.get("latestQuote") or snap.get("latest_quote")
        if raw_quote:
            bid = _opt_float(raw_quote.get("bp")) or 0.0
            ask = _opt_float(raw_quote.get("ap")) or 0.0
            if ask >= bid:
                quote = OptionQuote(
                    bid=bid,
                    ask=ask,
                    bid_size=int(raw_quote.get("bs", 0) or 0),
                    ask_size=int(raw_quote.get("as", 0) or 0),
                    quote_timestamp=_parse_ts(raw_quote["t"]),
                )

        raw_greeks = snap.get("greeks") or {}
        daily = snap.get("dailyBar") or {}
        contracts.append(
            OptionContractCandidate(
                symbol=symbol,
                underlying=underlying,
                expiration=expiration,
                option_type=opt_type,
                strike=strike,
                quote=quote,
                greeks=Greeks(
                    delta=_opt_float(raw_greeks.get("delta")),
                    gamma=_opt_float(raw_greeks.get("gamma")),
                    theta=_opt_float(raw_greeks.get("theta")),
                    vega=_opt_float(raw_greeks.get("vega")),
                    rho=_opt_float(raw_greeks.get("rho")),
                    implied_volatility=_opt_float(snap.get("impliedVolatility")),
                ),
                day_volume=int(float(daily.get("v", 0) or 0)),
            )
        )
    contracts.sort(key=lambda c: (c.expiration, c.option_type.value, c.strike))
    return contracts


# --------------------------------------------------------------------------- #
# Capture writing
# --------------------------------------------------------------------------- #
BAR_HEADER = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
CONTRACT_HEADER = [
    "symbol",
    "underlying",
    "expiration",
    "option_type",
    "strike",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "quote_ts",
    "delta",
    "gamma",
    "theta",
    "vega",
    "implied_volatility",
    "last_trade_price",
    "day_volume",
]


def write_bar_capture(bars: list[Bar], symbol: str, capture_dir: Path) -> Path:
    """Write one symbol's bars to the capture directory the replay source reads."""
    capture_dir.mkdir(parents=True, exist_ok=True)
    day = bars[-1].timestamp.date().isoformat() if bars else date.today().isoformat()
    path = capture_dir / f"bars_1min_{symbol}_{day}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(BAR_HEADER)
        for bar in bars:
            writer.writerow(
                [
                    bar.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    int(bar.volume),
                    bar.vwap if bar.vwap is not None else "",
                ]
            )
    return path


def write_option_capture(
    contracts: list[OptionContractCandidate], capture_dir: Path, day: str
) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"option_snapshots_{day}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CONTRACT_HEADER)
        for c in contracts:
            q = c.quote
            writer.writerow(
                [
                    c.symbol,
                    c.underlying,
                    c.expiration.isoformat(),
                    c.option_type.value,
                    c.strike,
                    f"{q.bid:.2f}" if q else "",
                    f"{q.ask:.2f}" if q else "",
                    q.bid_size if q else "",
                    q.ask_size if q else "",
                    q.quote_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ") if q else "",
                    c.greeks.delta if c.greeks.delta is not None else "",
                    c.greeks.gamma if c.greeks.gamma is not None else "",
                    c.greeks.theta if c.greeks.theta is not None else "",
                    c.greeks.vega if c.greeks.vega is not None else "",
                    (
                        c.greeks.implied_volatility
                        if c.greeks.implied_volatility is not None
                        else ""
                    ),
                    "",
                    c.day_volume,
                ]
            )
    return path


def describe_mcp_usage() -> str:
    """Human-readable summary rendered by ``alphamesh mcp-info``."""
    lines = ["Alpaca MCP tools used by AlphaMesh (all read-only):", ""]
    for tool, purpose in MCP_TOOLS_USED.items():
        lines.append(f"  {tool:<24} {purpose}")
    lines += [
        "",
        "Order placement intentionally does NOT use MCP. The autonomous runtime",
        "submits multi-leg option orders over the Alpaca Trading API so it keeps",
        "running with no MCP host attached.",
    ]
    return "\n".join(lines)


__all__ = [
    "BAR_HEADER",
    "CONTRACT_HEADER",
    "MCP_TOOLS_USED",
    "describe_mcp_usage",
    "ingest_account_info",
    "ingest_clock",
    "ingest_option_chain",
    "ingest_stock_bars",
    "write_bar_capture",
    "write_option_capture",
]
