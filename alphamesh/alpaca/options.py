"""Option-chain access.

Contracts are never synthesised. Every ``OptionContractCandidate`` that reaches
the strategy layer came from an Alpaca option-chain payload - either live over
REST, or replayed from a capture taken through the Alpaca MCP server. If no
real contract satisfies the filters, the answer is an empty list, which the
strategy layer turns into NO_TRADE.
"""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from alphamesh.models.domain import (
    Greeks,
    OptionContractCandidate,
    OptionQuote,
    OptionType,
)


@runtime_checkable
class OptionChainProvider(Protocol):
    def chain(
        self,
        underlying: str,
        option_type: OptionType,
        as_of: date,
        min_dte: int,
        max_dte: int,
    ) -> list[OptionContractCandidate]: ...


def _parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _opt_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_option_snapshots_csv(path: Path) -> list[OptionContractCandidate]:
    """Load captured option snapshots. Rows missing a quote keep ``quote=None``
    so downstream liquidity checks can reject them explicitly."""
    contracts: list[OptionContractCandidate] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            bid = _opt_float(row.get("bid"))
            ask = _opt_float(row.get("ask"))
            quote: OptionQuote | None = None
            if bid is not None and ask is not None and ask >= bid:
                quote = OptionQuote(
                    bid=bid,
                    ask=ask,
                    bid_size=int(row.get("bid_size") or 0),
                    ask_size=int(row.get("ask_size") or 0),
                    quote_timestamp=_parse_ts(row["quote_ts"]),
                )
            contracts.append(
                OptionContractCandidate(
                    symbol=row["symbol"],
                    underlying=row["underlying"],
                    expiration=date.fromisoformat(row["expiration"]),
                    option_type=OptionType(row["option_type"]),
                    strike=float(row["strike"]),
                    quote=quote,
                    greeks=Greeks(
                        delta=_opt_float(row.get("delta")),
                        gamma=_opt_float(row.get("gamma")),
                        theta=_opt_float(row.get("theta")),
                        vega=_opt_float(row.get("vega")),
                        implied_volatility=_opt_float(row.get("implied_volatility")),
                    ),
                    day_volume=int(float(row.get("day_volume") or 0)),
                )
            )
    return contracts


class CaptureOptionChain:
    """Replays option-chain snapshots captured from Alpaca via MCP."""

    def __init__(self, capture_dir: Path) -> None:
        self.capture_dir = Path(capture_dir)
        self._contracts: list[OptionContractCandidate] | None = None

    def _all(self) -> list[OptionContractCandidate]:
        if self._contracts is None:
            found: list[OptionContractCandidate] = []
            for path in sorted(self.capture_dir.glob("option_snapshots_*.csv")):
                found.extend(load_option_snapshots_csv(path))
            self._contracts = found
        return self._contracts

    def chain(
        self,
        underlying: str,
        option_type: OptionType,
        as_of: date,
        min_dte: int,
        max_dte: int,
    ) -> list[OptionContractCandidate]:
        return [
            c
            for c in self._all()
            if c.underlying.upper() == underlying.upper()
            and c.option_type is option_type
            and min_dte <= c.dte(as_of) <= max_dte
        ]


class AlpacaRestOptionChain:
    """Production option chain via ``alpaca-py``."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._client = None

    def _option_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            from alpaca.data.historical.option import OptionHistoricalDataClient

            # raw_data=True is required, not a preference. alpaca-py's
            # OptionsSnapshot model declares only symbol, latest_trade,
            # latest_quote, implied_volatility and greeks -- it silently drops
            # the dailyBar the API actually returns. Parsing the model would
            # make day_volume permanently 0 and fail every liquidity gate.
            self._client = OptionHistoricalDataClient(
                self._api_key, self._api_secret, raw_data=True
            )
        return self._client

    def chain(
        self,
        underlying: str,
        option_type: OptionType,
        as_of: date,
        min_dte: int,
        max_dte: int,
    ) -> list[OptionContractCandidate]:
        from datetime import timedelta

        from alpaca.data.requests import OptionChainRequest
        from alpaca.trading.enums import ContractType

        request = OptionChainRequest(
            underlying_symbol=underlying,
            type=ContractType(option_type.value),
            expiration_date_gte=as_of + timedelta(days=min_dte),
            expiration_date_lte=as_of + timedelta(days=max_dte),
        )
        raw = self._option_client().get_option_chain(request)
        return [
            c
            for symbol, snap in raw.items()
            if (c := self._to_candidate(symbol, underlying, snap)) is not None
        ]

    @staticmethod
    def _to_candidate(
        symbol: str, underlying: str, snap: Any
    ) -> OptionContractCandidate | None:
        """Map one raw Alpaca options snapshot to a domain candidate.

        ``snap`` is the verbatim JSON object for the contract, so the keys are
        the wire names (camelCase), not alpaca-py model attributes.
        """
        parsed = parse_occ_symbol(symbol)
        if parsed is None:
            return None
        expiration, opt_type, strike = parsed
        if not isinstance(snap, dict):
            return None

        quote: OptionQuote | None = None
        raw_quote = snap.get("latestQuote")
        if isinstance(raw_quote, dict):
            bid = float(raw_quote.get("bp") or 0)
            ask = float(raw_quote.get("ap") or 0)
            if ask >= bid:
                raw_ts = raw_quote.get("t")
                quote = OptionQuote(
                    bid=bid,
                    ask=ask,
                    bid_size=int(float(raw_quote.get("bs") or 0)),
                    ask_size=int(float(raw_quote.get("as") or 0)),
                    quote_timestamp=(
                        _parse_ts(raw_ts)
                        if isinstance(raw_ts, str)
                        else datetime.now(UTC)
                    ),
                )

        raw_greeks = snap.get("greeks")
        raw_greeks = raw_greeks if isinstance(raw_greeks, dict) else {}
        greeks = Greeks(
            delta=_opt_float(raw_greeks.get("delta")),
            gamma=_opt_float(raw_greeks.get("gamma")),
            theta=_opt_float(raw_greeks.get("theta")),
            vega=_opt_float(raw_greeks.get("vega")),
            implied_volatility=_opt_float(snap.get("impliedVolatility")),
        )

        daily = snap.get("dailyBar")
        day_volume = 0
        if isinstance(daily, dict):
            day_volume = int(float(daily.get("v") or 0))

        return OptionContractCandidate(
            symbol=symbol,
            underlying=underlying,
            expiration=expiration,
            option_type=opt_type,
            strike=strike,
            quote=quote,
            greeks=greeks,
            day_volume=day_volume,
        )


def parse_occ_symbol(symbol: str) -> tuple[date, OptionType, float] | None:
    """Decode an OCC option symbol, e.g. ``SPY260903C00770000``.

    Layout is root + YYMMDD + C/P + strike in thousandths of a dollar, so the
    fixed-width tail is parsed from the right and the root is whatever remains.
    """
    if len(symbol) < 16:
        return None
    tail = symbol[-15:]
    try:
        expiration = date(
            2000 + int(tail[0:2]), int(tail[2:4]), int(tail[4:6])
        )
        opt_type = OptionType.CALL if tail[6].upper() == "C" else OptionType.PUT
        if tail[6].upper() not in ("C", "P"):
            return None
        strike = int(tail[7:15]) / 1000.0
    except (ValueError, IndexError):
        return None
    if strike <= 0:
        return None
    return expiration, opt_type, strike


def occ_underlying(symbol: str) -> str | None:
    """Root ticker of an OCC option symbol, e.g. NVDA260902C00220000 -> NVDA.

    The OCC tail is fixed width, so the root is whatever precedes it.
    """
    if len(symbol) < 16:
        return None
    if parse_occ_symbol(symbol) is None:
        return None
    root = symbol[:-15].strip().upper()
    return root or None


def build_occ_symbol(
    underlying: str, expiration: date, option_type: OptionType, strike: float
) -> str:
    """Inverse of :func:`parse_occ_symbol`. Used only for tests and fixtures -
    never to invent a contract for submission."""
    letter = "C" if option_type is OptionType.CALL else "P"
    return (
        f"{underlying.upper()}"
        f"{expiration.strftime('%y%m%d')}"
        f"{letter}"
        f"{round(strike * 1000):08d}"
    )


__all__ = [
    "AlpacaRestOptionChain",
    "CaptureOptionChain",
    "OptionChainProvider",
    "build_occ_symbol",
    "load_option_snapshots_csv",
    "occ_underlying",
    "parse_occ_symbol",
]
