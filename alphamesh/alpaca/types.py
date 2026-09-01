"""Shared broker-facing value objects."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MarketClock(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None


class AccountState(BaseModel):
    """The subset of the Alpaca account that the risk layer needs.

    ``account_number`` is retained because the paper guard checks its prefix;
    it is never written to the journal or the dashboard.
    """

    model_config = ConfigDict(frozen=True)

    account_number: str = Field(repr=False)
    status: str
    equity: float
    last_equity: float
    cash: float
    buying_power: float
    options_buying_power: float
    options_trading_level: int
    trading_blocked: bool
    account_blocked: bool

    @property
    def is_tradeable(self) -> bool:
        return (
            self.status.upper() == "ACTIVE"
            and not self.trading_blocked
            and not self.account_blocked
        )

    @property
    def intraday_pnl(self) -> float:
        return self.equity - self.last_equity


class BrokerPosition(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: int
    avg_entry_price: float
    market_value: float
    unrealized_pl: float


class BrokerOrderLeg(BaseModel):
    """One leg of a multi-leg order as the broker reports it."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: str = ""
    position_intent: str = ""
    ratio: int = 1
    filled_avg_price: float | None = None


class BrokerOrderSummary(BaseModel):
    """A broker order plus its legs, used to pair positions with their entry.

    Multi-leg option orders carry no symbol of their own; the OCC symbols live
    on the legs, which is why adoption matches on the leg symbol set.
    """

    model_config = ConfigDict(frozen=True)

    client_order_id: str
    broker_order_id: str | None = None
    status: str = ""
    filled_quantity: int = 0
    filled_avg_price_cents: int | None = None
    submitted_at: datetime | None = None
    legs: tuple[BrokerOrderLeg, ...] = ()

    @property
    def leg_symbols(self) -> frozenset[str]:
        return frozenset(leg.symbol.upper() for leg in self.legs if leg.symbol)


__all__ = [
    "AccountState",
    "BrokerOrderLeg",
    "BrokerOrderSummary",
    "BrokerPosition",
    "MarketClock",
]
