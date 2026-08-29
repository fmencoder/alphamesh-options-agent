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


__all__ = ["AccountState", "BrokerPosition", "MarketClock"]
