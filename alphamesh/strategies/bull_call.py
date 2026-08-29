"""Bull call spread: buy a ~0.55-delta call, sell a higher-strike ~0.30-delta call.

Maximum loss is the net debit paid. Maximum profit is the strike width minus
that debit. Both are known before the order is sent, which is the whole reason
v1 trades only verticals.
"""

from __future__ import annotations

from datetime import date, datetime

from alphamesh.config import RiskLimits, StrategyConfig
from alphamesh.models.domain import OptionContractCandidate, Strategy
from alphamesh.strategies.contracts import SelectionResult, select_vertical_spread


def build_bull_call_spread(
    symbol: str,
    chain: list[OptionContractCandidate],
    strategies: StrategyConfig,
    limits: RiskLimits,
    now: datetime,
    as_of_date: date | None = None,
) -> SelectionResult:
    return select_vertical_spread(
        Strategy.BULL_CALL_SPREAD, symbol, chain, strategies, limits, now, as_of_date
    )


__all__ = ["build_bull_call_spread"]
