"""Bear put spread: buy a ~-0.55-delta put, sell a lower-strike ~-0.30-delta put.

Mirror image of the bull call spread. Maximum loss is the net debit paid.
"""

from __future__ import annotations

from datetime import date, datetime

from alphamesh.config import RiskLimits, StrategyConfig
from alphamesh.models.domain import OptionContractCandidate, Strategy
from alphamesh.strategies.contracts import SelectionResult, select_vertical_spread


def build_bear_put_spread(
    symbol: str,
    chain: list[OptionContractCandidate],
    strategies: StrategyConfig,
    limits: RiskLimits,
    now: datetime,
    as_of_date: date | None = None,
) -> SelectionResult:
    return select_vertical_spread(
        Strategy.BEAR_PUT_SPREAD, symbol, chain, strategies, limits, now, as_of_date
    )


__all__ = ["build_bear_put_spread"]
