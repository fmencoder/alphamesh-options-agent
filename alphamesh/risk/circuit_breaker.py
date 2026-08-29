"""Daily loss circuit breaker.

Once the session's combined realised and unrealised loss breaches the limit,
every new entry is refused for the rest of the day. The breaker is evaluated
fresh from portfolio state on each cycle, so a restart cannot reset it.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphamesh.config import RiskLimits
from alphamesh.models.domain import ReasonCode
from alphamesh.risk.money import to_dollars
from alphamesh.risk.portfolio import PortfolioState


@dataclass(frozen=True)
class BreakerStatus:
    tripped: bool
    session_pnl_cents: int
    limit_cents: int
    detail: str

    @property
    def reason_code(self) -> ReasonCode | None:
        return ReasonCode.DAILY_DRAWDOWN_LIMIT if self.tripped else None


def evaluate_circuit_breaker(state: PortfolioState, limits: RiskLimits) -> BreakerStatus:
    """Trip when session P&L is a loss of at least the configured magnitude."""
    limit_cents = round(limits.daily_loss_circuit_breaker * 100)
    pnl = state.session_pnl_cents
    tripped = pnl <= -limit_cents
    if tripped:
        detail = (
            f"session P&L {to_dollars(pnl):+.2f} breaches the "
            f"{limits.daily_loss_circuit_breaker:.2f} daily loss limit; "
            "new entries halted"
        )
    else:
        detail = (
            f"session P&L {to_dollars(pnl):+.2f} within the "
            f"{limits.daily_loss_circuit_breaker:.2f} daily loss limit"
        )
    return BreakerStatus(tripped, pnl, limit_cents, detail)


__all__ = ["BreakerStatus", "evaluate_circuit_breaker"]
