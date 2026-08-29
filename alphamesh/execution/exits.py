"""Deterministic exit management.

Exits are hard rules over marks and clocks. The LLM has no input here: it is
never asked whether to hold, and it cannot veto a stop, an end-of-day flatten
or a circuit-breaker exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from alphamesh.config import StrategyConfig
from alphamesh.execution.monitor import PositionMark
from alphamesh.models.domain import (
    Direction,
    ExitReason,
    PositionRecord,
    Regime,
    RegimeAssessment,
    Strategy,
)


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: ExitReason | None
    detail: str


NO_EXIT = ExitDecision(False, None, "position within all exit thresholds")


def _expected_direction(strategy: Strategy) -> Direction:
    return (
        Direction.BULLISH if strategy is Strategy.BULL_CALL_SPREAD else Direction.BEARISH
    )


def evaluate_exit(
    position: PositionRecord,
    mark: PositionMark | None,
    regime: RegimeAssessment | None,
    strategies: StrategyConfig,
    now: datetime,
    session_close: datetime | None = None,
    circuit_breaker_tripped: bool = False,
) -> ExitDecision:
    """Decide whether to flatten one position. Checks run hardest-first."""
    exits = strategies.exits

    # 1. Portfolio-wide emergency. Overrides everything else.
    if circuit_breaker_tripped:
        return ExitDecision(
            True, ExitReason.CIRCUIT_BREAKER, "daily loss circuit breaker is tripped"
        )

    # 2. Expiration. Never carry a spread into its own expiry.
    if position.expiration <= now.date():
        return ExitDecision(
            True, ExitReason.EXPIRED, f"expiration {position.expiration} has arrived"
        )

    # 3. End of day, when overnight is not permitted.
    if not exits.get("allow_overnight", False) and session_close is not None:
        cutoff = session_close - timedelta(
            minutes=int(exits.get("flatten_before_close_minutes", 20))
        )
        if now >= cutoff:
            return ExitDecision(
                True,
                ExitReason.END_OF_DAY,
                f"within {exits.get('flatten_before_close_minutes', 20)} minutes of the close",
            )

    # 4. Maximum holding time.
    max_minutes = float(exits.get("max_holding_minutes", 900))
    held = (now - position.opened_at).total_seconds() / 60.0
    if held >= max_minutes:
        return ExitDecision(
            True,
            ExitReason.MAX_HOLDING_TIME,
            f"held {held:.0f} minutes, limit {max_minutes:.0f}",
        )

    # 5. Mark-based exits. Skipped when the spread cannot be marked, rather
    #    than treated as a zero value.
    if mark is not None:
        target = float(exits.get("profit_target_pct_of_max_profit", 0.55))
        if mark.pct_of_max_profit >= target:
            return ExitDecision(
                True,
                ExitReason.PROFIT_TARGET,
                (
                    f"captured {mark.pct_of_max_profit:.0%} of defined max profit "
                    f"(target {target:.0%})"
                ),
            )
        stop = float(exits.get("max_loss_pct_of_defined_risk", 0.65))
        if mark.pct_of_defined_risk_lost >= stop:
            return ExitDecision(
                True,
                ExitReason.MAX_LOSS,
                (
                    f"lost {mark.pct_of_defined_risk_lost:.0%} of defined risk "
                    f"(stop {stop:.0%})"
                ),
            )

    # 6. Signal invalidation: the regime turned against the position's thesis.
    if exits.get("signal_invalidation_regime_flip", True) and regime is not None:
        wanted = _expected_direction(position.strategy)
        flipped = regime.direction is not Direction.NEUTRAL and regime.direction is not wanted
        broke = regime.regime in (Regime.UNSTABLE,)
        if flipped or broke:
            return ExitDecision(
                True,
                ExitReason.SIGNAL_INVALIDATED,
                (
                    f"regime moved to {regime.regime}/{regime.direction}, "
                    f"against a {position.strategy} thesis"
                ),
            )

    return NO_EXIT


def is_expiring_on(position: PositionRecord, day: date) -> bool:
    return position.expiration == day


__all__ = ["NO_EXIT", "ExitDecision", "evaluate_exit", "is_expiring_on"]
