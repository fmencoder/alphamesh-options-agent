"""Explicit execution state machine.

Every trade walks one path from DISCOVERED to a terminal state, and each hop is
persisted before the next is attempted. Illegal hops raise rather than being
silently accepted, so a bug in the orchestrator surfaces as a loud error
instead of an order in an impossible state.
"""

from __future__ import annotations

from alphamesh.models.domain import TERMINAL_STATES, TradeState

ALLOWED_TRANSITIONS: dict[TradeState, frozenset[TradeState]] = {
    TradeState.DISCOVERED: frozenset({TradeState.ANALYZED, TradeState.REJECTED}),
    TradeState.ANALYZED: frozenset({TradeState.AI_APPROVED, TradeState.REJECTED}),
    TradeState.AI_APPROVED: frozenset({TradeState.RISK_APPROVED, TradeState.REJECTED}),
    TradeState.RISK_APPROVED: frozenset({TradeState.CONSTRUCTED, TradeState.REJECTED}),
    TradeState.CONSTRUCTED: frozenset(
        {TradeState.SUBMITTED, TradeState.REJECTED, TradeState.FAILED}
    ),
    TradeState.SUBMITTED: frozenset(
        {
            TradeState.PARTIALLY_FILLED,
            TradeState.FILLED,
            TradeState.REJECTED,
            TradeState.FAILED,
            TradeState.CLOSED,
        }
    ),
    TradeState.PARTIALLY_FILLED: frozenset(
        {TradeState.FILLED, TradeState.MONITORING, TradeState.FAILED, TradeState.CLOSED}
    ),
    TradeState.FILLED: frozenset({TradeState.MONITORING, TradeState.EXIT_REQUESTED}),
    TradeState.MONITORING: frozenset(
        {TradeState.EXIT_REQUESTED, TradeState.CLOSED, TradeState.FAILED}
    ),
    TradeState.EXIT_REQUESTED: frozenset(
        {TradeState.CLOSED, TradeState.MONITORING, TradeState.FAILED}
    ),
    TradeState.CLOSED: frozenset(),
    TradeState.REJECTED: frozenset(),
    TradeState.FAILED: frozenset(),
}


class IllegalTransitionError(RuntimeError):
    def __init__(self, source: TradeState, target: TradeState) -> None:
        super().__init__(f"illegal state transition {source} -> {target}")
        self.source = source
        self.target = target


def can_transition(source: TradeState, target: TradeState) -> bool:
    return target in ALLOWED_TRANSITIONS.get(source, frozenset())


def transition(source: TradeState, target: TradeState) -> TradeState:
    """Validate and return the new state, or raise."""
    if not can_transition(source, target):
        raise IllegalTransitionError(source, target)
    return target


def is_terminal(state: TradeState) -> bool:
    return state in TERMINAL_STATES


def is_recoverable(state: TradeState) -> bool:
    """States that a restart must reconcile against the broker."""
    return state in {
        TradeState.CONSTRUCTED,
        TradeState.SUBMITTED,
        TradeState.PARTIALLY_FILLED,
        TradeState.FILLED,
        TradeState.MONITORING,
        TradeState.EXIT_REQUESTED,
    }


__all__ = [
    "ALLOWED_TRANSITIONS",
    "IllegalTransitionError",
    "can_transition",
    "is_recoverable",
    "is_terminal",
    "transition",
]
