"""Order and position monitoring.

Order status is always read back from the broker; the local journal is updated
from that reading, never the other way round. Marks for open spreads come from
the same option-chain provider that selected them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from alphamesh.alpaca.execution import Broker
from alphamesh.models.domain import (
    ExecutionRecord,
    OptionContractCandidate,
    PositionRecord,
    TradeState,
)
from alphamesh.persistence.journal import Journal
from alphamesh.risk.money import to_cents

log = logging.getLogger(__name__)

FILLED_STATUSES = frozenset({"filled"})
PARTIAL_STATUSES = frozenset({"partially_filled"})
DEAD_STATUSES = frozenset({"canceled", "expired", "rejected", "done_for_day", "replaced"})


def status_to_state(status: str, previous: TradeState) -> TradeState:
    """Map an Alpaca order status onto our state machine."""
    value = (status or "").lower()
    if value in FILLED_STATUSES:
        return TradeState.FILLED
    if value in PARTIAL_STATUSES:
        return TradeState.PARTIALLY_FILLED
    if value in DEAD_STATUSES:
        return TradeState.REJECTED
    return previous


def mark_spread_cents(
    long_contract: OptionContractCandidate, short_contract: OptionContractCandidate
) -> int | None:
    """Current mid-market value of the spread, in cents per spread.

    Returns ``None`` when either leg has no usable quote, which callers treat as
    "cannot mark" rather than as a value of zero.
    """
    if long_contract.quote is None or short_contract.quote is None:
        return None
    if long_contract.quote.bid <= 0 and long_contract.quote.ask <= 0:
        return None
    return to_cents(long_contract.quote.mid - short_contract.quote.mid)


@dataclass(frozen=True)
class PositionMark:
    position_id: str
    mark_cents: int
    unrealized_pnl_cents: int
    pct_of_max_profit: float
    pct_of_defined_risk_lost: float


def mark_position(position: PositionRecord, spread_mark_cents: int) -> PositionMark:
    """Value one open spread at the given per-spread mark."""
    from alphamesh.models.domain import OPTION_MULTIPLIER

    current_value = spread_mark_cents * OPTION_MULTIPLIER * position.quantity
    unrealized = current_value - position.entry_debit_cents
    max_profit = max(position.max_profit_cents, 1)
    max_loss = max(position.max_loss_cents, 1)
    return PositionMark(
        position_id=position.position_id,
        mark_cents=current_value,
        unrealized_pnl_cents=unrealized,
        pct_of_max_profit=max(0.0, unrealized) / max_profit,
        pct_of_defined_risk_lost=max(0, -unrealized) / max_loss,
    )


class OrderMonitor:
    def __init__(self, broker: Broker, journal: Journal) -> None:
        self.broker = broker
        self.journal = journal

    def refresh(self, client_order_id: str) -> ExecutionRecord | None:
        """Re-read one order from the broker and persist what it says."""
        record = self.broker.get_order_by_client_id(client_order_id)
        if record is None:
            return None
        row = self.journal.get_order(client_order_id)
        previous = TradeState(row["state"]) if row else TradeState.SUBMITTED
        new_state = status_to_state(record.status, previous)
        self.journal.update_order_execution(record, new_state)
        if new_state is not previous:
            self.journal.set_order_state(
                client_order_id, new_state, f"broker status {record.status}"
            )
        return record


__all__ = [
    "DEAD_STATUSES",
    "FILLED_STATUSES",
    "PARTIAL_STATUSES",
    "OrderMonitor",
    "PositionMark",
    "mark_position",
    "mark_spread_cents",
    "status_to_state",
]
