"""Restart recovery and reconciliation.

On startup every non-terminal order in the journal is re-read from the broker
by its client order id. Only after that reconciliation may the orchestrator
build new orders, which is what stops a restart mid-submit from opening a
second position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from alphamesh.alpaca.execution import Broker
from alphamesh.execution.monitor import status_to_state
from alphamesh.models.domain import TradeState
from alphamesh.persistence.journal import Journal

log = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    inspected: int = 0
    reconciled: int = 0
    orphaned: int = 0
    unchanged: int = 0
    details: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "inspected": self.inspected,
            "reconciled": self.reconciled,
            "orphaned": self.orphaned,
            "unchanged": self.unchanged,
            "details": list(self.details),
        }


def reconcile_open_orders(journal: Journal, broker: Broker) -> RecoveryReport:
    """Bring the journal back in line with the broker after a restart."""
    report = RecoveryReport()

    for row in journal.open_orders():
        client_order_id = row["client_order_id"]
        report.inspected += 1
        previous = TradeState(row["state"])

        try:
            record = broker.get_order_by_client_id(client_order_id)
        except Exception as exc:
            log.warning("could not reconcile %s: %s", client_order_id, exc)
            report.details.append(f"{client_order_id}: broker lookup failed ({exc})")
            continue

        if record is None:
            # We reserved the id but the broker has never seen it. The order was
            # never placed, so the reservation is safe to retire; it must not be
            # resubmitted under the same id in this cycle.
            if previous is TradeState.CONSTRUCTED:
                journal.set_order_state(
                    client_order_id,
                    TradeState.FAILED,
                    "reserved but never reached the broker; retired on recovery",
                )
                report.orphaned += 1
                report.details.append(f"{client_order_id}: never submitted, marked FAILED")
            else:
                report.details.append(
                    f"{client_order_id}: in {previous} but unknown to broker; left as-is"
                )
                report.unchanged += 1
            continue

        new_state = status_to_state(record.status, previous)
        journal.update_order_execution(record, new_state)
        if new_state is not previous:
            journal.set_order_state(
                client_order_id,
                new_state,
                f"recovery: broker reports {record.status}",
            )
            report.reconciled += 1
            report.details.append(
                f"{client_order_id}: {previous} -> {new_state} ({record.status})"
            )
        else:
            report.unchanged += 1

    return report


__all__ = ["RecoveryReport", "reconcile_open_orders"]
