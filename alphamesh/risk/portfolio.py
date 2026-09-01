"""Aggregate portfolio risk state used by the Risk Governor."""

from __future__ import annotations

from dataclasses import dataclass, field

from alphamesh.alpaca.types import AccountState
from alphamesh.config import RiskLimits
from alphamesh.models.domain import PositionRecord


@dataclass(frozen=True)
class PortfolioState:
    """A snapshot of everything the governor needs to reason about exposure."""

    account: AccountState
    open_positions: tuple[PositionRecord, ...] = ()
    realized_pnl_today_cents: int = 0
    unrealized_pnl_cents: int = 0
    open_client_order_ids: frozenset[str] = field(default_factory=frozenset)
    # Underlyings with a live order that has not reached a terminal state.
    # A working order is real exposure-in-waiting: it is not yet a position,
    # so the open-position gates cannot see it.
    working_order_symbols: frozenset[str] = field(default_factory=frozenset)

    # Broker truth, read from the account rather than the journal. The journal
    # can diverge; for blocking NEW exposure the broker is authoritative.
    broker_position_symbols: frozenset[str] = field(default_factory=frozenset)
    broker_working_symbols: frozenset[str] = field(default_factory=frozenset)
    broker_truth_available: bool = False
    # Underlyings the broker holds that could not be resolved into a managed
    # spread. Their risk is real but absent from every total below, so the
    # aggregate caps would be computed against an understated portfolio.
    unaccounted_broker_symbols: frozenset[str] = field(default_factory=frozenset)

    @property
    def exposure_fully_accounted(self) -> bool:
        """Whether the risk totals here can be trusted as complete.

        A broker position the agent could not resolve into a managed spread
        carries risk that appears in no total below, so every aggregate cap
        would be measured against an understated portfolio. That is a reason to
        refuse new risk rather than a rounding detail.

        Adoption is what keeps this true in the normal case: a spread the
        broker holds is pulled into the journal before risk is computed, so it
        is counted. This flag is the residue -- the spreads adoption could not
        make sense of, which are the ones nothing else can see.
        """
        return self.broker_truth_available and not self.unaccounted_broker_symbols

    def has_working_order_for(self, symbol: str) -> bool:
        return symbol.upper() in {s.upper() for s in self.working_order_symbols}

    def broker_has_position_for(self, symbol: str) -> bool:
        return symbol.upper() in {s.upper() for s in self.broker_position_symbols}

    def broker_has_working_order_for(self, symbol: str) -> bool:
        return symbol.upper() in {s.upper() for s in self.broker_working_symbols}

    @property
    def open_position_count(self) -> int:
        return len(self.open_positions)

    @property
    def total_defined_risk_cents(self) -> int:
        return sum(p.max_loss_cents for p in self.open_positions)

    @property
    def session_pnl_cents(self) -> int:
        """Realised plus unrealised profit and loss for the session."""
        return self.realized_pnl_today_cents + self.unrealized_pnl_cents

    def positions_in_group(self, limits: RiskLimits, group: str | None) -> list[PositionRecord]:
        if group is None:
            return []
        return [p for p in self.open_positions if limits.group_for(p.symbol) == group]

    def defined_risk_in_group_cents(self, limits: RiskLimits, group: str | None) -> int:
        return sum(p.max_loss_cents for p in self.positions_in_group(limits, group))

    def has_open_position_for(self, symbol: str) -> bool:
        return any(p.symbol.upper() == symbol.upper() for p in self.open_positions)


__all__ = ["PortfolioState"]
