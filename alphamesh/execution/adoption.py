"""Adopting live broker spreads the journal does not know about.

The journal can lose a position: a restart, a lost write, or -- as happened in
production -- an exit path that marked a position closed without ever sending a
closing order. Whatever the cause, a spread the broker is holding and the agent
has forgotten is unmanaged risk: no exit rule runs against it, and the
broker-truth entry guard blocks its underlying forever because the position
never goes away.

This module reconstructs those spreads from broker truth alone. It pairs option
legs into verticals, identifies the strategy from the strikes, and recovers the
real entry economics from the multi-leg order that opened them. Nothing here
submits anything: adoption discovers, reconstructs and reports, and the normal
management path decides what to do next.

Ambiguity fails closed. A leg set that is not unmistakably one of the two
defined-risk verticals the agent trades is reported as ambiguous and left
alone. Guessing a pairing could produce a "closing" order that flattens one leg
and leaves the other naked, which is worse than leaving the position untouched.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from alphamesh.alpaca.options import occ_underlying, parse_occ_symbol
from alphamesh.alpaca.types import BrokerOrderSummary, BrokerPosition
from alphamesh.models.domain import OPTION_MULTIPLIER, OptionType, Strategy

log = logging.getLogger(__name__)

#: Order statuses that mean the order actually put the position on.
_FILLED_STATUSES = frozenset({"filled", "partially_filled"})

#: How the entry cost basis was established, in decreasing order of authority.
ENTRY_BASIS_ORDER_FILL = "ORDER_FILL"
ENTRY_BASIS_POSITION_COST = "POSITION_COST"


@dataclass(frozen=True)
class AdoptedSpread:
    """A live broker spread, reconstructed well enough to manage and exit."""

    symbol: str
    strategy: Strategy
    quantity: int
    expiration: date
    option_type: OptionType
    long_symbol: str
    short_symbol: str
    long_strike: float
    short_strike: float
    entry_debit_cents: int
    strike_width_cents: int
    entry_basis: str
    client_order_id: str | None = None
    opened_at: datetime | None = None

    @property
    def max_loss_cents(self) -> int:
        """A vertical debit spread can lose exactly the premium paid."""
        return self.entry_debit_cents

    @property
    def max_profit_cents(self) -> int:
        width_total = self.strike_width_cents * OPTION_MULTIPLIER * self.quantity
        return max(0, width_total - self.entry_debit_cents)


@dataclass(frozen=True)
class AmbiguousSpread:
    """Broker legs that could not be resolved into one defined-risk vertical."""

    symbol: str
    reason: str
    legs: tuple[str, ...]


@dataclass
class AdoptionSummary:
    """What one adoption pass did, in terms callers can act on."""

    adopted: int = 0
    ambiguous: int = 0
    error: str | None = None
    detail: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        # The detail is nested rather than merged: it carries its own "spreads"
        # and "ambiguous" keys, and splatting it here silently replaced the
        # ambiguous COUNT with the ambiguous LIST, so the journalled payload
        # lost the number every caller reads.
        payload: dict[str, object] = {
            "adopted": self.adopted,
            "ambiguous": self.ambiguous,
            "detail": dict(self.detail),
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass
class AdoptionResult:
    spreads: list[AdoptedSpread] = field(default_factory=list)
    ambiguous: list[AmbiguousSpread] = field(default_factory=list)

    @property
    def ambiguous_symbols(self) -> frozenset[str]:
        return frozenset(a.symbol.upper() for a in self.ambiguous)

    def as_dict(self) -> dict[str, object]:
        return {
            "spreads": [
                {
                    "symbol": s.symbol,
                    "strategy": s.strategy.value,
                    "quantity": s.quantity,
                    "long_symbol": s.long_symbol,
                    "short_symbol": s.short_symbol,
                    "expiration": s.expiration.isoformat(),
                    "entry_debit_cents": s.entry_debit_cents,
                    "entry_basis": s.entry_basis,
                    "client_order_id": s.client_order_id,
                }
                for s in self.spreads
            ],
            "ambiguous": [
                {"symbol": a.symbol, "reason": a.reason, "legs": list(a.legs)}
                for a in self.ambiguous
            ],
        }


def _to_cents(value: float) -> int:
    return round(value * 100)


def reconstruct_spreads(
    positions: list[BrokerPosition],
    orders: list[BrokerOrderSummary] | None = None,
) -> AdoptionResult:
    """Rebuild vertical spreads from raw broker option positions.

    Legs are grouped by (underlying, expiration, option type). A group resolves
    only when it is exactly one long leg and one short leg of equal size whose
    strikes form one of the two supported debit verticals. Anything else --
    an odd number of legs, mismatched sizes, a credit structure, a naked leg --
    is reported ambiguous and never guessed at.
    """
    result = AdoptionResult()
    groups: dict[tuple[str, date, OptionType], list[tuple[BrokerPosition, float]]] = (
        defaultdict(list)
    )

    for position in positions:
        parsed = parse_occ_symbol(position.symbol)
        root = occ_underlying(position.symbol)
        if parsed is None or root is None:
            # Not an option (an equity leg, say). Nothing here understands it,
            # and it is not a spread leg, so it is not adopted.
            continue
        if position.quantity == 0:
            continue
        expiration, option_type, strike = parsed
        groups[(root, expiration, option_type)].append((position, strike))

    for (root, expiration, option_type), legs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2].value)
    ):
        symbols = tuple(sorted(p.symbol for p, _ in legs))
        if len(legs) != 2:
            result.ambiguous.append(
                AmbiguousSpread(
                    root,
                    f"expected exactly two legs for one vertical, found {len(legs)}",
                    symbols,
                )
            )
            continue

        longs = [(p, k) for p, k in legs if p.quantity > 0]
        shorts = [(p, k) for p, k in legs if p.quantity < 0]
        if len(longs) != 1 or len(shorts) != 1:
            result.ambiguous.append(
                AmbiguousSpread(
                    root, "not one long leg against one short leg", symbols
                )
            )
            continue

        long_pos, long_strike = longs[0]
        short_pos, short_strike = shorts[0]
        quantity = long_pos.quantity
        if quantity != abs(short_pos.quantity):
            result.ambiguous.append(
                AmbiguousSpread(
                    root,
                    (
                        f"leg sizes differ: long {long_pos.quantity} vs short "
                        f"{short_pos.quantity}"
                    ),
                    symbols,
                )
            )
            continue

        if option_type is OptionType.CALL and long_strike < short_strike:
            strategy = Strategy.BULL_CALL_SPREAD
        elif option_type is OptionType.PUT and long_strike > short_strike:
            strategy = Strategy.BEAR_PUT_SPREAD
        else:
            # A long leg further out of the money than the short one is a credit
            # spread, which this agent never opens and must not try to manage.
            result.ambiguous.append(
                AmbiguousSpread(
                    root,
                    (
                        f"{option_type.value} legs long {long_strike} / short "
                        f"{short_strike} are not a supported debit vertical"
                    ),
                    symbols,
                )
            )
            continue

        width_cents = abs(_to_cents(long_strike) - _to_cents(short_strike))
        entry = _entry_economics(
            long_pos, short_pos, quantity, orders or [], width_cents
        )
        if entry is None:
            result.ambiguous.append(
                AmbiguousSpread(
                    root,
                    "entry cost basis could not be established from broker data",
                    symbols,
                )
            )
            continue
        debit_cents, basis, client_order_id, opened_at = entry

        result.spreads.append(
            AdoptedSpread(
                symbol=root,
                strategy=strategy,
                quantity=quantity,
                expiration=expiration,
                option_type=option_type,
                long_symbol=long_pos.symbol,
                short_symbol=short_pos.symbol,
                long_strike=long_strike,
                short_strike=short_strike,
                entry_debit_cents=debit_cents,
                strike_width_cents=width_cents,
                entry_basis=basis,
                client_order_id=client_order_id,
                opened_at=opened_at,
            )
        )

    return result


def _entry_economics(
    long_pos: BrokerPosition,
    short_pos: BrokerPosition,
    quantity: int,
    orders: list[BrokerOrderSummary],
    width_cents: int,
) -> tuple[int, str, str | None, datetime | None] | None:
    """Total entry debit in cents, and where the number came from.

    The multi-leg order that opened the spread is authoritative: its net filled
    price is the debit actually paid. Failing that, the per-leg average entry
    prices the broker carries give the same figure leg by leg. Nothing is
    invented -- a spread whose basis cannot be established is not adopted.
    """
    legs = frozenset({long_pos.symbol.upper(), short_pos.symbol.upper()})

    # 1. The originating multi-leg order, newest first.
    candidates = [
        o
        for o in orders
        if o.leg_symbols == legs
        and o.status.lower() in _FILLED_STATUSES
        and o.filled_avg_price_cents is not None
        and o.filled_quantity > 0
        and _is_opening(o)
    ]
    candidates.sort(key=lambda o: (o.submitted_at or datetime.min.replace(tzinfo=UTC)))
    if candidates:
        order = candidates[-1]
        per_spread = order.filled_avg_price_cents or 0
        if 0 < per_spread < width_cents:
            return (
                per_spread * OPTION_MULTIPLIER * quantity,
                ENTRY_BASIS_ORDER_FILL,
                order.client_order_id or None,
                order.submitted_at,
            )

    # 2. Per-leg cost basis carried on the positions themselves.
    per_spread_cents = _to_cents(long_pos.avg_entry_price) - _to_cents(
        short_pos.avg_entry_price
    )
    if 0 < per_spread_cents < width_cents:
        return (
            per_spread_cents * OPTION_MULTIPLIER * quantity,
            ENTRY_BASIS_POSITION_COST,
            None,
            None,
        )
    return None


def _is_opening(order: BrokerOrderSummary) -> bool:
    """True when the order's legs opened exposure rather than closing it.

    Alpaca reports the intent per leg, so a closing order that happens to carry
    the same two contracts is never mistaken for the entry.
    """
    intents = {leg.position_intent.lower() for leg in order.legs if leg.position_intent}
    if not intents:
        # Older or sparser payloads omit the intent; the caller has already
        # matched the leg set, and a spread that is still open cannot have been
        # opened by a closing order.
        return True
    return all(intent.endswith("to_open") for intent in intents)


__all__ = [
    "ENTRY_BASIS_ORDER_FILL",
    "ENTRY_BASIS_POSITION_COST",
    "AdoptedSpread",
    "AdoptionResult",
    "AdoptionSummary",
    "AmbiguousSpread",
    "reconstruct_spreads",
]
