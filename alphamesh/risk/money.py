"""Integer-cent money helpers.

Every dollar amount that crosses the risk or execution boundary is carried as
an integer number of cents. Binary floats cannot represent 0.05 exactly, and a
cap that is off by a fraction of a cent is a cap that can be walked through, so
the conversion happens once, at the edge, and never again.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def to_cents(price: float | Decimal | str) -> int:
    """Convert a dollar price to integer cents, rounding half away from zero."""
    quantised = Decimal(str(price)).quantize(CENT, rounding=ROUND_HALF_UP)
    return int(quantised * 100)


def to_dollars(cents: int) -> float:
    """Convert integer cents back to a dollar float for display and reporting."""
    return float(Decimal(cents) / 100)


def dollars_to_cents(dollars: float) -> int:
    return to_cents(dollars)


__all__ = ["CENT", "dollars_to_cents", "to_cents", "to_dollars"]
