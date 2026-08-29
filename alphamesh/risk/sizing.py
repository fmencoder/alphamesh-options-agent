"""Position sizing.

Size is derived, never proposed. The AI supplies a confidence number and
nothing else; this module turns confidence into a per-trade dollar cap using
``risk.yaml``, then divides by the spread's exact per-contract maximum loss.
Anything that does not fit whole is rounded down, so the cap is never exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphamesh.config import RiskLimits
from alphamesh.models.domain import OPTION_MULTIPLIER, SpreadStructure


@dataclass(frozen=True)
class SizingResult:
    quantity: int
    per_contract_loss_cents: int
    cap_cents: int
    max_loss_cents: int
    detail: str


def size_spread(
    spread: SpreadStructure,
    confidence: float,
    limits: RiskLimits,
    available_risk_cents: int | None = None,
) -> SizingResult:
    """Largest whole number of spreads whose defined loss fits every cap.

    ``available_risk_cents`` lets the caller shrink the trade to whatever
    portfolio headroom is left, rather than rejecting it outright.
    """
    per_contract = spread.limit_price_cents * OPTION_MULTIPLIER
    cap = limits.cap_cents_for_confidence(confidence)
    effective_cap = cap if available_risk_cents is None else min(cap, available_risk_cents)

    if per_contract <= 0:
        return SizingResult(0, per_contract, cap, 0, "spread has no positive premium")
    if effective_cap < per_contract:
        return SizingResult(
            0,
            per_contract,
            cap,
            0,
            (
                f"one spread risks {per_contract} cents, above the "
                f"{effective_cap} cent budget"
            ),
        )

    quantity = effective_cap // per_contract
    return SizingResult(
        quantity=int(quantity),
        per_contract_loss_cents=per_contract,
        cap_cents=cap,
        max_loss_cents=int(quantity * per_contract),
        detail=f"{quantity} spread(s) at {per_contract} cents defined risk each",
    )


__all__ = ["SizingResult", "size_spread"]
