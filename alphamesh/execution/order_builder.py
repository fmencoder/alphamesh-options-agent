"""Order construction and idempotency keys.

The client order id is a pure function of the trade's economics. Two runs that
would place the same trade produce the same id, so Alpaca rejects the duplicate
even if our own journal were somehow lost. The id is also reserved in the
journal *before* the network call, which is what makes a crash mid-submit safe.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from alphamesh.models.domain import (
    OptionContractCandidate,
    OrderIntent,
    OrderSide,
    PositionIntent,
    PositionRecord,
    RiskDecision,
    SpreadLeg,
    SpreadStructure,
    Strategy,
    TradeDecision,
)

CLIENT_ORDER_ID_PREFIX = "alphamesh"
MAX_CLIENT_ORDER_ID_LEN = 48
"""Alpaca allows longer ids; we stay well short so the value is readable in the UI."""

STRATEGY_CODES: dict[Strategy, str] = {
    Strategy.BULL_CALL_SPREAD: "BCS",
    Strategy.BEAR_PUT_SPREAD: "BPS",
    Strategy.NO_TRADE: "NON",
}


def signal_hash(
    decision: TradeDecision, spread: SpreadStructure, quantity: int, limit_cents: int
) -> str:
    """Stable digest of everything that defines this specific trade."""
    payload = "|".join(
        [
            decision.decision_id,
            decision.symbol,
            decision.strategy.value,
            spread.long_leg.contract.symbol,
            spread.short_leg.contract.symbol,
            str(quantity),
            str(limit_cents),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_client_order_id(
    decision: TradeDecision, spread: SpreadStructure, quantity: int, limit_cents: int
) -> str:
    code = STRATEGY_CODES.get(decision.strategy, "UNK")
    digest = signal_hash(decision, spread, quantity, limit_cents)
    candidate = f"{CLIENT_ORDER_ID_PREFIX}-{decision.symbol.upper()}-{code}-{digest}"
    return candidate[:MAX_CLIENT_ORDER_ID_LEN]


def build_order_intent(
    decision: TradeDecision,
    spread: SpreadStructure,
    risk: RiskDecision,
    now: datetime,
) -> OrderIntent:
    """Assemble the submittable order. Refuses to build from a rejected risk verdict."""
    if not risk.approved:
        raise ValueError("cannot build an order from a rejected RiskDecision")
    if risk.quantity <= 0:
        raise ValueError("cannot build an order with non-positive quantity")

    client_order_id = build_client_order_id(
        decision, spread, risk.quantity, spread.limit_price_cents
    )
    return OrderIntent(
        client_order_id=client_order_id,
        decision_id=decision.decision_id,
        symbol=decision.symbol,
        strategy=decision.strategy,
        quantity=risk.quantity,
        limit_price_cents=spread.limit_price_cents,
        legs=(spread.long_leg, spread.short_leg),
        max_loss_cents=risk.max_loss_cents,
        created_at=now,
    )


EXIT_ID_INFIX = "X"
"""Marks a closing order id, so an exit is never mistaken for an entry."""


def build_exit_client_order_id(
    position: PositionRecord, limit_cents: int, attempt: int = 0
) -> str:
    """Deterministic id for the order that closes one position.

    Derived from the position, the price being quoted and the attempt number,
    so a restart between reserving the id and sending the order finds the same
    id and reconciles it instead of flattening the spread twice.

    ``attempt`` is the count of closing orders already raised for this position,
    read back from the journal. It only advances once a previous attempt is
    confirmed dead, which is what lets a close be re-quoted: an id is never
    reused, and a still-live attempt never mints a second one.
    """
    payload = "|".join(
        [
            position.position_id,
            position.long_symbol,
            position.short_symbol,
            str(position.quantity),
            str(limit_cents),
            f"exit{attempt}",
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    code = STRATEGY_CODES.get(position.strategy, "UNK")
    candidate = (
        f"{CLIENT_ORDER_ID_PREFIX}-{position.symbol.upper()}-{code}{EXIT_ID_INFIX}-{digest}"
    )
    return candidate[:MAX_CLIENT_ORDER_ID_LEN]


def build_exit_intent(
    position: PositionRecord,
    long_contract: OptionContractCandidate,
    short_contract: OptionContractCandidate,
    limit_price_cents: int,
    now: datetime,
    client_order_id: str | None = None,
    attempt: int = 0,
) -> OrderIntent:
    """Assemble the order that flattens an open spread.

    The legs are carried in their *opening* orientation. ``Broker.close_spread``
    mirrors them into SELL_TO_CLOSE / BUY_TO_CLOSE, so the closing intents are
    produced in exactly one place rather than being spelled out twice.
    """
    if limit_price_cents <= 0:
        raise ValueError("an exit limit price must be positive")
    if long_contract.symbol != position.long_symbol:
        raise ValueError("long contract does not match the position")
    if short_contract.symbol != position.short_symbol:
        raise ValueError("short contract does not match the position")

    return OrderIntent(
        client_order_id=(
            client_order_id
            or build_exit_client_order_id(position, limit_price_cents, attempt)
        ),
        decision_id=position.decision_id,
        symbol=position.symbol,
        strategy=position.strategy,
        quantity=position.quantity,
        limit_price_cents=limit_price_cents,
        legs=(
            SpreadLeg(
                contract=long_contract,
                side=OrderSide.BUY,
                ratio=1,
                position_intent=PositionIntent.BUY_TO_OPEN,
            ),
            SpreadLeg(
                contract=short_contract,
                side=OrderSide.SELL,
                ratio=1,
                position_intent=PositionIntent.SELL_TO_OPEN,
            ),
        ),
        max_loss_cents=max(position.max_loss_cents, 1),
        created_at=now,
    )


def to_alpaca_payload(intent: OrderIntent) -> dict[str, object]:
    """Render the intent as an Alpaca multi-leg (``mleg``) options order.

    The limit price is the net debit per spread, positive because these are
    always debit structures.
    """
    return {
        "order_class": "mleg",
        "qty": str(intent.quantity),
        "type": "limit",
        "time_in_force": "day",
        "limit_price": f"{intent.limit_price:.2f}",
        "client_order_id": intent.client_order_id,
        "legs": [
            {
                "symbol": leg.contract.symbol,
                "ratio_qty": str(leg.ratio),
                "side": leg.side.value,
                "position_intent": leg.position_intent.value,
            }
            for leg in intent.legs
        ],
    }


__all__ = [
    "CLIENT_ORDER_ID_PREFIX",
    "EXIT_ID_INFIX",
    "MAX_CLIENT_ORDER_ID_LEN",
    "STRATEGY_CODES",
    "build_client_order_id",
    "build_exit_client_order_id",
    "build_exit_intent",
    "build_order_intent",
    "signal_hash",
    "to_alpaca_payload",
]
