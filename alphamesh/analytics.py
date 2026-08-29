"""Competition analytics.

Performance is sliced by strategy, regime, symbol and confidence bucket. The
groundwork for adaptive strategy weighting lives here too - and it is
deliberately incapable of touching risk: :func:`adaptive_weights` returns
*advisory* multipliers bounded to a narrow band, which the scoring gate may
consult and which the Risk Governor never reads.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from alphamesh.risk.money import to_dollars

CONFIDENCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.50-0.65", 0.50, 0.65),
    ("0.65-0.75", 0.65, 0.75),
    ("0.75-0.85", 0.75, 0.85),
    ("0.85-1.00", 0.85, 1.01),
)

WEIGHT_FLOOR = 0.75
WEIGHT_CEILING = 1.25
"""Adaptive weights are clamped hard. Even a long losing streak can only nudge
the opportunity score; it can never widen a risk limit."""


def confidence_bucket(confidence: float) -> str:
    for label, lo, hi in CONFIDENCE_BUCKETS:
        if lo <= confidence < hi:
            return label
    return "<0.50" if confidence < 0.50 else CONFIDENCE_BUCKETS[-1][0]


@dataclass(frozen=True)
class GroupStats:
    key: str
    trades: int
    wins: int
    total_pnl_cents: int
    largest_win_cents: int
    largest_loss_cents: int
    total_holding_minutes: float
    total_return_on_risk: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def average_pnl_cents(self) -> float:
        return self.total_pnl_cents / self.trades if self.trades else 0.0

    @property
    def average_holding_minutes(self) -> float:
        return self.total_holding_minutes / self.trades if self.trades else 0.0

    @property
    def average_return_on_risk(self) -> float:
        return self.total_return_on_risk / self.trades if self.trades else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(to_dollars(self.total_pnl_cents), 2),
            "average_pnl": round(to_dollars(int(self.average_pnl_cents)), 2),
            "average_return_on_defined_risk": round(self.average_return_on_risk, 4),
            "average_holding_minutes": round(self.average_holding_minutes, 1),
            "largest_win": round(to_dollars(self.largest_win_cents), 2),
            "largest_loss": round(to_dollars(self.largest_loss_cents), 2),
        }


def _accumulate(rows: Iterable[dict[str, Any]], key_field: str) -> dict[str, GroupStats]:
    trades: dict[str, int] = defaultdict(int)
    wins: dict[str, int] = defaultdict(int)
    pnl: dict[str, int] = defaultdict(int)
    best: dict[str, int] = defaultdict(int)
    worst: dict[str, int] = defaultdict(int)
    holding: dict[str, float] = defaultdict(float)
    ror: dict[str, float] = defaultdict(float)

    for row in rows:
        if key_field == "confidence_bucket":
            key = confidence_bucket(float(row.get("confidence", 0.0)))
        else:
            key = str(row.get(key_field, "UNKNOWN"))
        value = int(row.get("realized_pnl_cents", 0))
        trades[key] += 1
        if value > 0:
            wins[key] += 1
        pnl[key] += value
        best[key] = max(best[key], value)
        worst[key] = min(worst[key], value)
        holding[key] += float(row.get("holding_minutes", 0.0))
        ror[key] += float(row.get("return_on_defined_risk", 0.0))

    return {
        key: GroupStats(
            key=key,
            trades=trades[key],
            wins=wins[key],
            total_pnl_cents=pnl[key],
            largest_win_cents=best[key],
            largest_loss_cents=worst[key],
            total_holding_minutes=holding[key],
            total_return_on_risk=ror[key],
        )
        for key in trades
    }


def build_report(outcomes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Full competition report across every slice."""
    overall = _accumulate(outcomes, "__all__") if outcomes else {}
    total = GroupStats("ALL", 0, 0, 0, 0, 0, 0.0, 0.0)
    if outcomes:
        merged = _accumulate([{**row, "__all__": "ALL"} for row in outcomes], "__all__")
        total = merged.get("ALL", total)

    return {
        "overall": total.as_dict(),
        "by_strategy": {k: v.as_dict() for k, v in _accumulate(outcomes, "strategy").items()},
        "by_regime": {k: v.as_dict() for k, v in _accumulate(outcomes, "regime").items()},
        "by_symbol": {k: v.as_dict() for k, v in _accumulate(outcomes, "symbol").items()},
        "by_confidence": {
            k: v.as_dict() for k, v in _accumulate(outcomes, "confidence_bucket").items()
        },
        "by_exit_reason": {
            k: v.as_dict() for k, v in _accumulate(outcomes, "exit_reason").items()
        },
        "_unused": len(overall),
    }


def adaptive_weights(
    outcomes: Sequence[dict[str, Any]], min_trades: int = 5
) -> dict[str, float]:
    """Advisory per-(strategy, regime) multipliers, hard-clamped.

    A pairing needs at least ``min_trades`` closed trades before it earns any
    adjustment, and the adjustment is bounded to +/-25%. This is scoring
    guidance only; the Risk Governor never reads it.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[f"{row.get('strategy')}|{row.get('regime')}"].append(row)

    weights: dict[str, float] = {}
    for key, rows in grouped.items():
        if len(rows) < min_trades:
            weights[key] = 1.0
            continue
        wins = sum(1 for r in rows if int(r.get("realized_pnl_cents", 0)) > 0)
        win_rate = wins / len(rows)
        # Centre on a 50% win rate; scale gently, then clamp.
        raw = 1.0 + (win_rate - 0.5) * 0.5
        weights[key] = max(WEIGHT_FLOOR, min(WEIGHT_CEILING, raw))
    return weights


__all__ = [
    "CONFIDENCE_BUCKETS",
    "WEIGHT_CEILING",
    "WEIGHT_FLOOR",
    "GroupStats",
    "adaptive_weights",
    "build_report",
    "confidence_bucket",
]
