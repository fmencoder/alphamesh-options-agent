"""Structured evidence packet handed to the AI reasoning council.

The council sees market evidence and nothing else. It never receives account
equity, buying power, risk limits, open positions or position sizes, so it
cannot reason about - let alone argue for - a larger allocation.
"""

from __future__ import annotations

import json
from typing import Any

from alphamesh.models.domain import QuantSignal, RegimeAssessment

EVIDENCE_FEATURES: tuple[str, ...] = (
    "ret_1m",
    "ret_5m",
    "ret_15m",
    "vwap_deviation",
    "atr_pct",
    "realized_vol",
    "volume_acceleration",
    "opening_range_position",
    "trend_strength",
    "distance_from_high",
    "distance_from_low",
)


def build_evidence(signal: QuantSignal, regime: RegimeAssessment) -> dict[str, Any]:
    """Whitelist the fields the council may see. Additive by construction."""
    return {
        "symbol": signal.symbol,
        "as_of": signal.as_of.isoformat(),
        "quant_score": round(signal.quant_score, 4),
        "quant_directional_bias": str(signal.directional_bias),
        "regime": str(regime.regime),
        "regime_direction": str(regime.direction),
        "regime_confidence": round(regime.confidence, 4),
        "regime_risk_flags": list(regime.risk_flags),
        "features": {
            key: round(signal.features[key], 6)
            for key in EVIDENCE_FEATURES
            if key in signal.features
        },
    }


def render_evidence(evidence: dict[str, Any]) -> str:
    return json.dumps(evidence, indent=2, sort_keys=True)


__all__ = ["EVIDENCE_FEATURES", "build_evidence", "render_evidence"]
