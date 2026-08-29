"""Deterministic quantitative opportunity score and directional bias.

This is the gate that decides whether the AI reasoning council runs at all.
Everything here is configuration-driven and free of I/O, so a score can be
recomputed exactly from the journalled feature vector.
"""

from __future__ import annotations

import math

from alphamesh.config import StrategyConfig, UniverseConfig
from alphamesh.intelligence.features import compute_features
from alphamesh.models.domain import Direction, MarketSnapshot, QuantSignal, ReasonCode


def _squash(value: float, scale: float) -> float:
    """Map an unbounded value into [0, 1] with a smooth, monotone curve."""
    if scale <= 0:
        return 0.0
    return 1.0 - math.exp(-abs(value) / scale)


# Component weights. They sum to 1.0 so the score is directly interpretable.
WEIGHTS: dict[str, float] = {
    "momentum": 0.30,
    "trend": 0.25,
    "vwap": 0.15,
    "participation": 0.15,
    "range_position": 0.15,
}

# Scales chosen so that a typical index-ETF intraday move lands mid-range.
MOMENTUM_SCALE = 0.0015
VWAP_SCALE = 0.0010
TREND_SCALE = 0.35


def score_components(features: dict[str, float]) -> dict[str, float]:
    """Individual [0, 1] sub-scores. Exposed so the dashboard can show them."""
    momentum = (
        0.5 * _squash(features.get("ret_5m", 0.0), MOMENTUM_SCALE)
        + 0.5 * _squash(features.get("ret_15m", 0.0), MOMENTUM_SCALE * 2)
    )
    trend = _squash(features.get("trend_strength", 0.0), TREND_SCALE)
    vwap = _squash(features.get("vwap_deviation", 0.0), VWAP_SCALE)
    accel = features.get("volume_acceleration", 1.0)
    participation = max(0.0, min(1.0, (accel - 0.8) / 1.2))
    range_pos = min(1.0, abs(features.get("opening_range_position", 0.0)))
    return {
        "momentum": momentum,
        "trend": trend,
        "vwap": vwap,
        "participation": participation,
        "range_position": range_pos,
    }


def directional_bias(features: dict[str, float]) -> Direction:
    """Sign consensus across the directional features.

    Requires agreement: mixed signs return NEUTRAL, which downstream code
    treats as a reason to stand aside rather than to guess.
    """
    votes = [
        math.copysign(1.0, features["ret_5m"]) if features.get("ret_5m") else 0.0,
        math.copysign(1.0, features["ret_15m"]) if features.get("ret_15m") else 0.0,
        math.copysign(1.0, features["trend_strength"]) if features.get("trend_strength") else 0.0,
        (
            math.copysign(1.0, features["vwap_deviation"])
            if features.get("vwap_deviation")
            else 0.0
        ),
    ]
    net = sum(votes)
    if net >= 2.0:
        return Direction.BULLISH
    if net <= -2.0:
        return Direction.BEARISH
    return Direction.NEUTRAL


def build_quant_signal(
    snapshot: MarketSnapshot,
    strategies: StrategyConfig,
    universe: UniverseConfig,
) -> QuantSignal:
    """Score one snapshot and decide whether it clears the AI-invocation gate."""
    features = compute_features(snapshot)
    reasons: list[ReasonCode] = []

    if len(snapshot.bars) < universe.min_bars_required:
        reasons.append(ReasonCode.INSUFFICIENT_MARKET_DATA)
        return QuantSignal(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            features=features,
            quant_score=0.0,
            directional_bias=Direction.NEUTRAL,
            passes_gate=False,
            reason_codes=tuple(reasons),
        )

    components = score_components(features)
    raw = sum(WEIGHTS[name] * value for name, value in components.items())
    score = max(0.0, min(1.0, raw))
    bias = directional_bias(features)

    passes = score >= strategies.quant_score_threshold and bias is not Direction.NEUTRAL
    if score < strategies.quant_score_threshold:
        reasons.append(ReasonCode.QUANT_SCORE_BELOW_THRESHOLD)
    if bias is Direction.NEUTRAL:
        reasons.append(ReasonCode.DIRECTION_CONFLICT)

    merged = dict(features)
    merged.update({f"score_{k}": v for k, v in components.items()})

    return QuantSignal(
        symbol=snapshot.symbol,
        as_of=snapshot.as_of,
        features=merged,
        quant_score=score,
        directional_bias=bias,
        passes_gate=passes,
        reason_codes=tuple(reasons),
    )


__all__ = [
    "WEIGHTS",
    "build_quant_signal",
    "directional_bias",
    "score_components",
]
