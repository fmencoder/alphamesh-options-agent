"""Deterministic quantitative opportunity score and directional bias.

This is the gate that decides whether the AI reasoning council runs at all.
Everything here is configuration-driven and free of I/O, so a score can be
recomputed exactly from the journalled feature vector.
"""

from __future__ import annotations

import math
from datetime import datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

from alphamesh.config import StrategyConfig, UniverseConfig
from alphamesh.intelligence.features import compute_features
from alphamesh.models.domain import (
    Direction,
    MarketSnapshot,
    QuantSignal,
    ReasonCode,
    Regime,
    RegimeAssessment,
)

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)


class ScoreProfile(StrEnum):
    OPENING = "OPENING"
    INTRADAY = "INTRADAY"


class EntryMode(StrEnum):
    """Which archetype the deterministic features look like.

    Advisory and observational: it never relaxes a gate. Both archetypes are
    expressed with the same two defined-risk vertical debit spreads.
    """

    MOMENTUM_BREAKOUT = "MOMENTUM_BREAKOUT"
    TREND_PULLBACK = "TREND_PULLBACK"
    UNCLASSIFIED = "UNCLASSIFIED"


def minutes_since_open(as_of: datetime) -> float:
    """Minutes elapsed since 09:30 ET on the snapshot's own trading day.

    Derived from the snapshot timestamp rather than wall-clock now, so a
    journalled feature vector always rescores to the identical value.
    """
    local = as_of.astimezone(MARKET_TZ)
    open_at = local.replace(
        hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0
    )
    return (local - open_at).total_seconds() / 60.0


def profile_for(as_of: datetime, strategies: StrategyConfig) -> ScoreProfile:
    elapsed = minutes_since_open(as_of)
    if 0.0 <= elapsed < strategies.opening_window_minutes:
        return ScoreProfile.OPENING
    return ScoreProfile.INTRADAY


def _squash(value: float, scale: float) -> float:
    """Map an unbounded value into [0, 1] with a smooth, monotone curve."""
    if scale <= 0:
        return 0.0
    return 1.0 - math.exp(-abs(value) / scale)


# Fallback component weights, used when config supplies none. Both profiles
# sum to 1.0 so the score is directly interpretable and the threshold carries
# the same meaning under either.
WEIGHTS: dict[str, float] = {
    "momentum": 0.30,
    "trend": 0.25,
    "vwap": 0.15,
    "participation": 0.15,
    "range_position": 0.15,
}

INTRADAY_WEIGHTS: dict[str, float] = {
    "momentum": 0.40,
    "trend": 0.35,
    "vwap": 0.25,
}


def weights_for(profile: ScoreProfile, strategies: StrategyConfig) -> dict[str, float]:
    if profile is ScoreProfile.OPENING:
        return dict(strategies.opening_weights or WEIGHTS)
    return dict(strategies.intraday_weights or INTRADAY_WEIGHTS)

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


def _sign(value: float) -> float:
    return math.copysign(1.0, value) if value else 0.0


def components_aligned(features: dict[str, float], bias: Direction) -> bool:
    """True when momentum, trend and VWAP all point the same way as the bias.

    This is the alignment requirement that gates the sub-normal threshold: a
    lower bar is only ever offered to a signal whose persistent components
    unanimously agree.
    """
    if bias is Direction.NEUTRAL:
        return False
    want = 1.0 if bias is Direction.BULLISH else -1.0
    return all(
        _sign(features.get(name, 0.0)) == want
        for name in ("ret_5m", "ret_15m", "trend_strength", "vwap_deviation")
    )


def classify_entry_mode(features: dict[str, float], bias: Direction) -> EntryMode:
    """Label the archetype. Observational only; never relaxes a gate."""
    if bias is Direction.NEUTRAL:
        return EntryMode.UNCLASSIFIED
    want = 1.0 if bias is Direction.BULLISH else -1.0
    trend_agrees = _sign(features.get("trend_strength", 0.0)) == want
    short_agrees = _sign(features.get("ret_5m", 0.0)) == want
    vwap_agrees = _sign(features.get("vwap_deviation", 0.0)) == want
    if trend_agrees and short_agrees and vwap_agrees:
        return EntryMode.MOMENTUM_BREAKOUT
    # A pullback is an established trend with a temporary counter-move that
    # has already reclaimed VWAP in the direction of that higher-order trend.
    if trend_agrees and vwap_agrees and not short_agrees:
        return EntryMode.TREND_PULLBACK
    return EntryMode.UNCLASSIFIED


def threshold_for(
    signal: QuantSignal, regime: RegimeAssessment, strategies: StrategyConfig
) -> tuple[float, str]:
    """Regime-conditioned entry threshold, and the band that produced it.

    The sub-normal band is deliberately hard to reach: it needs a trending
    regime, agreement between that regime and the signal's own bias, and
    unanimous alignment of the persistent components. Anything short of all
    three gets the normal bar or higher.
    """
    bands = strategies.regime_thresholds or {}
    normal = float(bands.get("normal", strategies.quant_score_threshold))
    strong = float(bands.get("strong_trend", normal))
    ranged = float(bands.get("range_bound", normal))
    floor = strategies.absolute_min_quant_threshold

    if regime.regime is Regime.RANGE_BOUND:
        return max(ranged, floor), "range_bound"

    trending = regime.regime in (Regime.BULLISH_TREND, Regime.BEARISH_TREND)
    if (
        trending
        and regime.direction is signal.directional_bias
        and components_aligned(signal.features, signal.directional_bias)
    ):
        # Never below the configured floor, whatever the config says.
        return max(strong, floor), "strong_trend"
    return max(normal, floor), "normal"


def evaluate_gate(
    signal: QuantSignal, regime: RegimeAssessment, strategies: StrategyConfig
) -> tuple[bool, float, str, tuple[ReasonCode, ...]]:
    """Apply the regime-conditioned gate. Returns (passes, threshold, band, codes)."""
    threshold, band = threshold_for(signal, regime, strategies)
    reasons: list[ReasonCode] = []
    if signal.directional_bias is Direction.NEUTRAL:
        reasons.append(ReasonCode.DIRECTION_CONFLICT)
    if signal.quant_score < threshold:
        reasons.append(ReasonCode.QUANT_SCORE_BELOW_THRESHOLD)
    # Insufficient data is fatal regardless of regime.
    if ReasonCode.INSUFFICIENT_MARKET_DATA in signal.reason_codes:
        reasons.append(ReasonCode.INSUFFICIENT_MARKET_DATA)
    return (not reasons), threshold, band, tuple(reasons)


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

    profile = profile_for(snapshot.as_of, strategies)
    components = score_components(features)
    weights = weights_for(profile, strategies)
    # Only the weighted components contribute. participation and
    # range_position are still computed and still journalled under the
    # intraday profile; they simply no longer depress a valid trend signal
    # once their information value has expired.
    raw = sum(weight * components.get(name, 0.0) for name, weight in weights.items())
    score = max(0.0, min(1.0, raw))
    bias = directional_bias(features)

    passes = score >= strategies.quant_score_threshold and bias is not Direction.NEUTRAL
    if score < strategies.quant_score_threshold:
        reasons.append(ReasonCode.QUANT_SCORE_BELOW_THRESHOLD)
    if bias is Direction.NEUTRAL:
        reasons.append(ReasonCode.DIRECTION_CONFLICT)

    merged = dict(features)
    merged.update({f"score_{k}": v for k, v in components.items()})
    merged["profile_is_opening"] = 1.0 if profile is ScoreProfile.OPENING else 0.0
    merged["minutes_since_open"] = minutes_since_open(snapshot.as_of)

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
    "EntryMode",
    "ScoreProfile",
    "build_quant_signal",
    "classify_entry_mode",
    "components_aligned",
    "directional_bias",
    "evaluate_gate",
    "minutes_since_open",
    "profile_for",
    "score_components",
    "threshold_for",
    "weights_for",
]
