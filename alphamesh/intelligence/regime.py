"""Market regime classification.

Six regimes, decided by explicit thresholds over the deterministic feature
vector. ``UNKNOWN`` and ``UNSTABLE`` are first-class outcomes and both push the
pipeline toward NO_TRADE rather than toward a guess.
"""

from __future__ import annotations

from alphamesh.config import UniverseConfig
from alphamesh.models.domain import (
    Direction,
    MarketSnapshot,
    Regime,
    RegimeAssessment,
)

# Classification thresholds. Tuned for 1-minute bars on liquid index ETFs.
TREND_STRONG = 0.20
TREND_WEAK = 0.05
VOL_EXPANSION_RATIO = 2.0
"""Realised vol above this multiple of its own ATR-implied baseline is an expansion."""
UNSTABLE_VOL_ANNUALISED = 0.60
CHOP_RANGE_PCT = 0.0012


def _baseline_vol(features: dict[str, float]) -> float:
    """ATR expressed as an annualised volatility, used as the calm baseline."""
    atr_pct = features.get("atr_pct", 0.0)
    # ATR is a per-minute range; scale to an annualised sigma equivalent.
    return atr_pct * (252 * 390) ** 0.5 * 0.5


def classify(
    snapshot: MarketSnapshot,
    features: dict[str, float],
    universe: UniverseConfig,
) -> RegimeAssessment:
    """Classify the regime for one symbol."""
    risk_flags: list[str] = []

    if len(snapshot.bars) < universe.min_bars_required or not features:
        return RegimeAssessment(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            regime=Regime.UNKNOWN,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            evidence={"bar_count": float(len(snapshot.bars))},
            risk_flags=("insufficient_history",),
        )

    trend = features.get("trend_strength", 0.0)
    realized = features.get("realized_vol", 0.0)
    baseline = _baseline_vol(features)
    vol_ratio = realized / baseline if baseline > 0 else 0.0
    hi_dist = features.get("distance_from_high", 0.0)
    lo_dist = features.get("distance_from_low", 0.0)
    span = hi_dist + lo_dist

    evidence = {
        "trend_strength": trend,
        "realized_vol": realized,
        "baseline_vol": baseline,
        "vol_ratio": vol_ratio,
        "range_span_pct": span,
        "volume_acceleration": features.get("volume_acceleration", 1.0),
        "vwap_deviation": features.get("vwap_deviation", 0.0),
    }

    # Instability dominates: a violently moving tape is not tradable by a
    # two-leg debit spread sized against a fixed dollar loss.
    if realized >= UNSTABLE_VOL_ANNUALISED:
        risk_flags.append("extreme_realized_volatility")
        return RegimeAssessment(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            regime=Regime.UNSTABLE,
            direction=Direction.NEUTRAL,
            confidence=min(1.0, realized / UNSTABLE_VOL_ANNUALISED / 2.0),
            evidence=evidence,
            risk_flags=tuple(risk_flags),
        )

    if vol_ratio >= VOL_EXPANSION_RATIO:
        risk_flags.append("volatility_expansion")
        direction = (
            Direction.BULLISH
            if trend > TREND_WEAK
            else Direction.BEARISH
            if trend < -TREND_WEAK
            else Direction.NEUTRAL
        )
        return RegimeAssessment(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            regime=Regime.VOLATILITY_EXPANSION,
            direction=direction,
            confidence=min(1.0, vol_ratio / (VOL_EXPANSION_RATIO * 2)),
            evidence=evidence,
            risk_flags=tuple(risk_flags),
        )

    if trend >= TREND_STRONG:
        return RegimeAssessment(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            regime=Regime.BULLISH_TREND,
            direction=Direction.BULLISH,
            confidence=min(1.0, trend / (TREND_STRONG * 2)),
            evidence=evidence,
            risk_flags=(),
        )

    if trend <= -TREND_STRONG:
        return RegimeAssessment(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            regime=Regime.BEARISH_TREND,
            direction=Direction.BEARISH,
            confidence=min(1.0, abs(trend) / (TREND_STRONG * 2)),
            evidence=evidence,
            risk_flags=(),
        )

    if abs(trend) < TREND_WEAK and span < CHOP_RANGE_PCT * 4:
        return RegimeAssessment(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            regime=Regime.RANGE_BOUND,
            direction=Direction.NEUTRAL,
            confidence=1.0 - abs(trend) / TREND_WEAK if TREND_WEAK else 0.5,
            evidence=evidence,
            risk_flags=("range_bound",),
        )

    # A drift that is neither a clean trend nor a tight range: no confident call.
    risk_flags.append("indeterminate_structure")
    return RegimeAssessment(
        symbol=snapshot.symbol,
        as_of=snapshot.as_of,
        regime=Regime.UNKNOWN,
        direction=Direction.NEUTRAL,
        confidence=0.2,
        evidence=evidence,
        risk_flags=tuple(risk_flags),
    )


__all__ = ["CHOP_RANGE_PCT", "TREND_STRONG", "TREND_WEAK", "classify"]
