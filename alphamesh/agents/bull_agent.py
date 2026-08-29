"""Bull agent: argues the strongest defensible bullish case from the evidence."""

from __future__ import annotations

from typing import Any

from alphamesh.agents.evidence import render_evidence
from alphamesh.intelligence.reasoning import (
    LLMUnavailableError,
    MalformedAIOutputError,
    ReasoningProvider,
)
from alphamesh.models.domain import AIArgument, Direction

SYSTEM = (
    "You are the BULL analyst on an options trading desk. You receive only "
    "structured intraday market evidence for one US index ETF. Argue the "
    "strongest honest bullish case that the evidence supports. Do not invent "
    "data. Do not mention position size, capital, or risk limits - you do not "
    "have that information and it is not your job.\n\n"
    "Respond with JSON only, no prose, in exactly this shape:\n"
    '{"thesis": "<=400 chars", "key_points": ["...", "..."], '
    '"conviction": 0.0-1.0}'
)


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return lo


DIRECTIONAL_FEATURES: tuple[str, ...] = (
    "ret_5m",
    "ret_15m",
    "trend_strength",
    "vwap_deviation",
)

# Conviction is built from the normalised quant score rather than from raw
# returns, so it lands on the same 0-1 scale the judge's confidence floor is
# expressed in. Raw index-ETF returns are far too small to use directly.
REGIME_ALIGNMENT_FACTOR: dict[str, float] = {
    "aligned": 1.0,
    "neutral": 0.62,
    "opposed": 0.25,
}
CONVICTION_GAIN = 1.6


def directional_support(evidence: dict[str, Any], bullish: bool) -> float:
    """Fraction of the directional features that agree with one side."""
    features = evidence.get("features", {})
    votes = 0
    counted = 0
    for name in DIRECTIONAL_FEATURES:
        if name not in features:
            continue
        counted += 1
        value = float(features[name])
        if (value > 0) == bullish and value != 0:
            votes += 1
    return votes / counted if counted else 0.0


def regime_factor(evidence: dict[str, Any], bullish: bool) -> float:
    direction = str(evidence.get("regime_direction", "NEUTRAL"))
    wanted = "BULLISH" if bullish else "BEARISH"
    if direction == wanted:
        return REGIME_ALIGNMENT_FACTOR["aligned"]
    if direction == "NEUTRAL":
        return REGIME_ALIGNMENT_FACTOR["neutral"]
    return REGIME_ALIGNMENT_FACTOR["opposed"]


def heuristic_conviction(evidence: dict[str, Any], bullish: bool) -> float:
    """Deterministic conviction on the same scale as an LLM's confidence.

    Three factors multiply: how strong the opportunity is at all (quant score),
    how many directional features agree, and whether the classified regime backs
    that direction. A strong score with split features stays low, which is the
    behaviour we want.
    """
    base = float(evidence.get("quant_score", 0.0))
    support = directional_support(evidence, bullish)
    return _clamp(base * support * regime_factor(evidence, bullish) * CONVICTION_GAIN)


def heuristic_bull_argument(evidence: dict[str, Any]) -> AIArgument:
    """Deterministic fallback used when no LLM is reachable or usable.

    Conviction is read straight off the quantitative evidence, so the council
    still produces a comparable bull/bear pair with no model in the loop.
    """
    features = evidence.get("features", {})
    trend = float(features.get("trend_strength", 0.0))
    ret15 = float(features.get("ret_15m", 0.0))
    vwap = float(features.get("vwap_deviation", 0.0))
    accel = float(features.get("volume_acceleration", 1.0))

    points: list[str] = []
    if trend > 0:
        points.append(f"Least-squares trend is positive at {trend:.3f}.")
    if ret15 > 0:
        points.append(f"15-minute return is +{ret15 * 100:.2f}%.")
    if vwap > 0:
        points.append(f"Price is {vwap * 100:.2f}% above session VWAP.")
    if accel > 1.2:
        points.append(f"Volume running {accel:.2f}x baseline supports continuation.")
    if not points:
        points.append("No supportive bullish evidence in the current feature set.")

    conviction = heuristic_conviction(evidence, bullish=True)
    return AIArgument(
        role="bull",
        stance=Direction.BULLISH,
        thesis=(
            f"Bullish continuation case for {evidence.get('symbol', '?')} "
            f"in a {evidence.get('regime', 'UNKNOWN')} regime."
        ),
        key_points=tuple(points),
        conviction=conviction,
        provider="heuristic",
    )


class BullAgent:
    """Produces one ``AIArgument``. Never fails: falls back to heuristics."""

    role = "bull"

    def __init__(self, provider: ReasoningProvider) -> None:
        self.provider = provider

    def argue(self, evidence: dict[str, Any]) -> AIArgument:
        if not self.provider.available():
            return heuristic_bull_argument(evidence)
        try:
            raw = self.provider.complete_json(
                system=SYSTEM, user=render_evidence(evidence), max_tokens=700
            )
            points = raw.get("key_points") or []
            if not isinstance(points, list) or not points:
                raise MalformedAIOutputError("key_points must be a non-empty list")
            thesis = str(raw.get("thesis", "")).strip()
            if not thesis:
                raise MalformedAIOutputError("thesis is required")
            return AIArgument(
                role=self.role,
                stance=Direction.BULLISH,
                thesis=thesis[:400],
                key_points=tuple(str(p)[:200] for p in points[:6]),
                conviction=_clamp(raw.get("conviction", 0.0)),
                provider=self.provider.name,
            )
        except (LLMUnavailableError, MalformedAIOutputError, ValueError, TypeError):
            return heuristic_bull_argument(evidence)


__all__ = [
    "CONVICTION_GAIN",
    "DIRECTIONAL_FEATURES",
    "REGIME_ALIGNMENT_FACTOR",
    "SYSTEM",
    "BullAgent",
    "directional_support",
    "heuristic_bull_argument",
    "heuristic_conviction",
    "regime_factor",
]
