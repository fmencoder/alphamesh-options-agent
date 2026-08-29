"""Bear agent: argues the strongest defensible bearish case from the evidence."""

from __future__ import annotations

from typing import Any

from alphamesh.agents.bull_agent import _clamp, heuristic_conviction
from alphamesh.agents.evidence import render_evidence
from alphamesh.intelligence.reasoning import (
    LLMUnavailableError,
    MalformedAIOutputError,
    ReasoningProvider,
)
from alphamesh.models.domain import AIArgument, Direction

SYSTEM = (
    "You are the BEAR analyst on an options trading desk. You receive only "
    "structured intraday market evidence for one US index ETF. Argue the "
    "strongest honest bearish case that the evidence supports. Do not invent "
    "data. Do not mention position size, capital, or risk limits - you do not "
    "have that information and it is not your job.\n\n"
    "Respond with JSON only, no prose, in exactly this shape:\n"
    '{"thesis": "<=400 chars", "key_points": ["...", "..."], '
    '"conviction": 0.0-1.0}'
)


def heuristic_bear_argument(evidence: dict[str, Any]) -> AIArgument:
    """Deterministic mirror of the bull fallback."""
    features = evidence.get("features", {})
    trend = float(features.get("trend_strength", 0.0))
    ret15 = float(features.get("ret_15m", 0.0))
    vwap = float(features.get("vwap_deviation", 0.0))
    realized = float(features.get("realized_vol", 0.0))

    points: list[str] = []
    if trend < 0:
        points.append(f"Least-squares trend is negative at {trend:.3f}.")
    if ret15 < 0:
        points.append(f"15-minute return is {ret15 * 100:.2f}%.")
    if vwap < 0:
        points.append(f"Price is {abs(vwap) * 100:.2f}% below session VWAP.")
    if realized > 0.25:
        points.append(f"Realised volatility of {realized:.2f} raises downside tail risk.")
    if not points:
        points.append("No supportive bearish evidence in the current feature set.")

    conviction = heuristic_conviction(evidence, bullish=False)
    return AIArgument(
        role="bear",
        stance=Direction.BEARISH,
        thesis=(
            f"Bearish reversal or breakdown case for {evidence.get('symbol', '?')} "
            f"in a {evidence.get('regime', 'UNKNOWN')} regime."
        ),
        key_points=tuple(points),
        conviction=conviction,
        provider="heuristic",
    )


class BearAgent:
    """Produces one ``AIArgument``. Never fails: falls back to heuristics."""

    role = "bear"

    def __init__(self, provider: ReasoningProvider) -> None:
        self.provider = provider

    def argue(self, evidence: dict[str, Any]) -> AIArgument:
        if not self.provider.available():
            return heuristic_bear_argument(evidence)
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
                stance=Direction.BEARISH,
                thesis=thesis[:400],
                key_points=tuple(str(p)[:200] for p in points[:6]),
                conviction=_clamp(raw.get("conviction", 0.0)),
                provider=self.provider.name,
            )
        except (LLMUnavailableError, MalformedAIOutputError, ValueError, TypeError):
            return heuristic_bear_argument(evidence)


__all__ = ["SYSTEM", "BearAgent", "heuristic_bear_argument"]
