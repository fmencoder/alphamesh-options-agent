"""Judge agent: the only AI role whose output steers the pipeline.

Its authority is deliberately tiny. It may return exactly one of three
strategies and a confidence. It cannot pick contracts, quantities, prices, or
anything touching the account. Every field is re-validated here; anything
outside the allowed set is downgraded to NO_TRADE with a machine-readable
reason code, never silently coerced into a trade.
"""

from __future__ import annotations

from typing import Any

from alphamesh.agents.evidence import render_evidence
from alphamesh.intelligence.reasoning import (
    LLMUnavailableError,
    MalformedAIOutputError,
    ReasoningProvider,
)
from alphamesh.models.domain import (
    AIArgument,
    Direction,
    JudgeVerdict,
    ReasonCode,
    Regime,
    RegimeAssessment,
    Strategy,
)

ALLOWED_STRATEGY_NAMES: frozenset[str] = frozenset(
    {Strategy.NO_TRADE.value, Strategy.BULL_CALL_SPREAD.value, Strategy.BEAR_PUT_SPREAD.value}
)

SYSTEM = (
    "You are the JUDGE on an options trading desk. You receive structured "
    "market evidence plus a bull argument and a bear argument. Weigh them and "
    "choose ONE strategy.\n\n"
    "You may ONLY choose from: NO_TRADE, BULL_CALL_SPREAD, BEAR_PUT_SPREAD.\n"
    "NO_TRADE is a valid and frequently correct answer. Choosing it is not a "
    "failure.\n"
    "You do NOT choose contracts, strikes, expirations, quantities, prices, or "
    "any amount of capital. Deterministic code does all of that.\n\n"
    "Respond with JSON only, no prose, in exactly this shape:\n"
    '{"strategy": "NO_TRADE|BULL_CALL_SPREAD|BEAR_PUT_SPREAD", '
    '"confidence": 0.0-1.0, "bull_score": 0.0-1.0, "bear_score": 0.0-1.0, '
    '"rationale": "<=400 chars"}'
)


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def render_judge_prompt(
    evidence: dict[str, Any], bull: AIArgument, bear: AIArgument
) -> str:
    return (
        f"MARKET EVIDENCE:\n{render_evidence(evidence)}\n\n"
        f"BULL ARGUMENT (conviction {bull.conviction:.2f}):\n"
        f"{bull.thesis}\n- " + "\n- ".join(bull.key_points) + "\n\n"
        f"BEAR ARGUMENT (conviction {bear.conviction:.2f}):\n"
        f"{bear.thesis}\n- " + "\n- ".join(bear.key_points) + "\n\n"
        "Return your JSON verdict now."
    )


def heuristic_verdict(
    evidence: dict[str, Any],
    bull: AIArgument,
    bear: AIArgument,
    regime: RegimeAssessment,
    extra_codes: tuple[ReasonCode, ...] = (),
) -> JudgeVerdict:
    """Deterministic judge used when no LLM is available or its output is unusable.

    The margin between the two convictions has to be decisive; a near-tie
    resolves to NO_TRADE rather than to a coin flip.
    """
    bull_score = _clamp(bull.conviction)
    bear_score = _clamp(bear.conviction)
    margin = abs(bull_score - bear_score)
    codes = list(extra_codes)

    if regime.favors_no_trade:
        codes.append(
            ReasonCode.REGIME_UNSTABLE
            if regime.regime is Regime.UNSTABLE
            else ReasonCode.REGIME_UNKNOWN
        )
        return JudgeVerdict(
            strategy=Strategy.NO_TRADE,
            confidence=0.0,
            rationale=f"Regime {regime.regime} does not support a directional debit spread.",
            bull_score=bull_score,
            bear_score=bear_score,
            provider="heuristic",
            reason_codes=tuple(codes),
        )

    if margin < 0.10:
        codes.append(ReasonCode.DIRECTION_CONFLICT)
        return JudgeVerdict(
            strategy=Strategy.NO_TRADE,
            confidence=margin,
            rationale=(
                f"Bull {bull_score:.2f} and bear {bear_score:.2f} convictions are "
                "too close to justify directional risk."
            ),
            bull_score=bull_score,
            bear_score=bear_score,
            provider="heuristic",
            reason_codes=tuple(codes),
        )

    if bull_score > bear_score:
        strategy, confidence = Strategy.BULL_CALL_SPREAD, bull_score
        direction = Direction.BULLISH
    else:
        strategy, confidence = Strategy.BEAR_PUT_SPREAD, bear_score
        direction = Direction.BEARISH

    # A directional call that fights the classified regime is not taken.
    if regime.direction is not Direction.NEUTRAL and regime.direction is not direction:
        codes.append(ReasonCode.DIRECTION_CONFLICT)
        return JudgeVerdict(
            strategy=Strategy.NO_TRADE,
            confidence=0.0,
            rationale=(
                f"Council leans {direction} but the regime is {regime.direction}; "
                "standing aside."
            ),
            bull_score=bull_score,
            bear_score=bear_score,
            provider="heuristic",
            reason_codes=tuple(codes),
        )

    return JudgeVerdict(
        strategy=strategy,
        confidence=confidence,
        rationale=(
            f"{direction} case wins on conviction margin {margin:.2f} inside a "
            f"{regime.regime} regime."
        ),
        bull_score=bull_score,
        bear_score=bear_score,
        provider="heuristic",
        reason_codes=tuple(codes),
    )


class JudgeAgent:
    """Validates and constrains the judge's output. Fails to NO_TRADE."""

    role = "judge"

    def __init__(self, provider: ReasoningProvider) -> None:
        self.provider = provider

    def judge(
        self,
        evidence: dict[str, Any],
        bull: AIArgument,
        bear: AIArgument,
        regime: RegimeAssessment,
    ) -> JudgeVerdict:
        if not self.provider.available():
            return heuristic_verdict(
                evidence, bull, bear, regime, (ReasonCode.AI_UNAVAILABLE,)
            )

        try:
            raw = self.provider.complete_json(
                system=SYSTEM,
                user=render_judge_prompt(evidence, bull, bear),
                max_tokens=600,
            )
        except (LLMUnavailableError, MalformedAIOutputError):
            return heuristic_verdict(
                evidence, bull, bear, regime, (ReasonCode.AI_UNAVAILABLE,)
            )

        name = str(raw.get("strategy", "")).strip().upper()
        if not name:
            return heuristic_verdict(
                evidence, bull, bear, regime, (ReasonCode.AI_MALFORMED_OUTPUT,)
            )
        if name not in ALLOWED_STRATEGY_NAMES:
            # The model asked for something outside its authority. Refuse it
            # outright rather than mapping it onto a strategy we do allow.
            return JudgeVerdict(
                strategy=Strategy.NO_TRADE,
                confidence=0.0,
                rationale=f"Judge proposed unsupported strategy {name!r}; refused.",
                bull_score=_clamp(raw.get("bull_score", bull.conviction)),
                bear_score=_clamp(raw.get("bear_score", bear.conviction)),
                provider=self.provider.name,
                reason_codes=(ReasonCode.AI_UNSUPPORTED_STRATEGY,),
            )

        strategy = Strategy(name)
        confidence = _clamp(raw.get("confidence", 0.0))
        rationale = str(raw.get("rationale", "")).strip()[:400]
        if not rationale:
            rationale = f"Judge selected {strategy} without stated rationale."

        codes: tuple[ReasonCode, ...] = ()
        if strategy is Strategy.NO_TRADE and confidence <= 0:
            confidence = 0.0

        # A directional verdict against a NO_TRADE regime is overridden here,
        # not left for downstream code to notice.
        if strategy is not Strategy.NO_TRADE and regime.favors_no_trade:
            return JudgeVerdict(
                strategy=Strategy.NO_TRADE,
                confidence=0.0,
                rationale=(
                    f"Judge chose {strategy} but regime {regime.regime} forbids "
                    "directional risk; overridden."
                ),
                bull_score=_clamp(raw.get("bull_score", bull.conviction)),
                bear_score=_clamp(raw.get("bear_score", bear.conviction)),
                provider=self.provider.name,
                reason_codes=(
                    ReasonCode.REGIME_UNSTABLE
                    if regime.regime is Regime.UNSTABLE
                    else ReasonCode.REGIME_UNKNOWN,
                ),
            )

        return JudgeVerdict(
            strategy=strategy,
            confidence=confidence,
            rationale=rationale,
            bull_score=_clamp(raw.get("bull_score", bull.conviction)),
            bear_score=_clamp(raw.get("bear_score", bear.conviction)),
            provider=self.provider.name,
            reason_codes=codes,
        )


__all__ = [
    "ALLOWED_STRATEGY_NAMES",
    "SYSTEM",
    "JudgeAgent",
    "heuristic_verdict",
    "render_judge_prompt",
]
