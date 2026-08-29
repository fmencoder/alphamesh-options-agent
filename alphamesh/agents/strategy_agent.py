"""Strategy agent: runs the reasoning council behind the quantitative gate and
emits one auditable ``TradeDecision``.

Order of authority, highest first:

1. The quantitative gate. Below threshold, the council is never invoked.
2. The regime. UNKNOWN/UNSTABLE force NO_TRADE regardless of the council.
3. The judge, constrained to three strategies.
4. The configured minimum judge confidence.

NO_TRADE is a first-class outcome at every step and always carries a reason.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime

from alphamesh.agents.bear_agent import BearAgent
from alphamesh.agents.bull_agent import BullAgent
from alphamesh.agents.evidence import build_evidence
from alphamesh.agents.judge_agent import JudgeAgent
from alphamesh.config import AppConfig
from alphamesh.intelligence.reasoning import ReasoningProvider
from alphamesh.models.domain import (
    AIArgument,
    Direction,
    JudgeVerdict,
    QuantSignal,
    ReasonCode,
    Regime,
    RegimeAssessment,
    Strategy,
    TradeDecision,
)

log = logging.getLogger(__name__)


def make_decision_id(symbol: str, as_of: datetime, quant_score: float) -> str:
    """Stable identifier for one evaluation of one symbol at one instant.

    Deterministic on purpose: re-evaluating the same bar window yields the same
    id, which is what lets the journal detect duplicate signals.
    """
    payload = f"{symbol}|{as_of.isoformat()}|{quant_score:.6f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CouncilResult:
    decision: TradeDecision
    bull: AIArgument | None
    bear: AIArgument | None
    verdict: JudgeVerdict | None


class StrategyAgent:
    def __init__(self, config: AppConfig, provider: ReasoningProvider) -> None:
        self.config = config
        self.provider = provider
        self.bull = BullAgent(provider)
        self.bear = BearAgent(provider)
        self.judge = JudgeAgent(provider)

    def _no_trade(
        self,
        signal: QuantSignal,
        regime: RegimeAssessment,
        reason: str,
        codes: tuple[ReasonCode, ...],
        bull_score: float = 0.0,
        bear_score: float = 0.0,
        provider: str = "none",
    ) -> TradeDecision:
        return TradeDecision(
            decision_id=make_decision_id(signal.symbol, signal.as_of, signal.quant_score),
            symbol=signal.symbol,
            timestamp=signal.as_of,
            regime=regime.regime,
            direction=regime.direction,
            strategy=Strategy.NO_TRADE,
            confidence=0.0,
            bull_score=bull_score,
            bear_score=bear_score,
            quant_score=signal.quant_score,
            reason_codes=codes,
            no_trade_reason=reason,
            ai_provider=provider,
        )

    def decide(self, signal: QuantSignal, regime: RegimeAssessment) -> CouncilResult:
        # 1. Quantitative gate: the AI is not consulted below threshold.
        if not signal.passes_gate:
            codes = signal.reason_codes or (ReasonCode.QUANT_SCORE_BELOW_THRESHOLD,)
            return CouncilResult(
                self._no_trade(
                    signal,
                    regime,
                    (
                        f"Quant score {signal.quant_score:.3f} below threshold "
                        f"{self.config.strategies.quant_score_threshold:.3f} "
                        "or directional bias unresolved; AI council not invoked."
                    ),
                    codes,
                ),
                None,
                None,
                None,
            )

        # 2. Regime veto, applied before any model cost is incurred.
        if regime.favors_no_trade:
            code = (
                ReasonCode.REGIME_UNSTABLE
                if regime.regime is Regime.UNSTABLE
                else ReasonCode.REGIME_UNKNOWN
            )
            return CouncilResult(
                self._no_trade(
                    signal,
                    regime,
                    f"Regime {regime.regime} strongly favours standing aside.",
                    (code,),
                ),
                None,
                None,
                None,
            )

        # 3. The council debates, then the judge rules.
        evidence = build_evidence(signal, regime)
        bull = self.bull.argue(evidence)
        bear = self.bear.argue(evidence)
        verdict = self.judge.judge(evidence, bull, bear, regime)

        if verdict.strategy is Strategy.NO_TRADE:
            return CouncilResult(
                self._no_trade(
                    signal,
                    regime,
                    verdict.rationale,
                    verdict.reason_codes or (ReasonCode.DIRECTION_CONFLICT,),
                    bull_score=verdict.bull_score,
                    bear_score=verdict.bear_score,
                    provider=verdict.provider,
                ),
                bull,
                bear,
                verdict,
            )

        # 4. Confidence floor.
        if verdict.confidence < self.config.strategies.min_judge_confidence:
            return CouncilResult(
                self._no_trade(
                    signal,
                    regime,
                    (
                        f"Judge confidence {verdict.confidence:.3f} below floor "
                        f"{self.config.strategies.min_judge_confidence:.3f}."
                    ),
                    (ReasonCode.LOW_JUDGE_CONFIDENCE,),
                    bull_score=verdict.bull_score,
                    bear_score=verdict.bear_score,
                    provider=verdict.provider,
                ),
                bull,
                bear,
                verdict,
            )

        direction = (
            Direction.BULLISH
            if verdict.strategy is Strategy.BULL_CALL_SPREAD
            else Direction.BEARISH
        )
        decision = TradeDecision(
            decision_id=make_decision_id(signal.symbol, signal.as_of, signal.quant_score),
            symbol=signal.symbol,
            timestamp=signal.as_of,
            regime=regime.regime,
            direction=direction,
            strategy=verdict.strategy,
            confidence=verdict.confidence,
            bull_score=verdict.bull_score,
            bear_score=verdict.bear_score,
            quant_score=signal.quant_score,
            reason_codes=verdict.reason_codes,
            no_trade_reason=None,
            ai_provider=verdict.provider,
        )
        return CouncilResult(decision, bull, bear, verdict)


__all__ = ["CouncilResult", "StrategyAgent", "make_decision_id"]
