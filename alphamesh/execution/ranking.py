"""The Global Opportunity Ranker.

Capital is scarce: the risk caps admit far fewer positions than the scan
produces qualified candidates for. Before this module existed, the order in
which candidates were offered capital was the scout's sort order, which is
``quant_score`` alone -- so the first symbol whose quant score happened to be
highest consumed the last open slot regardless of how the actual spread priced,
how tight its market was, or how much correlated exposure it piled on.

This module decides PRIORITY ONLY. It is a pure, deterministic function of data
the canonical path has already produced, and it has no authority whatsoever:

* it cannot approve a trade -- only :class:`~alphamesh.risk.governor.RiskGovernor`
  can, and every ranked candidate still runs the full governor,
* it cannot size a trade,
* it cannot reject a candidate. A score of exactly 0.0 still gets offered
  capital, last. Concentration, liquidity and payoff *rejection* remain where
  they already live.

Ranking a candidate first can therefore never turn a rejection into an
approval; it can only change which rejection happens first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from alphamesh.config import RiskLimits
from alphamesh.models.domain import (
    Direction,
    OptionQuote,
    Regime,
    RegimeAssessment,
    SpreadStructure,
    Strategy,
    TradeDecision,
)
from alphamesh.risk.portfolio import PortfolioState

# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

NEUTRAL_SCORE = 1.0
"""Value recorded for a component that is deliberately not modelled yet.

Both such components carry weight 0.0, so recording them as 1.0 documents
"no opinion" in the journal without adding a constant to every total. Wiring
one on later is a weight change, not a structural one.
"""

PAYOFF_TARGET_RATIO = 2.0
"""Reward-to-risk at which payoff efficiency earns full marks.

A vertical debit spread's reward:risk is ``(width - debit) / debit``. The
configured ``max_debit_to_width_ratio`` already refuses anything worse than a
fixed fraction of width, so this constant sets the top of the band rather than
its floor: paying a third of width for a spread (2:1) scores 1.0, paying half
(1:1) scores 0.5. Capped, so an almost-free spread cannot dominate the total.
"""

UNSTABLE_REGIME_MULTIPLIER = 0.25
"""Penalty applied to regime alignment when the regime is UNSTABLE or UNKNOWN.

``exits.evaluate_exit`` rule 6 treats an UNSTABLE regime as invalidating an
open thesis. Prioritising fresh capital into a regime whose own exit rule
would immediately want out is incoherent, so such candidates rank last without
being refused.
"""

SCORE_PRECISION = 6
"""Decimal places every component and total is rounded to.

Rounding at construction makes the stored score and the sort key the same
number, so ties are genuine ties and break on the documented keys rather than
on a floating-point artefact in the last bit.
"""

WEIGHTS: dict[str, float] = {
    # The only component measured directly from market data by machinery that
    # predates this module and is unit-tested on its own. It keeps the largest
    # single weight, but no longer a majority -- previously it was effectively
    # the entire allocation decision.
    "quant": 0.30,
    # The judge's calibrated conviction. Second, but deliberately below the
    # deterministic term: elsewhere in this system AI confidence may only
    # select between pre-configured caps, never set one.
    "judge": 0.20,
    # Cheap, robust directional agreement between the regime read and the
    # strategy's thesis.
    "regime": 0.15,
    # Structural properties of the actual spread that was constructed. They
    # reorder near-ties without overturning a strong signal.
    "payoff": 0.15,
    "liquidity": 0.12,
    # Bounded and deliberately smallest. Diversification informs priority; the
    # governor's correlation caps do the refusing.
    "diversification": 0.08,
    # Not modelled in v1. See NEUTRAL_SCORE.
    "execution_quality": 0.0,
    "learned_prior": 0.0,
    "persistence": 0.0,
}

if not math.isclose(sum(WEIGHTS.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
    raise RuntimeError(f"opportunity weights must sum to 1.0, got {sum(WEIGHTS.values())}")


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def _clamp(value: float) -> float:
    """Force a component into [0, 1]. A non-finite value scores zero.

    Every component funnels through here, which is what makes the "score is
    always in [0, 1]" invariant structural rather than a property of each
    formula being individually careful.
    """
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _fraction_below(value: float, cap: float) -> float:
    """Score a "smaller is better" quantity against its canonical rejection cap.

    Returns 1.0 at zero, 0.0 at or beyond ``cap``, linear in between. Anchoring
    to the limit the governor already enforces avoids inventing a second,
    divergent notion of what "good" means.
    """
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    if not math.isfinite(cap) or cap <= 0.0:
        return 0.0
    return _clamp(1.0 - value / cap)


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
#
# Every component maps to [0, 1], is monotonic in the direction that makes
# economic sense, is a pure function of its inputs, and returns 0.0 when an
# input it needs is missing or nonsensical. That last rule is uniform on
# purpose: a missing input can never raise a score.


def quant_component(decision: TradeDecision) -> float:
    """Quant conviction. Higher score, higher priority."""
    return _clamp(decision.quant_score)


def judge_component(decision: TradeDecision) -> float:
    """Judge confidence. Higher confidence, higher priority."""
    return _clamp(decision.confidence)


def regime_component(strategy: Strategy, regime: RegimeAssessment | None) -> float:
    """Agreement between the regime read and the strategy's directional thesis.

    Aligned beats neutral beats opposed; within each, a more confident read
    beats a less confident one. An UNSTABLE or UNKNOWN regime is penalised
    hard, whatever its direction.
    """
    if regime is None:
        return 0.0
    wanted = (
        Direction.BULLISH if strategy is Strategy.BULL_CALL_SPREAD else Direction.BEARISH
    )
    if regime.direction is wanted:
        base = 1.0
    elif regime.direction is Direction.NEUTRAL:
        base = 0.5
    else:
        base = 0.0
    score = base * (0.5 + 0.5 * _clamp(regime.confidence))
    if regime.regime in (Regime.UNSTABLE, Regime.UNKNOWN):
        score *= UNSTABLE_REGIME_MULTIPLIER
    return _clamp(score)


def payoff_component(spread: SpreadStructure) -> float:
    """Reward-to-risk of the constructed spread, capped.

    For a vertical debit spread the defined risk IS the debit and the maximum
    profit IS ``width - debit``, both exact. A debit at or above the width has
    no bounded payoff worth taking and scores zero rather than dividing by a
    negative.
    """
    debit = spread.limit_price_cents
    width = spread.strike_width_cents
    if debit <= 0 or width <= 0 or debit >= width:
        return 0.0
    ratio = (width - debit) / debit
    return _clamp(ratio / PAYOFF_TARGET_RATIO)


def _quote_quality(quote: OptionQuote | None, limits: RiskLimits) -> float:
    """Executable-market quality for one leg, against the canonical caps."""
    if quote is None:
        return 0.0
    if quote.bid <= 0.0 or quote.ask <= 0.0 or quote.mid <= 0.0:
        return 0.0
    if quote.ask < quote.bid:  # pragma: no cover - the model rejects these
        return 0.0
    relative = _fraction_below(quote.relative_spread, limits.max_relative_bid_ask_spread)
    absolute = _fraction_below(quote.absolute_spread, limits.max_absolute_bid_ask_spread)
    # The worse of the two readings. A leg that is tight in percentage terms
    # but wide in dollars is not cheap to trade.
    return min(relative, absolute)


def liquidity_component(spread: SpreadStructure, limits: RiskLimits) -> float:
    """Executable-market quality across BOTH legs.

    The worse leg governs: a spread is only as fillable as its harder side.
    This is ranking information only -- ``risk.liquidity`` still rejects, and
    the governor still re-checks both legs at approval time.
    """
    return _clamp(
        min(
            _quote_quality(spread.long_leg.contract.quote, limits),
            _quote_quality(spread.short_leg.contract.quote, limits),
        )
    )


def diversification_component(
    symbol: str, portfolio: PortfolioState, limits: RiskLimits
) -> float:
    """How little correlated concentration this candidate would add.

    Built entirely on the existing correlation-group architecture: the same
    groups, the same two caps the governor enforces. A candidate in an empty
    group scores 1.0; one in a group already at either cap scores 0.0 -- which
    deprioritises it, and nothing more. The governor decides whether a full
    group is a refusal.
    """
    group = limits.group_for(symbol)
    if group is None:
        # Outside every correlated bucket, so it adds to no group cap.
        return 1.0

    count_cap = limits.max_positions_per_correlation_group
    risk_cap_cents = round(limits.max_defined_risk_per_correlation_group * 100)
    if count_cap <= 0 or risk_cap_cents <= 0:
        return 0.0

    count_used = len(portfolio.positions_in_group(limits, group)) / count_cap
    risk_used = portfolio.defined_risk_in_group_cents(limits, group) / risk_cap_cents
    return _clamp(1.0 - max(count_used, risk_used))


# --------------------------------------------------------------------------- #
# Score
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScoreBreakdown:
    """Every component that produced a total, kept for the journal.

    Stored rounded to ``SCORE_PRECISION`` so the audit record and the sort key
    are the same numbers.
    """

    quant: float
    judge: float
    regime: float
    payoff: float
    liquidity: float
    diversification: float
    execution_quality: float
    learned_prior: float
    persistence: float | None
    total: float

    def as_dict(self) -> dict[str, float | None]:
        return {
            "quant_score": self.quant,
            "judge_confidence": self.judge,
            "regime_score": self.regime,
            "payoff_score": self.payoff,
            "liquidity_score": self.liquidity,
            "diversification_score": self.diversification,
            "execution_quality_score": self.execution_quality,
            "learned_prior_score": self.learned_prior,
            "persistence_score": self.persistence,
            "total_opportunity_score": self.total,
        }


def score_opportunity(
    decision: TradeDecision,
    spread: SpreadStructure,
    regime: RegimeAssessment | None,
    portfolio: PortfolioState,
    limits: RiskLimits,
) -> ScoreBreakdown:
    """Score one qualified candidate in [0, 1]. Pure and deterministic.

    ``portfolio`` is the pre-allocation snapshot for the whole cycle, so every
    candidate in a cycle is ranked against one consistent view of exposure.
    Positions opened later in the same cycle change what the governor allows,
    not the order candidates were offered capital in.
    """
    quant = quant_component(decision)
    judge = judge_component(decision)
    regime_score = regime_component(spread.strategy, regime)
    payoff = payoff_component(spread)
    liquidity = liquidity_component(spread, limits)
    diversification = diversification_component(decision.symbol, portfolio, limits)

    total = (
        WEIGHTS["quant"] * quant
        + WEIGHTS["judge"] * judge
        + WEIGHTS["regime"] * regime_score
        + WEIGHTS["payoff"] * payoff
        + WEIGHTS["liquidity"] * liquidity
        + WEIGHTS["diversification"] * diversification
        # Weight 0.0 in v1. Present so the arithmetic matches the documented
        # weight table rather than silently omitting two of its rows.
        + WEIGHTS["execution_quality"] * NEUTRAL_SCORE
        + WEIGHTS["learned_prior"] * NEUTRAL_SCORE
    )

    r = SCORE_PRECISION
    return ScoreBreakdown(
        quant=round(quant, r),
        judge=round(judge, r),
        regime=round(regime_score, r),
        payoff=round(payoff, r),
        liquidity=round(liquidity, r),
        diversification=round(diversification, r),
        execution_quality=NEUTRAL_SCORE,
        learned_prior=NEUTRAL_SCORE,
        # Signal persistence would need a second read path over persisted
        # decisions on every cycle. Left unmodelled rather than half-built.
        persistence=None,
        total=round(_clamp(total), r),
    )


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RankedCandidate:
    """A qualified candidate and its score, before any allocation is attempted.

    Holding one of these implies nothing has been reserved, submitted or
    approved. It is the constructed spread plus the reasons it might deserve
    capital first.
    """

    decision: TradeDecision
    spread: SpreadStructure
    regime: RegimeAssessment | None
    score: ScoreBreakdown

    @property
    def sort_key(self) -> tuple[float, float, float, str, str]:
        """Total, then quant, then judge, then a total order on identity.

        The last two keys mean the result never depends on incidental list
        order: two candidates that tie on every score still sort the same way
        on every run.
        """
        return (
            -self.score.total,
            -self.score.quant,
            -self.score.judge,
            self.decision.symbol,
            self.decision.strategy.value,
        )


def rank_candidates(candidates: Sequence[RankedCandidate]) -> list[RankedCandidate]:
    """Order candidates by descending opportunity, deterministically.

    Ranking only reorders: nothing is added, nothing is dropped.
    """
    return sorted(candidates, key=lambda c: c.sort_key)


__all__ = [
    "NEUTRAL_SCORE",
    "PAYOFF_TARGET_RATIO",
    "SCORE_PRECISION",
    "UNSTABLE_REGIME_MULTIPLIER",
    "WEIGHTS",
    "RankedCandidate",
    "ScoreBreakdown",
    "diversification_component",
    "judge_component",
    "liquidity_component",
    "payoff_component",
    "quant_component",
    "rank_candidates",
    "regime_component",
    "score_opportunity",
]
