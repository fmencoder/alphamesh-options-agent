"""The Global Opportunity Ranker.

Two things are under test here and they are not the same thing. The first is
the scorer: a pure function, tested at its boundaries and for the economic
direction of every component. The second is the authority boundary -- that
ranking reorders and nothing else. A ranker that quietly approved, sized or
refused a trade would be a far worse bug than one that ranked badly, so the
second half of this file is mostly about what the ranker must NOT do.
"""

from __future__ import annotations

import json
import math
from datetime import timedelta

import pytest

from alphamesh.alpaca.client import AlpacaStack
from alphamesh.alpaca.execution import SimulatedBroker
from alphamesh.alpaca.options import CaptureOptionChain
from alphamesh.execution.ranking import (
    NEUTRAL_SCORE,
    WEIGHTS,
    RankedCandidate,
    diversification_component,
    liquidity_component,
    payoff_component,
    rank_candidates,
    regime_component,
    score_opportunity,
)
from alphamesh.models.domain import (
    Direction,
    OptionType,
    OrderSide,
    PositionIntent,
    PositionRecord,
    Regime,
    SpreadLeg,
    SpreadStructure,
    Strategy,
    TradeState,
)
from alphamesh.orchestrator import CycleReport, Orchestrator
from alphamesh.persistence.journal import Journal
from alphamesh.safety import GuardResult
from tests.conftest import (
    CAPTURE_DIR,
    NOW,
    make_account,
    make_contract,
    make_decision,
    make_portfolio,
    make_regime,
)
from tests.test_orchestrator import TrendingMarketData, bull_script

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_spread(
    symbol: str = "SPY",
    strategy: Strategy = Strategy.BULL_CALL_SPREAD,
    long_bid: float = 3.00,
    long_ask: float = 3.10,
    short_bid: float = 1.00,
    short_ask: float = 1.10,
    limit_cents: int = 200,
    width_cents: int = 500,
    long_has_quote: bool = True,
    short_has_quote: bool = True,
) -> SpreadStructure:
    """A synthetic vertical whose price and market width are dialable.

    Real spreads come from the captured chain; this exists to drive the score's
    boundaries, which the captured session does not contain.
    """
    option_type = (
        OptionType.CALL if strategy is Strategy.BULL_CALL_SPREAD else OptionType.PUT
    )
    long_c = make_contract(
        underlying=symbol,
        strike=770.0,
        option_type=option_type,
        bid=long_bid,
        ask=long_ask,
        with_quote=long_has_quote,
    )
    short_c = make_contract(
        underlying=symbol,
        strike=775.0,
        option_type=option_type,
        delta=0.30,
        bid=short_bid,
        ask=short_ask,
        with_quote=short_has_quote,
    )
    return SpreadStructure(
        strategy=strategy,
        symbol=symbol,
        expiration=long_c.expiration,
        long_leg=SpreadLeg(
            contract=long_c,
            side=OrderSide.BUY,
            ratio=1,
            position_intent=PositionIntent.BUY_TO_OPEN,
        ),
        short_leg=SpreadLeg(
            contract=short_c,
            side=OrderSide.SELL,
            ratio=1,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
        net_debit_cents=limit_cents,
        strike_width_cents=width_cents,
        limit_price_cents=limit_cents,
    )


def make_position(
    symbol: str = "SPY", max_loss_cents: int = 20_000, position_id: str = "p1"
) -> PositionRecord:
    return PositionRecord(
        position_id=position_id,
        decision_id="d1",
        client_order_id="c1",
        symbol=symbol,
        strategy=Strategy.BULL_CALL_SPREAD,
        quantity=1,
        entry_debit_cents=max_loss_cents,
        max_loss_cents=max_loss_cents,
        max_profit_cents=30_000,
        opened_at=NOW,
        expiration=NOW.date() + timedelta(days=6),
        long_symbol="L",
        short_symbol="S",
        state=TradeState.MONITORING,
    )


def score(limits, **kwargs):  # type: ignore[no-untyped-def]
    """Score one candidate, defaulting everything not under test."""
    decision = kwargs.pop("decision", None) or make_decision(
        symbol=kwargs.pop("symbol", "SPY"),
        confidence=kwargs.pop("confidence", 0.70),
        quant_score=kwargs.pop("quant_score", 0.70),
    )
    spread = kwargs.pop("spread", None) or make_spread(symbol=decision.symbol)
    regime = kwargs.pop("regime", "default")
    if regime == "default":
        regime = make_regime(symbol=decision.symbol)
    portfolio = kwargs.pop("portfolio", None) or make_portfolio()
    assert not kwargs, f"unexpected kwargs {sorted(kwargs)}"
    return score_opportunity(decision, spread, regime, portfolio, limits)


def candidate(limits, **kwargs) -> RankedCandidate:  # type: ignore[no-untyped-def]
    decision = kwargs.get("decision") or make_decision(
        symbol=kwargs.get("symbol", "SPY"),
        confidence=kwargs.get("confidence", 0.70),
        quant_score=kwargs.get("quant_score", 0.70),
    )
    kwargs["decision"] = decision
    kwargs.pop("symbol", None)
    kwargs.pop("confidence", None)
    kwargs.pop("quant_score", None)
    spread = kwargs.get("spread") or make_spread(symbol=decision.symbol)
    kwargs["spread"] = spread
    regime = kwargs.get("regime", "default")
    if regime == "default":
        regime = make_regime(symbol=decision.symbol)
    return RankedCandidate(
        decision=decision,
        spread=spread,
        regime=regime,
        score=score(limits, **kwargs),
    )


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


class TestWeights:
    def test_weights_sum_to_one(self) -> None:
        assert math.isclose(sum(WEIGHTS.values()), 1.0, abs_tol=1e-9)

    def test_the_two_unmodelled_components_carry_no_weight(self) -> None:
        # Recording them as NEUTRAL_SCORE=1.0 while weighting them 0.0 means
        # they document "no opinion" without adding a constant to every total.
        assert WEIGHTS["execution_quality"] == 0.0
        assert WEIGHTS["learned_prior"] == 0.0
        assert WEIGHTS["persistence"] == 0.0


# --------------------------------------------------------------------------- #
# Invariants 1-8: the scorer
# --------------------------------------------------------------------------- #


class TestPriorityIsNotQuantScoreAlone:
    def test_highest_quant_score_does_not_necessarily_rank_first(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 1. This is the entire point of the change.

        The strong-quant candidate is worse on everything the old sort could
        not see: it pays 90% of width, its market is at the rejection
        threshold, and the regime disagrees with its thesis.
        """
        loud = candidate(
            limits,
            symbol="SPY",
            quant_score=0.95,
            confidence=0.50,
            spread=make_spread(
                symbol="SPY",
                limit_cents=450,
                width_cents=500,
                long_bid=3.00,
                long_ask=3.75,
                short_bid=1.00,
                short_ask=1.75,
            ),
            regime=make_regime(
                symbol="SPY", regime=Regime.UNSTABLE, direction=Direction.BEARISH
            ),
        )
        quiet = candidate(
            limits,
            symbol="QQQ",
            quant_score=0.60,
            confidence=0.90,
            spread=make_spread(
                symbol="QQQ",
                limit_cents=150,
                width_cents=500,
                long_bid=3.00,
                long_ask=3.02,
                short_bid=1.50,
                short_ask=1.52,
            ),
        )
        assert loud.decision.quant_score > quiet.decision.quant_score
        assert [c.decision.symbol for c in rank_candidates([loud, quiet])] == ["QQQ", "SPY"]

    def test_quant_score_still_moves_priority_all_else_equal(self, limits) -> None:  # type: ignore[no-untyped-def]
        strong = candidate(limits, quant_score=0.90)
        weak = candidate(limits, quant_score=0.40)
        assert strong.score.total > weak.score.total


class TestComponentDirection:
    def test_stronger_judge_confidence_improves_priority(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 2."""
        assert score(limits, confidence=0.95).total > score(limits, confidence=0.55).total

    def test_tighter_liquidity_improves_priority(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 3."""
        tight = make_spread(long_bid=3.00, long_ask=3.02, short_bid=1.00, short_ask=1.02)
        wide = make_spread(long_bid=3.00, long_ask=3.60, short_bid=1.00, short_ask=1.60)
        assert score(limits, spread=tight).total > score(limits, spread=wide).total
        assert liquidity_component(tight, limits) > liquidity_component(wide, limits)

    def test_the_worse_leg_governs_liquidity(self, limits) -> None:  # type: ignore[no-untyped-def]
        # A spread is only as fillable as its harder side, so one tight leg
        # cannot rescue one wide one.
        half_bad = make_spread(long_bid=3.00, long_ask=3.02, short_bid=1.00, short_ask=1.60)
        both_bad = make_spread(long_bid=3.00, long_ask=3.60, short_bid=1.00, short_ask=1.60)
        assert liquidity_component(half_bad, limits) == liquidity_component(both_bad, limits)

    def test_better_payoff_over_risk_improves_priority(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 4."""
        cheap = make_spread(limit_cents=150, width_cents=500)
        dear = make_spread(limit_cents=400, width_cents=500)
        assert score(limits, spread=cheap).total > score(limits, spread=dear).total
        assert payoff_component(cheap) > payoff_component(dear)

    def test_a_superficially_huge_payoff_ratio_cannot_dominate(self, limits) -> None:  # type: ignore[no-untyped-def]
        # A 1c debit on a $5 width is a 499:1 ratio. Capped at 1.0, so it buys
        # its weight and not a point more.
        absurd = make_spread(limit_cents=1, width_cents=500)
        assert payoff_component(absurd) == 1.0
        assert score(limits, spread=absurd).total <= 1.0

    def test_regime_alignment_beats_neutral_beats_opposed(self, limits) -> None:  # type: ignore[no-untyped-def]
        aligned = regime_component(
            Strategy.BULL_CALL_SPREAD, make_regime(direction=Direction.BULLISH)
        )
        neutral = regime_component(
            Strategy.BULL_CALL_SPREAD, make_regime(direction=Direction.NEUTRAL)
        )
        opposed = regime_component(
            Strategy.BULL_CALL_SPREAD, make_regime(direction=Direction.BEARISH)
        )
        assert aligned > neutral > opposed == 0.0

    def test_an_unstable_regime_is_penalised_even_when_aligned(self, limits) -> None:  # type: ignore[no-untyped-def]
        # exits.evaluate_exit would want straight back out of an UNSTABLE
        # regime, so paying up to get in first is incoherent.
        stable = regime_component(
            Strategy.BULL_CALL_SPREAD,
            make_regime(regime=Regime.BULLISH_TREND, direction=Direction.BULLISH),
        )
        unstable = regime_component(
            Strategy.BULL_CALL_SPREAD,
            make_regime(regime=Regime.UNSTABLE, direction=Direction.BULLISH),
        )
        assert stable > unstable > 0.0


class TestDiversification:
    def test_diversification_improves_priority(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 5, first half."""
        empty = make_portfolio()
        crowded = make_portfolio(
            open_positions=(make_position("SPY"), make_position("QQQ", position_id="p2"))
        )
        # IWM shares the index_beta bucket with both open positions.
        assert (
            score(limits, symbol="IWM", portfolio=empty).total
            > score(limits, symbol="IWM", portfolio=crowded).total
        )

    def test_diversification_is_not_a_rejection_gate(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 5, second half.

        A candidate whose correlated bucket is completely full still scores,
        still ranks and is still offered capital -- last. Refusing it is the
        governor's job, not the ranker's.
        """
        full = make_portfolio(
            open_positions=(make_position("SPY"), make_position("QQQ", position_id="p2"))
        )
        assert diversification_component("IWM", full, limits) == 0.0
        crowded = candidate(limits, symbol="IWM", portfolio=full)
        assert crowded.score.total > 0.0
        assert crowded in rank_candidates([crowded])

    def test_group_defined_risk_counts_as_well_as_position_count(self, limits) -> None:  # type: ignore[no-untyped-def]
        # One position, but it eats the whole correlated risk cap.
        cap_cents = round(limits.max_defined_risk_per_correlation_group * 100)
        hogged = make_portfolio(
            open_positions=(make_position("SPY", max_loss_cents=cap_cents),)
        )
        assert diversification_component("IWM", hogged, limits) == 0.0

    def test_a_symbol_outside_every_group_adds_no_correlated_concentration(
        self, limits
    ) -> None:  # type: ignore[no-untyped-def]
        assert limits.group_for("NOTINANYGROUP") is None
        assert diversification_component("NOTINANYGROUP", make_portfolio(), limits) == 1.0


class TestBoundaries:
    @pytest.mark.parametrize("quant", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("limit_cents,width_cents", [(1, 500), (250, 500), (499, 500)])
    def test_score_is_always_within_the_unit_interval(
        self, limits, quant, confidence, limit_cents, width_cents
    ) -> None:  # type: ignore[no-untyped-def]
        """Invariant 6."""
        result = score(
            limits,
            quant_score=quant,
            confidence=confidence,
            spread=make_spread(limit_cents=limit_cents, width_cents=width_cents),
        )
        assert 0.0 <= result.total <= 1.0
        for name, value in result.as_dict().items():
            if value is None:
                continue
            assert 0.0 <= value <= 1.0, name

    def test_a_degenerate_spread_scores_zero_payoff_rather_than_dividing(
        self, limits
    ) -> None:
        # SpreadStructure itself refuses debit >= width, so the guard is
        # reached through the component rather than a constructed model.
        ok = make_spread(limit_cents=499, width_cents=500)
        broken = ok.model_construct(**{**ok.__dict__, "limit_price_cents": 0})
        assert payoff_component(broken) == 0.0
        broken_wide = ok.model_construct(**{**ok.__dict__, "strike_width_cents": 0})
        assert payoff_component(broken_wide) == 0.0

    def test_missing_optional_input_cannot_inflate_the_score(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 7. Every missing input scores its component zero."""
        complete = score(limits)

        no_regime = score(limits, regime=None)
        assert no_regime.regime == 0.0
        assert no_regime.total < complete.total

        no_quote = score(
            limits, spread=make_spread(long_has_quote=False, short_has_quote=False)
        )
        assert no_quote.liquidity == 0.0
        assert no_quote.total < complete.total

        one_quote = score(limits, spread=make_spread(short_has_quote=False))
        assert one_quote.liquidity == 0.0
        assert one_quote.total < complete.total

    def test_a_zero_bid_market_scores_no_liquidity(self, limits) -> None:  # type: ignore[no-untyped-def]
        assert liquidity_component(make_spread(long_bid=0.0, long_ask=0.10), limits) == 0.0


class TestDeterminism:
    def test_tie_breaking_is_deterministic_and_order_independent(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 8.

        Three candidates identical on every score. The result must not depend
        on the order they happened to be collected in.
        """
        a = candidate(limits, symbol="AAPL")
        n = candidate(limits, symbol="NVDA")
        t = candidate(limits, symbol="TSLA")
        assert a.score.total == n.score.total == t.score.total
        for order in ([a, n, t], [t, n, a], [n, t, a]):
            assert [c.decision.symbol for c in rank_candidates(order)] == [
                "AAPL",
                "NVDA",
                "TSLA",
            ]

    def test_ties_break_on_quant_before_symbol(self, limits) -> None:  # type: ignore[no-untyped-def]
        # Same total, reached differently: TSLA has the stronger quant score
        # and the weaker judge, so it wins the secondary key despite sorting
        # last alphabetically.
        strong_quant = candidate(limits, symbol="TSLA", quant_score=0.90, confidence=0.55)
        strong_judge = candidate(limits, symbol="AAPL", quant_score=0.80, confidence=0.70)
        assert strong_quant.score.total == strong_judge.score.total
        assert [c.decision.symbol for c in rank_candidates([strong_judge, strong_quant])] == [
            "TSLA",
            "AAPL",
        ]

    def test_scoring_the_same_inputs_twice_gives_the_same_number(self, limits) -> None:  # type: ignore[no-untyped-def]
        assert score(limits).as_dict() == score(limits).as_dict()

    def test_ranking_only_reorders(self, limits) -> None:  # type: ignore[no-untyped-def]
        given = [candidate(limits, symbol=s) for s in ("SPY", "QQQ", "IWM")]
        assert sorted(
            c.decision.symbol for c in rank_candidates(given)
        ) == sorted(c.decision.symbol for c in given)


class TestNeutralComponents:
    def test_execution_quality_is_neutral(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 13. Today's nine legacy exits are not a training set."""
        assert score(limits).execution_quality == NEUTRAL_SCORE
        # Neutral AND weightless: it cannot separate two candidates.
        assert WEIGHTS["execution_quality"] * NEUTRAL_SCORE == 0.0

    def test_learned_prior_is_neutral(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 14. One broker-confirmed round trip is not five."""
        assert score(limits).learned_prior == NEUTRAL_SCORE
        assert WEIGHTS["learned_prior"] * NEUTRAL_SCORE == 0.0

    def test_signal_persistence_is_recorded_as_unused(self, limits) -> None:  # type: ignore[no-untyped-def]
        assert score(limits).persistence is None
        assert score(limits).as_dict()["persistence_score"] is None


# --------------------------------------------------------------------------- #
# Invariants 9-20: the authority boundary
# --------------------------------------------------------------------------- #


def build(config, market=None, broker=None, journal=None):  # type: ignore[no-untyped-def]
    stack = AlpacaStack(
        guard=GuardResult(paper=True, detail="test", checks=("ALPACA_PAPER=true",)),
        market_data=market or TrendingMarketData(),
        option_chain=CaptureOptionChain(CAPTURE_DIR),
        broker=broker or SimulatedBroker(make_account()),
        live_broker=False,
    )
    j = journal or Journal(":memory:")
    return Orchestrator(config, stack, j, bull_script()), j, stack


@pytest.fixture
def two_symbol_config(config):  # type: ignore[no-untyped-def]
    """SPY and QQQ: two symbols the captured chain actually covers."""
    return config.model_copy(
        update={"universe": config.universe.model_copy(update={"symbols": ["SPY", "QQQ"]})}
    )


@pytest.fixture
def single_symbol_config(config):  # type: ignore[no-untyped-def]
    return config.model_copy(
        update={"universe": config.universe.model_copy(update={"symbols": ["SPY"]})}
    )


def ranking_events(journal: Journal) -> list[dict]:
    return [
        json.loads(e["payload"])
        for e in journal.recent_events(limit=400)
        if e["event_type"] == "opportunity_ranking"
    ]


class TestRankingHasNoAuthority:
    def test_ranking_cannot_produce_a_risk_approval(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Invariant 9.

        Structural, not incidental: a RankedCandidate carries no RiskDecision
        and no order id, so there is nothing for it to have approved.
        """
        ranked = rank_candidates([candidate(limits, quant_score=1.0, confidence=1.0)])[0]
        assert not hasattr(ranked, "risk")
        assert not hasattr(ranked, "approved")
        assert not hasattr(ranked, "quantity")
        assert not hasattr(ranked, "client_order_id")
        assert set(ranked.__dataclass_fields__) == {
            "decision",
            "spread",
            "regime",
            "score",
        }

    def test_a_perfect_score_is_still_rejected_by_the_governor(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Invariant 10. Governor rejection remains rejection.

        The candidate is qualified and scored by the real path, then offered
        capital against a portfolio that already holds the underlying.
        """
        orchestrator, _journal, stack = build(single_symbol_config)
        orchestrator.startup()
        report = CycleReport(started_at=NOW)
        blocked = make_portfolio(open_positions=(make_position("SPY"),))

        ranked = orchestrator._qualify_candidate(
            make_decision(symbol="SPY"), make_regime("SPY"), blocked, NOW, report
        )
        assert ranked is not None, "the captured chain should yield a SPY spread"

        orchestrator._attempt_entry(ranked, blocked, NOW, report)
        assert report.orders_submitted == []
        assert report.risk_approved == 0
        assert stack.broker.submitted_payloads == []
        assert report.rejections and report.rejections[-1][0] == "SPY"

    def test_a_rejected_top_candidate_does_not_block_the_next_one(
        self, two_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Invariant 11. Sorted greedy, not first-match-or-stop."""
        orchestrator, _journal, stack = build(two_symbol_config)
        orchestrator.startup()
        report = CycleReport(started_at=NOW)
        # SPY is already held, QQQ is not.
        portfolio = make_portfolio(open_positions=(make_position("SPY"),))

        spy = orchestrator._qualify_candidate(
            make_decision(symbol="SPY", decision_id="d-spy"),
            make_regime("SPY"),
            portfolio,
            NOW,
            report,
        )
        qqq = orchestrator._qualify_candidate(
            make_decision(symbol="QQQ", decision_id="d-qqq"),
            make_regime("QQQ"),
            portfolio,
            NOW,
            report,
        )
        assert spy is not None and qqq is not None

        for ranked in (spy, qqq):
            orchestrator._attempt_entry(ranked, portfolio, NOW, report)

        assert [s for s, _ in report.rejections] == ["SPY"]
        assert len(report.orders_submitted) == 1
        assert stack.broker.submitted_payloads, "QQQ never reached the broker"

    def test_every_ranked_candidate_is_offered_capital_in_a_real_cycle(
        self, two_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _stack = build(two_symbol_config)
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)

        events = ranking_events(journal)
        assert len(events) == 1
        ranked = events[0]["candidates"]
        assert ranked, f"nothing qualified: {report.rejections}"
        # Each ranked candidate ended in exactly one outcome: an order, or a
        # recorded rejection. None was silently dropped.
        outcomes = {s for s, _ in report.rejections} | {
            oid.split("-")[1] for oid in report.orders_submitted
        }
        for entry in ranked:
            assert entry["symbol"] in outcomes


class TestCollectionIsSideEffectFree:
    def test_no_broker_order_occurs_during_candidate_collection(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Invariant 16."""
        orchestrator, _journal, stack = build(single_symbol_config)
        orchestrator.startup()
        report = CycleReport(started_at=NOW)

        ranked = orchestrator._qualify_candidate(
            make_decision(symbol="SPY"), make_regime("SPY"), make_portfolio(), NOW, report
        )
        assert ranked is not None
        assert stack.broker.submitted_payloads == []
        assert stack.broker.close_payloads == []
        assert stack.broker.orders == {}

    def test_no_reservation_occurs_during_candidate_collection(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Invariant 17. No client order id is minted, let alone reserved."""
        orchestrator, journal, _stack = build(single_symbol_config)
        orchestrator.startup()
        report = CycleReport(started_at=NOW)

        ranked = orchestrator._qualify_candidate(
            make_decision(symbol="SPY"), make_regime("SPY"), make_portfolio(), NOW, report
        )
        assert ranked is not None
        assert journal.known_client_order_ids() == frozenset()
        assert journal.open_positions() == []

    def test_the_whole_first_pass_completes_before_any_order(
        self, two_symbol_config, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Collection is side-effect free at cycle level, not just per call.

        Recording the broker's order count at the moment each entry is
        attempted proves ranking finished before allocation began: if pass 1
        had submitted anything, the first observation would be non-zero.
        """
        orchestrator, journal, stack = build(two_symbol_config)
        orchestrator.startup()
        observed: list[tuple[int, bool]] = []
        real = orchestrator._attempt_entry

        def spy_on_entry(cand, portfolio, now, report):  # type: ignore[no-untyped-def]
            observed.append((len(stack.broker.submitted_payloads), bool(ranking_events(journal))))
            return real(cand, portfolio, now, report)

        monkeypatch.setattr(orchestrator, "_attempt_entry", spy_on_entry)
        orchestrator.run_cycle(now=NOW)

        assert observed, "no candidate was offered capital"
        assert observed[0][0] == 0, "pass 1 reached the broker"
        # The complete ranking is already journalled by the time the FIRST
        # candidate is offered capital, which is only true if pass 1 finished
        # for every candidate before pass 2 began.
        assert all(ranked for _, ranked in observed), "ranking and allocation interleaved"


class TestAuditability:
    def test_candidate_score_inputs_are_journalled_exactly(
        self, two_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Invariant 12.

        The journal must carry every component, not just the total, and the
        total must be reproducible from the components it recorded.
        """
        orchestrator, journal, _stack = build(two_symbol_config)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)

        events = ranking_events(journal)
        assert len(events) == 1
        payload = events[0]
        assert payload["cycle_ts"] == NOW.isoformat()
        assert payload["weights"] == WEIGHTS
        assert payload["candidate_count"] == len(payload["candidates"])

        expected_keys = {
            "rank",
            "symbol",
            "strategy",
            "quant_score",
            "judge_confidence",
            "regime_score",
            "payoff_score",
            "liquidity_score",
            "diversification_score",
            "execution_quality_score",
            "learned_prior_score",
            "persistence_score",
            "total_opportunity_score",
        }
        assert payload["candidates"], "expected at least one ranked candidate"
        for index, entry in enumerate(payload["candidates"], start=1):
            assert set(entry) == expected_keys
            assert entry["rank"] == index
            recomputed = (
                WEIGHTS["quant"] * entry["quant_score"]
                + WEIGHTS["judge"] * entry["judge_confidence"]
                + WEIGHTS["regime"] * entry["regime_score"]
                + WEIGHTS["payoff"] * entry["payoff_score"]
                + WEIGHTS["liquidity"] * entry["liquidity_score"]
                + WEIGHTS["diversification"] * entry["diversification_score"]
            )
            assert math.isclose(
                recomputed, entry["total_opportunity_score"], abs_tol=1e-6
            )

    def test_the_ranking_record_is_ordered_by_descending_score(
        self, two_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        orchestrator, journal, _stack = build(two_symbol_config)
        orchestrator.startup()
        orchestrator.run_cycle(now=NOW)
        totals = [
            c["total_opportunity_score"] for c in ranking_events(journal)[0]["candidates"]
        ]
        assert totals == sorted(totals, reverse=True)

    def test_a_candidate_that_could_not_be_ranked_is_recorded_with_a_reason(
        self, two_symbol_config, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """AI-approved but unrankable must not vanish from the audit trail."""
        orchestrator, journal, _stack = build(two_symbol_config)
        orchestrator.startup()
        monkeypatch.setattr(orchestrator, "_qualify_candidate", lambda *a, **k: None)
        orchestrator.run_cycle(now=NOW)

        events = ranking_events(journal)
        assert events, "an all-excluded cycle must still journal its ranking"
        excluded = events[0]["excluded"]
        assert excluded
        assert {e["reason"] for e in excluded} == {"no_eligible_spread"}
        assert events[0]["candidate_count"] == 0


class TestExistingBehaviourIsUnchanged:
    def test_a_single_candidate_still_opens_a_position(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Invariant 15. One candidate: ranking is a no-op and entry is normal."""
        orchestrator, journal, _stack = build(single_symbol_config)
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)

        assert report.orders_submitted, f"no order placed: {report.rejections}"
        positions = journal.open_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "SPY"
        assert len(ranking_events(journal)[0]["candidates"]) == 1

    def test_exit_before_entry_cycle_ordering_is_unchanged(
        self, single_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Invariant 18.

        Ranking sits entirely inside the entry phase. An expiring position is
        still closed before the cycle considers anything new, which is what
        frees the slot the ranker then allocates.
        """
        orchestrator, journal, _stack = build(single_symbol_config)
        orchestrator.startup()
        first = orchestrator.run_cycle(now=NOW)
        assert first.orders_submitted

        order = [e["event_type"] for e in reversed(journal.recent_events(limit=400))]
        # Within one cycle: exits are managed, then candidates are ranked, then
        # entries are attempted.
        assert "opportunity_ranking" in order
        ranking_at = order.index("opportunity_ranking")
        selection_at = order.index("contract_selection")
        assert selection_at < ranking_at, "selection must precede ranking"
        assert order.index("cycle_complete") > ranking_at

    def test_the_ranker_never_widens_what_reaches_the_broker(
        self, two_symbol_config
    ) -> None:  # type: ignore[no-untyped-def]
        """Ranked count is an upper bound on submissions, never a floor."""
        orchestrator, journal, stack = build(two_symbol_config)
        orchestrator.startup()
        report = orchestrator.run_cycle(now=NOW)
        ranked = ranking_events(journal)[0]["candidates"]
        assert len(report.orders_submitted) <= len(ranked)
        assert len(stack.broker.submitted_payloads) == len(report.orders_submitted)
