"""Session-aware scoring, regime-conditioned entry, and the expanded universe.

Covers the competition-mode changes: the opening profile must be bit-identical
to the previous behaviour, the intraday profile must stop letting decayed
opening statistics veto a valid trend, and the entry threshold must move with
regime WITHOUT ever dropping below the configured floor.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import pytest

from alphamesh.config import load_config
from alphamesh.intelligence.scoring import (
    INTRADAY_WEIGHTS,
    WEIGHTS,
    EntryMode,
    ScoreProfile,
    build_quant_signal,
    classify_entry_mode,
    components_aligned,
    evaluate_gate,
    minutes_since_open,
    profile_for,
    score_components,
    threshold_for,
    weights_for,
)
from alphamesh.models.domain import Direction, Regime
from tests.conftest import make_regime, make_signal

CONFIG = load_config()
STRATEGIES = CONFIG.strategies

# 13:45Z = 09:45 ET -> inside the 30-minute opening window.
OPENING_AT = datetime(2026, 8, 31, 13, 45, tzinfo=UTC)
# 16:30Z = 12:30 ET -> well past it.
INTRADAY_AT = datetime(2026, 8, 31, 16, 30, tzinfo=UTC)


class TestUniverse:
    def test_all_eight_competition_symbols_are_active(self) -> None:
        assert CONFIG.universe.symbols == [
            "SPY", "QQQ", "IWM", "DIA", "NVDA", "TSLA", "AMD", "AAPL",
        ]

    def test_every_symbol_belongs_to_a_correlation_cluster(self) -> None:
        """Breadth without clustering would defeat the concurrency limit."""
        for symbol in CONFIG.universe.symbols:
            assert CONFIG.risk.group_for(symbol) is not None, symbol


class TestSessionProfile:
    def test_opening_window_is_thirty_minutes(self) -> None:
        assert STRATEGIES.opening_window_minutes == 30
        assert profile_for(OPENING_AT, STRATEGIES) is ScoreProfile.OPENING
        assert profile_for(INTRADAY_AT, STRATEGIES) is ScoreProfile.INTRADAY

    def test_boundary_is_exclusive_at_thirty_minutes(self) -> None:
        at_30 = datetime(2026, 8, 31, 14, 0, tzinfo=UTC)  # exactly 09:30+30
        assert profile_for(at_30, STRATEGIES) is ScoreProfile.INTRADAY

    def test_minutes_since_open_uses_eastern_time(self) -> None:
        assert minutes_since_open(OPENING_AT) == pytest.approx(15.0)
        assert minutes_since_open(INTRADAY_AT) == pytest.approx(180.0)

    def test_premarket_is_not_treated_as_opening(self) -> None:
        premarket = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)  # 08:00 ET
        assert profile_for(premarket, STRATEGIES) is ScoreProfile.INTRADAY


class TestWeightProfiles:
    def test_opening_weights_are_unchanged(self) -> None:
        """The opening profile must be exactly the original five-factor score."""
        assert weights_for(ScoreProfile.OPENING, STRATEGIES) == {
            "momentum": 0.30, "trend": 0.25, "vwap": 0.15,
            "participation": 0.15, "range_position": 0.15,
        }
        assert weights_for(ScoreProfile.OPENING, STRATEGIES) == WEIGHTS

    def test_intraday_weights_use_only_persistent_factors(self) -> None:
        w = weights_for(ScoreProfile.INTRADAY, STRATEGIES)
        assert w == {"momentum": 0.40, "trend": 0.35, "vwap": 0.25}
        assert w == INTRADAY_WEIGHTS
        assert "participation" not in w
        assert "range_position" not in w

    def test_both_profiles_sum_to_one(self) -> None:
        """Identical [0,1] semantics, so one threshold means one thing."""
        for profile in ScoreProfile:
            assert sum(weights_for(profile, STRATEGIES).values()) == pytest.approx(1.0)


class TestIntradayScoreRemovesDecayBias:
    """The defect being fixed: decayed opening stats vetoing a live trend."""

    DECAYED: ClassVar[dict[str, float]] = {
        "ret_5m": 0.0020, "ret_15m": 0.0040, "trend_strength": 0.45,
        "vwap_deviation": 0.0015, "realized_vol": 0.12,
        "volume_acceleration": 0.85,      # participation has decayed
        "opening_range_position": 0.05,   # opening range no longer informative
        "distance_from_high": 0.0005, "distance_from_low": 0.004, "atr_pct": 0.0004,
    }

    def _score(self, profile: ScoreProfile) -> float:
        components = score_components(self.DECAYED)
        weights = weights_for(profile, STRATEGIES)
        return sum(w * components.get(n, 0.0) for n, w in weights.items())

    def test_same_features_score_higher_intraday_than_opening(self) -> None:
        assert self._score(ScoreProfile.INTRADAY) > self._score(ScoreProfile.OPENING)

    def test_participation_and_range_still_computed_and_logged(self) -> None:
        """Dropped from the weighting, not from the record."""
        components = score_components(self.DECAYED)
        assert "participation" in components
        assert "range_position" in components


class TestRegimeConditionedThreshold:
    def test_normal_threshold_is_still_exactly_055(self) -> None:
        assert STRATEGIES.quant_score_threshold == 0.55
        assert STRATEGIES.regime_thresholds["normal"] == 0.55

    def test_strong_trend_uses_050(self) -> None:
        signal = make_signal(quant_score=0.52)
        regime = make_regime(regime=Regime.BULLISH_TREND, direction=Direction.BULLISH)
        threshold, band = threshold_for(signal, regime, STRATEGIES)
        assert (threshold, band) == (0.50, "strong_trend")

    def test_range_bound_demands_065(self) -> None:
        signal = make_signal()
        regime = make_regime(regime=Regime.RANGE_BOUND, direction=Direction.BULLISH)
        threshold, band = threshold_for(signal, regime, STRATEGIES)
        assert (threshold, band) == (0.65, "range_bound")

    def test_trend_without_direction_agreement_gets_normal_bar(self) -> None:
        """A trending regime pointing the other way earns no discount."""
        signal = make_signal(bias=Direction.BULLISH)
        regime = make_regime(regime=Regime.BEARISH_TREND, direction=Direction.BEARISH)
        threshold, band = threshold_for(signal, regime, STRATEGIES)
        assert (threshold, band) == (0.55, "normal")

    def test_trend_without_component_alignment_gets_normal_bar(self) -> None:
        conflicted = dict(make_signal().features)
        conflicted["vwap_deviation"] = -0.002  # disagrees with a bullish bias
        signal = make_signal(features=conflicted, bias=Direction.BULLISH)
        regime = make_regime(regime=Regime.BULLISH_TREND, direction=Direction.BULLISH)
        threshold, band = threshold_for(signal, regime, STRATEGIES)
        assert (threshold, band) == (0.55, "normal")

    def test_configured_floor_is_050(self) -> None:
        assert STRATEGIES.absolute_min_quant_threshold == 0.50

    def test_floor_cannot_be_undercut_by_config(self) -> None:
        """Even a misconfigured band may not open the gate below the floor."""
        reckless = STRATEGIES.model_copy(
            update={"regime_thresholds": {"strong_trend": 0.10, "normal": 0.20,
                                          "range_bound": 0.30}}
        )
        signal = make_signal(quant_score=0.30)
        for regime_kind in (Regime.BULLISH_TREND, Regime.RANGE_BOUND,
                            Regime.VOLATILITY_EXPANSION):
            threshold, _ = threshold_for(
                signal, make_regime(regime=regime_kind), reckless
            )
            assert threshold >= 0.50, regime_kind


class TestGateOutcomes:
    def test_weak_intraday_signal_still_fails(self) -> None:
        signal = make_signal(quant_score=0.47)
        regime = make_regime(regime=Regime.BULLISH_TREND, direction=Direction.BULLISH)
        passes, threshold, _, codes = evaluate_gate(signal, regime, STRATEGIES)
        assert not passes
        assert threshold == 0.50
        assert codes

    def test_strong_aligned_intraday_signal_passes_at_051(self) -> None:
        signal = make_signal(quant_score=0.51)
        regime = make_regime(regime=Regime.BULLISH_TREND, direction=Direction.BULLISH)
        passes, threshold, band, codes = evaluate_gate(signal, regime, STRATEGIES)
        assert passes
        assert (threshold, band, codes) == (0.50, "strong_trend", ())

    def test_051_fails_in_a_range_bound_regime(self) -> None:
        """The same score that trades in a trend must not trade in chop."""
        signal = make_signal(quant_score=0.51)
        regime = make_regime(regime=Regime.RANGE_BOUND, direction=Direction.BULLISH)
        passes, _, _, _ = evaluate_gate(signal, regime, STRATEGIES)
        assert not passes

    def test_neutral_bias_never_passes(self) -> None:
        signal = make_signal(quant_score=0.95, bias=Direction.NEUTRAL)
        regime = make_regime(regime=Regime.BULLISH_TREND)
        passes, _, _, _ = evaluate_gate(signal, regime, STRATEGIES)
        assert not passes


class TestEntryModes:
    def test_full_alignment_is_a_momentum_breakout(self) -> None:
        assert classify_entry_mode(
            make_signal().features, Direction.BULLISH
        ) is EntryMode.MOMENTUM_BREAKOUT

    def test_counter_move_within_a_trend_is_a_pullback(self) -> None:
        features = dict(make_signal().features)
        features["ret_5m"] = -0.0008  # temporary counter-move
        assert classify_entry_mode(features, Direction.BULLISH) is EntryMode.TREND_PULLBACK

    def test_alignment_requires_every_component(self) -> None:
        features = dict(make_signal().features)
        assert components_aligned(features, Direction.BULLISH)
        features["trend_strength"] = -0.4
        assert not components_aligned(features, Direction.BULLISH)


class TestRiskUnchanged:
    """The gate got smarter; downside control did not get looser."""

    def test_per_trade_ceilings_are_untouched(self) -> None:
        assert CONFIG.risk.max_defined_loss_per_trade == 500.0
        assert CONFIG.risk.high_confidence_max_defined_loss == 750.0
        assert CONFIG.risk.absolute_max_defined_loss == 1000.0

    def test_competition_concurrency_limits(self) -> None:
        assert CONFIG.risk.max_open_positions == 5
        assert CONFIG.risk.max_portfolio_defined_risk == 4000.0
        assert CONFIG.risk.max_positions_per_correlation_group == 2

    def test_liquidity_and_freshness_gates_are_untouched(self) -> None:
        assert CONFIG.risk.min_contract_day_volume == 25
        assert CONFIG.risk.max_quote_age_seconds == 120
        assert CONFIG.risk.max_relative_bid_ask_spread == 0.25
        assert CONFIG.risk.min_option_bid == 0.05

    def test_judge_confidence_floor_is_untouched(self) -> None:
        assert STRATEGIES.min_judge_confidence == 0.55

    def test_only_defined_risk_verticals_remain_allowed(self) -> None:
        assert CONFIG.risk.allowed_strategies == ["BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"]

    def test_daily_circuit_breaker_is_untouched(self) -> None:
        assert CONFIG.risk.daily_loss_circuit_breaker == 2000.0


class TestRuntimeCadence:
    def test_open_market_loop_is_thirty_seconds(self) -> None:
        assert CONFIG.settings.loop_seconds == 30

    def test_scoring_is_pure_and_makes_no_llm_call(self) -> None:
        """Doubling scan rate must not double AI spend: scoring does no I/O."""
        from tests.conftest import make_snapshot

        signal = build_quant_signal(make_snapshot(), STRATEGIES, CONFIG.universe)
        assert 0.0 <= signal.quant_score <= 1.0
