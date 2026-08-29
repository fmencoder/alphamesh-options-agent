"""Feature engine, opportunity score and regime classification."""

from __future__ import annotations

import pytest

from alphamesh.intelligence.features import (
    average_true_range,
    compute_features,
    opening_range_position,
    pct_return,
    realized_volatility,
    rolling_vwap,
    trend_strength,
    volume_acceleration,
    vwap_deviation,
)
from alphamesh.intelligence.regime import classify
from alphamesh.intelligence.scoring import (
    build_quant_signal,
    directional_bias,
    score_components,
)
from alphamesh.models.domain import Direction, ReasonCode, Regime
from tests.conftest import make_bars, make_snapshot


class TestFeaturePrimitives:
    def test_return_over_a_flat_series_is_zero(self) -> None:
        assert pct_return(make_bars(60, drift_per_bar=0.0), 5) == 0.0

    def test_return_tracks_a_known_drift(self) -> None:
        bars = make_bars(60, start_price=100.0, drift_per_bar=0.1)
        assert pct_return(bars, 5) == pytest.approx(0.5 / bars[-6].close, rel=1e-6)

    def test_return_with_insufficient_history_is_zero(self) -> None:
        assert pct_return(make_bars(3), 15) == 0.0

    def test_vwap_of_a_flat_series_equals_the_price(self) -> None:
        bars = make_bars(30, start_price=200.0, drift_per_bar=0.0)
        assert rolling_vwap(bars) == pytest.approx(200.0)

    def test_vwap_deviation_is_positive_in_an_uptrend(self) -> None:
        assert vwap_deviation(make_bars(60, drift_per_bar=0.05)) > 0

    def test_atr_is_non_negative(self) -> None:
        assert average_true_range(make_bars(60)) >= 0

    def test_realized_vol_rises_with_wobble(self) -> None:
        calm = realized_volatility(make_bars(60, wobble=0.0))
        choppy = realized_volatility(make_bars(60, wobble=1.0))
        assert choppy > calm

    def test_volume_acceleration_is_one_at_steady_volume(self) -> None:
        assert volume_acceleration(make_bars(80)) == pytest.approx(1.0)

    def test_trend_strength_is_signed(self) -> None:
        assert trend_strength(make_bars(60, drift_per_bar=0.1)) > 0
        assert trend_strength(make_bars(60, drift_per_bar=-0.1)) < 0
        assert trend_strength(make_bars(60, drift_per_bar=0.0)) == 0.0

    def test_trend_strength_is_bounded(self) -> None:
        assert -1.0 <= trend_strength(make_bars(60, drift_per_bar=50.0)) <= 1.0

    def test_opening_range_position_is_signed(self) -> None:
        assert opening_range_position(make_bars(90, drift_per_bar=0.1)) > 0
        assert opening_range_position(make_bars(90, drift_per_bar=-0.1)) < 0

    def test_features_are_deterministic(self) -> None:
        snapshot = make_snapshot()
        assert compute_features(snapshot) == compute_features(snapshot)

    def test_no_bars_yields_no_features(self) -> None:
        from alphamesh.models.domain import MarketSnapshot
        from tests.conftest import NOW

        empty = MarketSnapshot(symbol="SPY", as_of=NOW, last_price=1.0, bars=())
        assert compute_features(empty) == {}


class TestRealCapturedData:
    def test_features_compute_over_the_captured_spy_session(self, capture_chain) -> None:  # type: ignore[no-untyped-def]

        from alphamesh.alpaca.market_data import CaptureMarketData
        from tests.conftest import CAPTURE_DIR

        source = CaptureMarketData(CAPTURE_DIR)
        snapshot = source.snapshot("SPY", lookback_minutes=180)
        features = compute_features(snapshot)
        assert snapshot.bar_count >= 120
        assert features["last_price"] == pytest.approx(769.34)
        assert all(isinstance(v, float) for v in features.values())

    def test_both_universe_symbols_are_available(self) -> None:
        from alphamesh.alpaca.market_data import CaptureMarketData
        from tests.conftest import CAPTURE_DIR

        source = CaptureMarketData(CAPTURE_DIR)
        assert set(source.available_symbols()) == {"SPY", "QQQ"}


class TestScoring:
    def test_components_are_bounded(self) -> None:
        features = compute_features(make_snapshot(bars=make_bars(120, drift_per_bar=5.0)))
        for value in score_components(features).values():
            assert 0.0 <= value <= 1.0

    def test_score_is_bounded(self, config) -> None:  # type: ignore[no-untyped-def]
        for drift in (-1.0, -0.1, 0.0, 0.1, 1.0):
            snapshot = make_snapshot(
                bars=make_bars(120, start_price=1000.0, drift_per_bar=drift)
            )
            signal = build_quant_signal(snapshot, config.strategies, config.universe)
            assert 0.0 <= signal.quant_score <= 1.0

    def test_flat_tape_does_not_pass_the_gate(self, config) -> None:  # type: ignore[no-untyped-def]
        snapshot = make_snapshot(bars=make_bars(120, drift_per_bar=0.0))
        signal = build_quant_signal(snapshot, config.strategies, config.universe)
        assert not signal.passes_gate
        assert signal.directional_bias is Direction.NEUTRAL

    def test_insufficient_bars_blocks_the_gate(self, config) -> None:  # type: ignore[no-untyped-def]
        snapshot = make_snapshot(bars=make_bars(10, drift_per_bar=0.5))
        signal = build_quant_signal(snapshot, config.strategies, config.universe)
        assert not signal.passes_gate
        assert ReasonCode.INSUFFICIENT_MARKET_DATA in signal.reason_codes

    def test_threshold_is_configuration_driven(self, config) -> None:  # type: ignore[no-untyped-def]
        snapshot = make_snapshot(bars=make_bars(120, drift_per_bar=0.08))
        strict = config.strategies.model_copy(update={"quant_score_threshold": 0.99})
        lenient = config.strategies.model_copy(update={"quant_score_threshold": 0.01})
        assert not build_quant_signal(snapshot, strict, config.universe).passes_gate
        assert build_quant_signal(snapshot, lenient, config.universe).passes_gate

    def test_directional_bias_requires_agreement(self) -> None:
        assert (
            directional_bias(
                {
                    "ret_5m": 0.001,
                    "ret_15m": 0.001,
                    "trend_strength": 0.2,
                    "vwap_deviation": 0.001,
                }
            )
            is Direction.BULLISH
        )
        assert (
            directional_bias(
                {
                    "ret_5m": 0.001,
                    "ret_15m": -0.001,
                    "trend_strength": 0.2,
                    "vwap_deviation": -0.001,
                }
            )
            is Direction.NEUTRAL
        )


class TestRegime:
    def test_insufficient_history_is_unknown(self, config) -> None:  # type: ignore[no-untyped-def]
        snapshot = make_snapshot(bars=make_bars(10))
        assessment = classify(snapshot, {}, config.universe)
        assert assessment.regime is Regime.UNKNOWN
        assert assessment.favors_no_trade

    def test_strong_uptrend_is_bullish(self, config) -> None:  # type: ignore[no-untyped-def]
        snapshot = make_snapshot(bars=make_bars(120, start_price=500.0, drift_per_bar=0.15))
        assessment = classify(snapshot, compute_features(snapshot), config.universe)
        assert assessment.regime is Regime.BULLISH_TREND
        assert assessment.direction is Direction.BULLISH

    def test_strong_downtrend_is_bearish(self, config) -> None:  # type: ignore[no-untyped-def]
        snapshot = make_snapshot(bars=make_bars(120, start_price=500.0, drift_per_bar=-0.15))
        assessment = classify(snapshot, compute_features(snapshot), config.universe)
        assert assessment.regime is Regime.BEARISH_TREND
        assert assessment.direction is Direction.BEARISH

    def test_flat_tape_is_range_bound(self, config) -> None:  # type: ignore[no-untyped-def]
        snapshot = make_snapshot(bars=make_bars(120, drift_per_bar=0.0))
        assessment = classify(snapshot, compute_features(snapshot), config.universe)
        assert assessment.regime is Regime.RANGE_BOUND
        assert assessment.direction is Direction.NEUTRAL

    def test_violent_tape_is_unstable_and_favours_no_trade(self, config) -> None:  # type: ignore[no-untyped-def]
        snapshot = make_snapshot(
            bars=make_bars(120, start_price=500.0, drift_per_bar=0.0, wobble=8.0)
        )
        assessment = classify(snapshot, compute_features(snapshot), config.universe)
        assert assessment.regime in (Regime.UNSTABLE, Regime.VOLATILITY_EXPANSION)
        if assessment.regime is Regime.UNSTABLE:
            assert assessment.favors_no_trade

    def test_captured_session_classifies_without_error(self, config) -> None:  # type: ignore[no-untyped-def]
        from alphamesh.alpaca.market_data import CaptureMarketData
        from tests.conftest import CAPTURE_DIR

        source = CaptureMarketData(CAPTURE_DIR)
        for symbol in ("SPY", "QQQ"):
            snapshot = source.snapshot(symbol, 180)
            assessment = classify(
                snapshot, compute_features(snapshot), config.universe
            )
            assert assessment.regime in set(Regime)
            assert 0.0 <= assessment.confidence <= 1.0
