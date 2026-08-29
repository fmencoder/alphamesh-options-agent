"""Competition analytics and bounded adaptive weighting."""

from __future__ import annotations

import pytest

from alphamesh.analytics import (
    WEIGHT_CEILING,
    WEIGHT_FLOOR,
    adaptive_weights,
    build_report,
    confidence_bucket,
)


def outcome(
    pnl_cents: int,
    strategy: str = "BULL_CALL_SPREAD",
    regime: str = "BULLISH_TREND",
    symbol: str = "SPY",
    confidence: float = 0.8,
    holding: float = 60.0,
    ror: float = 0.2,
    exit_reason: str = "PROFIT_TARGET",
) -> dict:
    return {
        "realized_pnl_cents": pnl_cents,
        "strategy": strategy,
        "regime": regime,
        "symbol": symbol,
        "confidence": confidence,
        "holding_minutes": holding,
        "return_on_defined_risk": ror,
        "exit_reason": exit_reason,
    }


class TestConfidenceBuckets:
    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            (0.55, "0.50-0.65"),
            (0.70, "0.65-0.75"),
            (0.80, "0.75-0.85"),
            (0.95, "0.85-1.00"),
            (1.00, "0.85-1.00"),
            (0.10, "<0.50"),
        ],
    )
    def test_bucket_assignment(self, confidence, expected) -> None:  # type: ignore[no-untyped-def]
        assert confidence_bucket(confidence) == expected


class TestReport:
    def test_empty_history_reports_nothing_rather_than_guessing(self) -> None:
        report = build_report([])
        assert report["overall"]["trades"] == 0
        assert report["overall"]["total_pnl"] == 0
        assert report["by_strategy"] == {}

    def test_overall_totals(self) -> None:
        report = build_report([outcome(10_000), outcome(-4_000), outcome(2_500)])
        overall = report["overall"]
        assert overall["trades"] == 3
        assert overall["wins"] == 2
        assert overall["win_rate"] == pytest.approx(2 / 3, rel=1e-3)
        assert overall["total_pnl"] == 85.0
        assert overall["largest_win"] == 100.0
        assert overall["largest_loss"] == -40.0

    def test_slices_by_every_requested_dimension(self) -> None:
        rows = [
            outcome(10_000, strategy="BULL_CALL_SPREAD", symbol="SPY"),
            outcome(-5_000, strategy="BEAR_PUT_SPREAD", symbol="QQQ", regime="BEARISH_TREND"),
        ]
        report = build_report(rows)
        assert set(report["by_strategy"]) == {"BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"}
        assert set(report["by_symbol"]) == {"SPY", "QQQ"}
        assert set(report["by_regime"]) == {"BULLISH_TREND", "BEARISH_TREND"}
        assert report["by_confidence"]["0.75-0.85"]["trades"] == 2

    def test_averages_are_computed_per_group(self) -> None:
        report = build_report([outcome(10_000, holding=30.0), outcome(20_000, holding=90.0)])
        group = report["by_strategy"]["BULL_CALL_SPREAD"]
        assert group["average_pnl"] == 150.0
        assert group["average_holding_minutes"] == 60.0

    def test_exit_reasons_are_reported(self) -> None:
        rows = [
            outcome(1_000, exit_reason="PROFIT_TARGET"),
            outcome(-1_000, exit_reason="MAX_LOSS"),
        ]
        report = build_report(rows)
        assert set(report["by_exit_reason"]) == {"PROFIT_TARGET", "MAX_LOSS"}


class TestAdaptiveWeights:
    def test_thin_history_earns_no_adjustment(self) -> None:
        weights = adaptive_weights([outcome(1_000)] * 3)
        assert all(w == 1.0 for w in weights.values())

    def test_winning_pairing_is_nudged_up(self) -> None:
        weights = adaptive_weights([outcome(1_000)] * 10)
        assert weights["BULL_CALL_SPREAD|BULLISH_TREND"] > 1.0

    def test_losing_pairing_is_nudged_down(self) -> None:
        weights = adaptive_weights([outcome(-1_000)] * 10)
        assert weights["BULL_CALL_SPREAD|BULLISH_TREND"] < 1.0

    def test_weights_are_hard_clamped(self) -> None:
        """No streak, however long, can move a weight outside its band."""
        for rows in ([outcome(1_000)] * 500, [outcome(-1_000)] * 500):
            for weight in adaptive_weights(rows).values():
                assert WEIGHT_FLOOR <= weight <= WEIGHT_CEILING

    def test_weights_do_not_reference_any_risk_limit(self) -> None:
        """Adaptive output is scoring guidance; it names no risk field."""
        from alphamesh.config import RiskLimits

        weights = adaptive_weights([outcome(1_000)] * 10)
        assert not set(weights) & set(RiskLimits.model_fields)
