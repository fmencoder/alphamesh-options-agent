"""Money arithmetic and position sizing - the per-trade risk boundary."""

from __future__ import annotations

import pytest

from alphamesh.models.domain import (
    OPTION_MULTIPLIER,
    OrderSide,
    PositionIntent,
    SpreadLeg,
    SpreadStructure,
    Strategy,
)
from alphamesh.risk.money import to_cents, to_dollars
from alphamesh.risk.sizing import size_spread
from tests.conftest import make_contract


def build_spread(limit_cents: int, width_cents: int = 500) -> SpreadStructure:
    return SpreadStructure(
        strategy=Strategy.BULL_CALL_SPREAD,
        symbol="SPY",
        expiration=make_contract().expiration,
        long_leg=SpreadLeg(
            contract=make_contract(strike=770.0),
            side=OrderSide.BUY,
            position_intent=PositionIntent.BUY_TO_OPEN,
        ),
        short_leg=SpreadLeg(
            contract=make_contract(strike=775.0, delta=0.30, bid=1.0, ask=1.1),
            side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
        net_debit_cents=limit_cents,
        strike_width_cents=width_cents,
        limit_price_cents=limit_cents,
    )


class TestMoney:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.05, 5), (1.005, 101), (2.38, 238), (0.1, 10), (769.28, 76928), (0.0, 0)],
    )
    def test_to_cents_is_exact(self, value: float, expected: int) -> None:
        """Binary floats cannot hold 0.05 or 1.005; integer cents can."""
        assert to_cents(value) == expected

    def test_round_trip(self) -> None:
        assert to_dollars(to_cents(12.34)) == 12.34

    def test_no_float_drift_over_accumulation(self) -> None:
        total = sum(to_cents(0.1) for _ in range(10))
        assert total == 100
        assert to_dollars(total) == 1.0


class TestDefinedLoss:
    def test_max_loss_is_exactly_the_premium_paid(self) -> None:
        spread = build_spread(limit_cents=238, width_cents=500)
        assert spread.max_loss_cents(1) == 238 * OPTION_MULTIPLIER
        assert to_dollars(spread.max_loss_cents(1)) == 238.0

    def test_max_profit_is_width_minus_premium(self) -> None:
        spread = build_spread(limit_cents=238, width_cents=500)
        assert spread.max_profit_cents(1) == (500 - 238) * OPTION_MULTIPLIER

    def test_loss_and_profit_scale_linearly_with_quantity(self) -> None:
        spread = build_spread(limit_cents=200, width_cents=500)
        assert spread.max_loss_cents(3) == 3 * spread.max_loss_cents(1)
        assert spread.max_profit_cents(3) == 3 * spread.max_profit_cents(1)

    def test_premium_at_or_above_width_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="net debit must be below strike width"):
            build_spread(limit_cents=500, width_cents=500)


class TestSizing:
    def test_sizes_to_the_normal_cap(self, limits) -> None:  # type: ignore[no-untyped-def]
        # $1.00 debit -> $100 risk per spread; $500 cap -> 5 spreads.
        result = size_spread(build_spread(100), confidence=0.60, limits=limits)
        assert result.quantity == 5
        assert to_dollars(result.max_loss_cents) == 500.0

    def test_high_confidence_unlocks_the_elevated_cap(self, limits) -> None:  # type: ignore[no-untyped-def]
        low = size_spread(build_spread(100), confidence=0.60, limits=limits)
        high = size_spread(build_spread(100), confidence=0.90, limits=limits)
        assert low.cap_cents == 50_000
        assert high.cap_cents == 75_000
        assert high.quantity == 7

    def test_confidence_can_never_exceed_the_absolute_ceiling(self, limits) -> None:  # type: ignore[no-untyped-def]
        for confidence in (0.0, 0.5, 0.75, 0.99, 1.0):
            result = size_spread(build_spread(100), confidence, limits)
            assert result.cap_cents <= int(limits.absolute_max_defined_loss * 100)
            assert result.max_loss_cents <= int(limits.absolute_max_defined_loss * 100)

    def test_rounds_down_never_up(self, limits) -> None:  # type: ignore[no-untyped-def]
        # $1.20 debit -> $120 per spread; $500 / 120 = 4.16 -> 4 spreads.
        result = size_spread(build_spread(120), confidence=0.60, limits=limits)
        assert result.quantity == 4
        assert result.max_loss_cents == 48_000

    def test_spread_too_expensive_sizes_to_zero(self, limits) -> None:  # type: ignore[no-untyped-def]
        result = size_spread(build_spread(900, width_cents=1500), 0.60, limits)
        assert result.quantity == 0
        assert result.max_loss_cents == 0

    def test_portfolio_headroom_shrinks_the_trade(self, limits) -> None:  # type: ignore[no-untyped-def]
        result = size_spread(
            build_spread(100), confidence=0.90, limits=limits, available_risk_cents=25_000
        )
        assert result.quantity == 2
        assert result.max_loss_cents == 20_000

    def test_zero_headroom_sizes_to_zero(self, limits) -> None:  # type: ignore[no-untyped-def]
        result = size_spread(build_spread(100), 0.9, limits, available_risk_cents=0)
        assert result.quantity == 0
