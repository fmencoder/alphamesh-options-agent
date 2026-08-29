"""Exit management. Hard rules the LLM cannot override."""

from __future__ import annotations

from datetime import date, timedelta

from alphamesh.execution.exits import evaluate_exit
from alphamesh.execution.monitor import PositionMark
from alphamesh.models.domain import (
    Direction,
    ExitReason,
    PositionRecord,
    Regime,
    Strategy,
    TradeState,
)
from tests.conftest import NOW, make_regime


def position(
    strategy: Strategy = Strategy.BULL_CALL_SPREAD,
    opened_at=NOW,  # type: ignore[no-untyped-def]
    expiration: date = date(2026, 9, 3),
) -> PositionRecord:
    return PositionRecord(
        position_id="p1",
        decision_id="d1",
        client_order_id="c1",
        symbol="SPY",
        strategy=strategy,
        quantity=1,
        entry_debit_cents=23_800,
        max_loss_cents=23_800,
        max_profit_cents=26_200,
        opened_at=opened_at,
        expiration=expiration,
        long_symbol="SPY260903C00769000",
        short_symbol="SPY260903C00774000",
        state=TradeState.MONITORING,
    )


def mark(profit_pct: float = 0.0, loss_pct: float = 0.0) -> PositionMark:
    return PositionMark(
        position_id="p1",
        mark_cents=0,
        unrealized_pnl_cents=0,
        pct_of_max_profit=profit_pct,
        pct_of_defined_risk_lost=loss_pct,
    )


class TestNoExit:
    def test_healthy_position_is_held(self, strategies) -> None:  # type: ignore[no-untyped-def]
        decision = evaluate_exit(
            position(), mark(profit_pct=0.2), make_regime(), strategies, NOW
        )
        assert not decision.should_exit
        assert decision.reason is None


class TestProfitAndLoss:
    def test_profit_target_triggers(self, strategies) -> None:  # type: ignore[no-untyped-def]
        decision = evaluate_exit(
            position(), mark(profit_pct=0.60), make_regime(), strategies, NOW
        )
        assert decision.should_exit
        assert decision.reason is ExitReason.PROFIT_TARGET

    def test_just_below_target_holds(self, strategies) -> None:  # type: ignore[no-untyped-def]
        assert not evaluate_exit(
            position(), mark(profit_pct=0.54), make_regime(), strategies, NOW
        ).should_exit

    def test_max_loss_stop_triggers(self, strategies) -> None:  # type: ignore[no-untyped-def]
        decision = evaluate_exit(
            position(), mark(loss_pct=0.70), make_regime(), strategies, NOW
        )
        assert decision.should_exit
        assert decision.reason is ExitReason.MAX_LOSS

    def test_unmarkable_position_does_not_trigger_a_price_exit(self, strategies) -> None:  # type: ignore[no-untyped-def]
        """A missing mark must not be read as a zero value and stopped out."""
        decision = evaluate_exit(position(), None, make_regime(), strategies, NOW)
        assert not decision.should_exit


class TestTimeAndSession:
    def test_max_holding_time_triggers(self, strategies) -> None:  # type: ignore[no-untyped-def]
        later = NOW + timedelta(minutes=1000)
        decision = evaluate_exit(
            position(), mark(), make_regime(), strategies, later
        )
        assert decision.should_exit
        assert decision.reason is ExitReason.MAX_HOLDING_TIME

    def test_end_of_day_flatten_triggers(self, strategies) -> None:  # type: ignore[no-untyped-def]
        close = NOW + timedelta(minutes=10)
        decision = evaluate_exit(
            position(), mark(), make_regime(), strategies, NOW, session_close=close
        )
        assert decision.should_exit
        assert decision.reason is ExitReason.END_OF_DAY

    def test_no_flatten_when_overnight_is_permitted(self, strategies) -> None:  # type: ignore[no-untyped-def]
        overnight = strategies.model_copy(
            update={"exits": {**strategies.exits, "allow_overnight": True}}
        )
        close = NOW + timedelta(minutes=10)
        decision = evaluate_exit(
            position(), mark(), make_regime(), overnight, NOW, session_close=close
        )
        assert not decision.should_exit

    def test_expiration_day_forces_an_exit(self, strategies) -> None:  # type: ignore[no-untyped-def]
        decision = evaluate_exit(
            position(expiration=NOW.date()), mark(), make_regime(), strategies, NOW
        )
        assert decision.should_exit
        assert decision.reason is ExitReason.EXPIRED


class TestSignalInvalidation:
    def test_regime_flip_against_a_bull_position_exits(self, strategies) -> None:  # type: ignore[no-untyped-def]
        bearish = make_regime(regime=Regime.BEARISH_TREND, direction=Direction.BEARISH)
        decision = evaluate_exit(position(), mark(), bearish, strategies, NOW)
        assert decision.should_exit
        assert decision.reason is ExitReason.SIGNAL_INVALIDATED

    def test_regime_flip_against_a_bear_position_exits(self, strategies) -> None:  # type: ignore[no-untyped-def]
        bullish = make_regime(regime=Regime.BULLISH_TREND, direction=Direction.BULLISH)
        decision = evaluate_exit(
            position(strategy=Strategy.BEAR_PUT_SPREAD), mark(), bullish, strategies, NOW
        )
        assert decision.should_exit
        assert decision.reason is ExitReason.SIGNAL_INVALIDATED

    def test_unstable_regime_exits_regardless_of_direction(self, strategies) -> None:  # type: ignore[no-untyped-def]
        unstable = make_regime(regime=Regime.UNSTABLE, direction=Direction.NEUTRAL)
        decision = evaluate_exit(position(), mark(), unstable, strategies, NOW)
        assert decision.should_exit
        assert decision.reason is ExitReason.SIGNAL_INVALIDATED

    def test_aligned_regime_holds(self, strategies) -> None:  # type: ignore[no-untyped-def]
        assert not evaluate_exit(
            position(), mark(), make_regime(), strategies, NOW
        ).should_exit

    def test_invalidation_can_be_disabled_by_configuration(self, strategies) -> None:  # type: ignore[no-untyped-def]
        relaxed = strategies.model_copy(
            update={
                "exits": {**strategies.exits, "signal_invalidation_regime_flip": False}
            }
        )
        bearish = make_regime(regime=Regime.BEARISH_TREND, direction=Direction.BEARISH)
        assert not evaluate_exit(position(), mark(), bearish, relaxed, NOW).should_exit


class TestCircuitBreakerExit:
    def test_tripped_breaker_overrides_everything(self, strategies) -> None:  # type: ignore[no-untyped-def]
        decision = evaluate_exit(
            position(),
            mark(profit_pct=0.1),
            make_regime(),
            strategies,
            NOW,
            circuit_breaker_tripped=True,
        )
        assert decision.should_exit
        assert decision.reason is ExitReason.CIRCUIT_BREAKER

    def test_breaker_beats_a_profitable_hold(self, strategies) -> None:  # type: ignore[no-untyped-def]
        decision = evaluate_exit(
            position(),
            mark(profit_pct=0.0),
            make_regime(),
            strategies,
            NOW,
            circuit_breaker_tripped=True,
        )
        assert decision.reason is ExitReason.CIRCUIT_BREAKER
