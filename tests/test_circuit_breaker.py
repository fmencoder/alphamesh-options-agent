"""Daily loss circuit breaker."""

from __future__ import annotations

from alphamesh.models.domain import ReasonCode
from alphamesh.risk.circuit_breaker import evaluate_circuit_breaker
from tests.conftest import make_portfolio


class TestCircuitBreaker:
    def test_flat_session_is_not_tripped(self, limits) -> None:  # type: ignore[no-untyped-def]
        status = evaluate_circuit_breaker(make_portfolio(), limits)
        assert not status.tripped
        assert status.reason_code is None

    def test_exactly_at_the_limit_trips(self, limits) -> None:  # type: ignore[no-untyped-def]
        status = evaluate_circuit_breaker(
            make_portfolio(realized_pnl_today_cents=-200_000), limits
        )
        assert status.tripped
        assert status.reason_code is ReasonCode.DAILY_DRAWDOWN_LIMIT

    def test_one_cent_inside_the_limit_holds(self, limits) -> None:  # type: ignore[no-untyped-def]
        assert not evaluate_circuit_breaker(
            make_portfolio(realized_pnl_today_cents=-199_999), limits
        ).tripped

    def test_realised_and_unrealised_are_combined(self, limits) -> None:  # type: ignore[no-untyped-def]
        status = evaluate_circuit_breaker(
            make_portfolio(
                realized_pnl_today_cents=-150_000, unrealized_pnl_cents=-60_000
            ),
            limits,
        )
        assert status.tripped

    def test_unrealised_profit_can_offset_realised_loss(self, limits) -> None:  # type: ignore[no-untyped-def]
        assert not evaluate_circuit_breaker(
            make_portfolio(
                realized_pnl_today_cents=-250_000, unrealized_pnl_cents=100_000
            ),
            limits,
        ).tripped

    def test_profitable_session_never_trips(self, limits) -> None:  # type: ignore[no-untyped-def]
        assert not evaluate_circuit_breaker(
            make_portfolio(realized_pnl_today_cents=1_000_000), limits
        ).tripped

    def test_detail_is_human_readable(self, limits) -> None:  # type: ignore[no-untyped-def]
        status = evaluate_circuit_breaker(
            make_portfolio(realized_pnl_today_cents=-300_000), limits
        )
        assert "breaches" in status.detail
