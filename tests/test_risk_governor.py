"""Risk Governor. Every hard gate has a test that proves it cannot be walked past."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from alphamesh.models.domain import (
    OrderSide,
    PositionIntent,
    PositionRecord,
    ReasonCode,
    SpreadLeg,
    SpreadStructure,
    Strategy,
    TradeState,
)
from alphamesh.risk.governor import RiskGovernor
from alphamesh.risk.money import to_dollars
from tests.conftest import (
    NOW,
    make_account,
    make_contract,
    make_decision,
    make_portfolio,
)


def spread_with(limit_cents: int, width: float = 5.0, symbol: str = "SPY", **kw) -> SpreadStructure:  # type: ignore[no-untyped-def]
    long_c = make_contract(underlying=symbol, strike=770.0, **kw)
    short_c = make_contract(
        underlying=symbol, strike=770.0 + width, delta=0.30, bid=1.00, ask=1.05, **kw
    )
    return SpreadStructure(
        strategy=Strategy.BULL_CALL_SPREAD,
        symbol=symbol,
        expiration=long_c.expiration,
        long_leg=SpreadLeg(
            contract=long_c, side=OrderSide.BUY, position_intent=PositionIntent.BUY_TO_OPEN
        ),
        short_leg=SpreadLeg(
            contract=short_c,
            side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
        net_debit_cents=limit_cents,
        strike_width_cents=int(width * 100),
        limit_price_cents=limit_cents,
    )


def open_position(
    symbol: str = "SPY", max_loss_cents: int = 50_000, position_id: str = "p1"
) -> PositionRecord:
    return PositionRecord(
        position_id=position_id,
        decision_id="d",
        client_order_id=f"alphamesh-{symbol}-BCS-{position_id}",
        symbol=symbol,
        strategy=Strategy.BULL_CALL_SPREAD,
        quantity=1,
        entry_debit_cents=max_loss_cents,
        max_loss_cents=max_loss_cents,
        max_profit_cents=10_000,
        opened_at=NOW,
        expiration=date(2026, 9, 3),
        long_symbol="SPY260903C00770000",
        short_symbol="SPY260903C00775000",
        state=TradeState.MONITORING,
    )


@pytest.fixture
def governor(limits):  # type: ignore[no-untyped-def]
    return RiskGovernor(limits, paper_confirmed=True)


class TestPaperModeGate:
    def test_unconfirmed_paper_mode_blocks_everything(self, limits) -> None:  # type: ignore[no-untyped-def]
        governor = RiskGovernor(limits, paper_confirmed=False)
        result = governor.approve(
            make_decision(), spread_with(100), make_portfolio(), NOW, "cid-1"
        )
        assert not result.approved
        assert ReasonCode.LIVE_TRADING_FORBIDDEN in result.reason_codes

    def test_paper_check_runs_before_any_other_check(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Even an otherwise perfect trade is refused when paper is unproven."""
        governor = RiskGovernor(limits, paper_confirmed=False)
        result = governor.approve(
            make_decision(), spread_with(100), make_portfolio(), NOW, "cid-1"
        )
        assert result.reason_codes == (ReasonCode.LIVE_TRADING_FORBIDDEN,)


class TestHappyPath:
    def test_clean_trade_is_approved_and_sized(self, governor) -> None:  # type: ignore[no-untyped-def]
        result = governor.approve(
            make_decision(confidence=0.60), spread_with(100), make_portfolio(), NOW, "cid"
        )
        assert result.approved
        assert result.quantity == 5
        assert to_dollars(result.max_loss_cents) == 500.0
        assert result.reason_codes == (ReasonCode.APPROVED,)

    def test_every_gate_is_recorded_as_run(self, governor) -> None:  # type: ignore[no-untyped-def]
        result = governor.approve(
            make_decision(), spread_with(100), make_portfolio(), NOW, "cid"
        )
        assert {
            "paper_mode",
            "account_status",
            "strategy_allowlist",
            "defined_risk",
            "leg_liquidity",
            "duplicate_order",
            "daily_circuit_breaker",
            "max_open_positions",
            "correlated_exposure",
            "portfolio_risk",
            "position_sizing",
            "per_trade_cap",
            "aggregate_caps",
            "buying_power",
        } <= set(result.checks_run)


class TestPerTradeCap:
    def test_trade_exceeding_the_normal_cap_is_sized_down(self, governor) -> None:  # type: ignore[no-untyped-def]
        result = governor.approve(
            make_decision(confidence=0.60), spread_with(300, width=10.0), make_portfolio(), NOW, "c"
        )
        assert result.approved
        assert to_dollars(result.max_loss_cents) <= 500.0

    def test_single_spread_above_the_cap_is_rejected(self, governor) -> None:  # type: ignore[no-untyped-def]
        result = governor.approve(
            make_decision(confidence=0.60),
            spread_with(600, width=12.0),
            make_portfolio(),
            NOW,
            "c",
        )
        assert not result.approved
        assert ReasonCode.MAX_POSITION_RISK in result.reason_codes

    def test_high_confidence_never_exceeds_the_absolute_ceiling(self, governor, limits) -> None:  # type: ignore[no-untyped-def]
        for confidence in (0.75, 0.85, 0.95, 1.0):
            result = governor.approve(
                make_decision(confidence=confidence),
                spread_with(100),
                make_portfolio(),
                NOW,
                f"c{confidence}",
            )
            assert result.max_loss_cents <= int(limits.absolute_max_defined_loss * 100)


class TestPortfolioCap:
    def test_trade_is_shrunk_to_remaining_headroom(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(
            open_positions=(open_position("SPY", 140_000, "p1"),),
        )
        result = governor.approve(
            make_decision(symbol="QQQ", confidence=0.9),
            spread_with(100, symbol="QQQ"),
            portfolio,
            NOW,
            "c",
        )
        assert result.approved
        # $1500 correlated-group cap - $1400 already used = $100 of headroom.
        assert result.max_loss_cents <= 10_000

    def test_exhausted_portfolio_risk_rejects(self, governor) -> None:  # type: ignore[no-untyped-def]
        # Ungrouped symbols, so the portfolio cap is the binding constraint
        # rather than a correlation bucket. $1500 + $1500 + $1000 = the
        # $4000 aggregate ceiling exactly.
        portfolio = make_portfolio(
            open_positions=(
                open_position("AAA", 150_000, "p1"),
                open_position("BBB", 150_000, "p2"),
                open_position("CCC", 100_000, "p3"),
            )
        )
        result = governor.approve(
            make_decision(symbol="DDD"), spread_with(100, symbol="DDD"), portfolio, NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.MAX_PORTFOLIO_RISK in result.reason_codes


class TestOpenPositionCount:
    def test_at_the_limit_new_entries_are_refused(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(
            open_positions=tuple(
                open_position(s, 10_000, f"p{i}")
                for i, s in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"])
            )
        )
        result = governor.approve(
            make_decision(), spread_with(100), portfolio, NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.MAX_OPEN_POSITIONS in result.reason_codes


class TestCorrelatedExposure:
    def test_spy_and_qqq_share_one_exposure_bucket(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(
            open_positions=(
                open_position("SPY", 10_000, "p1"),
                open_position("QQQ", 10_000, "p2"),
            )
        )
        result = governor.approve(
            make_decision(symbol="SPY", decision_id="d2"),
            spread_with(100),
            portfolio,
            NOW,
            "c",
        )
        assert not result.approved
        assert ReasonCode.CORRELATED_EXPOSURE in result.reason_codes

    def test_group_defined_risk_cap_is_enforced(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(open_positions=(open_position("QQQ", 149_000, "p1"),))
        result = governor.approve(
            make_decision(symbol="SPY", confidence=0.9),
            spread_with(100),
            portfolio,
            NOW,
            "c",
        )
        # $1500 group cap - $1490 used = $10, below one $100 spread.
        assert not result.approved
        assert ReasonCode.CORRELATED_EXPOSURE in result.reason_codes


class TestDailyCircuitBreaker:
    def test_breached_daily_loss_halts_new_entries(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(realized_pnl_today_cents=-200_100)
        result = governor.approve(
            make_decision(), spread_with(100), portfolio, NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.DAILY_DRAWDOWN_LIMIT in result.reason_codes

    def test_unrealised_loss_counts_toward_the_breaker(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(
            realized_pnl_today_cents=-100_000, unrealized_pnl_cents=-100_000
        )
        result = governor.approve(
            make_decision(), spread_with(100), portfolio, NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.DAILY_DRAWDOWN_LIMIT in result.reason_codes

    def test_just_inside_the_limit_still_trades(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(realized_pnl_today_cents=-199_900)
        result = governor.approve(
            make_decision(), spread_with(100), portfolio, NOW, "c"
        )
        assert result.approved

    def test_profitable_session_does_not_trip(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(realized_pnl_today_cents=500_000)
        assert governor.approve(
            make_decision(), spread_with(100), portfolio, NOW, "c"
        ).approved


class TestDuplicateProtection:
    def test_known_client_order_id_is_refused(self, governor) -> None:  # type: ignore[no-untyped-def]
        result = governor.approve(
            make_decision(),
            spread_with(100),
            make_portfolio(),
            NOW,
            "cid-dup",
            known_client_order_ids=frozenset({"cid-dup"}),
        )
        assert not result.approved
        assert ReasonCode.DUPLICATE_ORDER in result.reason_codes

    def test_existing_position_in_the_symbol_is_refused(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(open_positions=(open_position("SPY", 10_000),))
        result = governor.approve(
            make_decision(symbol="SPY"), spread_with(100), portfolio, NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.DUPLICATE_ORDER in result.reason_codes


class TestStrategyAllowlist:
    def test_no_trade_can_never_be_approved(self, governor) -> None:  # type: ignore[no-untyped-def]
        decision = make_decision(strategy=Strategy.NO_TRADE)
        result = governor.approve(
            decision, spread_with(100), make_portfolio(), NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.UNSUPPORTED_STRATEGY in result.reason_codes

    def test_mismatched_spread_and_decision_are_refused(self, governor) -> None:  # type: ignore[no-untyped-def]
        decision = make_decision(strategy=Strategy.BEAR_PUT_SPREAD)
        result = governor.approve(
            decision, spread_with(100), make_portfolio(), NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.UNSUPPORTED_STRATEGY in result.reason_codes

    def test_removing_a_strategy_from_the_allowlist_blocks_it(self, limits) -> None:  # type: ignore[no-untyped-def]
        narrowed = limits.model_copy(update={"allowed_strategies": ["BEAR_PUT_SPREAD"]})
        governor = RiskGovernor(narrowed, paper_confirmed=True)
        result = governor.approve(
            make_decision(), spread_with(100), make_portfolio(), NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.UNSUPPORTED_STRATEGY in result.reason_codes


class TestAccountGates:
    def test_blocked_account_is_refused(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(account=make_account(trading_blocked=True))
        result = governor.approve(
            make_decision(), spread_with(100), portfolio, NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.ACCOUNT_NOT_TRADEABLE in result.reason_codes

    def test_insufficient_options_level_is_refused(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(account=make_account(options_level=2))
        result = governor.approve(
            make_decision(), spread_with(100), portfolio, NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.OPTIONS_LEVEL_INSUFFICIENT in result.reason_codes

    def test_insufficient_buying_power_is_refused(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(account=make_account(options_buying_power=10.0))
        result = governor.approve(
            make_decision(), spread_with(100), portfolio, NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.INSUFFICIENT_BUYING_POWER in result.reason_codes


class TestQuoteQualityAtApprovalTime:
    def test_quotes_that_went_stale_since_selection_are_refused(self, governor) -> None:  # type: ignore[no-untyped-def]
        stale = NOW + timedelta(minutes=30)
        result = governor.approve(
            make_decision(), spread_with(100), make_portfolio(), stale, "c"
        )
        assert not result.approved
        assert ReasonCode.STALE_QUOTES in result.reason_codes

    def test_mismatched_expirations_are_refused(self, governor) -> None:  # type: ignore[no-untyped-def]
        long_c = make_contract(strike=770.0, expiration=date(2026, 9, 3))
        short_c = make_contract(
            strike=775.0, delta=0.30, bid=1.0, ask=1.05, expiration=date(2026, 9, 3)
        )
        spread = SpreadStructure(
            strategy=Strategy.BULL_CALL_SPREAD,
            symbol="SPY",
            expiration=date(2026, 9, 3),
            long_leg=SpreadLeg(
                contract=long_c,
                side=OrderSide.BUY,
                position_intent=PositionIntent.BUY_TO_OPEN,
            ),
            short_leg=SpreadLeg(
                contract=short_c,
                side=OrderSide.SELL,
                position_intent=PositionIntent.SELL_TO_OPEN,
            ),
            net_debit_cents=100,
            strike_width_cents=500,
            limit_price_cents=100,
        )
        # Swap in a leg with a different expiration after construction.
        mismatched = spread.model_copy(
            update={
                "short_leg": spread.short_leg.model_copy(
                    update={
                        "contract": make_contract(
                            strike=775.0,
                            delta=0.30,
                            bid=1.0,
                            ask=1.05,
                            expiration=date(2026, 9, 10),
                        )
                    }
                )
            }
        )
        result = governor.approve(
            make_decision(), mismatched, make_portfolio(), NOW, "c"
        )
        assert not result.approved
        assert ReasonCode.EXPIRATION_MISMATCH in result.reason_codes


class TestGovernorImmutability:
    def test_limits_cannot_be_mutated_through_the_governor(self, governor) -> None:  # type: ignore[no-untyped-def]
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            governor.limits.max_defined_loss_per_trade = 999_999.0

    def test_ai_confidence_cannot_widen_the_absolute_ceiling(self, limits) -> None:  # type: ignore[no-untyped-def]
        """Confidence selects between two configured caps and nothing more."""
        for confidence in [0.0, 0.5, 0.749, 0.75, 0.99, 1.0, 5.0, -1.0]:
            cap = limits.cap_cents_for_confidence(confidence)
            assert cap <= int(limits.absolute_max_defined_loss * 100)
            assert cap >= int(limits.max_defined_loss_per_trade * 100) or confidence < 0.75


class TestWorkingOrderStacking:
    """Regression: three live AMD orders stacked in production on 2026-08-31.

    The id-based duplicate gate was defeated by a moving limit price. Each
    cycle rebuilt the same spread at a slightly different price, minting a
    fresh client_order_id, and the open-position gate could not see an order
    that had not filled yet. Three identical spreads went live at 4.30, 4.38
    and 4.32 within three minutes.
    """

    def test_a_working_order_blocks_a_second_entry(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(working_order_symbols=frozenset({"AMD"}))
        result = governor.approve(
            make_decision(symbol="AMD"),
            spread_with(100, symbol="AMD"),
            portfolio,
            NOW,
            "a-brand-new-client-order-id",
        )
        assert not result.approved
        assert ReasonCode.DUPLICATE_ORDER in result.reason_codes

    def test_a_different_limit_price_does_not_defeat_the_gate(self, governor) -> None:  # type: ignore[no-untyped-def]
        """The exact production failure: new id, same underlying, still blocked."""
        portfolio = make_portfolio(working_order_symbols=frozenset({"AMD"}))
        for client_order_id in ("alphamesh-AMD-BCS-aaaa", "alphamesh-AMD-BCS-bbbb"):
            result = governor.approve(
                make_decision(symbol="AMD"),
                spread_with(100, symbol="AMD"),
                portfolio,
                NOW,
                client_order_id,
            )
            assert not result.approved, client_order_id
            assert ReasonCode.DUPLICATE_ORDER in result.reason_codes

    def test_an_unrelated_symbol_is_unaffected(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(working_order_symbols=frozenset({"AMD"}))
        result = governor.approve(
            make_decision(symbol="SPY"), spread_with(100, symbol="SPY"), portfolio, NOW, "c"
        )
        assert result.approved

    def test_symbol_matching_is_case_insensitive(self, governor) -> None:  # type: ignore[no-untyped-def]
        portfolio = make_portfolio(working_order_symbols=frozenset({"amd"}))
        result = governor.approve(
            make_decision(symbol="AMD"), spread_with(100, symbol="AMD"), portfolio, NOW, "c"
        )
        assert not result.approved
