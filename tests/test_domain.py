"""Domain model invariants."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from alphamesh.models.domain import (
    TRADABLE_STRATEGIES,
    OptionQuote,
    OrderSide,
    PositionIntent,
    ReasonCode,
    RiskDecision,
    SpreadLeg,
    SpreadStructure,
    Strategy,
    TradeDecision,
)
from tests.conftest import NOW, make_contract, make_decision


class TestStrategySpace:
    def test_v1_offers_exactly_three_outcomes(self) -> None:
        assert set(Strategy) == {
            Strategy.NO_TRADE,
            Strategy.BULL_CALL_SPREAD,
            Strategy.BEAR_PUT_SPREAD,
        }

    def test_only_defined_risk_strategies_are_tradable(self) -> None:
        assert {
            Strategy.BULL_CALL_SPREAD,
            Strategy.BEAR_PUT_SPREAD,
        } == TRADABLE_STRATEGIES
        assert Strategy.NO_TRADE not in TRADABLE_STRATEGIES


class TestTradeDecision:
    def test_no_trade_must_explain_itself(self) -> None:
        with pytest.raises(ValidationError, match="no_trade_reason"):
            TradeDecision(
                decision_id="d",
                symbol="SPY",
                timestamp=NOW,
                regime="RANGE_BOUND",
                direction="NEUTRAL",
                strategy=Strategy.NO_TRADE,
                confidence=0.0,
                bull_score=0.0,
                bear_score=0.0,
                quant_score=0.0,
            )

    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            make_decision().model_copy(update={"confidence": 1.5}).model_validate(
                make_decision().model_dump() | {"confidence": 1.5}
            )

    def test_decisions_are_immutable(self) -> None:
        decision = make_decision()
        with pytest.raises(ValidationError):
            decision.strategy = Strategy.BEAR_PUT_SPREAD  # type: ignore[misc]


class TestOptionQuote:
    def test_relative_spread(self) -> None:
        quote = OptionQuote(
            bid=1.00, ask=1.10, bid_size=10, ask_size=10, quote_timestamp=NOW
        )
        assert quote.mid == pytest.approx(1.05)
        assert quote.absolute_spread == pytest.approx(0.10)
        assert quote.relative_spread == pytest.approx(0.10 / 1.05)

    def test_zero_mid_yields_infinite_relative_spread(self) -> None:
        quote = OptionQuote(bid=0.0, ask=0.0, bid_size=0, ask_size=0, quote_timestamp=NOW)
        assert quote.relative_spread == float("inf")

    def test_crossed_quote_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="crossed quote"):
            OptionQuote(bid=2.0, ask=1.0, bid_size=1, ask_size=1, quote_timestamp=NOW)


class TestSpreadStructure:
    def _legs(self, long_exp=date(2026, 9, 3), short_exp=date(2026, 9, 3), types=("call", "call")):  # type: ignore[no-untyped-def]
        from alphamesh.models.domain import OptionType

        return (
            SpreadLeg(
                contract=make_contract(
                    strike=770.0, expiration=long_exp, option_type=OptionType(types[0])
                ),
                side=OrderSide.BUY,
                position_intent=PositionIntent.BUY_TO_OPEN,
            ),
            SpreadLeg(
                contract=make_contract(
                    strike=775.0,
                    delta=0.3,
                    bid=1.0,
                    ask=1.05,
                    expiration=short_exp,
                    option_type=OptionType(types[1]),
                ),
                side=OrderSide.SELL,
                position_intent=PositionIntent.SELL_TO_OPEN,
            ),
        )

    def _build(self, **kw):  # type: ignore[no-untyped-def]
        long_leg, short_leg = kw.pop("legs", self._legs())
        return SpreadStructure(
            strategy=kw.pop("strategy", Strategy.BULL_CALL_SPREAD),
            symbol="SPY",
            expiration=date(2026, 9, 3),
            long_leg=long_leg,
            short_leg=short_leg,
            net_debit_cents=kw.pop("net_debit_cents", 200),
            strike_width_cents=kw.pop("strike_width_cents", 500),
            limit_price_cents=kw.pop("limit_price_cents", 200),
        )

    def test_no_trade_cannot_be_constructed_as_a_spread(self) -> None:
        with pytest.raises(ValidationError, match="not a constructible spread"):
            self._build(strategy=Strategy.NO_TRADE)

    def test_legs_must_share_an_expiration(self) -> None:
        with pytest.raises(ValidationError, match="one expiration"):
            self._build(legs=self._legs(short_exp=date(2026, 9, 10)))

    def test_legs_must_share_an_option_type(self) -> None:
        with pytest.raises(ValidationError, match="one option type"):
            self._build(legs=self._legs(types=("call", "put")))

    def test_debit_at_or_above_width_is_undefined_risk(self) -> None:
        with pytest.raises(ValidationError, match="net debit must be below strike width"):
            self._build(net_debit_cents=500, strike_width_cents=500)


class TestRiskDecision:
    def test_an_approval_must_size_something(self) -> None:
        with pytest.raises(ValidationError, match="must size at least one spread"):
            RiskDecision(
                approved=True,
                quantity=0,
                max_loss_cents=0,
                max_profit_cents=0,
                reason_codes=(ReasonCode.APPROVED,),
            )

    def test_a_rejection_cannot_claim_approval(self) -> None:
        with pytest.raises(ValidationError, match="must not carry APPROVED"):
            RiskDecision(
                approved=False,
                quantity=0,
                max_loss_cents=0,
                max_profit_cents=0,
                reason_codes=(ReasonCode.APPROVED,),
            )

    def test_a_rejection_always_carries_a_machine_readable_reason(self) -> None:
        decision = RiskDecision(
            approved=False,
            quantity=0,
            max_loss_cents=0,
            max_profit_cents=0,
            reason_codes=(ReasonCode.WIDE_SPREAD,),
        )
        assert decision.reason_codes
        assert all(isinstance(c, ReasonCode) for c in decision.reason_codes)


class TestReasonCodes:
    def test_every_documented_code_exists(self) -> None:
        for name in (
            "DAILY_DRAWDOWN_LIMIT",
            "CORRELATED_EXPOSURE",
            "MAX_PORTFOLIO_RISK",
            "MAX_POSITION_RISK",
            "STALE_QUOTES",
            "WIDE_SPREAD",
            "DUPLICATE_ORDER",
            "UNSUPPORTED_STRATEGY",
            "LIVE_TRADING_FORBIDDEN",
        ):
            assert ReasonCode(name)
