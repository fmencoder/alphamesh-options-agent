"""Option contract selection against the real captured Alpaca chain."""

from __future__ import annotations

from datetime import timedelta

import pytest

from alphamesh.models.domain import OptionType, ReasonCode, Strategy
from alphamesh.risk.liquidity import evaluate_contract
from alphamesh.strategies.bear_put import build_bear_put_spread
from alphamesh.strategies.bull_call import build_bull_call_spread
from alphamesh.strategies.contracts import select_vertical_spread
from tests.conftest import TODAY, make_contract


class TestRealBullCallSpread:
    def test_builds_from_the_captured_alpaca_chain(
        self, spy_calls, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        result = build_bull_call_spread(
            "SPY", spy_calls, strategies, limits, now, as_of_date=TODAY
        )
        assert result.ok, result.detail
        spread = result.spread
        long_c, short_c = spread.long_leg.contract, spread.short_leg.contract

        # Real OCC symbols returned by Alpaca, not constructed by us.
        assert long_c.symbol.startswith("SPY26") and long_c.symbol[-9] == "C"
        assert short_c.strike > long_c.strike
        assert long_c.expiration == short_c.expiration

    def test_selected_deltas_land_in_the_configured_bands(
        self, spy_calls, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        result = build_bull_call_spread(
            "SPY", spy_calls, strategies, limits, now, as_of_date=TODAY
        )
        long_delta = result.spread.long_leg.contract.greeks.delta
        short_delta = result.spread.short_leg.contract.greeks.delta
        lo, hi = strategies.bull_call_spread["long_delta_range"]
        assert lo <= long_delta <= hi
        lo, hi = strategies.bull_call_spread["short_delta_range"]
        assert lo <= short_delta <= hi

    def test_maximum_loss_is_defined_and_bounded_by_the_width(
        self, spy_calls, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        spread = build_bull_call_spread(
            "SPY", spy_calls, strategies, limits, now, as_of_date=TODAY
        ).spread
        assert 0 < spread.limit_price_cents < spread.strike_width_cents
        assert spread.max_loss_cents(1) + spread.max_profit_cents(1) == (
            spread.strike_width_cents * 100
        )

    def test_selection_is_deterministic(
        self, spy_calls, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        first = build_bull_call_spread("SPY", spy_calls, strategies, limits, now, TODAY)
        second = build_bull_call_spread("SPY", spy_calls, strategies, limits, now, TODAY)
        assert (
            first.spread.long_leg.contract.symbol == second.spread.long_leg.contract.symbol
        )
        assert first.spread.limit_price_cents == second.spread.limit_price_cents

    def test_dte_window_is_respected(
        self, spy_calls, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        spread = build_bull_call_spread(
            "SPY", spy_calls, strategies, limits, now, TODAY
        ).spread
        dte = (spread.expiration - TODAY).days
        assert strategies.min_dte <= dte <= strategies.max_dte


class TestRealBearPutSpread:
    def test_builds_from_the_captured_alpaca_chain(
        self, spy_puts, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        result = build_bear_put_spread(
            "SPY", spy_puts, strategies, limits, now, as_of_date=TODAY
        )
        assert result.ok, result.detail
        long_c, short_c = (
            result.spread.long_leg.contract,
            result.spread.short_leg.contract,
        )
        # In a bear put spread the short strike sits BELOW the long strike.
        assert short_c.strike < long_c.strike
        assert long_c.option_type is OptionType.PUT

    def test_selected_deltas_are_negative_and_in_band(
        self, spy_puts, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        spread = build_bear_put_spread(
            "SPY", spy_puts, strategies, limits, now, TODAY
        ).spread
        assert spread.long_leg.contract.greeks.delta < 0
        assert spread.short_leg.contract.greeks.delta < 0
        lo, hi = sorted(strategies.bear_put_spread["long_delta_range"])
        assert lo <= spread.long_leg.contract.greeks.delta <= hi


class TestQqqChain:
    def test_second_universe_symbol_also_builds(
        self, qqq_calls, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        result = build_bull_call_spread(
            "QQQ", qqq_calls, strategies, limits, now, TODAY
        )
        assert result.ok, result.detail
        assert result.spread.long_leg.contract.underlying == "QQQ"


class TestNoEligibleContracts:
    def test_empty_chain_returns_no_trade_reason(self, strategies, limits, now) -> None:  # type: ignore[no-untyped-def]
        result = build_bull_call_spread("SPY", [], strategies, limits, now, TODAY)
        assert not result.ok
        assert ReasonCode.NO_ELIGIBLE_CONTRACTS in result.reason_codes

    def test_wrong_option_type_is_not_substituted(
        self, spy_puts, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        """A bull call spread is never built out of puts."""
        result = build_bull_call_spread("SPY", spy_puts, strategies, limits, now, TODAY)
        assert not result.ok
        assert ReasonCode.NO_ELIGIBLE_CONTRACTS in result.reason_codes

    def test_stale_chain_is_refused(self, spy_calls, strategies, limits, now) -> None:  # type: ignore[no-untyped-def]
        stale_now = now + timedelta(hours=6)
        result = build_bull_call_spread(
            "SPY", spy_calls, strategies, limits, stale_now, TODAY
        )
        assert not result.ok
        assert ReasonCode.STALE_QUOTES in result.reason_codes

    def test_expirations_outside_the_window_are_refused(
        self, spy_calls, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        far_future = TODAY.replace(year=TODAY.year + 1)
        result = build_bull_call_spread(
            "SPY", spy_calls, strategies, limits, now, far_future
        )
        assert not result.ok

    def test_unsupported_strategy_is_refused(self, spy_calls, strategies, limits, now) -> None:  # type: ignore[no-untyped-def]
        result = select_vertical_spread(
            Strategy.NO_TRADE, "SPY", spy_calls, strategies, limits, now, TODAY
        )
        assert not result.ok
        assert ReasonCode.UNSUPPORTED_STRATEGY in result.reason_codes


class TestContractLiquidityGates:
    def test_healthy_contract_passes(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        assert evaluate_contract(make_contract(), limits, now).ok

    def test_missing_quote_is_rejected(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        check = evaluate_contract(make_contract(with_quote=False), limits, now)
        assert ReasonCode.NO_QUOTE in check.reason_codes

    def test_zero_bid_is_rejected(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        check = evaluate_contract(make_contract(bid=0.0, ask=0.10), limits, now)
        assert ReasonCode.NO_QUOTE in check.reason_codes

    def test_stale_quote_is_rejected(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        old = make_contract(quote_at=now - timedelta(minutes=30))
        check = evaluate_contract(old, limits, now)
        assert ReasonCode.STALE_QUOTES in check.reason_codes

    def test_quote_from_the_future_is_also_stale(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        """A quote timestamped ahead of us means the clocks disagree; refuse it."""
        future = make_contract(quote_at=now + timedelta(minutes=30))
        check = evaluate_contract(future, limits, now)
        assert ReasonCode.STALE_QUOTES in check.reason_codes

    def test_wide_relative_spread_is_rejected(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        check = evaluate_contract(make_contract(bid=1.00, ask=2.00), limits, now)
        assert ReasonCode.WIDE_SPREAD in check.reason_codes

    def test_wide_absolute_spread_is_rejected(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        check = evaluate_contract(make_contract(bid=10.00, ask=11.00), limits, now)
        assert ReasonCode.WIDE_SPREAD in check.reason_codes

    def test_missing_greeks_is_rejected(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        check = evaluate_contract(make_contract(delta=None), limits, now)
        assert ReasonCode.MISSING_GREEKS in check.reason_codes

    def test_thin_contract_is_rejected(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        check = evaluate_contract(make_contract(day_volume=1), limits, now)
        assert ReasonCode.ILLIQUID_CONTRACT in check.reason_codes

    def test_penny_bid_is_rejected(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        check = evaluate_contract(make_contract(bid=0.01, ask=0.02), limits, now)
        assert ReasonCode.ILLIQUID_CONTRACT in check.reason_codes

    def test_empty_book_is_rejected(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        check = evaluate_contract(make_contract(bid_size=0, ask_size=0), limits, now)
        assert ReasonCode.ILLIQUID_CONTRACT in check.reason_codes

    def test_reasons_accumulate_rather_than_short_circuit(self, limits, now) -> None:  # type: ignore[no-untyped-def]
        bad = make_contract(
            bid=1.0, ask=2.0, delta=None, quote_at=now - timedelta(hours=2), day_volume=0
        )
        codes = set(evaluate_contract(bad, limits, now).reason_codes)
        assert {
            ReasonCode.STALE_QUOTES,
            ReasonCode.WIDE_SPREAD,
            ReasonCode.MISSING_GREEKS,
            ReasonCode.ILLIQUID_CONTRACT,
        } <= codes


class TestPricing:
    def test_limit_sits_between_mid_and_natural(
        self, spy_calls, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        spread = build_bull_call_spread(
            "SPY", spy_calls, strategies, limits, now, TODAY
        ).spread
        long_q = spread.long_leg.contract.quote
        short_q = spread.short_leg.contract.quote
        mid = long_q.mid - short_q.mid
        natural = long_q.ask - short_q.bid
        assert mid * 100 <= spread.limit_price_cents <= natural * 100 + 1

    def test_debit_to_width_ratio_is_enforced(
        self, spy_calls, strategies, limits, now
    ) -> None:  # type: ignore[no-untyped-def]
        tight = strategies.model_copy(update={"max_debit_to_width_ratio": 0.05})
        result = build_bull_call_spread("SPY", spy_calls, tight, limits, now, TODAY)
        assert not result.ok
        assert ReasonCode.POOR_REWARD_RISK in result.reason_codes


@pytest.mark.parametrize("underlying", ["SPY", "QQQ"])
def test_no_contract_symbol_is_ever_invented(
    underlying, capture_chain, strategies, limits, now
) -> None:  # type: ignore[no-untyped-def]
    """Every selected symbol must exist verbatim in the captured Alpaca chain."""
    chain = capture_chain.chain(underlying, OptionType.CALL, TODAY, 2, 10)
    result = build_bull_call_spread(underlying, chain, strategies, limits, now, TODAY)
    if not result.ok:
        pytest.skip(f"no eligible spread for {underlying}: {result.detail}")
    known = {c.symbol for c in chain}
    assert result.spread.long_leg.contract.symbol in known
    assert result.spread.short_leg.contract.symbol in known
