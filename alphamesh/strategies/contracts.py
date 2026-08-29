"""Deterministic vertical-spread construction from a real option chain.

Contract selection is entirely mechanical: filter for liquidity, keep only
contracts whose delta lands inside the configured band, then pick the pair
closest to the target deltas. Ties break on fixed keys so the same chain always
produces the same spread. The AI never touches this path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from alphamesh.config import RiskLimits, StrategyConfig
from alphamesh.models.domain import (
    OptionContractCandidate,
    OptionType,
    OrderSide,
    PositionIntent,
    ReasonCode,
    SpreadLeg,
    SpreadStructure,
    Strategy,
)
from alphamesh.risk.liquidity import filter_contracts
from alphamesh.risk.money import to_cents


@dataclass(frozen=True)
class SelectionResult:
    """Either a spread, or the reasons no spread could be built."""

    spread: SpreadStructure | None
    reason_codes: tuple[ReasonCode, ...]
    detail: str
    rejected_contracts: dict[str, tuple[ReasonCode, ...]]

    @property
    def ok(self) -> bool:
        return self.spread is not None


def _params(strategies: StrategyConfig, strategy: Strategy) -> dict:
    return (
        strategies.bull_call_spread
        if strategy is Strategy.BULL_CALL_SPREAD
        else strategies.bear_put_spread
    )


def _in_range(value: float, bounds: list[float]) -> bool:
    lo, hi = min(bounds), max(bounds)
    return lo <= value <= hi


def _limit_price_cents(
    long_c: OptionContractCandidate,
    short_c: OptionContractCandidate,
    aggressiveness: float,
) -> tuple[int, int]:
    """Return ``(limit_cents, natural_debit_cents)`` for the spread.

    ``mid`` is the fair value of the pair; ``natural`` is what it costs to cross
    both spreads immediately. The limit sits a configurable fraction of the way
    from mid toward natural, so the agent pays for fills without lifting the
    whole offer.
    """
    assert long_c.quote is not None and short_c.quote is not None
    mid = long_c.quote.mid - short_c.quote.mid
    natural = long_c.quote.ask - short_c.quote.bid
    limit = mid + max(0.0, min(1.0, aggressiveness)) * (natural - mid)
    return to_cents(limit), to_cents(natural)


def select_vertical_spread(
    strategy: Strategy,
    symbol: str,
    chain: list[OptionContractCandidate],
    strategies: StrategyConfig,
    limits: RiskLimits,
    now: datetime,
    as_of_date: date | None = None,
) -> SelectionResult:
    """Build the best eligible vertical debit spread, or explain why not.

    For a bull call spread the short strike sits above the long strike; for a
    bear put spread it sits below. Either way the result is a debit spread whose
    maximum loss is the premium paid.
    """
    if strategy not in (Strategy.BULL_CALL_SPREAD, Strategy.BEAR_PUT_SPREAD):
        return SelectionResult(None, (ReasonCode.UNSUPPORTED_STRATEGY,), str(strategy), {})

    today = as_of_date or now.date()
    params = _params(strategies, strategy)
    is_bull = strategy is Strategy.BULL_CALL_SPREAD
    want_type = OptionType.CALL if is_bull else OptionType.PUT

    typed = [c for c in chain if c.option_type is want_type]
    if not typed:
        return SelectionResult(
            None,
            (ReasonCode.NO_ELIGIBLE_CONTRACTS,),
            f"chain contained no {want_type} contracts for {symbol}",
            {},
        )

    eligible, rejected = filter_contracts(typed, limits, now)
    if not eligible:
        codes: list[ReasonCode] = []
        for reasons in rejected.values():
            for code in reasons:
                if code not in codes:
                    codes.append(code)
        if not codes:
            codes = [ReasonCode.NO_ELIGIBLE_CONTRACTS]
        return SelectionResult(
            None,
            tuple(codes),
            f"all {len(typed)} {want_type} contracts failed liquidity or quote gates",
            rejected,
        )

    by_expiry: dict[date, list[OptionContractCandidate]] = {}
    for contract in eligible:
        dte = contract.dte(today)
        if strategies.min_dte <= dte <= strategies.max_dte:
            by_expiry.setdefault(contract.expiration, []).append(contract)

    if not by_expiry:
        return SelectionResult(
            None,
            (ReasonCode.EXPIRATION_MISMATCH,),
            (
                f"no eligible {symbol} contracts within "
                f"{strategies.min_dte}-{strategies.max_dte} DTE of {today}"
            ),
            rejected,
        )

    long_range = params["long_delta_range"]
    short_range = params["short_delta_range"]
    long_target = float(params["long_delta_target"])
    short_target = float(params["short_delta_target"])
    min_width = float(params["min_strike_width"])
    max_width = float(params["max_strike_width"])

    candidates: list[tuple[tuple[float, ...], SpreadStructure]] = []
    seen_reasons: list[ReasonCode] = []

    for expiration in sorted(by_expiry):
        pool = by_expiry[expiration]
        longs = [
            c for c in pool if c.greeks.delta is not None and _in_range(c.greeks.delta, long_range)
        ]
        shorts = [
            c
            for c in pool
            if c.greeks.delta is not None and _in_range(c.greeks.delta, short_range)
        ]
        if not longs or not shorts:
            if ReasonCode.NO_ELIGIBLE_CONTRACTS not in seen_reasons:
                seen_reasons.append(ReasonCode.NO_ELIGIBLE_CONTRACTS)
            continue

        for long_c in longs:
            for short_c in shorts:
                width = (
                    short_c.strike - long_c.strike if is_bull else long_c.strike - short_c.strike
                )
                if width < min_width or width > max_width:
                    continue
                limit_cents, natural_cents = _limit_price_cents(
                    long_c, short_c, strategies.limit_price_aggressiveness
                )
                width_cents = to_cents(width)
                if limit_cents <= 0 or natural_cents <= 0:
                    continue
                # A debit at or above the width has no defined profit and its
                # loss is not bounded by the premium in any useful sense.
                if limit_cents >= width_cents or natural_cents >= width_cents:
                    if ReasonCode.UNDEFINED_RISK not in seen_reasons:
                        seen_reasons.append(ReasonCode.UNDEFINED_RISK)
                    continue
                ratio = limit_cents / width_cents
                if ratio > strategies.max_debit_to_width_ratio:
                    if ReasonCode.POOR_REWARD_RISK not in seen_reasons:
                        seen_reasons.append(ReasonCode.POOR_REWARD_RISK)
                    continue

                spread = SpreadStructure(
                    strategy=strategy,
                    symbol=symbol,
                    expiration=expiration,
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
                    net_debit_cents=to_cents(
                        long_c.quote.mid - short_c.quote.mid  # type: ignore[union-attr]
                    )
                    or 1,
                    strike_width_cents=width_cents,
                    limit_price_cents=limit_cents,
                )
                delta_error = abs(
                    (long_c.greeks.delta or 0.0) - long_target
                ) + abs((short_c.greeks.delta or 0.0) - short_target)
                # Deterministic ordering: nearest expiry, closest to target
                # deltas, then cheapest relative to width, then strike.
                key = (
                    float(spread.expiration.toordinal()),
                    round(delta_error, 6),
                    round(ratio, 6),
                    long_c.strike,
                    short_c.strike,
                )
                candidates.append((key, spread))

    if not candidates:
        codes = seen_reasons or [ReasonCode.NO_ELIGIBLE_CONTRACTS]
        return SelectionResult(
            None,
            tuple(codes),
            f"no {strategy} pair for {symbol} satisfied delta, width and pricing rules",
            rejected,
        )

    candidates.sort(key=lambda item: item[0])
    return SelectionResult(candidates[0][1], (), "selected", rejected)


__all__ = ["SelectionResult", "select_vertical_spread"]
