"""The Risk Governor.

Authoritative and non-bypassable. Nothing reaches the broker without an
approving ``RiskDecision`` from this class, and this class reads its limits from
``risk.yaml`` at construction and never mutates them. No AI output is an input
to any limit: the judge's confidence can only select between two pre-configured
caps, both of which are themselves bounded by an absolute ceiling.

Every rejection carries at least one machine-readable ``ReasonCode``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from alphamesh.config import RiskLimits
from alphamesh.models.domain import (
    TRADABLE_STRATEGIES,
    ReasonCode,
    RiskDecision,
    SpreadStructure,
    Strategy,
    TradeDecision,
)
from alphamesh.risk.circuit_breaker import evaluate_circuit_breaker
from alphamesh.risk.liquidity import evaluate_contract
from alphamesh.risk.money import to_dollars
from alphamesh.risk.portfolio import PortfolioState
from alphamesh.risk.sizing import size_spread

log = logging.getLogger(__name__)

REQUIRED_OPTIONS_LEVEL = 3
"""Alpaca options level 3 is required to open multi-leg spreads."""


@dataclass
class _Verdict:
    reasons: list[ReasonCode]
    details: list[str]
    checks: list[str]

    def fail(self, code: ReasonCode, detail: str) -> None:
        if code not in self.reasons:
            self.reasons.append(code)
        self.details.append(detail)


class RiskGovernor:
    """Deterministic gate between an intended spread and the broker."""

    def __init__(self, limits: RiskLimits, paper_confirmed: bool) -> None:
        self._limits = limits
        self._paper_confirmed = paper_confirmed

    @property
    def limits(self) -> RiskLimits:
        """Read-only accessor. The governor never exposes a mutable handle."""
        return self._limits

    def approve(
        self,
        decision: TradeDecision,
        spread: SpreadStructure,
        portfolio: PortfolioState,
        now: datetime,
        client_order_id: str,
        known_client_order_ids: frozenset[str] = frozenset(),
    ) -> RiskDecision:
        v = _Verdict(reasons=[], details=[], checks=[])
        limits = self._limits

        # 1. Paper mode. Checked first and unconditionally.
        if not self._paper_confirmed:
            v.fail(
                ReasonCode.LIVE_TRADING_FORBIDDEN,
                "paper mode was not positively confirmed at startup",
            )
            return self._reject(v)
        v.checks.append("paper_mode")

        # 2. Account state.
        account = portfolio.account
        if not account.is_tradeable:
            v.fail(
                ReasonCode.ACCOUNT_NOT_TRADEABLE,
                f"account status {account.status} / blocked flags set",
            )
        if account.options_trading_level < REQUIRED_OPTIONS_LEVEL:
            v.fail(
                ReasonCode.OPTIONS_LEVEL_INSUFFICIENT,
                (
                    f"options level {account.options_trading_level} < "
                    f"{REQUIRED_OPTIONS_LEVEL} required for spreads"
                ),
            )
        v.checks.append("account_status")

        # 3. Strategy allowlist.
        allowed = {s.upper() for s in limits.allowed_strategies}
        if decision.strategy not in TRADABLE_STRATEGIES or decision.strategy.value not in allowed:
            v.fail(
                ReasonCode.UNSUPPORTED_STRATEGY,
                f"{decision.strategy} is not in the risk allowlist",
            )
        if spread.strategy is not decision.strategy:
            v.fail(
                ReasonCode.UNSUPPORTED_STRATEGY,
                f"spread strategy {spread.strategy} does not match decision "
                f"{decision.strategy}",
            )
        v.checks.append("strategy_allowlist")

        # 4. Defined risk. A vertical debit spread whose premium meets or
        #    exceeds the strike width has no bounded payoff worth taking.
        if spread.limit_price_cents >= spread.strike_width_cents:
            v.fail(
                ReasonCode.UNDEFINED_RISK,
                (
                    f"limit {spread.limit_price_cents}c >= width "
                    f"{spread.strike_width_cents}c"
                ),
            )
        if spread.long_leg.contract.expiration != spread.short_leg.contract.expiration:
            v.fail(ReasonCode.EXPIRATION_MISMATCH, "legs carry different expirations")
        v.checks.append("defined_risk")

        # 5. Live re-check of both legs' quotes at approval time. A chain that
        #    was fresh during selection can be stale by the time we get here.
        for leg in (spread.long_leg, spread.short_leg):
            check = evaluate_contract(leg.contract, limits, now)
            if not check.ok:
                for code in check.reason_codes:
                    v.fail(code, f"{leg.contract.symbol}: {code}")
        v.checks.append("leg_liquidity")

        # 6. Duplicate protection.
        if client_order_id in known_client_order_ids or client_order_id in (
            portfolio.open_client_order_ids
        ):
            v.fail(
                ReasonCode.DUPLICATE_ORDER,
                f"client_order_id {client_order_id} already exists",
            )
        if portfolio.has_open_position_for(decision.symbol):
            v.fail(
                ReasonCode.DUPLICATE_ORDER,
                f"an open position already exists for {decision.symbol}",
            )
        # An unfilled working order is exposure in waiting. Without this the
        # id-based check above is defeated by a moving limit price: every cycle
        # mints a fresh client_order_id for the same spread and stacks another
        # live order on top of the last one.
        if portfolio.has_working_order_for(decision.symbol):
            v.fail(
                ReasonCode.DUPLICATE_ORDER,
                f"a working order is already live for {decision.symbol}",
            )
        v.checks.append("duplicate_order")

        # 7. Daily circuit breaker.
        breaker = evaluate_circuit_breaker(portfolio, limits)
        if breaker.tripped:
            v.fail(ReasonCode.DAILY_DRAWDOWN_LIMIT, breaker.detail)
        v.checks.append("daily_circuit_breaker")

        # 8. Open-position count.
        if portfolio.open_position_count >= limits.max_open_positions:
            v.fail(
                ReasonCode.MAX_OPEN_POSITIONS,
                (
                    f"{portfolio.open_position_count} open positions at the "
                    f"{limits.max_open_positions} limit"
                ),
            )
        v.checks.append("max_open_positions")

        # 9. Correlated exposure. SPY and QQQ share one bucket.
        group = limits.group_for(decision.symbol)
        group_positions = portfolio.positions_in_group(limits, group)
        group_risk = portfolio.defined_risk_in_group_cents(limits, group)
        if group is not None and len(group_positions) >= limits.max_positions_per_correlation_group:
            v.fail(
                ReasonCode.CORRELATED_EXPOSURE,
                (
                    f"{len(group_positions)} positions already open in correlated "
                    f"group {group!r} (limit {limits.max_positions_per_correlation_group})"
                ),
            )
        v.checks.append("correlated_exposure")

        # Headroom left under each aggregate cap, so sizing can shrink to fit.
        portfolio_cap_cents = round(limits.max_portfolio_defined_risk * 100)
        portfolio_headroom = portfolio_cap_cents - portfolio.total_defined_risk_cents
        group_cap_cents = round(limits.max_defined_risk_per_correlation_group * 100)
        group_headroom = (
            group_cap_cents - group_risk if group is not None else portfolio_headroom
        )
        headroom = min(portfolio_headroom, group_headroom)
        # Which aggregate cap is actually binding, for reason-code attribution.
        binding_code = (
            ReasonCode.MAX_PORTFOLIO_RISK
            if portfolio_headroom <= group_headroom
            else ReasonCode.CORRELATED_EXPOSURE
        )

        if headroom <= 0:
            v.fail(
                binding_code,
                (
                    f"no defined-risk headroom left: portfolio "
                    f"{to_dollars(portfolio.total_defined_risk_cents):.2f} of "
                    f"{limits.max_portfolio_defined_risk:.2f}"
                    + (
                        f", correlated group {group!r} "
                        f"{to_dollars(group_risk):.2f} of "
                        f"{limits.max_defined_risk_per_correlation_group:.2f}"
                        if group is not None
                        else ""
                    )
                ),
            )
        v.checks.append("portfolio_risk")

        # 10. Sizing. Confidence selects between configured caps; nothing else.
        sizing = size_spread(
            spread,
            decision.confidence,
            limits,
            available_risk_cents=max(0, headroom),
        )
        if sizing.quantity <= 0:
            # Name the constraint that actually bound, so the journal explains
            # *why* nothing could be sized rather than only that nothing was.
            if sizing.per_contract_loss_cents > sizing.cap_cents:
                code = ReasonCode.MAX_POSITION_RISK
            elif headroom < sizing.per_contract_loss_cents:
                code = binding_code
            else:
                code = ReasonCode.SIZE_ROUNDS_TO_ZERO
            v.fail(code, sizing.detail)
        v.checks.append("position_sizing")

        max_loss_cents = sizing.max_loss_cents
        max_profit_cents = spread.max_profit_cents(sizing.quantity)

        # 11. Absolute per-trade ceiling, re-asserted on the final number.
        absolute_cap = round(limits.absolute_max_defined_loss * 100)
        if max_loss_cents > absolute_cap:
            v.fail(
                ReasonCode.MAX_POSITION_RISK,
                (
                    f"defined loss {to_dollars(max_loss_cents):.2f} exceeds the "
                    f"absolute {limits.absolute_max_defined_loss:.2f} ceiling"
                ),
            )
        if max_loss_cents > sizing.cap_cents:
            v.fail(
                ReasonCode.MAX_POSITION_RISK,
                (
                    f"defined loss {to_dollars(max_loss_cents):.2f} exceeds the "
                    f"{to_dollars(sizing.cap_cents):.2f} per-trade cap"
                ),
            )
        v.checks.append("per_trade_cap")

        # 12. Portfolio and group caps, re-asserted including this trade.
        if portfolio.total_defined_risk_cents + max_loss_cents > portfolio_cap_cents:
            v.fail(
                ReasonCode.MAX_PORTFOLIO_RISK,
                (
                    f"portfolio defined risk would reach "
                    f"{to_dollars(portfolio.total_defined_risk_cents + max_loss_cents):.2f} "
                    f"against a {limits.max_portfolio_defined_risk:.2f} cap"
                ),
            )
        if group is not None and group_risk + max_loss_cents > group_cap_cents:
            v.fail(
                ReasonCode.CORRELATED_EXPOSURE,
                (
                    f"correlated group {group!r} defined risk would reach "
                    f"{to_dollars(group_risk + max_loss_cents):.2f} against a "
                    f"{limits.max_defined_risk_per_correlation_group:.2f} cap"
                ),
            )
        v.checks.append("aggregate_caps")

        # 13. Buying power.
        required = to_dollars(max_loss_cents) * limits.min_buying_power_multiple
        available = max(account.options_buying_power, 0.0)
        if available < required:
            v.fail(
                ReasonCode.INSUFFICIENT_BUYING_POWER,
                (
                    f"options buying power {available:.2f} below the "
                    f"{required:.2f} required"
                ),
            )
        v.checks.append("buying_power")

        if v.reasons:
            return self._reject(v)

        return RiskDecision(
            approved=True,
            quantity=sizing.quantity,
            max_loss_cents=max_loss_cents,
            max_profit_cents=max_profit_cents,
            reason_codes=(ReasonCode.APPROVED,),
            detail=(
                f"approved {sizing.quantity} x {decision.strategy} on "
                f"{decision.symbol}: defined loss {to_dollars(max_loss_cents):.2f}, "
                f"defined profit {to_dollars(max_profit_cents):.2f}"
            ),
            checks_run=tuple(v.checks),
        )

    def reject_strategy(self, strategy: Strategy, detail: str) -> RiskDecision:
        """Explicit rejection helper for callers that never build a spread."""
        return RiskDecision(
            approved=False,
            quantity=0,
            max_loss_cents=0,
            max_profit_cents=0,
            reason_codes=(ReasonCode.UNSUPPORTED_STRATEGY,),
            detail=f"{strategy}: {detail}",
            checks_run=("strategy_allowlist",),
        )

    @staticmethod
    def _reject(v: _Verdict) -> RiskDecision:
        return RiskDecision(
            approved=False,
            quantity=0,
            max_loss_cents=0,
            max_profit_cents=0,
            reason_codes=tuple(v.reasons),
            detail="; ".join(v.details),
            checks_run=tuple(v.checks),
        )


__all__ = ["REQUIRED_OPTIONS_LEVEL", "RiskGovernor"]
