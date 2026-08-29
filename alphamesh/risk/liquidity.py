"""Per-contract quote-quality and liquidity gates.

These run before a spread is ever constructed. A contract that fails any check
is dropped from the candidate pool with a machine-readable reason, so the
journal records exactly why a chain produced no trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from alphamesh.config import RiskLimits
from alphamesh.models.domain import OptionContractCandidate, ReasonCode


@dataclass(frozen=True)
class ContractCheck:
    ok: bool
    reason_codes: tuple[ReasonCode, ...]

    @property
    def first_reason(self) -> ReasonCode | None:
        return self.reason_codes[0] if self.reason_codes else None


def quote_age_seconds(contract: OptionContractCandidate, now: datetime) -> float | None:
    if contract.quote is None:
        return None
    return (now - contract.quote.quote_timestamp).total_seconds()


def evaluate_contract(
    contract: OptionContractCandidate,
    limits: RiskLimits,
    now: datetime,
    require_greeks: bool = True,
) -> ContractCheck:
    """Run every per-contract gate. Reasons accumulate rather than short-circuit
    so the journal shows all the ways a contract was unsuitable."""
    reasons: list[ReasonCode] = []

    quote = contract.quote
    if quote is None:
        reasons.append(ReasonCode.NO_QUOTE)
    else:
        if quote.bid <= 0 or quote.ask <= 0:
            reasons.append(ReasonCode.NO_QUOTE)
        if quote.bid < limits.min_option_bid:
            reasons.append(ReasonCode.ILLIQUID_CONTRACT)
        age = (now - quote.quote_timestamp).total_seconds()
        if age > limits.max_quote_age_seconds or age < -limits.max_quote_age_seconds:
            reasons.append(ReasonCode.STALE_QUOTES)
        if quote.bid > 0 and quote.ask > 0:
            if quote.relative_spread > limits.max_relative_bid_ask_spread:
                reasons.append(ReasonCode.WIDE_SPREAD)
            if quote.absolute_spread > limits.max_absolute_bid_ask_spread:
                reasons.append(ReasonCode.WIDE_SPREAD)
        if (
            min(quote.bid_size, quote.ask_size) < limits.min_top_of_book_size
            and ReasonCode.ILLIQUID_CONTRACT not in reasons
        ):
            reasons.append(ReasonCode.ILLIQUID_CONTRACT)

    if (
        contract.day_volume < limits.min_contract_day_volume
        and ReasonCode.ILLIQUID_CONTRACT not in reasons
    ):
        reasons.append(ReasonCode.ILLIQUID_CONTRACT)

    if require_greeks and contract.greeks.delta is None:
        reasons.append(ReasonCode.MISSING_GREEKS)

    # De-duplicate while preserving first-seen order.
    seen: list[ReasonCode] = []
    for code in reasons:
        if code not in seen:
            seen.append(code)
    return ContractCheck(ok=not seen, reason_codes=tuple(seen))


def filter_contracts(
    contracts: list[OptionContractCandidate],
    limits: RiskLimits,
    now: datetime,
) -> tuple[list[OptionContractCandidate], dict[str, tuple[ReasonCode, ...]]]:
    """Split a chain into eligible contracts and a rejection map for the journal."""
    eligible: list[OptionContractCandidate] = []
    rejected: dict[str, tuple[ReasonCode, ...]] = {}
    for contract in contracts:
        check = evaluate_contract(contract, limits, now)
        if check.ok:
            eligible.append(contract)
        else:
            rejected[contract.symbol] = check.reason_codes
    return eligible, rejected


__all__ = ["ContractCheck", "evaluate_contract", "filter_contracts", "quote_age_seconds"]
