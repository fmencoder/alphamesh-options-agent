"""Strongly typed domain model for the AlphaMesh trading lifecycle.

Every object that crosses a module boundary is one of these models. The
AI reasoning council may only produce ``AIArgument`` and ``JudgeVerdict``;
it can never construct an ``OrderIntent``, a ``SpreadStructure`` or a
``RiskDecision``. Those are built exclusively by deterministic code.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OPTION_MULTIPLIER = 100
"""Shares of underlying per option contract. Fixed for standard US equity options."""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Direction(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Regime(StrEnum):
    BULLISH_TREND = "BULLISH_TREND"
    BEARISH_TREND = "BEARISH_TREND"
    RANGE_BOUND = "RANGE_BOUND"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    UNSTABLE = "UNSTABLE"
    UNKNOWN = "UNKNOWN"


class Strategy(StrEnum):
    """The complete v1 strategy space. Both tradable strategies are
    defined-risk vertical debit spreads: no naked options, no undefined risk."""

    NO_TRADE = "NO_TRADE"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"


TRADABLE_STRATEGIES: frozenset[Strategy] = frozenset(
    {Strategy.BULL_CALL_SPREAD, Strategy.BEAR_PUT_SPREAD}
)


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class PositionIntent(StrEnum):
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_OPEN = "sell_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_CLOSE = "sell_to_close"


class TradeState(StrEnum):
    """Execution state machine states. See ``alphamesh.execution.state_machine``."""

    DISCOVERED = "DISCOVERED"
    ANALYZED = "ANALYZED"
    AI_APPROVED = "AI_APPROVED"
    RISK_APPROVED = "RISK_APPROVED"
    CONSTRUCTED = "CONSTRUCTED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    MONITORING = "MONITORING"
    EXIT_REQUESTED = "EXIT_REQUESTED"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


TERMINAL_STATES: frozenset[TradeState] = frozenset(
    {TradeState.CLOSED, TradeState.REJECTED, TradeState.FAILED}
)


class ReasonCode(StrEnum):
    """Machine-readable outcome codes. Every rejection carries at least one."""

    # Safety
    LIVE_TRADING_FORBIDDEN = "LIVE_TRADING_FORBIDDEN"
    ACCOUNT_NOT_TRADEABLE = "ACCOUNT_NOT_TRADEABLE"
    OPTIONS_LEVEL_INSUFFICIENT = "OPTIONS_LEVEL_INSUFFICIENT"
    # Risk
    MAX_POSITION_RISK = "MAX_POSITION_RISK"
    MAX_PORTFOLIO_RISK = "MAX_PORTFOLIO_RISK"
    MAX_OPEN_POSITIONS = "MAX_OPEN_POSITIONS"
    DAILY_DRAWDOWN_LIMIT = "DAILY_DRAWDOWN_LIMIT"
    CORRELATED_EXPOSURE = "CORRELATED_EXPOSURE"
    INSUFFICIENT_BUYING_POWER = "INSUFFICIENT_BUYING_POWER"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    UNSUPPORTED_STRATEGY = "UNSUPPORTED_STRATEGY"
    UNDEFINED_RISK = "UNDEFINED_RISK"
    SIZE_ROUNDS_TO_ZERO = "SIZE_ROUNDS_TO_ZERO"
    # Liquidity / data quality
    STALE_QUOTES = "STALE_QUOTES"
    WIDE_SPREAD = "WIDE_SPREAD"
    NO_QUOTE = "NO_QUOTE"
    ILLIQUID_CONTRACT = "ILLIQUID_CONTRACT"
    MISSING_GREEKS = "MISSING_GREEKS"
    NO_ELIGIBLE_CONTRACTS = "NO_ELIGIBLE_CONTRACTS"
    EXPIRATION_MISMATCH = "EXPIRATION_MISMATCH"
    POOR_REWARD_RISK = "POOR_REWARD_RISK"
    INSUFFICIENT_MARKET_DATA = "INSUFFICIENT_MARKET_DATA"
    # Signal / AI
    QUANT_SCORE_BELOW_THRESHOLD = "QUANT_SCORE_BELOW_THRESHOLD"
    REGIME_UNKNOWN = "REGIME_UNKNOWN"
    REGIME_UNSTABLE = "REGIME_UNSTABLE"
    LOW_JUDGE_CONFIDENCE = "LOW_JUDGE_CONFIDENCE"
    AI_MALFORMED_OUTPUT = "AI_MALFORMED_OUTPUT"
    AI_UNSUPPORTED_STRATEGY = "AI_UNSUPPORTED_STRATEGY"
    AI_UNAVAILABLE = "AI_UNAVAILABLE"
    DIRECTION_CONFLICT = "DIRECTION_CONFLICT"
    # Broker truth. The journal can diverge from the account; for blocking new
    # exposure the broker is authoritative.
    BROKER_OPEN_POSITION = "BROKER_OPEN_POSITION"
    BROKER_WORKING_ORDER = "BROKER_WORKING_ORDER"
    # Market state
    MARKET_CLOSED = "MARKET_CLOSED"
    TOO_CLOSE_TO_CLOSE = "TOO_CLOSE_TO_CLOSE"
    # Approval
    APPROVED = "APPROVED"


class ExitReason(StrEnum):
    PROFIT_TARGET = "PROFIT_TARGET"
    MAX_LOSS = "MAX_LOSS"
    MAX_HOLDING_TIME = "MAX_HOLDING_TIME"
    SIGNAL_INVALIDATED = "SIGNAL_INVALIDATED"
    END_OF_DAY = "END_OF_DAY"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MANUAL = "MANUAL"
    EXPIRED = "EXPIRED"


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
class Bar(_Base):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None


class MarketSnapshot(_Base):
    """Point-in-time view of one underlying, plus the bar window behind it."""

    symbol: str
    as_of: datetime
    last_price: float = Field(gt=0)
    bid: float | None = None
    ask: float | None = None
    session_open: float | None = None
    prev_close: float | None = None
    bars: tuple[Bar, ...] = ()

    @property
    def bar_count(self) -> int:
        return len(self.bars)


class OptionQuote(_Base):
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    bid_size: int = Field(ge=0)
    ask_size: int = Field(ge=0)
    quote_timestamp: datetime

    @model_validator(mode="after")
    def _check_crossed(self) -> OptionQuote:
        if self.ask and self.bid and self.ask < self.bid:
            raise ValueError("crossed quote: ask < bid")
        return self

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def absolute_spread(self) -> float:
        return self.ask - self.bid

    @property
    def relative_spread(self) -> float:
        """Bid-ask spread as a fraction of mid. Infinite when mid is zero."""
        mid = self.mid
        if mid <= 0:
            return float("inf")
        return self.absolute_spread / mid


class Greeks(_Base):
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    implied_volatility: float | None = None


class OptionContractCandidate(_Base):
    """One real, tradable option contract as returned by Alpaca."""

    symbol: str
    underlying: str
    expiration: date
    option_type: OptionType
    strike: float = Field(gt=0)
    quote: OptionQuote | None = None
    greeks: Greeks = Greeks()
    day_volume: int = 0
    open_interest: int | None = None

    @field_validator("symbol")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("contract symbol must not be empty")
        return v

    def dte(self, as_of: date) -> int:
        return (self.expiration - as_of).days


# --------------------------------------------------------------------------- #
# Intelligence
# --------------------------------------------------------------------------- #
class QuantSignal(_Base):
    """Deterministic feature vector and opportunity score for one symbol."""

    symbol: str
    as_of: datetime
    features: dict[str, float]
    quant_score: float = Field(ge=0.0, le=1.0)
    directional_bias: Direction
    passes_gate: bool
    reason_codes: tuple[ReasonCode, ...] = ()


class RegimeAssessment(_Base):
    symbol: str
    as_of: datetime
    regime: Regime
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float]
    risk_flags: tuple[str, ...] = ()

    @property
    def favors_no_trade(self) -> bool:
        return self.regime in (Regime.UNKNOWN, Regime.UNSTABLE)


class AIArgument(_Base):
    """One side of the reasoning council's debate. Advisory only."""

    role: str
    stance: Direction
    thesis: str
    key_points: tuple[str, ...]
    conviction: float = Field(ge=0.0, le=1.0)
    provider: str


class JudgeVerdict(_Base):
    """Strict structured output from the judge. Constrained to three choices."""

    strategy: Strategy
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    bull_score: float = Field(ge=0.0, le=1.0)
    bear_score: float = Field(ge=0.0, le=1.0)
    provider: str
    reason_codes: tuple[ReasonCode, ...] = ()


class TradeDecision(_Base):
    """The agent's final, auditable decision for one symbol at one instant."""

    decision_id: str
    symbol: str
    timestamp: datetime
    regime: Regime
    direction: Direction
    strategy: Strategy
    confidence: float = Field(ge=0.0, le=1.0)
    bull_score: float = Field(ge=0.0, le=1.0)
    bear_score: float = Field(ge=0.0, le=1.0)
    quant_score: float = Field(ge=0.0, le=1.0)
    reason_codes: tuple[ReasonCode, ...] = ()
    no_trade_reason: str | None = None
    ai_provider: str = "none"

    @model_validator(mode="after")
    def _no_trade_requires_reason(self) -> TradeDecision:
        if self.strategy is Strategy.NO_TRADE and not self.no_trade_reason:
            raise ValueError("NO_TRADE decisions must carry a no_trade_reason")
        return self

    @property
    def is_tradable(self) -> bool:
        return self.strategy in TRADABLE_STRATEGIES


# --------------------------------------------------------------------------- #
# Strategy construction
# --------------------------------------------------------------------------- #
class SpreadLeg(_Base):
    contract: OptionContractCandidate
    side: OrderSide
    ratio: int = Field(default=1, gt=0)
    position_intent: PositionIntent


class SpreadStructure(_Base):
    """A two-leg vertical debit spread with an arithmetically defined max loss.

    All money is carried in integer cents to keep the risk boundary exact.
    """

    strategy: Strategy
    symbol: str
    expiration: date
    long_leg: SpreadLeg
    short_leg: SpreadLeg
    net_debit_cents: int = Field(gt=0)
    strike_width_cents: int = Field(gt=0)
    limit_price_cents: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate(self) -> SpreadStructure:
        if self.strategy not in TRADABLE_STRATEGIES:
            raise ValueError(f"{self.strategy} is not a constructible spread")
        if self.long_leg.contract.expiration != self.short_leg.contract.expiration:
            raise ValueError("both legs must share one expiration")
        if self.long_leg.contract.option_type != self.short_leg.contract.option_type:
            raise ValueError("both legs must share one option type")
        if self.net_debit_cents >= self.strike_width_cents:
            raise ValueError("net debit must be below strike width, else risk is undefined")
        return self

    def max_loss_cents(self, quantity: int) -> int:
        """Maximum loss in cents. For a vertical debit spread this is exactly
        the premium paid: there is no path in which more can be lost."""
        return self.limit_price_cents * OPTION_MULTIPLIER * quantity

    def max_profit_cents(self, quantity: int) -> int:
        return (
            (self.strike_width_cents - self.limit_price_cents) * OPTION_MULTIPLIER * quantity
        )


# --------------------------------------------------------------------------- #
# Risk and execution
# --------------------------------------------------------------------------- #
class RiskDecision(_Base):
    approved: bool
    quantity: int = Field(ge=0)
    max_loss_cents: int = Field(ge=0)
    max_profit_cents: int = Field(ge=0)
    reason_codes: tuple[ReasonCode, ...]
    detail: str = ""
    checks_run: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _approved_needs_size(self) -> RiskDecision:
        if self.approved and self.quantity <= 0:
            raise ValueError("an approved RiskDecision must size at least one spread")
        if not self.approved and ReasonCode.APPROVED in self.reason_codes:
            raise ValueError("a rejected RiskDecision must not carry APPROVED")
        return self


class OrderIntent(_Base):
    """A fully specified, idempotent multi-leg order, ready to submit."""

    client_order_id: str
    decision_id: str
    symbol: str
    strategy: Strategy
    quantity: int = Field(gt=0)
    limit_price_cents: int = Field(gt=0)
    legs: tuple[SpreadLeg, ...]
    max_loss_cents: int = Field(gt=0)
    created_at: datetime

    @model_validator(mode="after")
    def _two_legs(self) -> OrderIntent:
        if len(self.legs) != 2:
            raise ValueError("v1 supports exactly two-leg vertical spreads")
        return self

    @property
    def limit_price(self) -> float:
        return round(self.limit_price_cents / 100.0, 2)


class ExecutionRecord(_Base):
    client_order_id: str
    broker_order_id: str | None
    status: str
    filled_quantity: int = 0
    filled_avg_price_cents: int | None = None
    submitted_at: datetime | None = None
    updated_at: datetime | None = None
    raw_status: str = ""


class PositionRecord(_Base):
    position_id: str
    decision_id: str
    client_order_id: str
    symbol: str
    strategy: Strategy
    quantity: int = Field(gt=0)
    entry_debit_cents: int = Field(gt=0)
    max_loss_cents: int = Field(gt=0)
    max_profit_cents: int = Field(ge=0)
    opened_at: datetime
    expiration: date
    long_symbol: str
    short_symbol: str
    state: TradeState = TradeState.MONITORING


class TradeOutcome(_Base):
    position_id: str
    decision_id: str
    symbol: str
    strategy: Strategy
    regime: Regime
    confidence: float
    quantity: int
    entry_debit_cents: int
    exit_value_cents: int
    realized_pnl_cents: int
    return_on_defined_risk: float
    holding_minutes: float
    max_favorable_excursion_cents: int | None = None
    max_adverse_excursion_cents: int | None = None
    exit_reason: ExitReason
    opened_at: datetime
    closed_at: datetime

    @property
    def is_win(self) -> bool:
        return self.realized_pnl_cents > 0


__all__ = [
    "OPTION_MULTIPLIER",
    "TERMINAL_STATES",
    "TRADABLE_STRATEGIES",
    "AIArgument",
    "Bar",
    "Direction",
    "ExecutionRecord",
    "ExitReason",
    "Greeks",
    "JudgeVerdict",
    "MarketSnapshot",
    "OptionContractCandidate",
    "OptionQuote",
    "OptionType",
    "OrderIntent",
    "OrderSide",
    "PositionIntent",
    "PositionRecord",
    "QuantSignal",
    "ReasonCode",
    "Regime",
    "RegimeAssessment",
    "RiskDecision",
    "SpreadLeg",
    "SpreadStructure",
    "Strategy",
    "TradeDecision",
    "TradeOutcome",
    "TradeState",
]
