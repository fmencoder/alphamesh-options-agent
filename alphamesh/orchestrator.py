"""The autonomous lifecycle.

One ``run_cycle`` call is one complete pass:

    market data -> features -> quant gate -> regime -> AI council -> judge
    -> option chain -> deterministic contract selection -> Risk Governor
    -> order construction -> idempotent submission -> monitoring -> exits
    -> journal

No human is in the loop at any step. Every branch, including every NO_TRADE,
is written to the journal with machine-readable reason codes.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime

from alphamesh.agents.market_scout import MarketScout
from alphamesh.agents.regime_agent import RegimeAgent
from alphamesh.agents.strategy_agent import CouncilResult, StrategyAgent
from alphamesh.alpaca.client import AlpacaStack
from alphamesh.alpaca.execution import AmbiguousSubmissionError, BrokerError
from alphamesh.config import AppConfig
from alphamesh.execution.exits import evaluate_exit
from alphamesh.execution.monitor import (
    OrderMonitor,
    PositionMark,
    mark_position,
    mark_spread_cents,
)
from alphamesh.execution.order_builder import build_order_intent
from alphamesh.execution.recovery import reconcile_open_orders
from alphamesh.execution.state_machine import transition
from alphamesh.intelligence.reasoning import ReasoningProvider
from alphamesh.models.domain import (
    OPTION_MULTIPLIER,
    ExecutionRecord,
    ExitReason,
    OptionContractCandidate,
    OptionType,
    OrderIntent,
    PositionRecord,
    ReasonCode,
    Regime,
    RegimeAssessment,
    RiskDecision,
    Strategy,
    TradeDecision,
    TradeOutcome,
    TradeState,
)
from alphamesh.persistence.journal import Journal
from alphamesh.risk.circuit_breaker import evaluate_circuit_breaker
from alphamesh.risk.governor import RiskGovernor
from alphamesh.risk.portfolio import PortfolioState
from alphamesh.strategies.bear_put import build_bear_put_spread
from alphamesh.strategies.bull_call import build_bull_call_spread

log = logging.getLogger(__name__)


@dataclass
class CycleReport:
    """Everything that happened in one pass, for logs and the dashboard."""

    started_at: datetime
    symbols_scanned: int = 0
    decisions: list[TradeDecision] = field(default_factory=list)
    orders_submitted: list[str] = field(default_factory=list)
    exits_taken: list[str] = field(default_factory=list)
    rejections: list[tuple[str, tuple[ReasonCode, ...]]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    circuit_breaker_tripped: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at.isoformat(),
            "symbols_scanned": self.symbols_scanned,
            "decisions": [
                {
                    "symbol": d.symbol,
                    "strategy": d.strategy.value,
                    "confidence": round(d.confidence, 3),
                    "quant_score": round(d.quant_score, 3),
                    "regime": d.regime.value,
                    "reason_codes": [c.value for c in d.reason_codes],
                }
                for d in self.decisions
            ],
            "orders_submitted": list(self.orders_submitted),
            "exits_taken": list(self.exits_taken),
            "rejections": [(s, [c.value for c in codes]) for s, codes in self.rejections],
            "errors": list(self.errors),
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
        }


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        stack: AlpacaStack,
        journal: Journal,
        provider: ReasoningProvider,
    ) -> None:
        self.config = config
        self.stack = stack
        self.journal = journal
        self.scout = MarketScout(config, stack.market_data)
        self.regime_agent = RegimeAgent(config)
        self.strategy_agent = StrategyAgent(config, provider)
        self.governor = RiskGovernor(config.risk, paper_confirmed=stack.guard.paper)
        self.monitor = OrderMonitor(stack.broker, journal)
        self._regimes: dict[str, RegimeAssessment] = {}

    # ------------------------------------------------------------- lifecycle
    def startup(self) -> dict[str, object]:
        """Reconcile with the broker before anything new can be created."""
        report = reconcile_open_orders(self.journal, self.stack.broker)
        self.journal.record_event("startup_recovery", report.as_dict())
        log.info(
            "recovery: inspected=%d reconciled=%d orphaned=%d",
            report.inspected,
            report.reconciled,
            report.orphaned,
        )
        return report.as_dict()

    def portfolio_state(self, now: datetime | None = None) -> PortfolioState:
        account = self.stack.broker.account()
        positions = tuple(self.journal.open_positions())
        unrealized = self._unrealized_pnl_cents(positions, now)
        today = (now or datetime.now(UTC)).date().isoformat()
        return PortfolioState(
            account=account,
            open_positions=positions,
            realized_pnl_today_cents=self.journal.realized_pnl_cents(since_iso=today),
            unrealized_pnl_cents=unrealized,
            open_client_order_ids=frozenset(p.client_order_id for p in positions),
        )

    def _unrealized_pnl_cents(
        self, positions: tuple[PositionRecord, ...], now: datetime | None = None
    ) -> int:
        total = 0
        for position in positions:
            mark = self._mark_for(position, now)
            if mark is not None:
                total += mark.unrealized_pnl_cents
        return total

    def _contracts_for(
        self, position: PositionRecord, now: datetime | None = None
    ) -> dict[str, OptionContractCandidate]:
        found: dict[str, OptionContractCandidate] = {}
        for option_type in (OptionType.CALL, OptionType.PUT):
            try:
                chain = self.stack.option_chain.chain(
                    position.symbol,
                    option_type,
                    as_of=(now or datetime.now(UTC)).date(),
                    min_dte=0,
                    max_dte=self.config.strategies.max_dte + 5,
                )
            except Exception:
                continue
            for contract in chain:
                if contract.symbol in (position.long_symbol, position.short_symbol):
                    found[contract.symbol] = contract
        return found

    def _mark_for(  # type: ignore[no-untyped-def]
        self, position: PositionRecord, now: datetime | None = None
    ):
        contracts = self._contracts_for(position, now)
        long_c = contracts.get(position.long_symbol)
        short_c = contracts.get(position.short_symbol)
        if long_c is None or short_c is None:
            return None
        spread_mark = mark_spread_cents(long_c, short_c)
        if spread_mark is None:
            return None
        return mark_position(position, spread_mark)

    # ---------------------------------------------------------------- cycle
    def run_cycle(self, now: datetime | None = None) -> CycleReport:
        """Run one full pass.

        ``now`` may be pinned so a captured session can be replayed
        deterministically; production passes ``None`` and uses the wall clock.
        """
        now = now or datetime.now(UTC)
        report = CycleReport(started_at=now)

        try:
            portfolio = self.portfolio_state(now)
        except Exception as exc:
            log.exception("could not read portfolio state")
            report.errors.append(f"portfolio_state: {exc}")
            self.journal.record_event("cycle_error", {"stage": "portfolio", "error": str(exc)})
            return report

        breaker = evaluate_circuit_breaker(portfolio, self.config.risk)
        report.circuit_breaker_tripped = breaker.tripped
        if breaker.tripped:
            self.journal.record_event("circuit_breaker", {"detail": breaker.detail})

        # 1. Manage what is already open, before considering anything new.
        self._manage_positions(portfolio, now, breaker.tripped, report)

        # 2. Scan and decide.
        scanned = self.scout.scan()
        report.symbols_scanned = len(scanned)

        for snapshot, signal in scanned:
            regime = self.regime_agent.assess(snapshot, signal)
            self._regimes[snapshot.symbol] = regime
            council = self.strategy_agent.decide(signal, regime)
            decision = council.decision
            report.decisions.append(decision)
            self.journal.record_decision(
                decision,
                features=signal.features,
                regime=regime,
                bull=council.bull,
                bear=council.bear,
                verdict=council.verdict,
            )

            if not decision.is_tradable:
                continue

            if breaker.tripped:
                self._record_rejection(
                    decision, (ReasonCode.DAILY_DRAWDOWN_LIMIT,), breaker.detail, report
                )
                continue

            try:
                self._attempt_entry(decision, council, portfolio, now, report)
            except Exception as exc:
                log.exception("entry attempt failed for %s", decision.symbol)
                report.errors.append(f"{decision.symbol}: {exc}")
                self.journal.record_event(
                    "entry_error",
                    {"error": str(exc)},
                    decision_id=decision.decision_id,
                    symbol=decision.symbol,
                )

            # Re-read state so a fill inside this cycle counts against limits.
            try:
                portfolio = self.portfolio_state(now)
            except Exception:
                break

        self.journal.record_event("cycle_complete", report.as_dict())
        return report

    # ---------------------------------------------------------------- entry
    def _attempt_entry(
        self,
        decision: TradeDecision,
        council: CouncilResult,
        portfolio: PortfolioState,
        now: datetime,
        report: CycleReport,
    ) -> None:
        state = transition(TradeState.DISCOVERED, TradeState.ANALYZED)
        state = transition(state, TradeState.AI_APPROVED)

        option_type = (
            OptionType.CALL
            if decision.strategy is Strategy.BULL_CALL_SPREAD
            else OptionType.PUT
        )
        chain = self.stack.option_chain.chain(
            decision.symbol,
            option_type,
            as_of=now.date(),
            min_dte=self.config.strategies.min_dte,
            max_dte=self.config.strategies.max_dte,
        )

        builder = (
            build_bull_call_spread
            if decision.strategy is Strategy.BULL_CALL_SPREAD
            else build_bear_put_spread
        )
        selection = builder(
            decision.symbol,
            chain,
            self.config.strategies,
            self.config.risk,
            now,
            as_of_date=now.date(),
        )
        self.journal.record_event(
            "contract_selection",
            {
                "strategy": decision.strategy.value,
                "chain_size": len(chain),
                "ok": selection.ok,
                "reason_codes": [c.value for c in selection.reason_codes],
                "detail": selection.detail,
                "rejected": {
                    k: [c.value for c in v] for k, v in selection.rejected_contracts.items()
                },
                "selected": (
                    {
                        "long": selection.spread.long_leg.contract.symbol,
                        "short": selection.spread.short_leg.contract.symbol,
                        "limit_price_cents": selection.spread.limit_price_cents,
                        "width_cents": selection.spread.strike_width_cents,
                    }
                    if selection.spread
                    else None
                ),
            },
            decision_id=decision.decision_id,
            symbol=decision.symbol,
        )

        if selection.spread is None:
            self._record_rejection(
                decision,
                selection.reason_codes or (ReasonCode.NO_ELIGIBLE_CONTRACTS,),
                selection.detail,
                report,
            )
            return

        spread = selection.spread
        # The client order id must be known before the governor runs, so the
        # duplicate check sees the exact id we would submit.
        from alphamesh.execution.order_builder import build_client_order_id

        provisional_qty = 1
        provisional_id = build_client_order_id(
            decision, spread, provisional_qty, spread.limit_price_cents
        )
        risk = self.governor.approve(
            decision,
            spread,
            portfolio,
            now,
            client_order_id=provisional_id,
            known_client_order_ids=self.journal.known_client_order_ids(),
        )
        self.journal.record_risk_decision(decision.decision_id, risk)

        if not risk.approved:
            report.rejections.append((decision.symbol, risk.reason_codes))
            log.info(
                "risk rejected %s %s: %s",
                decision.symbol,
                decision.strategy,
                [c.value for c in risk.reason_codes],
            )
            return

        state = transition(state, TradeState.RISK_APPROVED)
        intent = build_order_intent(decision, spread, risk, now)
        state = transition(state, TradeState.CONSTRUCTED)

        # Reserve the id BEFORE the network call. A crash between here and the
        # broker leaves a CONSTRUCTED row that recovery reconciles, rather than
        # a silent duplicate on the next start.
        if not self.journal.reserve_order(intent):
            self._record_rejection(
                decision,
                (ReasonCode.DUPLICATE_ORDER,),
                f"client_order_id {intent.client_order_id} is already reserved",
                report,
            )
            return

        try:
            record = self.stack.broker.submit_spread(intent)
        except AmbiguousSubmissionError as exc:
            # Do not resubmit. Leave the reservation for recovery to reconcile.
            log.error("ambiguous submission for %s: %s", intent.client_order_id, exc)
            self.journal.record_event(
                "ambiguous_submission",
                {"client_order_id": intent.client_order_id, "cause": exc.cause},
                decision_id=decision.decision_id,
                symbol=decision.symbol,
            )
            report.errors.append(f"{decision.symbol}: ambiguous submission, will reconcile")
            return
        except BrokerError as exc:
            self.journal.set_order_state(
                intent.client_order_id, TradeState.REJECTED, f"broker rejected: {exc}"
            )
            report.rejections.append((decision.symbol, (ReasonCode.DUPLICATE_ORDER,)))
            return

        self.journal.set_order_state(
            intent.client_order_id, TradeState.SUBMITTED, "submitted to Alpaca paper"
        )
        self.journal.update_order_execution(record, TradeState.SUBMITTED)
        report.orders_submitted.append(intent.client_order_id)

        refreshed = self.monitor.refresh(intent.client_order_id) or record
        if refreshed.filled_quantity > 0:
            self._open_position(decision, intent, refreshed, risk, now)

    def _open_position(
        self,
        decision: TradeDecision,
        intent: OrderIntent,
        record: ExecutionRecord,
        risk: RiskDecision,
        now: datetime,
    ) -> None:
        fill_cents = record.filled_avg_price_cents or intent.limit_price_cents
        entry_debit = fill_cents * OPTION_MULTIPLIER * record.filled_quantity
        width = intent.legs[1].contract.strike - intent.legs[0].contract.strike
        width_cents = abs(round(width * 100))
        max_profit = max(
            0, (width_cents - fill_cents) * OPTION_MULTIPLIER * record.filled_quantity
        )
        position = PositionRecord(
            position_id=uuid.uuid4().hex[:16],
            decision_id=decision.decision_id,
            client_order_id=intent.client_order_id,
            symbol=decision.symbol,
            strategy=decision.strategy,
            quantity=record.filled_quantity,
            entry_debit_cents=entry_debit,
            max_loss_cents=entry_debit,
            max_profit_cents=max_profit,
            opened_at=now,
            expiration=intent.legs[0].contract.expiration,
            long_symbol=intent.legs[0].contract.symbol,
            short_symbol=intent.legs[1].contract.symbol,
            state=TradeState.MONITORING,
        )
        self.journal.record_position(position)
        # The monitor may already have moved this order to FILLED when it read
        # the broker back. Only record the hop if it has not happened yet, so
        # the transition log stays a faithful history rather than a repeat.
        row = self.journal.get_order(intent.client_order_id)
        if row is None or row["state"] != TradeState.FILLED.value:
            self.journal.set_order_state(
                intent.client_order_id, TradeState.FILLED, "filled at broker"
            )
        self.journal.set_order_state(
            intent.client_order_id, TradeState.MONITORING, "position under management"
        )
        self.journal.record_event(
            "position_opened",
            {
                "position_id": position.position_id,
                "quantity": position.quantity,
                "entry_debit_cents": entry_debit,
                "max_loss_cents": position.max_loss_cents,
            },
            decision_id=decision.decision_id,
            symbol=decision.symbol,
        )

    # ----------------------------------------------------------- management
    def _manage_positions(
        self,
        portfolio: PortfolioState,
        now: datetime,
        breaker_tripped: bool,
        report: CycleReport,
    ) -> None:
        clock = None
        with suppress(Exception):
            # A clock outage only costs us the session-close exit; the
            # time-based and mark-based rules still run without it.
            clock = self.stack.market_data.clock()

        for position in portfolio.open_positions:
            mark = self._mark_for(position, now)
            if mark is not None:
                previous_mfe, previous_mae = self.journal.position_excursions(
                    position.position_id
                )
                mfe = max(previous_mfe or 0, mark.unrealized_pnl_cents)
                mae = min(previous_mae or 0, mark.unrealized_pnl_cents)
                self.journal.update_excursions(position.position_id, mfe, mae)

            decision = evaluate_exit(
                position,
                mark,
                self._regimes.get(position.symbol),
                self.config.strategies,
                now,
                session_close=clock.next_close if clock else None,
                circuit_breaker_tripped=breaker_tripped,
            )
            if not decision.should_exit:
                continue

            try:
                self._exit_position(position, mark, decision.reason, decision.detail, now)
                report.exits_taken.append(f"{position.symbol}:{decision.reason}")
            except Exception as exc:
                log.exception("exit failed for %s", position.position_id)
                report.errors.append(f"exit {position.position_id}: {exc}")

    def _exit_position(
        self,
        position: PositionRecord,
        mark: PositionMark | None,
        reason: ExitReason | None,
        detail: str,
        now: datetime,
    ) -> None:
        exit_value = mark.mark_cents if mark is not None else 0
        realized = exit_value - position.entry_debit_cents
        holding = (now - position.opened_at).total_seconds() / 60.0
        mfe, mae = self.journal.position_excursions(position.position_id)
        assessment = self._regimes.get(position.symbol)
        regime = assessment.regime if assessment is not None else Regime.UNKNOWN

        self.journal.set_order_state(
            position.client_order_id, TradeState.EXIT_REQUESTED, detail
        )
        self.journal.set_position_state(position.position_id, TradeState.EXIT_REQUESTED)

        outcome = TradeOutcome(
            position_id=position.position_id,
            decision_id=position.decision_id,
            symbol=position.symbol,
            strategy=position.strategy,
            regime=regime,
            confidence=0.0,
            quantity=position.quantity,
            entry_debit_cents=position.entry_debit_cents,
            exit_value_cents=exit_value,
            realized_pnl_cents=realized,
            return_on_defined_risk=(
                realized / position.max_loss_cents if position.max_loss_cents else 0.0
            ),
            holding_minutes=holding,
            max_favorable_excursion_cents=mfe,
            max_adverse_excursion_cents=mae,
            exit_reason=reason or ExitReason.MANUAL,
            opened_at=position.opened_at,
            closed_at=now,
        )
        self.journal.record_outcome(outcome)
        self.journal.set_order_state(
            position.client_order_id, TradeState.CLOSED, f"exit: {reason}"
        )
        self.journal.record_event(
            "position_closed",
            {
                "position_id": position.position_id,
                "exit_reason": str(reason),
                "realized_pnl_cents": realized,
                "detail": detail,
            },
            decision_id=position.decision_id,
            symbol=position.symbol,
        )

    def _record_rejection(
        self,
        decision: TradeDecision,
        codes: tuple[ReasonCode, ...],
        detail: str,
        report: CycleReport,
    ) -> None:
        risk = RiskDecision(
            approved=False,
            quantity=0,
            max_loss_cents=0,
            max_profit_cents=0,
            reason_codes=codes,
            detail=detail,
            checks_run=("pre_risk",),
        )
        self.journal.record_risk_decision(decision.decision_id, risk)
        report.rejections.append((decision.symbol, codes))


__all__ = ["CycleReport", "Orchestrator"]
