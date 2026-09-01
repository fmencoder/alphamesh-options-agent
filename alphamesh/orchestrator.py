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
from dataclasses import dataclass, field
from datetime import UTC, datetime

from alphamesh.agents.market_scout import MarketScout
from alphamesh.agents.regime_agent import RegimeAgent
from alphamesh.agents.strategy_agent import CouncilResult, StrategyAgent
from alphamesh.alpaca.client import AlpacaStack
from alphamesh.alpaca.execution import AmbiguousSubmissionError, BrokerError
from alphamesh.alpaca.options import occ_underlying
from alphamesh.alpaca.types import MarketClock
from alphamesh.config import AppConfig
from alphamesh.execution.adoption import (
    AdoptedSpread,
    AdoptionSummary,
    reconstruct_spreads,
)
from alphamesh.execution.exits import evaluate_exit
from alphamesh.execution.monitor import (
    DEAD_STATUSES,
    OrderMonitor,
    mark_position,
    mark_spread_cents,
)
from alphamesh.execution.order_builder import build_exit_intent, build_order_intent
from alphamesh.execution.recovery import reconcile_open_orders
from alphamesh.execution.state_machine import is_terminal, transition
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
from alphamesh.persistence.journal import (
    ORDER_KIND_ENTRY,
    ORDER_KIND_EXIT,
    PHANTOM_CLOSE_NOTE,
    POSITION_ORIGIN_ADOPTED,
    Journal,
)
from alphamesh.risk.circuit_breaker import evaluate_circuit_breaker
from alphamesh.risk.governor import RiskGovernor
from alphamesh.risk.portfolio import PortfolioState
from alphamesh.safety import LiveTradingForbiddenError, check_trading_endpoint
from alphamesh.strategies.bear_put import build_bear_put_spread
from alphamesh.strategies.bull_call import build_bull_call_spread

log = logging.getLogger(__name__)



def _parse_journal_ts(raw: object) -> datetime | None:
    """Parse a journal timestamp, normalising naive values to UTC."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass
class CycleReport:
    """Everything that happened in one pass, for logs and the dashboard."""

    started_at: datetime
    market_open: bool = True
    next_open: datetime | None = None
    execution_mode: str = "UNKNOWN"
    symbols_scanned: int = 0
    decisions: list[TradeDecision] = field(default_factory=list)
    orders_submitted: list[str] = field(default_factory=list)
    exits_taken: list[str] = field(default_factory=list)
    rejections: list[tuple[str, tuple[ReasonCode, ...]]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    circuit_breaker_tripped: bool = False
    # Funnel counters. Each stage increments exactly once per candidate, so the
    # sequence reads as a strict narrowing from scan to fill.
    quant_passes: int = 0
    ai_tradable: int = 0
    entry_fills: int = 0
    stale_orders_cancelled: int = 0
    contracts_selected: int = 0
    risk_approved: int = 0
    open_positions: int = 0
    realized_pnl_cents: int = 0
    unrealized_pnl_cents: int = 0
    # Exits. ``exit_orders_submitted`` is what reached the broker;
    # ``exits_taken`` is only what the broker confirmed as closed. They are
    # deliberately separate: a submitted exit is not a closed position, and
    # conflating them is what let phantom closes look like real ones.
    exit_orders_submitted: list[str] = field(default_factory=list)
    exit_orders_repriced: int = 0
    positions_adopted: int = 0
    ambiguous_broker_positions: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at.isoformat(),
            "market_open": self.market_open,
            "next_open": self.next_open.isoformat() if self.next_open else None,
            "execution_mode": self.execution_mode,
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
            "quant_passes": self.quant_passes,
            "ai_tradable": self.ai_tradable,
            "entry_fills": self.entry_fills,
            "stale_orders_cancelled": self.stale_orders_cancelled,
            "contracts_selected": self.contracts_selected,
            "risk_approved": self.risk_approved,
            "open_positions": self.open_positions,
            "realized_pnl_cents": self.realized_pnl_cents,
            "unrealized_pnl_cents": self.unrealized_pnl_cents,
            "exit_orders_submitted": list(self.exit_orders_submitted),
            "exit_orders_repriced": self.exit_orders_repriced,
            "positions_adopted": self.positions_adopted,
            "ambiguous_broker_positions": self.ambiguous_broker_positions,
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
        self._clock: MarketClock | None = None
        # Underlyings the broker holds that could not be resolved into a
        # managed spread. Exposure is real but uncounted, so new risk is
        # refused until it is understood.
        self._ambiguous_broker_symbols: frozenset[str] = frozenset()

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
        adoption = self.adopt_broker_positions()
        summary = report.as_dict()
        summary["adoption"] = adoption.as_dict()
        return summary

    # ------------------------------------------------------------ adoption
    def adopt_broker_positions(self, now: datetime | None = None) -> AdoptionSummary:
        """Take ownership of live broker spreads the journal has forgotten.

        This only reads and journals. It never submits an order, so it is safe
        to run outside market hours -- discovery is not execution, and an
        adopted position is exited by the normal management path against live
        quotes, not by a queued order waiting for the bell.
        """
        now = now or datetime.now(UTC)
        try:
            positions = self.stack.broker.positions()
        except Exception as exc:
            log.warning("adoption could not read broker positions: %s", exc)
            return AdoptionSummary(error=str(exc))

        try:
            orders = self.stack.broker.recent_orders()
        except Exception as exc:
            # Without order history the entry debit can still come from the
            # per-leg cost basis, so this degrades rather than fails.
            log.warning("adoption could not read broker order history: %s", exc)
            orders = []

        result = reconstruct_spreads(positions, orders)
        known = {
            (p.long_symbol, p.short_symbol) for p in self.journal.open_positions()
        }
        adopted: list[str] = []

        for spread in result.spreads:
            if (spread.long_symbol, spread.short_symbol) in known:
                continue
            position = self._adopt_one(spread, now)
            adopted.append(position.position_id)

        self._ambiguous_broker_symbols = result.ambiguous_symbols
        for item in result.ambiguous:
            log.warning(
                "broker_position_ambiguous symbol=%s reason=%s legs=%s; "
                "not adopted, and new exposure stays blocked",
                item.symbol,
                item.reason,
                list(item.legs),
            )

        summary = AdoptionSummary(
            adopted=len(adopted),
            ambiguous=len(result.ambiguous),
            detail=result.as_dict(),
        )
        if adopted or result.ambiguous:
            self.journal.record_event("broker_position_adoption", summary.as_dict())
            log.info(
                "adoption: adopted=%d ambiguous=%d broker_option_legs=%d",
                len(adopted),
                len(result.ambiguous),
                len(positions),
            )
        return summary

    def _adopt_one(self, spread: AdoptedSpread, now: datetime) -> PositionRecord:
        """Journal one reconstructed spread, annotating any phantom close."""
        position = PositionRecord(
            position_id=uuid.uuid4().hex[:16],
            decision_id=f"adopted-{uuid.uuid4().hex[:12]}",
            client_order_id=spread.client_order_id or f"adopted-{uuid.uuid4().hex[:12]}",
            symbol=spread.symbol,
            strategy=spread.strategy,
            quantity=spread.quantity,
            entry_debit_cents=spread.entry_debit_cents,
            max_loss_cents=spread.max_loss_cents,
            max_profit_cents=spread.max_profit_cents,
            # An unknown open time must not fabricate a holding period. Falling
            # back to now means the max-holding-time rule starts from adoption,
            # which delays a time exit rather than forcing one on bad data.
            opened_at=spread.opened_at or now,
            expiration=spread.expiration,
            long_symbol=spread.long_symbol,
            short_symbol=spread.short_symbol,
            state=TradeState.MONITORING,
        )
        self.journal.record_position(
            position, origin=POSITION_ORIGIN_ADOPTED, entry_basis=spread.entry_basis
        )
        self._annotate_phantom_closes(spread, position)
        log.info(
            "position_adopted position_id=%s symbol=%s strategy=%s qty=%d "
            "long=%s short=%s entry_debit_cents=%d basis=%s expiration=%s",
            position.position_id,
            position.symbol,
            position.strategy.value,
            position.quantity,
            position.long_symbol,
            position.short_symbol,
            position.entry_debit_cents,
            spread.entry_basis,
            position.expiration.isoformat(),
        )
        return position

    def _had_real_closing_order(self, position_id: str) -> bool:
        """Whether a filled closing order stands behind this position's close.

        This is the whole distinction between a phantom close and a real one:
        a real close has an exit order that the broker filled. A phantom has
        nothing but a journal row.
        """
        exit_order = self.journal.exit_order_for(position_id)
        if exit_order is None:
            return False
        return int(exit_order.get("filled_quantity") or 0) > 0

    def _annotate_phantom_closes(
        self, spread: AdoptedSpread, adopted: PositionRecord
    ) -> None:
        """Record that an earlier journal close had no broker order behind it.

        Nothing is deleted. The outcome row stays exactly as it was written and
        gains a note, which both preserves the competition audit trail and takes
        its invented money out of every realised total.
        """
        for row in self.journal.positions_for_legs(
            spread.long_symbol, spread.short_symbol
        ):
            if str(row["position_id"]) == adopted.position_id:
                continue
            if TradeState(str(row["state"])) is not TradeState.CLOSED:
                continue
            position_id = str(row["position_id"])
            if self._had_real_closing_order(position_id):
                # A genuine close that was later re-opened over the same two
                # contracts. Its money was real; annotating it would erase a
                # true result to tidy up a false one.
                continue
            self.journal.annotate_outcome(position_id, PHANTOM_CLOSE_NOTE)
            log.warning(
                "phantom_close_reconciled symbol=%s journal_position_id=%s "
                "adopted_position_id=%s broker_position_present=true "
                "reason=no_broker_exit_order",
                spread.symbol,
                position_id,
                adopted.position_id,
            )
            self.journal.record_event(
                "phantom_close_reconciled",
                {
                    "symbol": spread.symbol,
                    "journal_position_id": position_id,
                    "adopted_position_id": adopted.position_id,
                    "broker_position_present": True,
                    "reason": "no_broker_exit_order",
                    "long_symbol": spread.long_symbol,
                    "short_symbol": spread.short_symbol,
                },
                symbol=spread.symbol,
            )

    def market_clock(self) -> MarketClock | None:
        """Read the trading calendar. ``None`` means it could not be read, which
        callers must treat as closed rather than as open."""
        try:
            self._clock = self.stack.market_data.clock()
        except Exception as exc:
            log.warning("could not read the market clock: %s", exc)
            self._clock = None
        return self._clock

    def portfolio_state(self, now: datetime | None = None) -> PortfolioState:
        account = self.stack.broker.account()
        positions = tuple(self.journal.open_positions())
        unrealized = self._unrealized_pnl_cents(positions, now)
        today = (now or datetime.now(UTC)).date().isoformat()
        journal_working = frozenset(
            str(o["symbol"]) for o in self.journal.open_orders() if o.get("symbol")
        )
        broker_positions, broker_working, broker_ok = self._broker_exposure()
        journal_exposed = {p.symbol.upper() for p in positions} | {
            s.upper() for s in journal_working
        }
        broker_exposed = {s.upper() for s in broker_positions} | {
            s.upper() for s in broker_working
        }
        if broker_ok and journal_exposed != broker_exposed:
            only_broker = sorted(broker_exposed - journal_exposed)
            only_journal = sorted(journal_exposed - broker_exposed)
            log.warning(
                "exposure_state_mismatch only_broker=%s only_journal=%s "
                "broker_positions=%s broker_working=%s journal_positions=%s "
                "journal_working=%s resolution=BROKER_WINS_FOR_BLOCKING",
                only_broker,
                only_journal,
                sorted(broker_positions),
                sorted(broker_working),
                sorted(p.symbol for p in positions),
                sorted(journal_working),
            )
            self.journal.record_event(
                "exposure_state_mismatch",
                {
                    "only_broker": only_broker,
                    "only_journal": only_journal,
                    "resolution": "BROKER_WINS_FOR_BLOCKING",
                },
            )
        return PortfolioState(
            account=account,
            open_positions=positions,
            realized_pnl_today_cents=self.journal.realized_pnl_cents(since_iso=today),
            unrealized_pnl_cents=unrealized,
            open_client_order_ids=frozenset(p.client_order_id for p in positions),
            working_order_symbols=journal_working,
            broker_position_symbols=broker_positions,
            broker_working_symbols=broker_working,
            broker_truth_available=broker_ok,
            unaccounted_broker_symbols=self._ambiguous_broker_symbols,
        )

    def _broker_exposure(self) -> tuple[frozenset[str], frozenset[str], bool]:
        """Underlyings the ACCOUNT says are exposed, by position and by order.

        Returns ``available=False`` only when the broker cannot be read at all.
        Callers must treat that as ambiguous state and refuse new exposure
        rather than fall back to the journal, which is exactly the blind spot
        this guard exists to close.
        """
        try:
            positions = frozenset(
                root
                for p in self.stack.broker.positions()
                if (root := (occ_underlying(p.symbol) or p.symbol.upper()))
            )
            working = frozenset(self.stack.broker.working_order_symbols())
        except Exception as exc:
            log.warning("broker exposure unreadable; refusing new exposure: %s", exc)
            return frozenset(), frozenset(), False
        return positions, working, True

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
        report = CycleReport(started_at=now, execution_mode=self.stack.execution_mode)

        # Market-hours gate, before any other work. A closed market means every
        # quote is dead, so scanning, invoking the AI council and pulling option
        # chains would burn cost on data that cannot be traded. A clock we
        # cannot read is treated as closed: fail closed, never open.
        clock = self.market_clock()
        if clock is None:
            report.market_open = False
            report.errors.append("market clock unavailable; treating the market as closed")
            self.journal.record_event("market_closed", {"reason": "clock_unavailable"})
            return report
        report.market_open = bool(clock.is_open)
        report.next_open = clock.next_open
        if not clock.is_open:
            self.journal.record_event(
                "market_closed",
                {
                    "next_open": clock.next_open.isoformat() if clock.next_open else None,
                    "execution_mode": report.execution_mode,
                },
            )
            return report

        # 1. Take ownership of anything the broker holds that the journal has
        #    forgotten, before any risk number is computed from it. This runs
        #    first on purpose: an unadopted spread is exposure that nothing
        #    manages and no total counts, so a portfolio read taken ahead of it
        #    would understate real risk.
        adoption = self.adopt_broker_positions(now)
        report.positions_adopted = adoption.adopted
        report.ambiguous_broker_positions = adoption.ambiguous

        try:
            portfolio = self.portfolio_state(now)
        except Exception as exc:
            log.exception("could not read portfolio state")
            report.errors.append(f"portfolio_state: {exc}")
            self.journal.record_event("cycle_error", {"stage": "portfolio", "error": str(exc)})
            return report

        report.open_positions = portfolio.open_position_count
        report.realized_pnl_cents = portfolio.realized_pnl_today_cents
        report.unrealized_pnl_cents = portfolio.unrealized_pnl_cents

        breaker = evaluate_circuit_breaker(portfolio, self.config.risk)
        report.circuit_breaker_tripped = breaker.tripped
        if breaker.tripped:
            self.journal.record_event("circuit_breaker", {"detail": breaker.detail})

        # 2. Retire entry orders the market has walked away from, before
        #    anything else reads portfolio state -- a stale working order holds
        #    the per-symbol duplicate lock. Exit orders are swept separately and
        #    on their own clock: an unfilled close is re-quoted, never abandoned.
        self._expire_stale_exit_orders(now, report)
        self._expire_stale_entry_orders(now, report)
        if report.stale_orders_cancelled:
            try:
                portfolio = self.portfolio_state(now)
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("could not re-read portfolio after stale sweep")
                report.errors.append(f"portfolio_state: {exc}")
                return report

        # 3. Manage what is already open, before considering anything new.
        self._manage_positions(portfolio, now, breaker.tripped, report)
        if report.exits_taken or report.exit_orders_submitted:
            # A close changes exposure, so limits must be recomputed before
            # any new candidate is sized against them.
            try:
                portfolio = self.portfolio_state(now)
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("could not re-read portfolio after exits")
                report.errors.append(f"portfolio_state: {exc}")
                return report

        # 4. Scan and decide.
        scanned = self.scout.scan()
        report.symbols_scanned = len(scanned)

        for snapshot, signal in scanned:
            regime = self.regime_agent.assess(snapshot, signal)
            self._regimes[snapshot.symbol] = regime
            council = self.strategy_agent.decide(signal, regime)
            decision = council.decision
            if council.gate_passed:
                report.quant_passes += 1
            if decision.is_tradable:
                report.ai_tradable += 1
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

    # -------------------------------------------------------- stale orders
    def _expire_stale_entry_orders(self, now: datetime, report: CycleReport) -> None:
        """Abandon zero-fill entry orders older than the configured TTL.

        A working order holds the per-symbol duplicate lock. A limit the market
        has walked away from therefore blocks that symbol for the rest of the
        session while never filling. Retiring it frees the symbol; it does NOT
        place anything. Any replacement must earn its way back through a fresh
        quant -> AI -> contract -> risk cycle, at prices current at that time.

        A partially filled order is never touched here: it is real exposure and
        belongs to the position and exit paths, not to a timeout.
        """
        ttl = self.config.settings.entry_order_ttl_seconds
        if ttl <= 0:
            return

        for row in self.journal.open_orders(kind=ORDER_KIND_ENTRY):
            try:
                state = TradeState(str(row["state"]))
            except ValueError:
                continue
            # Only orders that are on the wire and not yet filled at all.
            if state is not TradeState.SUBMITTED:
                continue

            client_order_id = str(row["client_order_id"])
            symbol = str(row.get("symbol") or "")
            created = _parse_journal_ts(row.get("created_at"))
            if created is None:
                continue
            age = (now - created).total_seconds()
            if age < ttl:
                continue

            # Re-read the broker before acting. The journal can be behind, and
            # cancelling something that has just filled would be far worse than
            # leaving a stale order in place.
            try:
                record = self.monitor.refresh(client_order_id)
            except Exception as exc:
                log.warning("stale sweep could not refresh %s: %s", client_order_id, exc)
                continue
            if record is None:
                continue
            if record.filled_quantity > 0:
                log.info(
                    "stale_order_skipped client_order_id=%s symbol=%s age_s=%.0f "
                    "reason=PARTIAL_OR_FULL_FILL filled_qty=%d",
                    client_order_id,
                    symbol,
                    age,
                    record.filled_quantity,
                )
                continue
            if record.broker_order_id is None:
                continue

            try:
                self.stack.broker.cancel_order(record.broker_order_id)
            except BrokerError as exc:
                log.warning(
                    "stale_order_cancel_failed client_order_id=%s symbol=%s "
                    "age_s=%.0f error=%s",
                    client_order_id,
                    symbol,
                    age,
                    exc,
                )
                continue

            # Confirm with the broker before treating the lock as released.
            # Marking it terminal on an unconfirmed cancel would free the symbol
            # while the order is still live and could still fill.
            confirmed = self.monitor.refresh(client_order_id)
            after = self.journal.get_order(client_order_id)
            released = after is not None and is_terminal(TradeState(str(after["state"])))
            # "Not filled" is not confirmation: an order the broker still shows
            # as live could fill a moment later. Only a broker-reported dead
            # status -- which the monitor persists as a terminal state -- proves
            # the cancel landed and makes it safe to free the symbol.
            if confirmed is None or confirmed.filled_quantity > 0 or not released:
                log.warning(
                    "stale_order_cancel_unconfirmed client_order_id=%s symbol=%s "
                    "age_s=%.0f status=%s filled_qty=%s",
                    client_order_id,
                    symbol,
                    age,
                    confirmed.status if confirmed else "unknown",
                    confirmed.filled_quantity if confirmed else "unknown",
                )
                continue

            report.stale_orders_cancelled += 1
            log.info(
                "stale_order_cancelled client_order_id=%s symbol=%s age_s=%.0f "
                "ttl_s=%d limit_price_cents=%s broker_status=%s filled_qty=%d "
                "duplicate_lock_released=%s",
                client_order_id,
                symbol,
                age,
                ttl,
                row.get("limit_price_cents"),
                confirmed.status,
                confirmed.filled_quantity,
                released,
            )
            self.journal.record_event(
                "stale_entry_order_cancelled",
                {
                    "client_order_id": client_order_id,
                    "age_seconds": round(age, 1),
                    "ttl_seconds": ttl,
                    "broker_status": confirmed.status,
                    "duplicate_lock_released": released,
                },
                symbol=symbol or None,
            )

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
            codes = selection.reason_codes or (ReasonCode.NO_ELIGIBLE_CONTRACTS,)
            # Governor rejections already log their codes; selection rejections
            # did not, which made a 100% selection failure invisible in
            # production and diagnosable only from the SQLite journal.
            log.info(
                "contract_selection_rejected symbol=%s strategy=%s chain_size=%d "
                "reason_codes=%s detail=%s",
                decision.symbol,
                decision.strategy.value,
                len(chain),
                [c.value for c in codes],
                selection.detail,
            )
            self._record_rejection(decision, codes, selection.detail, report)
            return

        spread = selection.spread
        report.contracts_selected += 1
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

        report.risk_approved += 1
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

        # Final paper revalidation, immediately before the only call in the
        # system that can create a position. The startup guard and the
        # cycle-level account read both happened earlier; this closes the gap
        # between them and the wire.
        try:
            self._assert_paper_before_submit()
        except LiveTradingForbiddenError as exc:
            self.journal.set_order_state(
                intent.client_order_id, TradeState.REJECTED, f"paper recheck failed: {exc.detail}"
            )
            self.journal.record_event(
                "live_trading_blocked",
                {"client_order_id": intent.client_order_id, "detail": exc.detail},
                decision_id=decision.decision_id,
                symbol=decision.symbol,
            )
            report.rejections.append((decision.symbol, (ReasonCode.LIVE_TRADING_FORBIDDEN,)))
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
            report.entry_fills += 1
            self._open_position(decision, intent, refreshed, risk, now)

    def _assert_paper_before_submit(self) -> None:
        """Re-prove paper mode at the wire. Raises rather than returning."""
        if not self.stack.guard.paper:
            raise LiveTradingForbiddenError("startup paper guard is not satisfied")
        check_trading_endpoint(self.config.settings.base_url)
        # AlpacaPaperBroker.account() re-checks the PA account prefix on every
        # call and raises if it is absent.
        self.stack.broker.account()

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
                "execution_mode": self.stack.execution_mode,
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
        # Reuse the clock already read by run_cycle's market-hours gate.
        clock = self._clock

        # Settle anything already on the wire first: a position with a live
        # closing order must not be re-evaluated into a second one.
        self._reconcile_exit_orders(now, report)

        for position in self.journal.open_positions():
            mark = self._mark_for(position, now)
            if mark is not None:
                previous_mfe, previous_mae = self.journal.position_excursions(
                    position.position_id
                )
                mfe = max(previous_mfe or 0, mark.unrealized_pnl_cents)
                mae = min(previous_mae or 0, mark.unrealized_pnl_cents)
                self.journal.update_excursions(position.position_id, mfe, mae)

            existing = self.journal.exit_order_for(position.position_id)
            if existing is not None and not is_terminal(TradeState(str(existing["state"]))):
                # Already exiting. Re-pricing is handled by the exit TTL sweep,
                # which cancels and confirms before anything new is raised.
                continue

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
                self._request_exit(position, decision.reason, decision.detail, now, report)
            except Exception as exc:
                log.exception("exit request failed for %s", position.position_id)
                report.errors.append(f"exit {position.position_id}: {exc}")

    # --------------------------------------------------------------- exits
    def _request_exit(
        self,
        position: PositionRecord,
        reason: ExitReason | None,
        detail: str,
        now: datetime,
        report: CycleReport,
    ) -> None:
        """Send the order that actually flattens the spread.

        Nothing here closes the position in the journal. A closing order is a
        request, not an outcome: only a broker-confirmed fill, reconciled by
        :meth:`_reconcile_exit_orders`, may retire a position or realise money.
        """
        if not self._market_is_open():
            # Recovery and management run outside the session too. Discovering
            # that a position should be exited is not a licence to queue an
            # order against a dead book; the next open re-evaluates it against
            # live quotes.
            log.info(
                "exit_deferred_market_closed position_id=%s symbol=%s reason=%s",
                position.position_id,
                position.symbol,
                reason,
            )
            return

        contracts = self._contracts_for(position, now)
        long_c = contracts.get(position.long_symbol)
        short_c = contracts.get(position.short_symbol)
        if long_c is None or short_c is None:
            self._log_unresolved_exit(position, reason, "legs not present in the chain", now)
            return

        spread_mark = mark_spread_cents(long_c, short_c)
        if spread_mark is None:
            self._log_unresolved_exit(position, reason, "no usable quote on both legs", now)
            return

        # Selling the spread back: the limit is what we ask to receive. A
        # spread marked at or below zero is still worth closing, so the order
        # is quoted at the minimum tick rather than abandoned.
        limit_cents = max(1, spread_mark)
        intent = build_exit_intent(
            position,
            long_c,
            short_c,
            limit_cents,
            now,
            attempt=self.journal.exit_order_count(position.position_id),
        )

        # Reserve before the wire, exactly as an entry does, so a crash between
        # here and the broker leaves a row for recovery instead of a spread that
        # gets flattened twice.
        if not self.journal.reserve_order(
            intent, kind=ORDER_KIND_EXIT, position_id=position.position_id
        ):
            log.info(
                "exit_order_already_reserved client_order_id=%s position_id=%s",
                intent.client_order_id,
                position.position_id,
            )
            return

        self.journal.set_position_state(position.position_id, TradeState.EXIT_REQUESTED)

        try:
            self._assert_paper_before_submit()
        except LiveTradingForbiddenError as exc:
            self.journal.set_order_state(
                intent.client_order_id,
                TradeState.REJECTED,
                f"paper recheck failed: {exc.detail}",
            )
            self.journal.set_position_state(position.position_id, TradeState.MONITORING)
            report.errors.append(f"exit {position.symbol}: {exc.detail}")
            return

        try:
            record = self.stack.broker.close_spread(
                intent, limit_cents, intent.client_order_id
            )
        except AmbiguousSubmissionError as exc:
            # The close may already be live. Never send a second one; the
            # reservation stays for reconciliation to resolve.
            log.error("ambiguous exit submission for %s: %s", intent.client_order_id, exc)
            self.journal.record_event(
                "ambiguous_exit_submission",
                {"client_order_id": intent.client_order_id, "cause": exc.cause},
                decision_id=position.decision_id,
                symbol=position.symbol,
            )
            report.errors.append(f"{position.symbol}: ambiguous exit, will reconcile")
            return
        except BrokerError as exc:
            self.journal.set_order_state(
                intent.client_order_id, TradeState.REJECTED, f"broker rejected: {exc}"
            )
            self.journal.set_position_state(position.position_id, TradeState.MONITORING)
            report.errors.append(f"exit {position.symbol}: broker rejected ({exc})")
            return

        self.journal.set_order_state(
            intent.client_order_id, TradeState.SUBMITTED, f"exit requested: {reason}"
        )
        self.journal.update_order_execution(record, TradeState.SUBMITTED)
        report.exit_orders_submitted.append(intent.client_order_id)
        log.info(
            "exit_order_submitted client_order_id=%s position_id=%s symbol=%s "
            "reason=%s limit_cents=%d qty=%d",
            intent.client_order_id,
            position.position_id,
            position.symbol,
            reason,
            limit_cents,
            position.quantity,
        )
        self.journal.record_event(
            "exit_order_submitted",
            {
                "position_id": position.position_id,
                "client_order_id": intent.client_order_id,
                "exit_reason": str(reason),
                "limit_price_cents": limit_cents,
                "detail": detail,
            },
            decision_id=position.decision_id,
            symbol=position.symbol,
        )

        # A close can fill immediately. Settle it now rather than waiting a
        # cycle, but through the same confirmation path as everything else.
        self._reconcile_exit_orders(now, report)

    def _log_unresolved_exit(
        self,
        position: PositionRecord,
        reason: ExitReason | None,
        why: str,
        now: datetime,
    ) -> None:
        """An exit that is due but cannot be priced. Never silent.

        An expiring position that cannot be closed is the loudest case: it will
        settle on its own terms rather than ours, so it is logged at warning and
        journalled rather than skipped.
        """
        expiring = position.expiration <= now.date()
        (log.warning if expiring else log.info)(
            "exit_unresolved position_id=%s symbol=%s reason=%s expiration=%s "
            "expiring=%s why=%s",
            position.position_id,
            position.symbol,
            reason,
            position.expiration.isoformat(),
            expiring,
            why,
        )
        self.journal.record_event(
            "exit_unresolved",
            {
                "position_id": position.position_id,
                "exit_reason": str(reason),
                "expiration": position.expiration.isoformat(),
                "expiring": expiring,
                "why": why,
            },
            decision_id=position.decision_id,
            symbol=position.symbol,
        )

    def _reconcile_exit_orders(self, now: datetime, report: CycleReport) -> None:
        """Resolve closing orders against the broker, and only then close.

        Four outcomes, and only the first retires a position:

        * fully filled -- realise from the actual fill and close;
        * partially filled -- real exposure is still open, so the position stays
          open and the order is left alone;
        * dead (cancelled, rejected, expired) -- the position returns to
          management and may be re-quoted next cycle;
        * still working -- nothing to do.
        """
        for row in self.journal.open_orders(kind=ORDER_KIND_EXIT):
            client_order_id = str(row["client_order_id"])
            position_id = str(row["position_id"] or "")
            position = self.journal.get_position(position_id) if position_id else None
            if position is None:
                continue

            try:
                record = self.monitor.refresh(client_order_id)
            except Exception as exc:
                log.warning("could not refresh exit %s: %s", client_order_id, exc)
                continue

            if record is None:
                # The broker has never heard of it. For a reservation that never
                # reached the wire -- an ambiguous submission, a crash between
                # reserving and sending -- that is conclusive: retire it and
                # give the position back to management, or it stays in
                # EXIT_REQUESTED forever with nothing able to raise a new close.
                # For an order we know was submitted, a lookup returning nothing
                # is ambiguous, not proof, so it is left alone.
                if TradeState(str(row["state"])) is TradeState.CONSTRUCTED:
                    self.journal.set_order_state(
                        client_order_id,
                        TradeState.FAILED,
                        "exit reserved but never reached the broker; retired",
                    )
                    self.journal.set_position_state(
                        position.position_id, TradeState.MONITORING
                    )
                    log.info(
                        "exit_reservation_retired client_order_id=%s position_id=%s; "
                        "position returned to management",
                        client_order_id,
                        position.position_id,
                    )
                continue

            status = record.status.lower()
            if status in DEAD_STATUSES:
                self.journal.set_position_state(position.position_id, TradeState.MONITORING)
                log.info(
                    "exit_order_dead client_order_id=%s position_id=%s status=%s; "
                    "position returned to management",
                    client_order_id,
                    position.position_id,
                    record.status,
                )
                continue

            if record.filled_quantity <= 0:
                continue

            if record.filled_quantity < position.quantity:
                # Never close on a partial. The remaining spreads are still real
                # exposure and the order is still working against them.
                log.info(
                    "exit_partially_filled client_order_id=%s position_id=%s "
                    "filled=%d of %d; position stays open",
                    client_order_id,
                    position.position_id,
                    record.filled_quantity,
                    position.quantity,
                )
                self.journal.record_event(
                    "exit_partially_filled",
                    {
                        "position_id": position.position_id,
                        "client_order_id": client_order_id,
                        "filled_quantity": record.filled_quantity,
                        "position_quantity": position.quantity,
                    },
                    symbol=position.symbol,
                )
                continue

            if record.filled_avg_price_cents is None:
                # Filled but unpriced: realised money cannot be established yet,
                # and inventing it is exactly the defect this replaced.
                log.warning(
                    "exit_filled_without_price client_order_id=%s position_id=%s; "
                    "leaving unresolved rather than inventing a realised value",
                    client_order_id,
                    position.position_id,
                )
                continue

            self._finalise_exit(position, record, client_order_id, now, report)

    def _finalise_exit(
        self,
        position: PositionRecord,
        record: ExecutionRecord,
        client_order_id: str,
        now: datetime,
        report: CycleReport,
    ) -> None:
        """Close one position from its actual closing fill.

        Realised money is entry fill against exit fill. No mark, no estimate,
        no fallback: both sides are prices the broker actually traded.
        """
        exit_price_cents = record.filled_avg_price_cents or 0
        exit_value = exit_price_cents * OPTION_MULTIPLIER * record.filled_quantity
        realized = exit_value - position.entry_debit_cents
        holding = (now - position.opened_at).total_seconds() / 60.0
        mfe, mae = self.journal.position_excursions(position.position_id)
        assessment = self._regimes.get(position.symbol)
        regime = assessment.regime if assessment is not None else Regime.UNKNOWN
        reason = self._exit_reason_for(client_order_id)

        outcome = TradeOutcome(
            position_id=position.position_id,
            decision_id=position.decision_id,
            symbol=position.symbol,
            strategy=position.strategy,
            regime=regime,
            confidence=0.0,
            quantity=record.filled_quantity,
            entry_debit_cents=position.entry_debit_cents,
            exit_value_cents=exit_value,
            realized_pnl_cents=realized,
            return_on_defined_risk=(
                realized / position.max_loss_cents if position.max_loss_cents else 0.0
            ),
            holding_minutes=holding,
            max_favorable_excursion_cents=mfe,
            max_adverse_excursion_cents=mae,
            exit_reason=reason,
            opened_at=position.opened_at,
            closed_at=now,
        )
        self.journal.record_outcome(outcome)
        self.journal.set_order_state(
            client_order_id, TradeState.CLOSED, f"exit filled: {reason}"
        )
        report.exits_taken.append(f"{position.symbol}:{reason}")
        log.info(
            "position_closed position_id=%s symbol=%s exit_fill_cents=%d "
            "entry_debit_cents=%d realized_cents=%d reason=%s",
            position.position_id,
            position.symbol,
            exit_price_cents,
            position.entry_debit_cents,
            realized,
            reason,
        )
        self.journal.record_event(
            "position_closed",
            {
                "position_id": position.position_id,
                "client_order_id": client_order_id,
                "exit_reason": str(reason),
                "exit_fill_price_cents": exit_price_cents,
                "exit_value_cents": exit_value,
                "realized_pnl_cents": realized,
                "basis": "BROKER_EXIT_FILL",
            },
            decision_id=position.decision_id,
            symbol=position.symbol,
        )

    def _exit_reason_for(self, client_order_id: str) -> ExitReason:
        """Recover the reason recorded when the closing order was raised."""
        for row in self.journal.transitions_for(client_order_id):
            detail = str(row.get("detail") or "")
            if not detail.startswith("exit requested: "):
                continue
            raw = detail.split(": ", 1)[1].strip()
            for candidate in ExitReason:
                if raw in (candidate.value, str(candidate)):
                    return candidate
        return ExitReason.MANUAL

    def _market_is_open(self) -> bool:
        """Whether the session is open, from the clock this cycle already read."""
        return bool(self._clock is not None and self._clock.is_open)

    def _expire_stale_exit_orders(self, now: datetime, report: CycleReport) -> None:
        """Re-quote a closing order the market has walked away from.

        An exit that never fills leaves real exposure unmanaged, which is the
        same failure as never sending one. The order is cancelled, the cancel is
        confirmed, and the position returns to management so the next cycle
        prices it against the current chain. A partially filled exit is never
        touched: cancelling one would strand the balance.
        """
        ttl = self.config.settings.exit_order_ttl_seconds
        if ttl <= 0 or not self._market_is_open():
            return

        for row in self.journal.open_orders(kind=ORDER_KIND_EXIT):
            if TradeState(str(row["state"])) is not TradeState.SUBMITTED:
                continue
            client_order_id = str(row["client_order_id"])
            created = _parse_journal_ts(row.get("created_at"))
            if created is None or (now - created).total_seconds() < ttl:
                continue

            try:
                record = self.monitor.refresh(client_order_id)
            except Exception as exc:
                log.warning("exit sweep could not refresh %s: %s", client_order_id, exc)
                continue
            if record is None or record.broker_order_id is None:
                continue
            if record.filled_quantity > 0:
                continue

            try:
                self.stack.broker.cancel_order(record.broker_order_id)
            except BrokerError as exc:
                log.warning("stale_exit_cancel_failed %s: %s", client_order_id, exc)
                continue

            confirmed = self.monitor.refresh(client_order_id)
            after = self.journal.get_order(client_order_id)
            released = after is not None and is_terminal(TradeState(str(after["state"])))
            if confirmed is None or confirmed.filled_quantity > 0 or not released:
                log.warning(
                    "stale_exit_cancel_unconfirmed client_order_id=%s status=%s",
                    client_order_id,
                    confirmed.status if confirmed else "unknown",
                )
                continue

            position_id = str(row["position_id"] or "")
            if position_id:
                self.journal.set_position_state(position_id, TradeState.MONITORING)
            report.exit_orders_repriced += 1
            log.info(
                "stale_exit_order_retired client_order_id=%s position_id=%s "
                "ttl_s=%d limit_price_cents=%s; will be re-quoted",
                client_order_id,
                position_id,
                ttl,
                row.get("limit_price_cents"),
            )
            self.journal.record_event(
                "stale_exit_order_retired",
                {
                    "client_order_id": client_order_id,
                    "position_id": position_id,
                    "ttl_seconds": ttl,
                },
                symbol=str(row.get("symbol") or "") or None,
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
