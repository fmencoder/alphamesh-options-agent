"""AlphaMesh command-line entrypoint.

Subcommands
-----------
``run``         The autonomous agent loop. Runs headless; no operator required.
``once``        A single lifecycle cycle, for smoke tests and CI.
``replay``      One cycle over captured Alpaca data with the clock pinned to it.
``replay-session`` Walk-forward replay of the whole captured session (simulated fills).
``preflight``   Safety, configuration and connectivity report. Places no orders.
``report``      Competition analytics from the journal.
``mcp-info``    Exactly which Alpaca MCP tools AlphaMesh uses, and why.
``cli-info``    Whether the Alpaca CLI operational path is available here.
``health``      One-line JSON health record for a deployment probe.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any

from alphamesh import __version__
from alphamesh.alpaca.cli_adapter import AlpacaCliAdapter
from alphamesh.alpaca.client import build_stack
from alphamesh.alpaca.market_data import (
    LOOKBACK_CALENDAR_DAYS,
    CaptureMarketData,
    MarketDataUnavailableError,
)
from alphamesh.alpaca.mcp_adapter import describe_mcp_usage
from alphamesh.analytics import build_report
from alphamesh.config import AppConfig, load_config
from alphamesh.intelligence.reasoning import build_provider
from alphamesh.orchestrator import CycleReport, Orchestrator
from alphamesh.persistence.journal import Journal
from alphamesh.safety import LiveTradingForbiddenError, banner

log = logging.getLogger("alphamesh")

_SHUTDOWN = False


def _handle_signal(signum: int, frame: FrameType | None) -> None:
    global _SHUTDOWN
    _SHUTDOWN = True
    log.info("signal %s received; finishing the current cycle then stopping", signum)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _load() -> AppConfig:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return load_config()


def _build(config: AppConfig) -> tuple[Orchestrator, Journal]:
    stack = build_stack(config.settings)
    journal = Journal(config.settings.database_path)
    provider = build_provider(
        config.settings.anthropic_api_key, config.settings.llm_model
    )
    return Orchestrator(config, stack, journal, provider), journal


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def _interruptible_sleep(seconds: float) -> None:
    """Sleep in short slices so SIGTERM is honoured promptly.

    Railway sends SIGTERM and then SIGKILLs after a grace period; a long
    uninterruptible sleep would be killed mid-cycle instead of shutting down.
    """
    deadline = time.monotonic() + seconds
    while not _SHUTDOWN:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def _closed_market_wait(report: Any, closed_interval: int) -> float:
    """Seconds to wait while closed: the backoff, but never past the next open."""
    if report.next_open is None:
        return float(closed_interval)
    until_open = (report.next_open - datetime.now(UTC)).total_seconds()
    if until_open <= 0:
        return float(min(closed_interval, 60))
    return float(max(5.0, min(closed_interval, until_open)))


class OrderSubmissionForbiddenError(RuntimeError):
    """Raised if anything in preflight tries to place an order."""


class _ZeroOrderBroker:
    """Read-only proxy around a broker.

    Preflight is run against the real competition account, so "it places no
    orders" must be structural rather than a promise. Every read is forwarded;
    every write raises.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def account(self) -> Any:
        return self._inner.account()

    def positions(self) -> Any:
        return self._inner.positions()

    def get_order_by_client_id(self, client_order_id: str) -> Any:
        return self._inner.get_order_by_client_id(client_order_id)

    def working_order_symbols(self) -> Any:
        return self._inner.working_order_symbols()

    def recent_orders(self, *a: Any, **k: Any) -> Any:
        return self._inner.recent_orders(*a, **k)

    def submit_spread(self, *_a: Any, **_k: Any) -> Any:
        raise OrderSubmissionForbiddenError("preflight must never submit an order")

    def close_spread(self, *_a: Any, **_k: Any) -> Any:
        raise OrderSubmissionForbiddenError("preflight must never close a position")

    def cancel_order(self, *_a: Any, **_k: Any) -> Any:
        raise OrderSubmissionForbiddenError("preflight must never cancel an order")


# Checks whose failure makes autonomous execution unsafe or impossible.
EXECUTION_CRITICAL = (
    "PREFLIGHT_PAPER_MODE",
    "PREFLIGHT_ACCOUNT",
    "PREFLIGHT_MARKET_DATA",
    "PREFLIGHT_OPTIONS_CHAIN",
    "PREFLIGHT_GREEKS",
    "PREFLIGHT_JOURNAL",
    "PREFLIGHT_RECOVERY",
)


def cmd_preflight(config: AppConfig) -> int:
    """Authoritative readiness report. Places ZERO orders, structurally.

    Exits non-zero when any execution-critical dependency fails, so it can gate
    a deploy or a start script.
    """
    from alphamesh.models.domain import OptionType

    print(banner(config.settings))

    flags: dict[str, str] = {
        "PREFLIGHT_PAPER_MODE": "FAIL",
        "PREFLIGHT_ACCOUNT": "FAIL",
        "PREFLIGHT_MARKET_DATA": "FAIL",
        "PREFLIGHT_OPTIONS_CHAIN": "FAIL",
        "PREFLIGHT_GREEKS": "FAIL",
        "PREFLIGHT_JOURNAL": "FAIL",
        "PREFLIGHT_RECOVERY": "FAIL",
        "PREFLIGHT_AI_PROVIDER": "FAIL",
        "PREFLIGHT_READY": "NO",
    }
    detail: dict[str, Any] = {
        "version": __version__,
        "settings": config.settings.redacted(),
        "risk_limits": {
            "max_defined_loss_per_trade": config.risk.max_defined_loss_per_trade,
            "high_confidence_max_defined_loss": config.risk.high_confidence_max_defined_loss,
            "absolute_max_defined_loss": config.risk.absolute_max_defined_loss,
            "max_open_positions": config.risk.max_open_positions,
            "max_portfolio_defined_risk": config.risk.max_portfolio_defined_risk,
            "daily_loss_circuit_breaker": config.risk.daily_loss_circuit_breaker,
            "allowed_strategies": config.risk.allowed_strategies,
        },
        "universe": config.universe.symbols,
        "orders_submitted": 0,
    }

    # ---------------------------------------------------------- paper mode
    try:
        stack = build_stack(config.settings)
    except LiveTradingForbiddenError as exc:
        detail["paper_guard"] = {"passed": False, "detail": exc.detail}
        return _emit_preflight(detail, flags, exit_code=2)
    except Exception as exc:
        detail["stack"] = {"built": False, "error": str(exc)}
        return _emit_preflight(detail, flags, exit_code=3)

    flags["PREFLIGHT_PAPER_MODE"] = "PASS"
    detail["paper_guard"] = {"passed": True, "checks": list(stack.guard.checks)}
    detail["execution_mode"] = stack.execution_mode
    detail["broker"] = {"live_broker": stack.live_broker, "kind": type(stack.broker).__name__}
    if not stack.live_broker:
        detail["execution_mode_warning"] = (
            "SIMULATED: ALPHAMESH_DRY_RUN is true, so no order would reach Alpaca "
            "and any P&L recorded would be simulated."
        )

    broker = _ZeroOrderBroker(stack.broker)

    # ------------------------------------------------------------- account
    try:
        account = broker.account()
        tradeable = account.is_tradeable and account.options_trading_level >= 3
        detail["account"] = {
            "paper_prefix_verified": True,
            "status": account.status,
            "active": account.status.upper() == "ACTIVE",
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
            "options_trading_level": account.options_trading_level,
            "equity": account.equity,
            "buying_power": account.buying_power,
            "options_buying_power": account.options_buying_power,
            "tradeable": tradeable,
        }
        flags["PREFLIGHT_ACCOUNT"] = "PASS" if tradeable else "FAIL"
    except Exception as exc:
        detail["account"] = {"reachable": False, "error": str(exc)}

    # --------------------------------------------------------------- clock
    try:
        clock = stack.market_data.clock()
        detail["clock"] = {
            "is_open": clock.is_open,
            "timestamp": clock.timestamp.isoformat(),
            "next_open": clock.next_open.isoformat() if clock.next_open else None,
            "next_close": clock.next_close.isoformat() if clock.next_close else None,
        }
    except Exception as exc:
        detail["clock"] = {"reachable": False, "error": str(exc)}

    # --------------------------------------------------------- market data
    # Two independent probes per symbol, because a closed market must not look
    # like a broken feed:
    #   latest_bar  - proves live REST market-data access even on a weekend or
    #                 holiday, since it returns the last completed session's bar
    #   snapshot    - proves enough real historical depth for the feature engine
    # MARKET_CLOSED is reported, never treated as a failure. Only genuinely
    # absent or unreadable data fails the gate.
    market: dict[str, Any] = {}
    market_ok = True
    for symbol in config.universe.symbols:
        entry: dict[str, Any] = {"feed": getattr(stack.market_data, "feed", "n/a")}
        latest_ok = False
        depth_ok = False

        try:
            latest = stack.market_data.latest_bar(symbol)
            entry["latest_bar_timestamp"] = latest.timestamp.isoformat()
            entry["latest_bar_close"] = latest.close
            entry["latest_bar_source"] = "alpaca_latest_bar"
            latest_ok = True
        except Exception as exc:
            entry["latest_bar_error"] = _describe_data_error(exc)

        try:
            snapshot = stack.market_data.snapshot(
                symbol, lookback_minutes=config.universe.bar_lookback_minutes
            )
            depth_ok = snapshot.bar_count >= config.universe.min_bars_required
            entry.update(
                {
                    "bars": snapshot.bar_count,
                    "min_required": config.universe.min_bars_required,
                    "sufficient": depth_ok,
                    "last_price": snapshot.last_price,
                    "as_of": snapshot.as_of.isoformat(),
                    "lookback_calendar_days": LOOKBACK_CALENDAR_DAYS,
                }
            )
        except Exception as exc:
            entry["historical_error"] = _describe_data_error(exc)

        entry["status"] = (
            "OK" if latest_ok and depth_ok else "MARKET_DATA_UNAVAILABLE"
        )
        market[symbol] = entry
        market_ok = market_ok and latest_ok and depth_ok

    detail["market_data"] = market
    detail["equities_feed"] = config.settings.equities_feed
    detail["market_data_note"] = (
        "A closed market is not a failure: latest_bar returns the last completed "
        "session, so this gate distinguishes MARKET_CLOSED from "
        "MARKET_DATA_UNAVAILABLE."
    )
    flags["PREFLIGHT_MARKET_DATA"] = "PASS" if market_ok and market else "FAIL"

    # -------------------------------------------------------- option chain
    chains: dict[str, Any] = {}
    chain_ok = True
    greeks_ok = True
    for symbol in config.universe.symbols:
        for option_type in (OptionType.CALL, OptionType.PUT):
            key = f"{symbol}:{option_type.value}"
            try:
                chain = stack.option_chain.chain(
                    symbol,
                    option_type,
                    as_of=datetime.now(UTC).date(),
                    min_dte=config.strategies.min_dte,
                    max_dte=config.strategies.max_dte,
                )
                with_greeks = sum(1 for c in chain if c.greeks.delta is not None)
                with_quotes = sum(
                    1 for c in chain if c.quote is not None and c.quote.bid > 0
                )
                chains[key] = {
                    "contracts": len(chain),
                    "with_greeks": with_greeks,
                    "with_usable_quotes": with_quotes,
                }
                chain_ok = chain_ok and len(chain) > 0
                greeks_ok = greeks_ok and with_greeks > 0 and with_quotes > 0
            except Exception as exc:
                chains[key] = {"error": str(exc)}
                chain_ok = False
                greeks_ok = False
    detail["option_chains"] = chains
    flags["PREFLIGHT_OPTIONS_CHAIN"] = "PASS" if chain_ok and chains else "FAIL"
    flags["PREFLIGHT_GREEKS"] = "PASS" if greeks_ok and chains else "FAIL"

    # ------------------------------------------------- journal and recovery
    journal: Journal | None = None
    try:
        journal = Journal(config.settings.database_path)
        journal.record_event("preflight", {"version": __version__})
        detail["journal"] = {
            "path": str(config.settings.database_path),
            "writable": True,
            "open_positions": len(journal.open_positions()),
            "closed_trades": len(journal.outcomes()),
            "open_orders": len(journal.open_orders()),
        }
        flags["PREFLIGHT_JOURNAL"] = "PASS"
    except Exception as exc:
        detail["journal"] = {"writable": False, "error": str(exc)}

    if journal is not None:
        try:
            provider = build_provider(
                config.settings.anthropic_api_key, config.settings.llm_model
            )
            probe = Orchestrator(config, stack, journal, provider)
            probe.stack.broker = broker  # type: ignore[assignment]
            recovery = probe.startup()
            detail["recovery"] = recovery
            flags["PREFLIGHT_RECOVERY"] = "PASS"
        except Exception as exc:
            detail["recovery"] = {"error": str(exc)}
        finally:
            journal.close()

    # --------------------------------------------------------- AI provider
    provider = build_provider(config.settings.anthropic_api_key, config.settings.llm_model)
    detail["ai_provider"] = {
        "provider": provider.name,
        "available": provider.available(),
        "model": config.settings.llm_model if provider.available() else None,
        "fallback": "deterministic heuristic council",
        "note": (
            "No key configured: the council runs on deterministic heuristics. "
            "The agent still trades."
            if not provider.available()
            else "LLM council active; heuristics remain the fallback."
        ),
    }
    # The heuristic fallback is always present, so this check passes either way;
    # it reports which path is live rather than gating on the LLM.
    flags["PREFLIGHT_AI_PROVIDER"] = "PASS"

    failed = [k for k in EXECUTION_CRITICAL if flags[k] != "PASS"]
    detail["failed_critical_checks"] = failed
    flags["PREFLIGHT_READY"] = "YES" if not failed else "NO"
    return _emit_preflight(detail, flags, exit_code=0 if not failed else 1)


def _describe_data_error(exc: Exception) -> str:
    """Classify a market-data failure precisely instead of reporting a bare string.

    Entitlement and auth problems must never be reported as "no data" - they
    need a different fix from an empty window.
    """
    text = str(exc)
    lowered = text.lower()
    if "401" in text or "unauthorized" in lowered:
        return f"AUTH_FAILED (401): {text[:300]}"
    if "403" in text or "forbidden" in lowered or "subscription" in lowered:
        return f"FEED_NOT_ENTITLED (403): {text[:300]}"
    if "no bars" in lowered or "no latest bar" in lowered or "empty" in lowered:
        return f"NO_DATA_RETURNED: {text[:300]}"
    return f"{type(exc).__name__}: {text[:300]}"


def _emit_preflight(detail: dict[str, Any], flags: dict[str, str], exit_code: int) -> int:
    """Print the human report then the machine-readable flag block."""
    print(json.dumps(detail, indent=2, default=str))
    print()
    for key, value in flags.items():
        print(f"{key}={value}")
    if exit_code != 0:
        print(
            f"\nPREFLIGHT NOT READY (exit {exit_code}): "
            f"{', '.join(detail.get('failed_critical_checks') or ['paper mode'])}",
            file=sys.stderr,
        )
    return exit_code


def cmd_replay(config: AppConfig) -> int:
    """Replay the captured session with the clock pinned to the capture.

    This exercises the entire lifecycle - scoring, regime, council, contract
    selection, risk approval, order construction and submission - against real
    Alpaca data, without a live market and without a live broker. Orders go to
    the in-process simulator, never to Alpaca.
    """
    if config.settings.data_source != "mcp_capture":
        print(
            "replay requires ALPHAMESH_DATA_SOURCE=mcp_capture", file=sys.stderr
        )
        return 2

    print(banner(config.settings))
    orchestrator, journal = _build(config)
    try:
        orchestrator.startup()
        # Pin the clock just after the newest captured bar so the freshness
        # gates evaluate the captured quotes as current, exactly as they were.
        #
        # The capture covers the symbols that were recorded, which is not
        # necessarily the whole configured universe: the universe grew to eight
        # symbols while the capture holds two. A replay is a walk over captured
        # data, so an uncaptured symbol is simply absent here rather than a
        # failure -- it is skipped and named. An empty capture is still an
        # error, so the command keeps its value as a CI gate.
        as_of: list[datetime] = []
        missing: list[str] = []
        for symbol in config.universe.symbols:
            try:
                as_of.append(
                    orchestrator.stack.market_data.snapshot(
                        symbol, lookback_minutes=config.universe.bar_lookback_minutes
                    ).as_of
                )
            except MarketDataUnavailableError:
                missing.append(symbol)
        if not as_of:
            print(
                f"no captured bars for any universe symbol in "
                f"{config.settings.capture_dir}",
                file=sys.stderr,
            )
            return 4
        if missing:
            print(f"  NOT CAPTURED: {', '.join(missing)} (skipped)")
        latest = max(as_of)
        pinned = latest.replace(tzinfo=UTC) + timedelta(seconds=30)
        print(f"  REPLAY AS-OF: {pinned.isoformat()} (pinned to captured data)\n")
        report = orchestrator.run_cycle(now=pinned)
        print(json.dumps(report.as_dict(), indent=2))
    finally:
        journal.close()
    return 0


def cmd_replay_session(config: AppConfig, args: argparse.Namespace) -> int:
    """Walk the captured session bar by bar, running the full lifecycle each step.

    Every decision, risk verdict, order and exit is real code on real Alpaca
    market data. Fills, however, come from the in-process simulator, so any P&L
    printed here is SIMULATED and is not Alpaca paper P&L. It is a behavioural
    demonstration, not a track record.
    """
    if config.settings.data_source != "mcp_capture":
        print("replay-session requires ALPHAMESH_DATA_SOURCE=mcp_capture", file=sys.stderr)
        return 2

    # Simulated fills must never land in the journal a judge reads for P&L, so
    # this command always writes to its own database beside the real one.
    real_db = Path(config.settings.database_path)
    replay_db = real_db.with_name(f"{real_db.stem}.replay{real_db.suffix or '.db'}")
    config = config.model_copy(
        update={"settings": config.settings.model_copy(update={"database_path": replay_db})}
    )

    overrides: dict[str, float] = {}
    if args.threshold is not None:
        overrides["quant_score_threshold"] = float(args.threshold)
    if args.min_confidence is not None:
        overrides["min_judge_confidence"] = float(args.min_confidence)
    if overrides:
        config = config.model_copy(
            update={"strategies": config.strategies.model_copy(update=overrides)}
        )

    print(banner(config.settings))
    print("=" * 72)
    print("  SIMULATED WALK-FORWARD REPLAY OVER CAPTURED ALPACA DATA")
    print("  Fills come from the in-process simulator. Any P&L below is")
    print("  SIMULATED and is NOT Alpaca paper account P&L.")
    print(f"  journal: {replay_db}  (kept separate from the live journal)")
    print(f"  quant_score_threshold  = {config.strategies.quant_score_threshold}")
    print(f"  min_judge_confidence   = {config.strategies.min_judge_confidence}")
    print("=" * 72)

    orchestrator, journal = _build(config)
    source = orchestrator.stack.market_data
    if not isinstance(source, CaptureMarketData):
        print("replay-session needs the capture market-data source", file=sys.stderr)
        journal.close()
        return 2

    try:
        orchestrator.startup()
        for symbol in config.universe.symbols:
            source._load(symbol)
        total = min(len(bars) for bars in source._bars.values())
        start = config.universe.min_bars_required
        stop = min(total, start + int(args.steps)) if args.steps else total

        scores: list[float] = []
        gate_passes = 0
        orders: list[str] = []
        exits: list[str] = []
        rejections: dict[str, int] = {}

        for end in range(start, stop + 1):
            for symbol in config.universe.symbols:
                source._cursor[symbol] = end
            pinned = source.snapshot(
                config.universe.symbols[0], config.universe.bar_lookback_minutes
            ).as_of + timedelta(seconds=30)
            report = orchestrator.run_cycle(now=pinned)
            for d in report.decisions:
                scores.append(d.quant_score)
                if d.strategy.value != "NO_TRADE":
                    gate_passes += 1
            orders.extend(report.orders_submitted)
            exits.extend(report.exits_taken)
            for _symbol, codes in report.rejections:
                for code in codes:
                    rejections[code.value] = rejections.get(code.value, 0) + 1

        outcomes = journal.outcomes()
        summary = {
            "steps": stop - start + 1,
            "decisions_evaluated": len(scores),
            "max_quant_score": round(max(scores), 4) if scores else None,
            "mean_quant_score": round(sum(scores) / len(scores), 4) if scores else None,
            "tradable_decisions": gate_passes,
            "orders_submitted": len(orders),
            "exits_taken": len(exits),
            "risk_rejections_by_code": rejections,
            "closed_trades": len(outcomes),
            "simulated_pnl_usd": round(
                sum(int(o["realized_pnl_cents"]) for o in outcomes) / 100, 2
            ),
            "pnl_note": "SIMULATED fills. Not Alpaca paper account P&L.",
        }
        print(json.dumps(summary, indent=2))
        if len(outcomes) > 0:
            print("\nCompetition report (SIMULATED):")
            rep = build_report(outcomes)
            rep.pop("_unused", None)
            print(json.dumps(rep, indent=2))
    finally:
        journal.close()
    return 0


def cmd_once(config: AppConfig) -> int:
    print(banner(config.settings))
    orchestrator, journal = _build(config)
    try:
        orchestrator.startup()
        report = orchestrator.run_cycle()
        print(json.dumps(report.as_dict(), indent=2))
    finally:
        journal.close()
    return 0


@dataclass
class _Funnel:
    """Rolling aggregate of the discovery-to-fill funnel.

    Emitted on a wall-clock cadence rather than per cycle: at a 30-second loop
    the per-cycle line is too noisy to read, and the funnel is what actually
    shows where candidates are being lost.
    """

    scans: int = 0
    quant_pass: int = 0
    ai_tradable: int = 0
    contract_selected: int = 0
    risk_approved: int = 0
    orders_submitted: int = 0
    fills: int = 0
    exit_orders: int = 0
    exits: int = 0
    adopted: int = 0
    ambiguous: int = 0
    open_positions: int = 0
    realized_pnl_cents: int = 0
    unrealized_pnl_cents: int = 0

    def absorb(self, report: CycleReport) -> None:
        self.scans += report.symbols_scanned
        self.quant_pass += report.quant_passes
        self.ai_tradable += report.ai_tradable
        self.contract_selected += report.contracts_selected
        self.risk_approved += report.risk_approved
        self.orders_submitted += len(report.orders_submitted)
        # Entry fills, not exits. Counting exits here reported FILLS=0 while
        # three spreads were filled and open.
        self.fills += report.entry_fills
        # EXITS counts positions the broker confirmed closed. Exit orders
        # that are merely on the wire are reported separately: a submitted
        # close is not a closed position.
        self.exits += len(report.exits_taken)
        self.exit_orders += len(report.exit_orders_submitted)
        self.adopted += report.positions_adopted
        self.ambiguous = report.ambiguous_broker_positions
        # Point-in-time, not cumulative.
        self.open_positions = report.open_positions
        self.realized_pnl_cents = report.realized_pnl_cents
        self.unrealized_pnl_cents = report.unrealized_pnl_cents

    def render(self) -> str:
        return (
            f"funnel SCANS={self.scans} QUANT_PASS={self.quant_pass} "
            f"AI_TRADABLE={self.ai_tradable} CONTRACT_SELECTED={self.contract_selected} "
            f"RISK_APPROVED={self.risk_approved} ORDERS_SUBMITTED={self.orders_submitted} "
            f"FILLS={self.fills} EXIT_ORDERS={self.exit_orders} EXITS={self.exits} "
            f"ADOPTED={self.adopted} AMBIGUOUS={self.ambiguous} "
            f"OPEN_POSITIONS={self.open_positions} "
            f"REALIZED_PNL=${self.realized_pnl_cents / 100:.2f} "
            f"UNREALIZED_PNL=${self.unrealized_pnl_cents / 100:.2f}"
        )


FUNNEL_INTERVAL_SECONDS = 300


def cmd_run(config: AppConfig) -> int:
    print(banner(config.settings))
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    orchestrator, journal = _build(config)
    interval = max(5, config.settings.loop_seconds)
    closed_interval = max(interval, config.settings.closed_poll_seconds)
    log.info(
        "autonomous loop starting; open=%ss closed=%ss mode=%s",
        interval,
        closed_interval,
        orchestrator.stack.execution_mode,
    )
    funnel = _Funnel()
    last_funnel = time.monotonic()
    try:
        orchestrator.startup()
        while not _SHUTDOWN:
            started = time.monotonic()
            wait: float = float(interval)
            try:
                report = orchestrator.run_cycle()
                if report.market_open:
                    funnel.absorb(report)
                    log.info(
                        "cycle: scanned=%d quant_pass=%d ai_tradable=%d selected=%d "
                        "risk_approved=%d orders=%d exit_orders=%d exits=%d "
                        "rejections=%d errors=%d",
                        report.symbols_scanned,
                        report.quant_passes,
                        report.ai_tradable,
                        report.contracts_selected,
                        report.risk_approved,
                        len(report.orders_submitted),
                        len(report.exit_orders_submitted),
                        len(report.exits_taken),
                        len(report.rejections),
                        len(report.errors),
                    )
                    if time.monotonic() - last_funnel >= FUNNEL_INTERVAL_SECONDS:
                        log.info("%s", funnel.render())
                        last_funnel = time.monotonic()
                else:
                    # Closed market: back off hard rather than re-scanning dead
                    # quotes every interval. Never sleep past the next open.
                    wait = _closed_market_wait(report, closed_interval)
                    log.info(
                        "market closed; next open %s, sleeping %ss",
                        report.next_open.isoformat() if report.next_open else "unknown",
                        wait,
                    )
            except Exception:
                log.exception("cycle failed; continuing")
            elapsed = time.monotonic() - started
            _interruptible_sleep(max(0.0, wait - elapsed))
    finally:
        journal.close()
        log.info("autonomous loop stopped")
    return 0


def cmd_report(config: AppConfig) -> int:
    journal = Journal(config.settings.database_path)
    try:
        outcomes = journal.outcomes()
        report = build_report(outcomes)
        report.pop("_unused", None)
        report["closed_trades"] = len(outcomes)
        if not outcomes:
            report["note"] = "No closed trades recorded. No P&L to report."
        print(json.dumps(report, indent=2))
    finally:
        journal.close()
    return 0


def cmd_mcp_info(_: AppConfig) -> int:
    print(describe_mcp_usage())
    return 0


def cmd_cli_info(config: AppConfig) -> int:
    adapter = AlpacaCliAdapter(config.settings.alpaca_cli_path)
    print(json.dumps(adapter.status(), indent=2))
    return 0


def cmd_health(config: AppConfig) -> int:
    """Machine-readable health line for a deployment probe."""
    payload: dict[str, Any] = {
        "service": "alphamesh",
        "version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "paper": config.settings.paper,
        "dry_run": config.settings.dry_run,
        "data_source": config.settings.data_source,
    }
    try:
        stack = build_stack(config.settings)
        payload["paper_guard"] = "passed"
        payload["broker"] = type(stack.broker).__name__
        payload["status"] = "ok"
    except LiveTradingForbiddenError as exc:
        payload["paper_guard"] = "failed"
        payload["detail"] = exc.detail
        payload["status"] = "blocked"
        print(json.dumps(payload))
        return 2
    except Exception as exc:
        payload["status"] = "degraded"
        payload["detail"] = str(exc)
        print(json.dumps(payload))
        return 1
    try:
        journal = Journal(config.settings.database_path)
        payload["open_positions"] = len(journal.open_positions())
        payload["closed_trades"] = len(journal.outcomes())
        journal.close()
    except Exception as exc:
        payload["journal"] = f"unavailable: {exc}"
    print(json.dumps(payload))
    return 0


# Commands taking only the config. ``replay-session`` also needs the parsed
# arguments, so it is dispatched separately rather than widening this signature.
COMMANDS: dict[str, Callable[[AppConfig], int]] = {
    "run": cmd_run,
    "once": cmd_once,
    "replay": cmd_replay,
    "preflight": cmd_preflight,
    "report": cmd_report,
    "mcp-info": cmd_mcp_info,
    "cli-info": cmd_cli_info,
    "health": cmd_health,
}
ARG_COMMANDS: dict[str, Callable[[AppConfig, argparse.Namespace], int]] = {
    "replay-session": cmd_replay_session,
}
ALL_COMMANDS = sorted({*COMMANDS, *ARG_COMMANDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alphamesh", description="AlphaMesh autonomous options agent (PAPER ONLY)"
    )
    parser.add_argument("command", choices=ALL_COMMANDS, help="subcommand to run")
    parser.add_argument("--version", action="version", version=f"alphamesh {__version__}")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="replay-session only: override the quant score gate for this run",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="replay-session only: override the judge confidence floor for this run",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="replay-session only: limit the number of walk-forward steps (0 = whole capture)",
    )
    args = parser.parse_args(argv)

    config = _load()
    configure_logging(config.settings.log_level)

    try:
        if args.command in ARG_COMMANDS:
            return ARG_COMMANDS[args.command](config, args)
        return COMMANDS[args.command](config)
    except LiveTradingForbiddenError as exc:
        log.error("STARTUP BLOCKED: %s", exc.detail)
        print(
            f"\nSTARTUP BLOCKED - {exc.detail}\n"
            "AlphaMesh only runs against Alpaca paper trading.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
