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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any

from alphamesh import __version__
from alphamesh.alpaca.cli_adapter import AlpacaCliAdapter
from alphamesh.alpaca.client import build_stack
from alphamesh.alpaca.market_data import CaptureMarketData
from alphamesh.alpaca.mcp_adapter import describe_mcp_usage
from alphamesh.analytics import build_report
from alphamesh.config import AppConfig, load_config
from alphamesh.intelligence.reasoning import build_provider
from alphamesh.orchestrator import Orchestrator
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
def cmd_preflight(config: AppConfig) -> int:
    """Report on safety, configuration and reachability. Never places an order."""
    print(banner(config.settings))
    result: dict[str, Any] = {
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
    }

    try:
        stack = build_stack(config.settings)
    except LiveTradingForbiddenError as exc:
        result["paper_guard"] = {"passed": False, "detail": exc.detail}
        print(json.dumps(result, indent=2))
        print("\nPREFLIGHT FAILED: paper mode could not be established.", file=sys.stderr)
        return 2
    except Exception as exc:
        result["stack"] = {"built": False, "error": str(exc)}
        print(json.dumps(result, indent=2))
        return 3

    result["paper_guard"] = {"passed": True, "checks": list(stack.guard.checks)}
    result["broker"] = {
        "live_broker": stack.live_broker,
        "kind": type(stack.broker).__name__,
    }

    try:
        account = stack.broker.account()
        result["account"] = {
            "paper_prefix_verified": True,
            "status": account.status,
            "equity": account.equity,
            "options_trading_level": account.options_trading_level,
            "options_buying_power": account.options_buying_power,
            "tradeable": account.is_tradeable,
        }
    except Exception as exc:
        result["account"] = {"reachable": False, "error": str(exc)}

    market: dict[str, Any] = {}
    for symbol in config.universe.symbols:
        try:
            snapshot = stack.market_data.snapshot(
                symbol, lookback_minutes=config.universe.bar_lookback_minutes
            )
            market[symbol] = {
                "bars": snapshot.bar_count,
                "last_price": snapshot.last_price,
                "as_of": snapshot.as_of.isoformat(),
            }
        except Exception as exc:
            market[symbol] = {"error": str(exc)}
    result["market_data"] = market

    chains: dict[str, Any] = {}
    from alphamesh.models.domain import OptionType

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
                chains[key] = {
                    "contracts": len(chain),
                    "with_greeks": sum(1 for c in chain if c.greeks.delta is not None),
                    "with_quotes": sum(1 for c in chain if c.quote is not None),
                }
            except Exception as exc:
                chains[key] = {"error": str(exc)}
    result["option_chains"] = chains

    print(json.dumps(result, indent=2))
    return 0


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
        latest = max(
            orchestrator.stack.market_data.snapshot(
                symbol, lookback_minutes=config.universe.bar_lookback_minutes
            ).as_of
            for symbol in config.universe.symbols
        )
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


def cmd_run(config: AppConfig) -> int:
    print(banner(config.settings))
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    orchestrator, journal = _build(config)
    interval = max(5, config.settings.loop_seconds)
    log.info("autonomous loop starting; interval=%ss", interval)
    try:
        orchestrator.startup()
        while not _SHUTDOWN:
            started = time.monotonic()
            try:
                report = orchestrator.run_cycle()
                log.info(
                    "cycle: scanned=%d orders=%d exits=%d rejections=%d errors=%d",
                    report.symbols_scanned,
                    len(report.orders_submitted),
                    len(report.exits_taken),
                    len(report.rejections),
                    len(report.errors),
                )
            except Exception:
                log.exception("cycle failed; continuing")
            elapsed = time.monotonic() - started
            for _ in range(int(max(0.0, interval - elapsed))):
                if _SHUTDOWN:
                    break
                time.sleep(1)
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
