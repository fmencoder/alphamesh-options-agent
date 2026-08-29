#!/usr/bin/env python3
"""Turn saved Alpaca MCP responses into AlphaMesh capture files.

Workflow, run from any MCP host connected to the Alpaca MCP server
(Claude Code, Claude Desktop, or another MCP client):

    1. Call the Alpaca MCP tools and save each raw response to a JSON file:

         get_account_info                            -> get_account_info.json
         get_clock                                   -> get_clock.json
         get_stock_bars(symbols="SPY,QQQ",
                        timeframe="1Min", feed="sip") -> bars.json
         get_option_chain(underlying_symbol="SPY",
                          type="call", ...)           -> chain_SPY_call.json

    2. Run this script over them:

         python scripts/mcp_capture.py --bars bars.json \
             --chain SPY=chain_SPY_call.json --out data/mcp_capture

The parsing is the same production code the agent uses
(``alphamesh.alpaca.mcp_adapter``), so what lands on disk is exactly what
Alpaca returned. Nothing is synthesised: given no input file, nothing is
written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphamesh.alpaca.mcp_adapter import (
    ingest_account_info,
    ingest_clock,
    ingest_option_chain,
    ingest_stock_bars,
    write_bar_capture,
    write_option_capture,
)
from alphamesh.safety import LiveTradingForbiddenError, check_account_number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, help="saved get_stock_bars response")
    parser.add_argument(
        "--chain",
        action="append",
        default=[],
        metavar="SYMBOL=PATH",
        help="saved get_option_chain response, e.g. SPY=chain_SPY_call.json",
    )
    parser.add_argument("--account", type=Path, help="saved get_account_info response")
    parser.add_argument("--clock", type=Path, help="saved get_clock response")
    parser.add_argument("--day", default=None, help="capture date stamp, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=Path("data/mcp_capture"))
    args = parser.parse_args(argv)

    written: list[str] = []

    if args.account:
        account = ingest_account_info(json.loads(args.account.read_text()))
        try:
            check_account_number(account.account_number)
        except LiveTradingForbiddenError as exc:
            print(f"REFUSING capture: {exc.detail}", file=sys.stderr)
            return 2
        print(
            f"account: PAPER verified, status={account.status}, "
            f"options level={account.options_trading_level}, equity={account.equity}"
        )

    if args.clock:
        clock = ingest_clock(json.loads(args.clock.read_text()))
        print(f"clock: market_open={clock.is_open} at {clock.timestamp.isoformat()}")

    if args.bars:
        for symbol, bars in ingest_stock_bars(json.loads(args.bars.read_text())).items():
            if not bars:
                continue
            path = write_bar_capture(bars, symbol, args.out)
            written.append(f"{path} ({len(bars)} bars)")

    contracts = []
    for spec in args.chain:
        if "=" not in spec:
            parser.error(f"--chain expects SYMBOL=PATH, got {spec!r}")
        symbol, _, path = spec.partition("=")
        contracts.extend(
            ingest_option_chain(json.loads(Path(path).read_text()), symbol.upper())
        )
    if contracts:
        day = args.day or contracts[0].expiration.isoformat()
        path = write_option_capture(contracts, args.out, day)
        written.append(f"{path} ({len(contracts)} contracts)")

    if not written:
        print("nothing to write; pass --bars and/or --chain", file=sys.stderr)
        return 1
    for line in written:
        print(f"wrote {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
