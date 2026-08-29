"""Alpaca MCP ingestion, tested against verbatim MCP responses.

The files under ``data/mcp_capture/raw`` are real Alpaca MCP tool output. These
tests parse them with production code and cross-check the results against the
CSV captures the agent actually replays, which is what ties the committed
fixtures back to the MCP server they came from.
"""

from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from alphamesh.alpaca.mcp_adapter import (
    MCP_TOOLS_USED,
    describe_mcp_usage,
    ingest_account_info,
    ingest_clock,
    ingest_option_chain,
    ingest_stock_bars,
    write_bar_capture,
    write_option_capture,
)
from alphamesh.alpaca.options import build_occ_symbol, parse_occ_symbol
from alphamesh.models.domain import OptionType
from alphamesh.safety import check_account_number
from tests.conftest import CAPTURE_DIR

RAW = CAPTURE_DIR / "raw"


def load(name: str) -> dict:
    return json.loads((RAW / name).read_text())


class TestAccountIngest:
    def test_parses_the_real_response(self) -> None:
        account = ingest_account_info(load("get_account_info.json"))
        assert account.status == "ACTIVE"
        assert account.equity == 100_000.0
        assert account.options_trading_level == 3
        assert account.options_buying_power == 100_000.0
        assert account.is_tradeable

    def test_the_captured_account_passes_the_paper_guard(self) -> None:
        """The captured account number begins with PA, so it is a paper account."""
        account = ingest_account_info(load("get_account_info.json"))
        check_account_number(account.account_number)

    def test_account_number_is_excluded_from_the_repr(self) -> None:
        account = ingest_account_info(load("get_account_info.json"))
        assert account.account_number not in repr(account)


class TestClockIngest:
    def test_parses_the_real_response(self) -> None:
        clock = ingest_clock(load("get_clock.json"))
        assert clock.is_open is False
        assert clock.next_open is not None
        assert clock.next_close > clock.next_open


class TestBarIngest:
    def test_parses_both_symbols(self) -> None:
        bars = ingest_stock_bars(load("get_stock_bars_sample.json"))
        assert set(bars) == {"SPY", "QQQ"}
        assert len(bars["SPY"]) == 5

    def test_field_mapping_is_correct(self) -> None:
        spy = ingest_stock_bars(load("get_stock_bars_sample.json"))["SPY"]
        last = spy[-1]
        assert last.open == 769.61
        assert last.high == 769.75
        assert last.low == 769.07
        assert last.close == 769.34
        assert last.volume == 1_292_176
        assert last.vwap == pytest.approx(769.341694)

    def test_bars_come_back_in_chronological_order(self) -> None:
        spy = ingest_stock_bars(load("get_stock_bars_sample.json"))["SPY"]
        assert spy == sorted(spy, key=lambda b: b.timestamp)

    def test_ingested_bars_match_the_committed_csv_capture(self) -> None:
        """The CSV the agent replays carries the same values MCP returned."""
        ingested = {
            b.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"): b
            for b in ingest_stock_bars(load("get_stock_bars_sample.json"))["SPY"]
        }
        with (CAPTURE_DIR / "bars_1min_SPY_2026-08-28.csv").open() as fh:
            rows = {r["timestamp"]: r for r in csv.DictReader(fh)}
        assert set(ingested) <= set(rows)
        for ts, bar in ingested.items():
            row = rows[ts]
            assert float(row["open"]) == bar.open
            assert float(row["high"]) == bar.high
            assert float(row["low"]) == bar.low
            assert float(row["close"]) == bar.close
            assert float(row["volume"]) == bar.volume


class TestOptionChainIngest:
    def test_parses_contracts_with_greeks_and_quotes(self) -> None:
        contracts = ingest_option_chain(load("get_option_chain_sample.json"), "SPY")
        assert len(contracts) == 3
        by_symbol = {c.symbol: c for c in contracts}
        long_call = by_symbol["SPY260903C00769000"]
        assert long_call.strike == 769.0
        assert long_call.option_type is OptionType.CALL
        assert long_call.expiration == date(2026, 9, 3)
        assert long_call.greeks.delta == pytest.approx(0.5391)
        assert long_call.quote.bid == 3.86
        assert long_call.quote.ask == 3.94
        assert long_call.day_volume == 563

    def test_puts_are_parsed_with_negative_delta(self) -> None:
        contracts = ingest_option_chain(load("get_option_chain_sample.json"), "SPY")
        put = next(c for c in contracts if c.option_type is OptionType.PUT)
        assert put.greeks.delta == pytest.approx(-0.5016)
        assert put.expiration == date(2026, 9, 4)

    def test_ingested_contracts_match_the_committed_csv_capture(self) -> None:
        ingested = {
            c.symbol: c
            for c in ingest_option_chain(load("get_option_chain_sample.json"), "SPY")
        }
        with (CAPTURE_DIR / "option_snapshots_2026-08-28.csv").open() as fh:
            rows = {r["symbol"]: r for r in csv.DictReader(fh)}
        for symbol, contract in ingested.items():
            row = rows[symbol]
            assert float(row["bid"]) == contract.quote.bid
            assert float(row["ask"]) == contract.quote.ask
            assert float(row["delta"]) == pytest.approx(contract.greeks.delta)
            assert float(row["strike"]) == contract.strike
            assert row["expiration"] == contract.expiration.isoformat()

    def test_unparseable_symbols_are_dropped_not_guessed(self) -> None:
        payload = {"data": {"snapshots": {"NOT_AN_OPTION": {"greeks": {"delta": 0.5}}}}}
        assert ingest_option_chain(payload, "SPY") == []

    def test_security_envelope_is_stripped_and_not_obeyed(self) -> None:
        """The MCP trust block is metadata; its text is never treated as input."""
        payload = load("get_option_chain_sample.json")
        assert "_alpaca_mcp_security" in payload
        contracts = ingest_option_chain(payload, "SPY")
        assert all(c.symbol.startswith("SPY") for c in contracts)


class TestOccSymbols:
    @pytest.mark.parametrize(
        ("symbol", "expiration", "option_type", "strike"),
        [
            ("SPY260903C00769000", date(2026, 9, 3), OptionType.CALL, 769.0),
            ("SPY260904P00770000", date(2026, 9, 4), OptionType.PUT, 770.0),
            ("QQQ260903C00715000", date(2026, 9, 3), OptionType.CALL, 715.0),
        ],
    )
    def test_parses_real_symbols(self, symbol, expiration, option_type, strike) -> None:  # type: ignore[no-untyped-def]
        assert parse_occ_symbol(symbol) == (expiration, option_type, strike)

    @pytest.mark.parametrize("symbol", ["", "SPY", "NOTAREALSYMBOL", "SPY260903X00769000"])
    def test_rejects_malformed_symbols(self, symbol: str) -> None:
        assert parse_occ_symbol(symbol) is None

    def test_round_trips(self) -> None:
        symbol = build_occ_symbol("SPY", date(2026, 9, 3), OptionType.CALL, 769.0)
        assert symbol == "SPY260903C00769000"
        assert parse_occ_symbol(symbol)[2] == 769.0


class TestCaptureWriting:
    def test_bar_capture_round_trips_through_the_replay_reader(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from alphamesh.alpaca.market_data import load_bars_csv

        bars = ingest_stock_bars(load("get_stock_bars_sample.json"))["SPY"]
        path = write_bar_capture(bars, "SPY", tmp_path)
        reloaded = load_bars_csv(path)
        assert [b.close for b in reloaded] == [b.close for b in bars]

    def test_option_capture_round_trips(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from alphamesh.alpaca.options import load_option_snapshots_csv

        contracts = ingest_option_chain(load("get_option_chain_sample.json"), "SPY")
        path = write_option_capture(contracts, tmp_path, "2026-08-28")
        reloaded = {c.symbol: c for c in load_option_snapshots_csv(path)}
        for contract in contracts:
            other = reloaded[contract.symbol]
            assert other.strike == contract.strike
            assert other.quote.bid == contract.quote.bid
            assert other.greeks.delta == pytest.approx(contract.greeks.delta)


class TestUsageDocumentation:
    def test_every_documented_tool_is_read_only(self) -> None:
        """No order-placing MCP tool is in the documented set."""
        for tool in MCP_TOOLS_USED:
            assert not any(
                verb in tool for verb in ("place", "cancel", "close", "exercise", "update")
            )

    def test_description_names_the_tools(self) -> None:
        text = describe_mcp_usage()
        for tool in MCP_TOOLS_USED:
            assert tool in text
        assert "does NOT use MCP" in text
