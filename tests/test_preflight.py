"""Preflight must be authoritative and structurally incapable of trading.

It is run against the real competition paper account, so "it places no orders"
has to be enforced by construction rather than promised.
"""

from __future__ import annotations

import pytest

from alphamesh.alpaca.execution import SimulatedBroker
from alphamesh.config import Settings, load_config
from alphamesh.main import (
    EXECUTION_CRITICAL,
    OrderSubmissionForbiddenError,
    _ZeroOrderBroker,
    cmd_preflight,
)
from tests.conftest import CAPTURE_DIR, make_account


def parse_flags(text: str) -> dict[str, str]:
    return {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in text.splitlines()
        if line.startswith("PREFLIGHT_") and "=" in line
    }


@pytest.fixture
def preflight_config(tmp_path):  # type: ignore[no-untyped-def]
    return load_config(
        settings=Settings(
            paper=True,
            dry_run=True,
            data_source="mcp_capture",
            capture_dir=CAPTURE_DIR,
            database_path=tmp_path / "preflight.db",
        )
    )


class TestZeroOrderBroker:
    """The proxy is what makes 'places no orders' structural."""

    def test_reads_are_forwarded(self) -> None:
        broker = _ZeroOrderBroker(SimulatedBroker(make_account()))
        assert broker.account().is_tradeable
        assert broker.positions() == []
        assert broker.get_order_by_client_id("nope") is None

    @pytest.mark.parametrize("method", ["submit_spread", "close_spread", "cancel_order"])
    def test_every_write_raises(self, method: str) -> None:
        broker = _ZeroOrderBroker(SimulatedBroker(make_account()))
        with pytest.raises(OrderSubmissionForbiddenError):
            getattr(broker, method)("anything")

    def test_the_proxy_covers_the_whole_broker_write_surface(self) -> None:
        """If a write method is added to Broker, this fails until the proxy blocks it."""
        from alphamesh.alpaca.execution import Broker

        writes = {"submit_spread", "close_spread", "cancel_order"}
        protocol_methods = {
            name
            for name in dir(Broker)
            if not name.startswith("_") and callable(getattr(Broker, name, None))
        }
        assert writes <= protocol_methods
        unguarded = protocol_methods - writes - {"account", "positions", "get_order_by_client_id"}
        assert unguarded == set(), f"unguarded broker methods: {unguarded}"


class TestPreflightOutcome:
    def test_healthy_configuration_is_ready_and_exits_zero(
        self, preflight_config, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        code = cmd_preflight(preflight_config)
        flags = parse_flags(capsys.readouterr().out)
        assert code == 0
        assert flags["PREFLIGHT_READY"] == "YES"
        for key in EXECUTION_CRITICAL:
            assert flags[key] == "PASS", f"{key} should pass"

    def test_all_required_flags_are_emitted(self, preflight_config, capsys) -> None:  # type: ignore[no-untyped-def]
        cmd_preflight(preflight_config)
        flags = parse_flags(capsys.readouterr().out)
        assert set(flags) == {
            "PREFLIGHT_PAPER_MODE",
            "PREFLIGHT_ACCOUNT",
            "PREFLIGHT_MARKET_DATA",
            "PREFLIGHT_OPTIONS_CHAIN",
            "PREFLIGHT_GREEKS",
            "PREFLIGHT_JOURNAL",
            "PREFLIGHT_RECOVERY",
            "PREFLIGHT_AI_PROVIDER",
            "PREFLIGHT_READY",
        }

    def test_reports_zero_orders_submitted(self, preflight_config, capsys) -> None:  # type: ignore[no-untyped-def]
        cmd_preflight(preflight_config)
        assert '"orders_submitted": 0' in capsys.readouterr().out

    def test_live_endpoint_fails_closed_with_exit_two(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        config = load_config(
            settings=Settings(
                paper=True,
                dry_run=True,
                base_url="https://api.alpaca.markets",
                data_source="mcp_capture",
                capture_dir=CAPTURE_DIR,
                database_path=tmp_path / "p.db",
            )
        )
        code = cmd_preflight(config)
        out = capsys.readouterr().out
        assert code == 2
        assert parse_flags(out)["PREFLIGHT_PAPER_MODE"] == "FAIL"
        assert parse_flags(out)["PREFLIGHT_READY"] == "NO"

    def test_paper_flag_false_fails_closed(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        config = load_config(
            settings=Settings(
                paper=False,
                dry_run=True,
                data_source="mcp_capture",
                capture_dir=CAPTURE_DIR,
                database_path=tmp_path / "p.db",
            )
        )
        assert cmd_preflight(config) == 2

    def test_missing_option_chain_is_not_ready(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        """An empty capture directory must fail, not quietly report success."""
        empty = tmp_path / "empty"
        empty.mkdir()
        config = load_config(
            settings=Settings(
                paper=True,
                dry_run=True,
                data_source="mcp_capture",
                capture_dir=empty,
                database_path=tmp_path / "p.db",
            )
        )
        code = cmd_preflight(config)
        flags = parse_flags(capsys.readouterr().out)
        assert code == 1
        assert flags["PREFLIGHT_READY"] == "NO"
        assert flags["PREFLIGHT_MARKET_DATA"] == "FAIL"
        assert flags["PREFLIGHT_OPTIONS_CHAIN"] == "FAIL"

    def test_simulated_execution_is_surfaced(self, preflight_config, capsys) -> None:  # type: ignore[no-untyped-def]
        cmd_preflight(preflight_config)
        out = capsys.readouterr().out
        assert '"execution_mode": "SIMULATED"' in out
        assert "execution_mode_warning" in out

    def test_ai_provider_state_and_fallback_are_reported(
        self, preflight_config, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        cmd_preflight(preflight_config)
        out = capsys.readouterr().out
        assert '"fallback": "deterministic heuristic council"' in out
        assert parse_flags(out)["PREFLIGHT_AI_PROVIDER"] == "PASS"

    def test_journal_and_recovery_are_exercised_not_assumed(
        self, preflight_config, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        cmd_preflight(preflight_config)
        out = capsys.readouterr().out
        assert '"writable": true' in out
        assert '"inspected"' in out  # the recovery sweep actually ran
        assert preflight_config.settings.database_path.exists()

    def test_no_secret_value_is_printed(self, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
        config = load_config(
            settings=Settings(
                paper=True,
                dry_run=True,
                api_key_id="PKLEAKKEY123",
                api_secret_key="SUPERSECRETVALUE",
                anthropic_api_key="sk-ant-LEAKME",
                data_source="mcp_capture",
                capture_dir=CAPTURE_DIR,
                database_path=tmp_path / "p.db",
            )
        )
        cmd_preflight(config)
        out = capsys.readouterr().out
        assert "SUPERSECRETVALUE" not in out
        assert "sk-ant-LEAKME" not in out
        assert "PKLEAKKEY123" not in out
        assert "<set>" in out
