"""Paper-mode enforcement. These are the tests that stop real money moving."""

from __future__ import annotations

import pytest

from alphamesh.alpaca.client import build_stack
from alphamesh.alpaca.execution import AlpacaPaperBroker
from alphamesh.config import Settings
from alphamesh.safety import (
    LiveTradingForbiddenError,
    banner,
    check_account_number,
    check_data_endpoint,
    check_trading_endpoint,
    enforce_paper_mode,
)


class TestTradingEndpoint:
    def test_paper_endpoint_accepted(self) -> None:
        check_trading_endpoint("https://paper-api.alpaca.markets")

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.alpaca.markets",
            "https://api.alpaca.markets/v2",
            "api.alpaca.markets",
            "https://broker-api.alpaca.markets",
        ],
    )
    def test_live_endpoints_rejected(self, url: str) -> None:
        with pytest.raises(LiveTradingForbiddenError, match="LIVE_TRADING_FORBIDDEN"):
            check_trading_endpoint(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://alpaca.example.com",
            "https://paper-api.alpaca.markets.evil.com",
            "https://localhost:8080",
            "",
        ],
    )
    def test_unrecognised_endpoints_fail_closed(self, url: str) -> None:
        """An unknown host is not proof of paper trading, so it is refused."""
        with pytest.raises(LiveTradingForbiddenError):
            check_trading_endpoint(url)


class TestDataEndpoint:
    def test_data_host_accepted(self) -> None:
        check_data_endpoint("https://data.alpaca.markets")

    def test_order_routing_host_rejected_as_data(self) -> None:
        with pytest.raises(LiveTradingForbiddenError):
            check_data_endpoint("https://api.alpaca.markets")


class TestAccountNumber:
    def test_paper_prefix_accepted(self) -> None:
        check_account_number("PA0000EXAMPLE")

    @pytest.mark.parametrize("value", ["", "  ", "920123456", "LIVE12345", "XA000001"])
    def test_non_paper_account_rejected(self, value: str) -> None:
        with pytest.raises(LiveTradingForbiddenError):
            check_account_number(value)


class TestEnforcePaperMode:
    def test_passes_for_paper_settings(self) -> None:
        result = enforce_paper_mode(Settings(paper=True))
        assert result.paper is True
        assert len(result.checks) == 3

    def test_paper_flag_false_blocks_startup(self) -> None:
        with pytest.raises(LiveTradingForbiddenError, match="ALPACA_PAPER is not true"):
            enforce_paper_mode(Settings(paper=False))

    def test_live_base_url_blocks_startup(self) -> None:
        settings = Settings(paper=True, base_url="https://api.alpaca.markets")
        with pytest.raises(LiveTradingForbiddenError):
            enforce_paper_mode(settings)


class TestBrokerConstruction:
    def test_broker_refuses_live_endpoint_at_construction(self) -> None:
        with pytest.raises(LiveTradingForbiddenError):
            AlpacaPaperBroker("k", "s", base_url="https://api.alpaca.markets")

    def test_broker_accepts_paper_endpoint(self) -> None:
        broker = AlpacaPaperBroker("k", "s")
        assert broker.base_url == "https://paper-api.alpaca.markets"


class TestStackConstruction:
    def test_stack_refuses_live_settings(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        settings = Settings(
            paper=True,
            base_url="https://api.alpaca.markets",
            data_source="mcp_capture",
            capture_dir=tmp_path,
        )
        with pytest.raises(LiveTradingForbiddenError):
            build_stack(settings)

    def test_dry_run_uses_simulator_not_alpaca(self, config) -> None:  # type: ignore[no-untyped-def]
        stack = build_stack(config.settings)
        assert stack.live_broker is False
        assert type(stack.broker).__name__ == "SimulatedBroker"


class TestBanner:
    def test_banner_states_paper_mode(self) -> None:
        assert "PAPER MODE" in banner(Settings(paper=True))

    def test_banner_flags_non_paper_configuration(self) -> None:
        assert "STARTUP BLOCKED" in banner(Settings(paper=False))
