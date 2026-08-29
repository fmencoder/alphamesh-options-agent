"""Factory that wires the Alpaca-facing components together.

Every construction path runs the paper guard first. There is no code path in
this module that can produce a broker pointed at a live endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from alphamesh.alpaca.execution import AlpacaPaperBroker, Broker, SimulatedBroker
from alphamesh.alpaca.market_data import (
    AlpacaRestMarketData,
    CaptureMarketData,
    MarketDataProvider,
)
from alphamesh.alpaca.options import (
    AlpacaRestOptionChain,
    CaptureOptionChain,
    OptionChainProvider,
)
from alphamesh.alpaca.types import AccountState
from alphamesh.config import Settings
from alphamesh.safety import GuardResult, enforce_paper_mode

log = logging.getLogger(__name__)

SIMULATED_ACCOUNT = AccountState(
    account_number="PA000000SIM0",
    status="ACTIVE",
    equity=100_000.0,
    last_equity=100_000.0,
    cash=100_000.0,
    buying_power=200_000.0,
    options_buying_power=100_000.0,
    options_trading_level=3,
    trading_blocked=False,
    account_blocked=False,
)
"""Stand-in account used only for dry runs and tests. The ``PA`` prefix keeps it
consistent with the paper guard; it is never used against a real endpoint."""


@dataclass
class AlpacaStack:
    guard: GuardResult
    market_data: MarketDataProvider
    option_chain: OptionChainProvider
    broker: Broker
    live_broker: bool


def build_stack(settings: Settings) -> AlpacaStack:
    """Construct the data and execution stack for the configured mode.

    ``ALPHAMESH_DATA_SOURCE=rest`` uses Alpaca over the network and requires
    credentials. ``mcp_capture`` replays real payloads captured through the
    Alpaca MCP server, which is how the system runs where outbound access to
    Alpaca is blocked.
    """
    guard = enforce_paper_mode(settings)

    if settings.data_source == "mcp_capture":
        market_data: MarketDataProvider = CaptureMarketData(settings.capture_dir)
        option_chain: OptionChainProvider = CaptureOptionChain(settings.capture_dir)
    elif settings.has_credentials:
        market_data = AlpacaRestMarketData(settings.api_key_id, settings.api_secret_key)
        option_chain = AlpacaRestOptionChain(settings.api_key_id, settings.api_secret_key)
    else:
        raise RuntimeError(
            "ALPHAMESH_DATA_SOURCE=rest requires APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY. Set them, or use ALPHAMESH_DATA_SOURCE=mcp_capture."
        )

    if settings.dry_run or not settings.has_credentials:
        broker: Broker = SimulatedBroker(SIMULATED_ACCOUNT)
        live_broker = False
        log.info("dry run: using the in-process simulated broker, no orders will be sent")
    else:
        broker = AlpacaPaperBroker(
            settings.api_key_id, settings.api_secret_key, settings.base_url
        )
        live_broker = True

    return AlpacaStack(
        guard=guard,
        market_data=market_data,
        option_chain=option_chain,
        broker=broker,
        live_broker=live_broker,
    )


__all__ = ["SIMULATED_ACCOUNT", "AlpacaStack", "build_stack"]
