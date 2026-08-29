"""Paper-trading environment guard.

This module fails closed. AlphaMesh will not start, and will not construct a
broker client, unless paper trading is *positively* established:

  * ``ALPACA_PAPER`` must be true,
  * the trading base URL must be a known Alpaca paper host,
  * the URL must not match any known live host,
  * the account (when reachable) must report a paper account number.

Anything ambiguous is treated as live and rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from alphamesh.config import Settings
from alphamesh.models.domain import ReasonCode

PAPER_HOSTS: frozenset[str] = frozenset(
    {
        "paper-api.alpaca.markets",
        "broker-api.sandbox.alpaca.markets",
    }
)
"""Hosts positively identified as Alpaca paper/sandbox trading endpoints."""

LIVE_HOSTS: frozenset[str] = frozenset(
    {
        "api.alpaca.markets",
        "broker-api.alpaca.markets",
    }
)
"""Hosts positively identified as Alpaca LIVE-money endpoints. Always rejected."""

DATA_HOSTS: frozenset[str] = frozenset(
    {
        "data.alpaca.markets",
        "stream.data.alpaca.markets",
    }
)
"""Market-data hosts. Read-only; they carry no order-routing capability."""

PAPER_ACCOUNT_PREFIX = "PA"
"""Alpaca paper account numbers begin with 'PA'. Live accounts do not."""


class LiveTradingForbiddenError(RuntimeError):
    """Raised whenever the environment could route a real-money order."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"{ReasonCode.LIVE_TRADING_FORBIDDEN}: {detail}")
        self.detail = detail
        self.reason_code = ReasonCode.LIVE_TRADING_FORBIDDEN


@dataclass(frozen=True)
class GuardResult:
    paper: bool
    detail: str
    checks: tuple[str, ...]


def _host_of(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return (parsed.hostname or "").lower()


def check_trading_endpoint(base_url: str) -> None:
    """Reject any trading endpoint that is not a known Alpaca paper host."""
    host = _host_of(base_url)
    if not host:
        raise LiveTradingForbiddenError(f"trading base URL {base_url!r} has no resolvable host")
    if host in LIVE_HOSTS:
        raise LiveTradingForbiddenError(f"{host} is an Alpaca LIVE trading endpoint")
    if host not in PAPER_HOSTS:
        # Fail closed: an unrecognised host is not proof of paper trading.
        raise LiveTradingForbiddenError(
            f"{host} is not a recognised Alpaca paper endpoint; "
            f"expected one of {sorted(PAPER_HOSTS)}"
        )


def check_data_endpoint(data_url: str) -> None:
    host = _host_of(data_url)
    if host in LIVE_HOSTS:
        raise LiveTradingForbiddenError(f"{host} is an order-routing host, not a data host")
    if host and host not in DATA_HOSTS and host not in PAPER_HOSTS:
        raise LiveTradingForbiddenError(f"{host} is not a recognised Alpaca market-data endpoint")


def check_account_number(account_number: str) -> None:
    """Alpaca paper accounts are numbered ``PA...``. Anything else is live."""
    value = (account_number or "").strip().upper()
    if not value:
        raise LiveTradingForbiddenError("account number is empty; cannot prove paper mode")
    if not value.startswith(PAPER_ACCOUNT_PREFIX):
        raise LiveTradingForbiddenError(
            f"account number {value[:2]}... is not a paper account "
            f"(expected prefix {PAPER_ACCOUNT_PREFIX})"
        )


def enforce_paper_mode(settings: Settings) -> GuardResult:
    """Run every startup safety check. Raises rather than returning on failure."""
    checks: list[str] = []

    if not settings.paper:
        raise LiveTradingForbiddenError("ALPACA_PAPER is not true")
    checks.append("ALPACA_PAPER=true")

    check_trading_endpoint(settings.base_url)
    checks.append(f"trading endpoint {_host_of(settings.base_url)} is a paper host")

    check_data_endpoint(settings.data_url)
    checks.append(f"data endpoint {_host_of(settings.data_url)} is a data host")

    return GuardResult(
        paper=True,
        detail="PAPER MODE confirmed by environment guard",
        checks=tuple(checks),
    )


BANNER = r"""
    _    _       _         __  __           _
   / \  | |_ __ | |__   __ |  \/  | ___ ___| |__
  / _ \ | | '_ \| '_ \ / _` | |\/| |/ _ / __| '_ \
 / ___ \| | |_) | | | | (_| | |  | |  __\__ \ | | |
/_/   \_|_| .__/|_| |_|\__,_|_|  |_|\___|___/_| |_|
          |_|
       AUTONOMOUS OPTIONS INTELLIGENCE
"""


def banner(settings: Settings) -> str:
    """Startup banner. Always states the mode in unmissable terms."""
    mode = "PAPER MODE" if settings.paper else "!!! NON-PAPER CONFIGURATION - STARTUP BLOCKED !!!"
    return (
        f"{BANNER}\n"
        f"  MODE      : {mode}\n"
        f"  TRADING   : {settings.base_url}\n"
        f"  DATA      : {settings.data_url}\n"
        f"  DRY RUN   : {settings.dry_run}\n"
        f"  SOURCE    : {settings.data_source}\n"
    )


__all__ = [
    "BANNER",
    "DATA_HOSTS",
    "LIVE_HOSTS",
    "PAPER_ACCOUNT_PREFIX",
    "PAPER_HOSTS",
    "GuardResult",
    "LiveTradingForbiddenError",
    "banner",
    "check_account_number",
    "check_data_endpoint",
    "check_trading_endpoint",
    "enforce_paper_mode",
]
