"""Broker adapters.

``AlpacaPaperBroker`` is the production path: it submits real multi-leg option
orders to the Alpaca *paper* trading API and refuses to be constructed against
any other endpoint. ``SimulatedBroker`` is a deterministic in-process stand-in
used by the test suite and by dry runs; it never opens a socket.

An ambiguous submission - a timeout, a connection reset - is never retried
blind. ``submit`` raises ``AmbiguousSubmissionError`` and the caller reconciles
by client order id before deciding anything.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from alphamesh.alpaca.options import occ_underlying
from alphamesh.alpaca.types import AccountState, BrokerPosition
from alphamesh.execution.order_builder import to_alpaca_payload
from alphamesh.models.domain import ExecutionRecord, OrderIntent
from alphamesh.safety import check_account_number, check_trading_endpoint

log = logging.getLogger(__name__)


class AmbiguousSubmissionError(RuntimeError):
    """The order may or may not have reached the broker. Reconcile, never retry."""

    def __init__(self, client_order_id: str, cause: str) -> None:
        super().__init__(
            f"submission of {client_order_id} is ambiguous ({cause}); "
            "reconcile with the broker before any retry"
        )
        self.client_order_id = client_order_id
        self.cause = cause


class BrokerError(RuntimeError):
    """The broker positively rejected the request."""


@runtime_checkable
class Broker(Protocol):
    def account(self) -> AccountState: ...

    def submit_spread(self, intent: OrderIntent) -> ExecutionRecord: ...

    def get_order_by_client_id(self, client_order_id: str) -> ExecutionRecord | None: ...

    def cancel_order(self, broker_order_id: str) -> None: ...

    def close_spread(
        self, intent: OrderIntent, limit_price_cents: int, client_order_id: str
    ) -> ExecutionRecord: ...

    def positions(self) -> list[BrokerPosition]: ...

    def working_order_symbols(self) -> frozenset[str]: ...


# --------------------------------------------------------------------------- #
# Alpaca paper broker
# --------------------------------------------------------------------------- #
class AlpacaPaperBroker:
    """Alpaca paper trading over REST.

    The constructor runs the paper endpoint guard, and :meth:`account` runs the
    account-number guard on every call, so a mid-session configuration change
    cannot quietly move order flow to a live account.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets",
        timeout: float = 20.0,
    ) -> None:
        check_trading_endpoint(base_url)
        self._api_key = api_key
        self._api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "content-type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import httpx

        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.request(method, url, headers=self._headers(), **kwargs)
        if response.status_code >= 400:
            raise BrokerError(f"{method} {path} -> {response.status_code}: {response.text[:400]}")
        return response.json() if response.content else {}

    def account(self) -> AccountState:
        data = self._request("GET", "/v2/account")
        account_number = str(data.get("account_number", ""))
        # Fail closed on every account read, not just at startup.
        check_account_number(account_number)
        return AccountState(
            account_number=account_number,
            status=str(data.get("status", "")),
            equity=float(data.get("equity", 0) or 0),
            last_equity=float(data.get("last_equity", 0) or 0),
            cash=float(data.get("cash", 0) or 0),
            buying_power=float(data.get("buying_power", 0) or 0),
            options_buying_power=float(data.get("options_buying_power", 0) or 0),
            options_trading_level=int(data.get("options_trading_level", 0) or 0),
            trading_blocked=bool(data.get("trading_blocked", False)),
            account_blocked=bool(data.get("account_blocked", False)),
        )

    def submit_spread(self, intent: OrderIntent) -> ExecutionRecord:
        import httpx

        payload = to_alpaca_payload(intent)
        try:
            data = self._request("POST", "/v2/orders", json=payload)
        except httpx.HTTPError as exc:
            # Network-level failure: the order may already be live at Alpaca.
            raise AmbiguousSubmissionError(
                intent.client_order_id, type(exc).__name__
            ) from exc
        return self._to_record(intent.client_order_id, data)

    def close_spread(
        self, intent: OrderIntent, limit_price_cents: int, client_order_id: str
    ) -> ExecutionRecord:
        """Submit the mirror-image order that flattens the spread."""
        import httpx

        from alphamesh.models.domain import OrderSide, PositionIntent

        flipped = []
        for leg in intent.legs:
            side = OrderSide.SELL if leg.side is OrderSide.BUY else OrderSide.BUY
            closing = (
                PositionIntent.SELL_TO_CLOSE
                if side is OrderSide.SELL
                else PositionIntent.BUY_TO_CLOSE
            )
            flipped.append(
                {
                    "symbol": leg.contract.symbol,
                    "ratio_qty": str(leg.ratio),
                    "side": side.value,
                    "position_intent": closing.value,
                }
            )
        payload = {
            "order_class": "mleg",
            "qty": str(intent.quantity),
            "type": "limit",
            "time_in_force": "day",
            "limit_price": f"{limit_price_cents / 100:.2f}",
            "client_order_id": client_order_id,
            "legs": flipped,
        }
        try:
            data = self._request("POST", "/v2/orders", json=payload)
        except httpx.HTTPError as exc:
            raise AmbiguousSubmissionError(client_order_id, type(exc).__name__) from exc
        return self._to_record(client_order_id, data)

    def get_order_by_client_id(self, client_order_id: str) -> ExecutionRecord | None:
        try:
            data = self._request(
                "GET",
                "/v2/orders:by_client_order_id",
                params={"client_order_id": client_order_id},
            )
        except BrokerError:
            return None
        if not data:
            return None
        return self._to_record(client_order_id, data)

    def cancel_order(self, broker_order_id: str) -> None:
        self._request("DELETE", f"/v2/orders/{broker_order_id}")

    def working_order_symbols(self) -> frozenset[str]:
        """Underlyings with a live order at the broker, from the account itself.

        Multi-leg option orders carry the OCC symbols on their legs, so the
        parent's own symbol field is empty and the roots come from the legs.
        """
        data = self._request("GET", "/v2/orders", params={"status": "open", "nested": "true"})
        roots: set[str] = set()
        for order in data or []:
            candidates = [order, *(order.get("legs") or [])]
            for item in candidates:
                symbol = str(item.get("symbol") or "")
                if not symbol:
                    continue
                root = occ_underlying(symbol) or symbol.upper()
                roots.add(root)
        return frozenset(roots)

    def positions(self) -> list[BrokerPosition]:
        data = self._request("GET", "/v2/positions")
        return [
            BrokerPosition(
                symbol=str(p.get("symbol", "")),
                quantity=int(float(p.get("qty", 0) or 0)),
                avg_entry_price=float(p.get("avg_entry_price", 0) or 0),
                market_value=float(p.get("market_value", 0) or 0),
                unrealized_pl=float(p.get("unrealized_pl", 0) or 0),
            )
            for p in data
        ]

    @staticmethod
    def _to_record(client_order_id: str, data: dict[str, Any]) -> ExecutionRecord:
        filled_price = data.get("filled_avg_price")
        return ExecutionRecord(
            client_order_id=client_order_id,
            broker_order_id=str(data.get("id")) if data.get("id") else None,
            status=str(data.get("status", "unknown")),
            filled_quantity=int(float(data.get("filled_qty", 0) or 0)),
            filled_avg_price_cents=(
                round(float(filled_price) * 100) if filled_price else None
            ),
            submitted_at=_parse_dt(data.get("submitted_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            raw_status=str(data.get("status", "")),
        )


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
_SIM_LIVE_STATUSES = frozenset({"new", "accepted", "pending_new", "partially_filled"})


class SimulatedBroker:
    """Deterministic in-process broker for tests and dry runs.

    Fills at the submitted limit price. It enforces client-order-id uniqueness
    exactly as Alpaca does, so duplicate-protection tests exercise the real
    failure mode.
    """

    def __init__(
        self,
        account: AccountState,
        fill: bool = True,
        fail_next_submit_with: Exception | None = None,
    ) -> None:
        self._account = account
        self.fill = fill
        self.orders: dict[str, ExecutionRecord] = {}
        self.submitted_payloads: list[dict[str, Any]] = []
        self._fail_next = fail_next_submit_with
        self._seq = 0
        # OCC symbols per order, so working exposure can be reported by
        # underlying exactly as the real broker does.
        self.legs_by_client_order_id: dict[str, tuple[str, ...]] = {}

    def account(self) -> AccountState:
        check_account_number(self._account.account_number)
        return self._account

    def set_account(self, account: AccountState) -> None:
        self._account = account

    def submit_spread(self, intent: OrderIntent) -> ExecutionRecord:
        if self._fail_next is not None:
            exc, self._fail_next = self._fail_next, None
            raise exc
        if intent.client_order_id in self.orders:
            raise BrokerError(f"duplicate client_order_id {intent.client_order_id}")
        self.submitted_payloads.append(to_alpaca_payload(intent))
        self.legs_by_client_order_id[intent.client_order_id] = tuple(
            leg.contract.symbol for leg in intent.legs
        )
        return self._record(intent.client_order_id, intent.quantity, intent.limit_price_cents)

    def close_spread(
        self, intent: OrderIntent, limit_price_cents: int, client_order_id: str
    ) -> ExecutionRecord:
        if client_order_id in self.orders:
            raise BrokerError(f"duplicate client_order_id {client_order_id}")
        return self._record(client_order_id, intent.quantity, limit_price_cents)

    def _record(self, client_order_id: str, quantity: int, price_cents: int) -> ExecutionRecord:
        self._seq += 1
        now = datetime.now(UTC)
        record = ExecutionRecord(
            client_order_id=client_order_id,
            broker_order_id=f"sim-{self._seq:06d}",
            status="filled" if self.fill else "new",
            filled_quantity=quantity if self.fill else 0,
            filled_avg_price_cents=price_cents if self.fill else None,
            submitted_at=now,
            updated_at=now,
            raw_status="filled" if self.fill else "new",
        )
        self.orders[client_order_id] = record
        return record

    def get_order_by_client_id(self, client_order_id: str) -> ExecutionRecord | None:
        return self.orders.get(client_order_id)

    def cancel_order(self, broker_order_id: str) -> None:
        for cid, record in list(self.orders.items()):
            if record.broker_order_id == broker_order_id:
                self.orders[cid] = record.model_copy(
                    update={"status": "canceled", "raw_status": "canceled"}
                )

    def working_order_symbols(self) -> frozenset[str]:
        roots: set[str] = set()
        for record in self.orders.values():
            if record.status.lower() not in _SIM_LIVE_STATUSES:
                continue
            for leg in self.legs_by_client_order_id.get(record.client_order_id, ()):
                root = occ_underlying(leg) or leg.upper()
                roots.add(root)
        return frozenset(roots)

    def positions(self) -> list[BrokerPosition]:
        return []


__all__ = [
    "AlpacaPaperBroker",
    "AmbiguousSubmissionError",
    "Broker",
    "BrokerError",
    "SimulatedBroker",
]
