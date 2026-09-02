"""Conformance to Alpaca's documented Trading API contract.

Checked against the Alpaca OpenAPI reference for POST /v2/orders and
GET /v2/orders:by_client_order_id, retrieved through the Alpaca MCP server.
These lock in an audit that cannot be re-run offline.
"""

from __future__ import annotations

import pytest

from alphamesh.execution.monitor import status_to_state
from alphamesh.execution.order_builder import (
    MAX_CLIENT_ORDER_ID_LEN,
    build_order_intent,
    to_alpaca_payload,
)
from alphamesh.models.domain import TradeState
from tests.conftest import NOW, make_decision
from tests.test_execution import approved, make_spread

# Enums exactly as documented by Alpaca.
ORDER_CLASS = {"simple", "bracket", "oco", "oto", "mleg", ""}
ORDER_TYPE_MULTILEG = {"market", "limit"}
TIME_IN_FORCE_OPTIONS = {"day", "gtc"}
ORDER_SIDE = {"buy", "sell"}
POSITION_INTENT = {"buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"}
ALPACA_CLIENT_ORDER_ID_MAX = 128

# Every status Alpaca documents for an order.
ALPACA_ORDER_STATUSES = {
    "new", "partially_filled", "filled", "done_for_day", "canceled", "expired",
    "replaced", "pending_cancel", "pending_replace", "accepted", "pending_new",
    "accepted_for_bidding", "stopped", "rejected", "suspended", "calculated", "held",
}


@pytest.fixture
def payload():  # type: ignore[no-untyped-def]
    intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
    return to_alpaca_payload(intent)


class TestMultiLegOrderPayload:
    def test_order_class_is_mleg(self, payload) -> None:  # type: ignore[no-untyped-def]
        assert payload["order_class"] == "mleg"
        assert payload["order_class"] in ORDER_CLASS

    def test_parent_omits_symbol_and_side(self, payload) -> None:  # type: ignore[no-untyped-def]
        """Alpaca: symbol is 'required for all order classes except mleg', and
        side is 'required for all order classes except for mleg'."""
        assert "symbol" not in payload
        assert "side" not in payload

    def test_qty_is_present_and_a_string(self, payload) -> None:  # type: ignore[no-untyped-def]
        """Alpaca: qty is 'required if order class is mleg'."""
        assert isinstance(payload["qty"], str)
        assert int(payload["qty"]) > 0

    def test_type_is_supported_for_multileg(self, payload) -> None:  # type: ignore[no-untyped-def]
        assert payload["type"] in ORDER_TYPE_MULTILEG

    def test_time_in_force_is_supported_for_options(self, payload) -> None:  # type: ignore[no-untyped-def]
        assert payload["time_in_force"] in TIME_IN_FORCE_OPTIONS

    def test_limit_price_is_a_two_decimal_string(self, payload) -> None:  # type: ignore[no-untyped-def]
        price = payload["limit_price"]
        assert isinstance(price, str)
        assert price == f"{float(price):.2f}"
        assert float(price) > 0

    def test_client_order_id_is_within_alpacas_limit(self, payload) -> None:  # type: ignore[no-untyped-def]
        assert len(payload["client_order_id"]) <= ALPACA_CLIENT_ORDER_ID_MAX
        assert MAX_CLIENT_ORDER_ID_LEN <= ALPACA_CLIENT_ORDER_ID_MAX

    def test_legs_carry_every_required_field(self, payload) -> None:  # type: ignore[no-untyped-def]
        assert len(payload["legs"]) == 2
        for leg in payload["legs"]:
            assert set(leg) == {"symbol", "ratio_qty", "side", "position_intent"}
            assert leg["symbol"]
            assert isinstance(leg["ratio_qty"], str) and int(leg["ratio_qty"]) > 0
            assert leg["side"] in ORDER_SIDE
            assert leg["position_intent"] in POSITION_INTENT

    def test_opening_a_debit_spread_buys_one_leg_and_sells_the_other(
        self, payload
    ) -> None:  # type: ignore[no-untyped-def]
        sides = [leg["side"] for leg in payload["legs"]]
        intents = [leg["position_intent"] for leg in payload["legs"]]
        assert sides == ["buy", "sell"]
        assert intents == ["buy_to_open", "sell_to_open"]

    def test_no_unexpected_top_level_keys(self, payload) -> None:  # type: ignore[no-untyped-def]
        assert set(payload) == {
            "order_class", "qty", "type", "time_in_force",
            "limit_price", "client_order_id", "legs",
        }


class TestClosingPayload:
    def test_close_flips_sides_to_closing_intents(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from alphamesh.alpaca.execution import AlpacaPaperBroker

        broker = AlpacaPaperBroker("k", "s")
        captured: dict = {}
        monkeypatch.setattr(
            broker,
            "_request",
            lambda m, p, **kw: (
                captured.update(kw.get("json", {})),
                {"id": "x", "status": "new"},
            )[1],
        )
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        broker.close_spread(intent, 300, "alphamesh-SPY-BCS-close")

        assert captured["order_class"] == "mleg"
        assert captured["time_in_force"] in TIME_IN_FORCE_OPTIONS
        assert [leg["side"] for leg in captured["legs"]] == ["sell", "buy"]
        assert [leg["position_intent"] for leg in captured["legs"]] == [
            "sell_to_close",
            "buy_to_close",
        ]
        for leg in captured["legs"]:
            assert leg["position_intent"] in POSITION_INTENT

    def test_a_credit_close_is_submitted_as_a_negative_limit_price(
        self, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Alpaca mleg notation: negative limit_price = credit received.

        Sending the positive magnitude tells Alpaca "I will PAY up to $7.55 to
        get out" when we mean "pay me $7.55". The order still fills, but with no
        downside protection whatsoever -- the limit stops constraining the one
        direction it exists to constrain.
        """
        from alphamesh.alpaca.execution import AlpacaPaperBroker

        broker = AlpacaPaperBroker("k", "s")
        captured: dict = {}
        monkeypatch.setattr(
            broker,
            "_request",
            lambda m, p, **kw: (
                captured.update(kw.get("json", {})),
                {"id": "x", "status": "new"},
            )[1],
        )
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)

        broker.close_spread(intent, 755, "alphamesh-SPY-BCS-close")

        assert captured["limit_price"] == "-7.55", (
            "a credit close must be negative in Alpaca mleg notation"
        )
        assert float(captured["limit_price"]) < 0
        # The magnitude is untouched; only the sign carries the direction.
        assert abs(float(captured["limit_price"])) == 7.55
        # And the rest of the contract is unchanged.
        assert captured["order_class"] == "mleg"
        assert captured["type"] == "limit"
        assert captured["time_in_force"] == "day"

    def test_an_unsigned_magnitude_is_never_double_negated(
        self, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Callers pass an unsigned credit magnitude; the sign is applied once."""
        from alphamesh.alpaca.execution import AlpacaPaperBroker

        broker = AlpacaPaperBroker("k", "s")
        captured: dict = {}
        monkeypatch.setattr(
            broker,
            "_request",
            lambda m, p, **kw: (
                captured.update(kw.get("json", {})),
                {"id": "x", "status": "new"},
            )[1],
        )
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)

        # Even if a caller ever handed this a negative value, the wire result is
        # a credit exactly once -- never flipped back to a debit.
        broker.close_spread(intent, -755, "alphamesh-SPY-BCS-close")

        assert captured["limit_price"] == "-7.55"

    def test_the_entry_debit_limit_price_stays_positive(self) -> None:
        """The other half of the notation: a debit open IS a positive limit."""
        from alphamesh.execution.order_builder import to_alpaca_payload

        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        payload = to_alpaca_payload(intent)

        assert float(payload["limit_price"]) > 0
        assert payload["limit_price"] == f"{intent.limit_price_cents / 100:.2f}"
        assert [leg["position_intent"] for leg in payload["legs"]] == [
            "buy_to_open",
            "sell_to_open",
        ]


class TestOrderStatusHandling:
    def test_every_documented_status_maps_to_a_valid_state(self) -> None:
        for status in ALPACA_ORDER_STATUSES:
            state = status_to_state(status, TradeState.SUBMITTED)
            assert isinstance(state, TradeState)

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("filled", TradeState.FILLED),
            ("partially_filled", TradeState.PARTIALLY_FILLED),
            ("canceled", TradeState.REJECTED),
            ("expired", TradeState.REJECTED),
            ("rejected", TradeState.REJECTED),
            ("done_for_day", TradeState.REJECTED),
            ("replaced", TradeState.REJECTED),
        ],
    )
    def test_terminal_statuses_map_correctly(self, status, expected) -> None:  # type: ignore[no-untyped-def]
        assert status_to_state(status, TradeState.SUBMITTED) is expected

    @pytest.mark.parametrize(
        "status",
        ["new", "accepted", "pending_new", "accepted_for_bidding", "held",
         "pending_cancel", "pending_replace", "stopped", "suspended", "calculated"],
    )
    def test_in_flight_statuses_never_invent_a_fill_or_a_rejection(self, status) -> None:  # type: ignore[no-untyped-def]
        """An unrecognised or in-flight status must hold the current state, never
        be optimistically read as FILLED or pessimistically as REJECTED."""
        assert status_to_state(status, TradeState.SUBMITTED) is TradeState.SUBMITTED
        assert status_to_state(status, TradeState.MONITORING) is TradeState.MONITORING


class TestEndpointPaths:
    def test_lookup_by_client_order_id_uses_the_documented_path(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from alphamesh.alpaca.execution import AlpacaPaperBroker

        broker = AlpacaPaperBroker("k", "s")
        seen: dict = {}

        def capture(method, path, **kwargs):  # type: ignore[no-untyped-def]
            seen["method"], seen["path"], seen["params"] = method, path, kwargs.get("params")
            return {"id": "x", "status": "new"}

        monkeypatch.setattr(broker, "_request", capture)
        broker.get_order_by_client_id("alphamesh-SPY-BCS-abc")
        assert seen["method"] == "GET"
        assert seen["path"] == "/v2/orders:by_client_order_id"
        assert seen["params"] == {"client_order_id": "alphamesh-SPY-BCS-abc"}

    def test_paper_base_url_matches_the_documented_paper_server(self) -> None:
        from alphamesh.alpaca.execution import AlpacaPaperBroker

        assert AlpacaPaperBroker("k", "s").base_url == "https://paper-api.alpaca.markets"
