"""Order construction, idempotency, the state machine, monitoring and recovery."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from alphamesh.alpaca.execution import (
    AmbiguousSubmissionError,
    BrokerError,
    SimulatedBroker,
)
from alphamesh.execution.monitor import (
    OrderMonitor,
    mark_position,
    mark_spread_cents,
    status_to_state,
)
from alphamesh.execution.order_builder import (
    build_client_order_id,
    build_order_intent,
    signal_hash,
    to_alpaca_payload,
)
from alphamesh.execution.recovery import reconcile_open_orders
from alphamesh.execution.state_machine import (
    ALLOWED_TRANSITIONS,
    IllegalTransitionError,
    can_transition,
    is_recoverable,
    is_terminal,
    transition,
)
from alphamesh.models.domain import (
    OrderSide,
    PositionIntent,
    PositionRecord,
    ReasonCode,
    RiskDecision,
    SpreadLeg,
    SpreadStructure,
    Strategy,
    TradeState,
)
from alphamesh.persistence.journal import Journal
from tests.conftest import NOW, make_account, make_contract, make_decision


def make_spread(limit_cents: int = 238, symbol: str = "SPY") -> SpreadStructure:
    long_c = make_contract(underlying=symbol, strike=769.0)
    short_c = make_contract(
        underlying=symbol, strike=774.0, delta=0.30, bid=1.50, ask=1.60
    )
    return SpreadStructure(
        strategy=Strategy.BULL_CALL_SPREAD,
        symbol=symbol,
        expiration=long_c.expiration,
        long_leg=SpreadLeg(
            contract=long_c, side=OrderSide.BUY, position_intent=PositionIntent.BUY_TO_OPEN
        ),
        short_leg=SpreadLeg(
            contract=short_c,
            side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
        net_debit_cents=limit_cents,
        strike_width_cents=500,
        limit_price_cents=limit_cents,
    )


def approved(quantity: int = 2, max_loss_cents: int = 47_600) -> RiskDecision:
    return RiskDecision(
        approved=True,
        quantity=quantity,
        max_loss_cents=max_loss_cents,
        max_profit_cents=52_400,
        reason_codes=(ReasonCode.APPROVED,),
        detail="ok",
        checks_run=("paper_mode",),
    )


@pytest.fixture
def journal() -> Journal:
    j = Journal(":memory:")
    yield j
    j.close()


class TestClientOrderId:
    def test_is_deterministic_for_identical_trades(self) -> None:
        decision, spread = make_decision(), make_spread()
        assert build_client_order_id(decision, spread, 2, 238) == build_client_order_id(
            decision, spread, 2, 238
        )

    def test_changes_when_the_trade_changes(self) -> None:
        decision, spread = make_decision(), make_spread()
        base = build_client_order_id(decision, spread, 2, 238)
        assert build_client_order_id(decision, spread, 3, 238) != base
        assert build_client_order_id(decision, spread, 2, 240) != base
        assert (
            build_client_order_id(make_decision(symbol="QQQ"), spread, 2, 238) != base
        )

    def test_has_the_documented_shape(self) -> None:
        cid = build_client_order_id(make_decision(), make_spread(), 2, 238)
        assert cid.startswith("alphamesh-SPY-BCS-")
        assert len(cid) <= 48

    def test_signal_hash_is_stable(self) -> None:
        decision, spread = make_decision(), make_spread()
        assert signal_hash(decision, spread, 1, 100) == signal_hash(decision, spread, 1, 100)


class TestOrderIntent:
    def test_builds_a_two_leg_debit_order(self) -> None:
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        assert len(intent.legs) == 2
        assert intent.quantity == 2
        assert intent.limit_price == 2.38

    def test_refuses_to_build_from_a_rejected_risk_decision(self) -> None:
        rejected = RiskDecision(
            approved=False,
            quantity=0,
            max_loss_cents=0,
            max_profit_cents=0,
            reason_codes=(ReasonCode.MAX_POSITION_RISK,),
        )
        with pytest.raises(ValueError, match="rejected RiskDecision"):
            build_order_intent(make_decision(), make_spread(), rejected, NOW)

    def test_alpaca_payload_is_a_multi_leg_limit_order(self) -> None:
        payload = to_alpaca_payload(
            build_order_intent(make_decision(), make_spread(), approved(), NOW)
        )
        assert payload["order_class"] == "mleg"
        assert payload["type"] == "limit"
        assert payload["time_in_force"] == "day"
        assert payload["limit_price"] == "2.38"
        assert [leg["side"] for leg in payload["legs"]] == ["buy", "sell"]
        assert [leg["position_intent"] for leg in payload["legs"]] == [
            "buy_to_open",
            "sell_to_open",
        ]

    def test_payload_carries_real_contract_symbols(self) -> None:
        spread = make_spread()
        payload = to_alpaca_payload(
            build_order_intent(make_decision(), spread, approved(), NOW)
        )
        assert payload["legs"][0]["symbol"] == spread.long_leg.contract.symbol
        assert payload["legs"][1]["symbol"] == spread.short_leg.contract.symbol


class TestStateMachine:
    def test_happy_path_is_legal(self) -> None:
        state = TradeState.DISCOVERED
        for target in (
            TradeState.ANALYZED,
            TradeState.AI_APPROVED,
            TradeState.RISK_APPROVED,
            TradeState.CONSTRUCTED,
            TradeState.SUBMITTED,
            TradeState.FILLED,
            TradeState.MONITORING,
            TradeState.EXIT_REQUESTED,
            TradeState.CLOSED,
        ):
            state = transition(state, target)
        assert state is TradeState.CLOSED

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (TradeState.DISCOVERED, TradeState.SUBMITTED),
            (TradeState.DISCOVERED, TradeState.FILLED),
            (TradeState.ANALYZED, TradeState.CONSTRUCTED),
            (TradeState.CLOSED, TradeState.SUBMITTED),
            (TradeState.REJECTED, TradeState.SUBMITTED),
            (TradeState.RISK_APPROVED, TradeState.FILLED),
        ],
    )
    def test_illegal_transitions_raise(self, source, target) -> None:  # type: ignore[no-untyped-def]
        assert not can_transition(source, target)
        with pytest.raises(IllegalTransitionError):
            transition(source, target)

    def test_terminal_states_have_no_successors(self) -> None:
        for state in (TradeState.CLOSED, TradeState.REJECTED, TradeState.FAILED):
            assert is_terminal(state)
            assert ALLOWED_TRANSITIONS[state] == frozenset()

    def test_recoverable_states_exclude_terminals(self) -> None:
        for state in TradeState:
            assert not (is_terminal(state) and is_recoverable(state))

    def test_every_state_has_a_transition_entry(self) -> None:
        assert set(ALLOWED_TRANSITIONS) == set(TradeState)


class TestStatusMapping:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("filled", TradeState.FILLED),
            ("partially_filled", TradeState.PARTIALLY_FILLED),
            ("canceled", TradeState.REJECTED),
            ("expired", TradeState.REJECTED),
            ("rejected", TradeState.REJECTED),
        ],
    )
    def test_known_statuses_map(self, status, expected) -> None:  # type: ignore[no-untyped-def]
        assert status_to_state(status, TradeState.SUBMITTED) is expected

    def test_unknown_status_leaves_the_state_alone(self) -> None:
        assert status_to_state("new", TradeState.SUBMITTED) is TradeState.SUBMITTED
        assert status_to_state("", TradeState.MONITORING) is TradeState.MONITORING


class TestIdempotentSubmission:
    def test_order_is_reserved_before_it_is_sent(self, journal) -> None:  # type: ignore[no-untyped-def]
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        assert journal.reserve_order(intent) is True
        assert journal.order_exists(intent.client_order_id)
        assert journal.get_order(intent.client_order_id)["state"] == "CONSTRUCTED"

    def test_second_reservation_of_the_same_id_is_refused(self, journal) -> None:  # type: ignore[no-untyped-def]
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        assert journal.reserve_order(intent) is True
        assert journal.reserve_order(intent) is False

    def test_broker_refuses_a_duplicate_client_order_id(self) -> None:
        broker = SimulatedBroker(make_account())
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        broker.submit_spread(intent)
        with pytest.raises(BrokerError, match="duplicate client_order_id"):
            broker.submit_spread(intent)

    def test_ambiguous_submission_is_never_retried_blind(self) -> None:
        broker = SimulatedBroker(
            make_account(),
            fail_next_submit_with=AmbiguousSubmissionError("cid", "ReadTimeout"),
        )
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        with pytest.raises(AmbiguousSubmissionError):
            broker.submit_spread(intent)
        # Nothing was recorded, so the caller must reconcile rather than resend.
        assert broker.orders == {}
        assert broker.submitted_payloads == []


class TestRecovery:
    def test_reconciles_a_submitted_order_that_actually_filled(self, journal) -> None:  # type: ignore[no-untyped-def]
        broker = SimulatedBroker(make_account())
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        journal.reserve_order(intent)
        broker.submit_spread(intent)
        journal.set_order_state(intent.client_order_id, TradeState.SUBMITTED, "sent")

        report = reconcile_open_orders(journal, broker)
        assert report.inspected == 1
        assert report.reconciled == 1
        assert journal.get_order(intent.client_order_id)["state"] == "FILLED"

    def test_reserved_but_never_sent_order_is_retired(self, journal) -> None:  # type: ignore[no-untyped-def]
        """A crash between reservation and submission must not leave a live id."""
        broker = SimulatedBroker(make_account())
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        journal.reserve_order(intent)

        report = reconcile_open_orders(journal, broker)
        assert report.orphaned == 1
        assert journal.get_order(intent.client_order_id)["state"] == "FAILED"

    def test_recovery_never_submits_anything(self, journal) -> None:  # type: ignore[no-untyped-def]
        broker = SimulatedBroker(make_account())
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        journal.reserve_order(intent)
        reconcile_open_orders(journal, broker)
        assert broker.submitted_payloads == []

    def test_broker_lookup_failure_does_not_stop_the_sweep(self, journal) -> None:  # type: ignore[no-untyped-def]
        class Exploding(SimulatedBroker):
            def get_order_by_client_id(self, client_order_id: str):  # type: ignore[no-untyped-def]
                raise httpx.ConnectError("down")

        broker = Exploding(make_account())
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        journal.reserve_order(intent)
        report = reconcile_open_orders(journal, broker)
        assert report.inspected == 1
        assert any("broker lookup failed" in d for d in report.details)

    def test_terminal_orders_are_not_reinspected(self, journal) -> None:  # type: ignore[no-untyped-def]
        broker = SimulatedBroker(make_account())
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        journal.reserve_order(intent)
        journal.set_order_state(intent.client_order_id, TradeState.REJECTED, "done")
        assert reconcile_open_orders(journal, broker).inspected == 0


class TestMonitor:
    def test_refresh_persists_the_brokers_view(self, journal) -> None:  # type: ignore[no-untyped-def]
        broker = SimulatedBroker(make_account())
        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        journal.reserve_order(intent)
        broker.submit_spread(intent)
        journal.set_order_state(intent.client_order_id, TradeState.SUBMITTED, "sent")

        record = OrderMonitor(broker, journal).refresh(intent.client_order_id)
        assert record.filled_quantity == 2
        assert journal.get_order(intent.client_order_id)["state"] == "FILLED"

    def test_refresh_of_an_unknown_order_returns_none(self, journal) -> None:  # type: ignore[no-untyped-def]
        broker = SimulatedBroker(make_account())
        assert OrderMonitor(broker, journal).refresh("nope") is None


class TestMarking:
    def test_spread_mark_uses_both_mids(self) -> None:
        long_c = make_contract(bid=4.00, ask=4.10)
        short_c = make_contract(strike=775.0, delta=0.3, bid=1.50, ask=1.60)
        assert mark_spread_cents(long_c, short_c) == 250

    def test_missing_quote_cannot_be_marked(self) -> None:
        assert mark_spread_cents(make_contract(with_quote=False), make_contract()) is None

    def test_position_mark_reports_progress_toward_max_profit(self) -> None:
        position = PositionRecord(
            position_id="p",
            decision_id="d",
            client_order_id="c",
            symbol="SPY",
            strategy=Strategy.BULL_CALL_SPREAD,
            quantity=1,
            entry_debit_cents=23_800,
            max_loss_cents=23_800,
            max_profit_cents=26_200,
            opened_at=NOW,
            expiration=date(2026, 9, 3),
            long_symbol="L",
            short_symbol="S",
        )
        mark = mark_position(position, 380)
        assert mark.unrealized_pnl_cents == 38_000 - 23_800
        assert mark.pct_of_max_profit == pytest.approx(14_200 / 26_200)
        assert mark.pct_of_defined_risk_lost == 0.0

    def test_losing_position_reports_risk_consumed(self) -> None:
        position = PositionRecord(
            position_id="p",
            decision_id="d",
            client_order_id="c",
            symbol="SPY",
            strategy=Strategy.BULL_CALL_SPREAD,
            quantity=1,
            entry_debit_cents=20_000,
            max_loss_cents=20_000,
            max_profit_cents=30_000,
            opened_at=NOW,
            expiration=date(2026, 9, 3),
            long_symbol="L",
            short_symbol="S",
        )
        mark = mark_position(position, 50)
        assert mark.unrealized_pnl_cents == -15_000
        assert mark.pct_of_defined_risk_lost == pytest.approx(0.75)
