"""TTL for zero-fill entry orders.

A working order holds the per-symbol duplicate lock. On 2026-08-31 five AMD
spreads sat unfilled at limits the market had walked away from; once duplicate
protection was fixed, a single such order would have blocked AMD for the rest
of the session while never filling. The TTL retires the order and frees the
symbol. It never places a replacement: the next entry must earn its way back
through a fresh quant -> AI -> contract -> risk cycle.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from alphamesh.alpaca.client import AlpacaStack
from alphamesh.alpaca.execution import BrokerError, SimulatedBroker
from alphamesh.alpaca.options import CaptureOptionChain
from alphamesh.execution.order_builder import build_order_intent
from alphamesh.models.domain import ExecutionRecord, RiskDecision, TradeState
from alphamesh.orchestrator import CycleReport, Orchestrator
from alphamesh.persistence.journal import Journal
from alphamesh.safety import GuardResult
from tests.conftest import CAPTURE_DIR, NOW, make_account, make_decision
from tests.test_orchestrator import TrendingMarketData, bull_script
from tests.test_risk_governor import spread_with

TTL = 180


def build_orchestrator(config, broker):  # type: ignore[no-untyped-def]
    stack = AlpacaStack(
        guard=GuardResult(paper=True, detail="test", checks=("ALPACA_PAPER=true",)),
        market_data=TrendingMarketData(),
        option_chain=CaptureOptionChain(CAPTURE_DIR),
        broker=broker,
        live_broker=False,
    )
    journal = Journal(":memory:")
    return Orchestrator(config, stack, journal, bull_script()), journal


class RecordingBroker(SimulatedBroker):
    """Simulated broker that records cancels and can fake a stuck cancel."""

    def __init__(self, account, confirm_cancel: bool = True) -> None:  # type: ignore[no-untyped-def]
        super().__init__(account)
        self.cancelled: list[str] = []
        self.confirm_cancel = confirm_cancel
        self.fail_cancel = False

    def cancel_order(self, broker_order_id: str) -> None:
        if self.fail_cancel:
            raise BrokerError("broker refused the cancel")
        self.cancelled.append(broker_order_id)
        if self.confirm_cancel:
            super().cancel_order(broker_order_id)


def seed_working_order(
    journal: Journal,
    broker: RecordingBroker,
    *,
    symbol: str = "AMD",
    age_seconds: int,
    filled_quantity: int = 0,
    state: TradeState = TradeState.SUBMITTED,
) -> str:
    """Put one order on the books in the journal and in the broker."""
    created = NOW - timedelta(seconds=age_seconds)
    intent = build_order_intent(
        make_decision(symbol=symbol),
        spread_with(400, symbol=symbol),
        RiskDecision(
            approved=True,
            quantity=1,
            max_loss_cents=40_000,
            max_profit_cents=10_000,
            reason_codes=(),
            detail="test",
            checks_run=("test",),
        ),
        created,
    )
    assert journal.reserve_order(intent)
    client_order_id = intent.client_order_id
    if state is not TradeState.CONSTRUCTED:
        journal.set_order_state(client_order_id, state, "seeded for test")
    broker.orders[client_order_id] = ExecutionRecord(
        client_order_id=client_order_id,
        broker_order_id=f"broker-{client_order_id}",
        status="partially_filled" if filled_quantity else "new",
        filled_quantity=filled_quantity,
        raw_status="partially_filled" if filled_quantity else "new",
    )
    return client_order_id


@pytest.fixture
def ttl_config(config):  # type: ignore[no-untyped-def]
    return config.model_copy(
        update={"settings": config.settings.model_copy(update={"entry_order_ttl_seconds": TTL})}
    )


class TestTtlConfiguration:
    def test_default_ttl_is_180_seconds(self, config) -> None:  # type: ignore[no-untyped-def]
        assert config.settings.entry_order_ttl_seconds == 180


class TestStaleCancellation:
    def test_order_past_ttl_is_cancelled_and_lock_released(self, ttl_config) -> None:  # type: ignore[no-untyped-def]
        broker = RecordingBroker(make_account())
        orch, journal = build_orchestrator(ttl_config, broker)
        cid = seed_working_order(journal, broker, age_seconds=TTL + 30)

        report = CycleReport(started_at=NOW)
        orch._expire_stale_entry_orders(NOW, report)

        assert report.stale_orders_cancelled == 1
        assert broker.cancelled == [f"broker-{cid}"]
        # Terminal state means open_orders no longer reports it, which is what
        # releases the per-symbol duplicate lock.
        assert TradeState(journal.get_order(cid)["state"]) is TradeState.REJECTED
        assert cid not in {o["client_order_id"] for o in journal.open_orders()}

    def test_order_inside_ttl_is_left_alone(self, ttl_config) -> None:  # type: ignore[no-untyped-def]
        broker = RecordingBroker(make_account())
        orch, journal = build_orchestrator(ttl_config, broker)
        cid = seed_working_order(journal, broker, age_seconds=TTL - 30)

        report = CycleReport(started_at=NOW)
        orch._expire_stale_entry_orders(NOW, report)

        assert report.stale_orders_cancelled == 0
        assert broker.cancelled == []
        assert cid in {o["client_order_id"] for o in journal.open_orders()}


class TestPartialFillsAreExempt:
    """The explicit requirement: a partial fill is real exposure, never a timeout."""

    def test_partially_filled_order_is_never_ttl_cancelled(self, ttl_config) -> None:  # type: ignore[no-untyped-def]
        broker = RecordingBroker(make_account())
        orch, journal = build_orchestrator(ttl_config, broker)
        seed_working_order(journal, broker, age_seconds=TTL * 10, filled_quantity=2)

        report = CycleReport(started_at=NOW)
        orch._expire_stale_entry_orders(NOW, report)

        assert report.stale_orders_cancelled == 0
        assert broker.cancelled == []

    def test_fully_filled_order_is_never_ttl_cancelled(self, ttl_config) -> None:  # type: ignore[no-untyped-def]
        broker = RecordingBroker(make_account())
        orch, journal = build_orchestrator(ttl_config, broker)
        seed_working_order(journal, broker, age_seconds=TTL * 10, filled_quantity=4)

        report = CycleReport(started_at=NOW)
        orch._expire_stale_entry_orders(NOW, report)

        assert report.stale_orders_cancelled == 0
        assert broker.cancelled == []


class TestCancelMustBeConfirmed:
    def test_unconfirmed_cancel_does_not_release_the_lock(self, ttl_config) -> None:  # type: ignore[no-untyped-def]
        """Freeing the symbol while the order could still fill is the worst case."""
        broker = RecordingBroker(make_account(), confirm_cancel=False)
        orch, journal = build_orchestrator(ttl_config, broker)
        cid = seed_working_order(journal, broker, age_seconds=TTL + 30)

        report = CycleReport(started_at=NOW)
        orch._expire_stale_entry_orders(NOW, report)

        assert report.stale_orders_cancelled == 0
        assert cid in {o["client_order_id"] for o in journal.open_orders()}

    def test_broker_error_leaves_the_order_intact(self, ttl_config) -> None:  # type: ignore[no-untyped-def]
        broker = RecordingBroker(make_account())
        broker.fail_cancel = True
        orch, journal = build_orchestrator(ttl_config, broker)
        cid = seed_working_order(journal, broker, age_seconds=TTL + 30)

        report = CycleReport(started_at=NOW)
        orch._expire_stale_entry_orders(NOW, report)

        assert report.stale_orders_cancelled == 0
        assert cid in {o["client_order_id"] for o in journal.open_orders()}


class TestNoBlindReplacement:
    def test_sweep_places_no_order_of_its_own(self, ttl_config) -> None:  # type: ignore[no-untyped-def]
        """Retiring an order frees a symbol; it must never create one."""
        broker = RecordingBroker(make_account())
        orch, journal = build_orchestrator(ttl_config, broker)
        seed_working_order(journal, broker, age_seconds=TTL + 30)
        before = set(broker.orders)

        report = CycleReport(started_at=NOW)
        orch._expire_stale_entry_orders(NOW, report)

        assert set(broker.orders) == before, "sweep must not submit anything"
        assert report.orders_submitted == []


class TestScope:
    def test_ttl_zero_disables_the_sweep(self, config) -> None:  # type: ignore[no-untyped-def]
        disabled = config.model_copy(
            update={"settings": config.settings.model_copy(update={"entry_order_ttl_seconds": 0})}
        )
        broker = RecordingBroker(make_account())
        orch, journal = build_orchestrator(disabled, broker)
        seed_working_order(journal, broker, age_seconds=99_999)

        report = CycleReport(started_at=NOW)
        orch._expire_stale_entry_orders(NOW, report)

        assert report.stale_orders_cancelled == 0
        assert broker.cancelled == []

    def test_only_submitted_orders_are_swept(self, ttl_config) -> None:  # type: ignore[no-untyped-def]
        """A CONSTRUCTED reservation never reached the wire; recovery owns it."""
        broker = RecordingBroker(make_account())
        orch, journal = build_orchestrator(ttl_config, broker)
        seed_working_order(
            journal, broker, age_seconds=TTL * 5, state=TradeState.CONSTRUCTED
        )

        report = CycleReport(started_at=NOW)
        orch._expire_stale_entry_orders(NOW, report)

        assert report.stale_orders_cancelled == 0
        assert broker.cancelled == []
