"""Audit journal: persistence, reconstructability and secret redaction."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from alphamesh.models.domain import (
    ExitReason,
    PositionRecord,
    ReasonCode,
    Regime,
    RiskDecision,
    Strategy,
    TradeOutcome,
    TradeState,
)
from alphamesh.persistence.journal import REDACTED, Journal, redact
from tests.conftest import NOW, make_decision, make_regime


@pytest.fixture
def journal() -> Journal:
    j = Journal(":memory:")
    yield j
    j.close()


class TestRedaction:
    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "APCA_API_KEY_ID",
            "apca_api_secret_key",
            "ANTHROPIC_API_KEY",
            "secret",
            "access_token",
            "password",
            "x-api-key",
            "Authorization",
            "account_number",
            "private_key",
        ],
    )
    def test_credential_shaped_keys_are_blanked(self, key: str) -> None:
        assert redact({key: "super-secret"})[key] == REDACTED

    def test_redaction_recurses_into_nested_structures(self) -> None:
        payload = {"outer": {"list": [{"api_key": "sk-live-123"}], "safe": 1}}
        result = redact(payload)
        assert result["outer"]["list"][0]["api_key"] == REDACTED
        assert result["outer"]["safe"] == 1

    def test_ordinary_fields_survive(self) -> None:
        payload = {"symbol": "SPY", "strike": 770.0, "delta": 0.55}
        assert redact(payload) == payload

    def test_secrets_never_reach_the_database(self, journal) -> None:  # type: ignore[no-untyped-def]
        journal.record_event(
            "config_snapshot",
            {
                "APCA_API_SECRET_KEY": "sk-live-abcdef",
                "anthropic_api_key": "sk-ant-123",
                "symbol": "SPY",
            },
        )
        stored = journal.recent_events(1)[0]["payload"]
        assert "sk-live-abcdef" not in stored
        assert "sk-ant-123" not in stored
        assert "SPY" in stored

    def test_account_number_is_never_journalled(self, journal) -> None:  # type: ignore[no-untyped-def]
        journal.record_event("account", {"account_number": "PA3D6YAUG6DT"})
        assert "PA3D6YAUG6DT" not in journal.recent_events(1)[0]["payload"]


class TestDecisionPersistence:
    def test_decision_is_recorded_with_full_context(self, journal) -> None:  # type: ignore[no-untyped-def]
        decision = make_decision()
        journal.record_decision(
            decision,
            features={"trend_strength": 0.35},
            regime=make_regime(),
        )
        row = journal.latest_decision()
        assert row["decision_id"] == decision.decision_id
        assert row["strategy"] == "BULL_CALL_SPREAD"
        assert json.loads(row["features"])["trend_strength"] == 0.35
        assert json.loads(row["regime_evidence"])["regime"] == "BULLISH_TREND"

    def test_no_trade_decision_records_its_reason(self, journal) -> None:  # type: ignore[no-untyped-def]
        decision = make_decision(strategy=Strategy.NO_TRADE)
        journal.record_decision(decision)
        row = journal.latest_decision()
        assert row["no_trade_reason"] == "test"

    def test_reason_codes_round_trip(self, journal) -> None:  # type: ignore[no-untyped-def]
        risk = RiskDecision(
            approved=False,
            quantity=0,
            max_loss_cents=0,
            max_profit_cents=0,
            reason_codes=(ReasonCode.STALE_QUOTES, ReasonCode.WIDE_SPREAD),
            detail="bad quotes",
        )
        journal.record_risk_decision("d1", risk)
        row = journal.recent_rejections(1)[0]
        assert json.loads(row["reason_codes"]) == ["STALE_QUOTES", "WIDE_SPREAD"]

    def test_only_rejections_appear_in_the_rejection_feed(self, journal) -> None:  # type: ignore[no-untyped-def]
        journal.record_risk_decision(
            "ok",
            RiskDecision(
                approved=True,
                quantity=1,
                max_loss_cents=100,
                max_profit_cents=200,
                reason_codes=(ReasonCode.APPROVED,),
            ),
        )
        assert journal.recent_rejections() == []


class TestStateTransitions:
    def test_transitions_are_persisted_in_order(self, journal) -> None:  # type: ignore[no-untyped-def]
        from alphamesh.execution.order_builder import build_order_intent
        from tests.test_execution import approved, make_spread

        intent = build_order_intent(make_decision(), make_spread(), approved(), NOW)
        journal.reserve_order(intent)
        journal.set_order_state(intent.client_order_id, TradeState.SUBMITTED, "sent")
        journal.set_order_state(intent.client_order_id, TradeState.FILLED, "filled")

        rows = journal.transitions_for(intent.client_order_id)
        assert [r["to_state"] for r in rows] == ["SUBMITTED", "FILLED"]
        assert rows[0]["from_state"] == "CONSTRUCTED"


class TestPositionsAndOutcomes:
    def _position(self) -> PositionRecord:
        return PositionRecord(
            position_id="pos1",
            decision_id="d1",
            client_order_id="c1",
            symbol="SPY",
            strategy=Strategy.BULL_CALL_SPREAD,
            quantity=2,
            entry_debit_cents=47_600,
            max_loss_cents=47_600,
            max_profit_cents=52_400,
            opened_at=NOW,
            expiration=date(2026, 9, 3),
            long_symbol="SPY260903C00769000",
            short_symbol="SPY260903C00774000",
        )

    def test_position_round_trips(self, journal) -> None:  # type: ignore[no-untyped-def]
        journal.record_position(self._position())
        loaded = journal.get_position("pos1")
        assert loaded.symbol == "SPY"
        assert loaded.quantity == 2
        assert loaded.expiration == date(2026, 9, 3)

    def test_open_positions_excludes_closed(self, journal) -> None:  # type: ignore[no-untyped-def]
        journal.record_position(self._position())
        assert len(journal.open_positions()) == 1
        journal.set_position_state("pos1", TradeState.CLOSED)
        assert journal.open_positions() == []

    def test_excursions_are_tracked(self, journal) -> None:  # type: ignore[no-untyped-def]
        journal.record_position(self._position())
        journal.update_excursions("pos1", 5_000, -3_000)
        assert journal.position_excursions("pos1") == (5_000, -3_000)

    def test_outcome_closes_the_position(self, journal) -> None:  # type: ignore[no-untyped-def]
        journal.record_position(self._position())
        outcome = TradeOutcome(
            position_id="pos1",
            decision_id="d1",
            symbol="SPY",
            strategy=Strategy.BULL_CALL_SPREAD,
            regime=Regime.BULLISH_TREND,
            confidence=0.8,
            quantity=2,
            entry_debit_cents=47_600,
            exit_value_cents=62_000,
            realized_pnl_cents=14_400,
            return_on_defined_risk=0.30,
            holding_minutes=45.0,
            exit_reason=ExitReason.PROFIT_TARGET,
            opened_at=NOW,
            closed_at=NOW + timedelta(minutes=45),
        )
        journal.record_outcome(outcome)
        assert journal.open_positions() == []
        assert journal.realized_pnl_cents() == 14_400
        assert journal.outcomes()[0]["exit_reason"] == "PROFIT_TARGET"

    def test_realized_pnl_is_zero_with_no_trades(self, journal) -> None:  # type: ignore[no-untyped-def]
        assert journal.realized_pnl_cents() == 0
        assert journal.outcomes() == []


class TestDurability:
    def test_journal_survives_reopening(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "alphamesh.db"
        first = Journal(path)
        first.record_decision(make_decision())
        first.close()

        second = Journal(path)
        assert second.latest_decision()["symbol"] == "SPY"
        second.close()

    def test_schema_creates_its_parent_directory(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "nested" / "deeper" / "alphamesh.db"
        journal = Journal(path)
        assert path.exists()
        journal.close()
