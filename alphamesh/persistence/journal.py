"""SQLite audit journal.

Everything the agent observes, computes, argues, decides, sends and earns is
written here. The journal is also the source of truth for restart recovery:
non-terminal orders are reloaded from it and reconciled against the broker
before any new order can be built.

Secrets never enter the journal. :func:`redact` strips any key whose name looks
like a credential before a payload is serialised.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alphamesh.models.domain import (
    AIArgument,
    ExecutionRecord,
    JudgeVerdict,
    OrderIntent,
    PositionRecord,
    RegimeAssessment,
    RiskDecision,
    Strategy,
    TradeDecision,
    TradeOutcome,
    TradeState,
)
from alphamesh.persistence.models import (
    POST_MIGRATION_INDEXES,
    SCHEMA,
    SCHEMA_VERSION,
)

SECRET_KEY_MARKERS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "authorization",
    "x-api-key",
    "account_number",
    "private",
)
REDACTED = "<redacted>"

ORDER_KIND_ENTRY = "ENTRY"
ORDER_KIND_EXIT = "EXIT"

POSITION_ORIGIN_AGENT = "AGENT"
POSITION_ORIGIN_ADOPTED = "ADOPTED"

#: Marks an outcome that was written without a broker closing order behind it.
PHANTOM_CLOSE_NOTE = "PHANTOM_NO_BROKER_EXIT_ORDER"


def redact(value: Any) -> Any:
    """Recursively blank out anything that looks like a credential.

    Matching is on the key name, case-insensitively, by substring: a field
    called ``apca_api_secret_key`` is caught by the ``secret`` marker without
    needing an exact-name allowlist to stay in sync.
    """
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in SECRET_KEY_MARKERS):
                out[str(key)] = REDACTED
            else:
                out[str(key)] = redact(item)
        return out
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


def _dumps(payload: Any) -> str:
    return json.dumps(redact(payload), default=str, sort_keys=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Journal:
    """Thin, synchronous SQLite wrapper. One journal per process."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    # A journal written by an older build already exists on the production
    # volume, and CREATE TABLE IF NOT EXISTS silently leaves its columns alone.
    # Every column added after v1 therefore has to be applied by hand, additively
    # -- nothing is dropped or rewritten, so an old row keeps its history and
    # simply takes the default for the new column.
    _ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
        ("orders", "kind", "TEXT NOT NULL DEFAULT 'ENTRY'"),
        ("orders", "position_id", "TEXT"),
        ("positions", "origin", "TEXT NOT NULL DEFAULT 'AGENT'"),
        ("positions", "entry_basis", "TEXT NOT NULL DEFAULT 'FILL'"),
        ("outcomes", "reconciliation_note", "TEXT"),
    )

    def _migrate(self) -> None:
        for table, column, ddl in self._ADDED_COLUMNS:
            existing = {
                str(row["name"])
                for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not existing or column in existing:
                continue
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        # Only now that every column exists can they be indexed.
        for statement in POST_MIGRATION_INDEXES:
            self._conn.execute(statement)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ----------------------------------------------------------------- events
    def record_event(
        self,
        event_type: str,
        payload: Any,
        decision_id: str | None = None,
        symbol: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO events(ts, event_type, decision_id, symbol, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (_now(), event_type, decision_id, symbol, _dumps(payload)),
            )

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- decisions
    def record_decision(
        self,
        decision: TradeDecision,
        features: dict[str, float] | None = None,
        regime: RegimeAssessment | None = None,
        bull: AIArgument | None = None,
        bear: AIArgument | None = None,
        verdict: JudgeVerdict | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO decisions(
                    decision_id, ts, symbol, regime, direction, strategy, confidence,
                    bull_score, bear_score, quant_score, reason_codes, no_trade_reason,
                    ai_provider, features, regime_evidence, bull_argument,
                    bear_argument, judge_verdict)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    decision.decision_id,
                    decision.timestamp.isoformat(),
                    decision.symbol,
                    decision.regime.value,
                    decision.direction.value,
                    decision.strategy.value,
                    decision.confidence,
                    decision.bull_score,
                    decision.bear_score,
                    decision.quant_score,
                    json.dumps([c.value for c in decision.reason_codes]),
                    decision.no_trade_reason,
                    decision.ai_provider,
                    _dumps(features or {}),
                    _dumps(regime.model_dump() if regime else {}),
                    _dumps(bull.model_dump() if bull else {}),
                    _dumps(bear.model_dump() if bear else {}),
                    _dumps(verdict.model_dump() if verdict else {}),
                ),
            )

    def recent_decisions(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_decision(self) -> dict[str, Any] | None:
        rows = self.recent_decisions(1)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ risk
    def record_risk_decision(self, decision_id: str, risk: RiskDecision) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO risk_decisions(
                    decision_id, ts, approved, quantity, max_loss_cents,
                    max_profit_cents, reason_codes, detail, checks_run)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    decision_id,
                    _now(),
                    int(risk.approved),
                    risk.quantity,
                    risk.max_loss_cents,
                    risk.max_profit_cents,
                    json.dumps([c.value for c in risk.reason_codes]),
                    risk.detail,
                    json.dumps(list(risk.checks_run)),
                ),
            )

    def recent_rejections(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT r.*, d.symbol, d.strategy FROM risk_decisions r "
            "LEFT JOIN decisions d ON d.decision_id = r.decision_id "
            "WHERE r.approved = 0 ORDER BY r.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- orders
    def reserve_order(
        self,
        intent: OrderIntent,
        kind: str = ORDER_KIND_ENTRY,
        position_id: str | None = None,
    ) -> bool:
        """Persist the order intent *before* it is sent.

        Returns False when this ``client_order_id`` is already reserved, which
        is what makes submission idempotent across a crash: the row exists
        before the network call, so a restart finds it and reconciles instead of
        sending a second order.

        ``kind`` separates the order that opens a position from the one that
        closes it. The entry TTL sweep must never retire an exit order, and an
        exit must be findable from its position.
        """
        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO orders(
                        client_order_id, decision_id, symbol, strategy, quantity,
                        limit_price_cents, max_loss_cents, legs, state,
                        created_at, updated_at, kind, position_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        intent.client_order_id,
                        intent.decision_id,
                        intent.symbol,
                        intent.strategy.value,
                        intent.quantity,
                        intent.limit_price_cents,
                        intent.max_loss_cents,
                        _dumps([leg.model_dump() for leg in intent.legs]),
                        TradeState.CONSTRUCTED.value,
                        intent.created_at.isoformat(),
                        _now(),
                        kind,
                        position_id,
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def exit_order_count(self, position_id: str) -> int:
        """How many closing orders have been raised for this position.

        Used as the attempt number in the exit order id, so a close that was
        confirmed dead can be re-quoted under a fresh id. It is read from the
        journal rather than held in memory, so it survives a restart.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM orders WHERE position_id = ? AND kind = ?",
            (position_id, ORDER_KIND_EXIT),
        ).fetchone()
        return int(row["n"])

    def exit_order_for(self, position_id: str) -> dict[str, Any] | None:
        """The most recent closing order raised for one position, if any."""
        row = self._conn.execute(
            "SELECT * FROM orders WHERE position_id = ? AND kind = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (position_id, ORDER_KIND_EXIT),
        ).fetchone()
        return dict(row) if row else None

    def order_exists(self, client_order_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return row is not None

    def known_client_order_ids(self) -> frozenset[str]:
        rows = self._conn.execute("SELECT client_order_id FROM orders").fetchall()
        return frozenset(r["client_order_id"] for r in rows)

    def update_order_execution(self, record: ExecutionRecord, state: TradeState) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE orders SET broker_order_id = ?, broker_status = ?,
                    filled_quantity = ?, filled_avg_price_cents = ?,
                    state = ?, updated_at = ?
                WHERE client_order_id = ?
                """,
                (
                    record.broker_order_id,
                    record.status,
                    record.filled_quantity,
                    record.filled_avg_price_cents,
                    state.value,
                    _now(),
                    record.client_order_id,
                ),
            )

    def set_order_state(
        self, client_order_id: str, state: TradeState, detail: str = ""
    ) -> None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM orders WHERE client_order_id = ?", (client_order_id,)
            ).fetchone()
            previous = row["state"] if row else None
            conn.execute(
                "UPDATE orders SET state = ?, updated_at = ? WHERE client_order_id = ?",
                (state.value, _now(), client_order_id),
            )
            conn.execute(
                "INSERT INTO state_transitions(client_order_id, from_state, to_state, "
                "ts, detail) VALUES (?,?,?,?,?)",
                (client_order_id, previous, state.value, _now(), detail),
            )

    def get_order(self, client_order_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return dict(row) if row else None

    def open_orders(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Orders in a non-terminal state, for restart recovery.

        ``kind`` narrows to entry or exit orders; the TTL sweep uses it so a
        closing order is never mistaken for a stale entry and cancelled.
        """
        terminal = (
            TradeState.CLOSED.value,
            TradeState.REJECTED.value,
            TradeState.FAILED.value,
        )
        sql = f"SELECT * FROM orders WHERE state NOT IN ({','.join('?' * len(terminal))})"
        params: tuple[Any, ...] = terminal
        if kind is not None:
            sql += " AND kind = ?"
            params = (*terminal, kind)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def transitions_for(self, client_order_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM state_transitions WHERE client_order_id = ? ORDER BY id",
            (client_order_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- positions
    def record_position(
        self,
        position: PositionRecord,
        origin: str = POSITION_ORIGIN_AGENT,
        entry_basis: str = "FILL",
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO positions(
                    position_id, decision_id, client_order_id, symbol, strategy,
                    quantity, entry_debit_cents, max_loss_cents, max_profit_cents,
                    opened_at, expiration, long_symbol, short_symbol, state,
                    origin, entry_basis)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    position.position_id,
                    position.decision_id,
                    position.client_order_id,
                    position.symbol,
                    position.strategy.value,
                    position.quantity,
                    position.entry_debit_cents,
                    position.max_loss_cents,
                    position.max_profit_cents,
                    position.opened_at.isoformat(),
                    position.expiration.isoformat(),
                    position.long_symbol,
                    position.short_symbol,
                    position.state.value,
                    origin,
                    entry_basis,
                ),
            )

    def update_excursions(self, position_id: str, mfe: int | None, mae: int | None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE positions SET mfe_cents = ?, mae_cents = ? WHERE position_id = ?",
                (mfe, mae, position_id),
            )

    def set_position_state(self, position_id: str, state: TradeState) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE positions SET state = ? WHERE position_id = ?",
                (state.value, position_id),
            )

    def open_positions(self) -> list[PositionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE state NOT IN (?, ?, ?)",
            (TradeState.CLOSED.value, TradeState.REJECTED.value, TradeState.FAILED.value),
        ).fetchall()
        return [self._row_to_position(dict(r)) for r in rows]

    def get_position(self, position_id: str) -> PositionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM positions WHERE position_id = ?", (position_id,)
        ).fetchone()
        return self._row_to_position(dict(row)) if row else None

    def position_excursions(self, position_id: str) -> tuple[int | None, int | None]:
        row = self._conn.execute(
            "SELECT mfe_cents, mae_cents FROM positions WHERE position_id = ?",
            (position_id,),
        ).fetchone()
        if not row:
            return None, None
        return row["mfe_cents"], row["mae_cents"]

    @staticmethod
    def _row_to_position(row: dict[str, Any]) -> PositionRecord:
        from datetime import date

        return PositionRecord(
            position_id=row["position_id"],
            decision_id=row["decision_id"],
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            strategy=Strategy(row["strategy"]),
            quantity=row["quantity"],
            entry_debit_cents=row["entry_debit_cents"],
            max_loss_cents=row["max_loss_cents"],
            max_profit_cents=row["max_profit_cents"],
            opened_at=datetime.fromisoformat(row["opened_at"]),
            expiration=date.fromisoformat(row["expiration"]),
            long_symbol=row["long_symbol"],
            short_symbol=row["short_symbol"],
            state=TradeState(row["state"]),
        )

    # -------------------------------------------------------------- outcomes
    def record_outcome(
        self, outcome: TradeOutcome, reconciliation_note: str | None = None
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO outcomes(
                    position_id, decision_id, symbol, strategy, regime, confidence,
                    quantity, entry_debit_cents, exit_value_cents, realized_pnl_cents,
                    return_on_defined_risk, holding_minutes, mfe_cents, mae_cents,
                    exit_reason, opened_at, closed_at, reconciliation_note)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    outcome.position_id,
                    outcome.decision_id,
                    outcome.symbol,
                    outcome.strategy.value,
                    outcome.regime.value,
                    outcome.confidence,
                    outcome.quantity,
                    outcome.entry_debit_cents,
                    outcome.exit_value_cents,
                    outcome.realized_pnl_cents,
                    outcome.return_on_defined_risk,
                    outcome.holding_minutes,
                    outcome.max_favorable_excursion_cents,
                    outcome.max_adverse_excursion_cents,
                    outcome.exit_reason.value,
                    outcome.opened_at.isoformat(),
                    outcome.closed_at.isoformat(),
                    reconciliation_note,
                ),
            )
            conn.execute(
                "UPDATE positions SET state = ? WHERE position_id = ?",
                (TradeState.CLOSED.value, outcome.position_id),
            )

    def outcomes(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM outcomes ORDER BY closed_at DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def realized_pnl_cents(self, since_iso: str | None = None) -> int:
        """Realised profit and loss from outcomes backed by a real closing fill.

        An outcome carrying a ``reconciliation_note`` was written without a
        broker closing order behind it. The row is kept for the audit trail, but
        its money was never earned, so it is excluded from every total rather
        than being allowed to misreport performance.
        """
        if since_iso:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(realized_pnl_cents), 0) AS total FROM outcomes "
                "WHERE closed_at >= ? AND reconciliation_note IS NULL",
                (since_iso,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(realized_pnl_cents), 0) AS total FROM outcomes "
                "WHERE reconciliation_note IS NULL"
            ).fetchone()
        return int(row["total"])

    def annotate_outcome(self, position_id: str, note: str) -> None:
        """Mark an outcome as not corresponding to a real broker exit.

        The row is never deleted: the competition record has to show that the
        earlier close was journalled without a closing order behind it.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE outcomes SET reconciliation_note = ? WHERE position_id = ?",
                (note, position_id),
            )

    def positions_for_legs(self, long_symbol: str, short_symbol: str) -> list[dict[str, Any]]:
        """Every journalled position over this exact pair of contracts, any state."""
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE long_symbol = ? AND short_symbol = ? "
            "ORDER BY opened_at",
            (long_symbol, short_symbol),
        ).fetchall()
        return [dict(r) for r in rows]


__all__ = [
    "ORDER_KIND_ENTRY",
    "ORDER_KIND_EXIT",
    "PHANTOM_CLOSE_NOTE",
    "POSITION_ORIGIN_ADOPTED",
    "POSITION_ORIGIN_AGENT",
    "REDACTED",
    "SECRET_KEY_MARKERS",
    "Journal",
    "redact",
]
