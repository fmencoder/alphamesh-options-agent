"""SQLite schema for the AlphaMesh audit journal.

One decision, one risk verdict, one order, one position and one outcome are all
linked by ``decision_id``, so any trade can be reconstructed end to end: what
the agent saw, what it computed, what each agent argued, what the judge ruled,
what the governor allowed, what was sent, what filled and what it earned.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    decision_id TEXT,
    symbol      TEXT,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_decision ON events(decision_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id     TEXT PRIMARY KEY,
    ts              TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    regime          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    confidence      REAL NOT NULL,
    bull_score      REAL NOT NULL,
    bear_score      REAL NOT NULL,
    quant_score     REAL NOT NULL,
    reason_codes    TEXT NOT NULL,
    no_trade_reason TEXT,
    ai_provider     TEXT NOT NULL,
    features        TEXT,
    regime_evidence TEXT,
    bull_argument   TEXT,
    bear_argument   TEXT,
    judge_verdict   TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol_ts ON decisions(symbol, ts);

CREATE TABLE IF NOT EXISTS risk_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     TEXT NOT NULL,
    ts              TEXT NOT NULL,
    approved        INTEGER NOT NULL,
    quantity        INTEGER NOT NULL,
    max_loss_cents  INTEGER NOT NULL,
    max_profit_cents INTEGER NOT NULL,
    reason_codes    TEXT NOT NULL,
    detail          TEXT NOT NULL,
    checks_run      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_risk_decision ON risk_decisions(decision_id);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id       TEXT PRIMARY KEY,
    decision_id           TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    strategy              TEXT NOT NULL,
    quantity              INTEGER NOT NULL,
    limit_price_cents     INTEGER NOT NULL,
    max_loss_cents        INTEGER NOT NULL,
    legs                  TEXT NOT NULL,
    state                 TEXT NOT NULL,
    broker_order_id       TEXT,
    broker_status         TEXT,
    filled_quantity       INTEGER NOT NULL DEFAULT 0,
    filled_avg_price_cents INTEGER,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state);

CREATE TABLE IF NOT EXISTS state_transitions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id TEXT NOT NULL,
    from_state      TEXT,
    to_state        TEXT NOT NULL,
    ts              TEXT NOT NULL,
    detail          TEXT
);
CREATE INDEX IF NOT EXISTS idx_transitions_order ON state_transitions(client_order_id);

CREATE TABLE IF NOT EXISTS positions (
    position_id        TEXT PRIMARY KEY,
    decision_id        TEXT NOT NULL,
    client_order_id    TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    strategy           TEXT NOT NULL,
    quantity           INTEGER NOT NULL,
    entry_debit_cents  INTEGER NOT NULL,
    max_loss_cents     INTEGER NOT NULL,
    max_profit_cents   INTEGER NOT NULL,
    opened_at          TEXT NOT NULL,
    expiration         TEXT NOT NULL,
    long_symbol        TEXT NOT NULL,
    short_symbol       TEXT NOT NULL,
    state              TEXT NOT NULL,
    mfe_cents          INTEGER,
    mae_cents          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_positions_state ON positions(state);

CREATE TABLE IF NOT EXISTS outcomes (
    position_id            TEXT PRIMARY KEY,
    decision_id            TEXT NOT NULL,
    symbol                 TEXT NOT NULL,
    strategy               TEXT NOT NULL,
    regime                 TEXT NOT NULL,
    confidence             REAL NOT NULL,
    quantity               INTEGER NOT NULL,
    entry_debit_cents      INTEGER NOT NULL,
    exit_value_cents       INTEGER NOT NULL,
    realized_pnl_cents     INTEGER NOT NULL,
    return_on_defined_risk REAL NOT NULL,
    holding_minutes        REAL NOT NULL,
    mfe_cents              INTEGER,
    mae_cents              INTEGER,
    exit_reason            TEXT NOT NULL,
    opened_at              TEXT NOT NULL,
    closed_at              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcomes_closed ON outcomes(closed_at);
"""

__all__ = ["SCHEMA", "SCHEMA_VERSION"]
