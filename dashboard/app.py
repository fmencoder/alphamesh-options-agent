"""AlphaMesh Streamlit dashboard.

Built for a hackathon judge: the point is to make it obvious *why* a trade
happened or was refused, not to look pretty. Everything shown is read from the
audit journal, so nothing on this page is computed twice or differently from
what the agent actually did.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from alphamesh.analytics import build_report
from alphamesh.config import load_config
from alphamesh.persistence.journal import Journal
from alphamesh.risk.money import to_dollars
from alphamesh.safety import LiveTradingForbiddenError, enforce_paper_mode

COMPETITION_START_EQUITY = 100_000.0
REFRESH_SECONDS = 30


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _load_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_list(raw: Any) -> list[Any]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


st.set_page_config(page_title="AlphaMesh", page_icon="AM", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:0'>ALPHAMESH</h1>"
    "<p style='margin-top:0;letter-spacing:0.25em;color:#888'>"
    "AUTONOMOUS OPTIONS INTELLIGENCE</p>",
    unsafe_allow_html=True,
)

config = load_config()

# --------------------------------------------------------------------------- #
# Mode banner. This is the first thing on the page, deliberately.
# --------------------------------------------------------------------------- #
try:
    guard = enforce_paper_mode(config.settings)
    st.success(
        f"**PAPER MODE** - trading endpoint `{config.settings.base_url}` verified. "
        "No live-money order can be placed by this system."
    )
except LiveTradingForbiddenError as exc:
    st.error(f"**LIVE TRADING BLOCKED** - {exc.detail}")
    st.stop()

journal = Journal(config.settings.database_path)

outcomes = journal.outcomes()
open_positions = journal.open_positions()
decisions = journal.recent_decisions(limit=40)
rejections = journal.recent_rejections(limit=25)
events = journal.recent_events(limit=40)

realized_cents = sum(int(o["realized_pnl_cents"]) for o in outcomes)
today = datetime.now(UTC).date().isoformat()
daily_cents = journal.realized_pnl_cents(since_iso=today)
open_risk_cents = sum(p.max_loss_cents for p in open_positions)
wins = sum(1 for o in outcomes if int(o["realized_pnl_cents"]) > 0)
win_rate = wins / len(outcomes) if outcomes else 0.0
equity = COMPETITION_START_EQUITY + to_dollars(realized_cents)

# --------------------------------------------------------------------------- #
# Account and P&L
# --------------------------------------------------------------------------- #
st.subheader("Account")
cols = st.columns(6)
cols[0].metric("Equity (booked)", _fmt_money(equity))
cols[1].metric("Competition start", _fmt_money(COMPETITION_START_EQUITY))
cols[2].metric(
    "Total realised P&L",
    _fmt_money(to_dollars(realized_cents)),
    delta=f"{to_dollars(realized_cents):+,.2f}",
)
cols[3].metric("Today's realised P&L", _fmt_money(to_dollars(daily_cents)))
cols[4].metric("Win rate", f"{win_rate:.0%}" if outcomes else "n/a")
cols[5].metric("Closed trades", len(outcomes))

cols = st.columns(3)
cols[0].metric("Open positions", len(open_positions))
cols[1].metric("Open defined risk", _fmt_money(to_dollars(open_risk_cents)))
cols[2].metric(
    "Portfolio risk cap", _fmt_money(config.risk.max_portfolio_defined_risk)
)

if not outcomes:
    st.info(
        "No closed trades recorded yet, so there is no P&L to report. "
        "This figure is left empty rather than estimated."
    )

# --------------------------------------------------------------------------- #
# Latest decision - the "why" panel
# --------------------------------------------------------------------------- #
st.subheader("Latest AI decision")
if not decisions:
    st.write("No decisions recorded yet. Start the agent with `alphamesh run`.")
else:
    latest = decisions[0]
    strategy = latest["strategy"]
    badge = {
        "NO_TRADE": ":grey[NO TRADE]",
        "BULL_CALL_SPREAD": ":green[BULL CALL SPREAD]",
        "BEAR_PUT_SPREAD": ":red[BEAR PUT SPREAD]",
    }.get(strategy, strategy)

    cols = st.columns(6)
    cols[0].metric("Symbol", latest["symbol"])
    cols[1].metric("Regime", latest["regime"])
    cols[2].metric("Confidence", f"{float(latest['confidence']):.2f}")
    cols[3].metric("Bull score", f"{float(latest['bull_score']):.2f}")
    cols[4].metric("Bear score", f"{float(latest['bear_score']):.2f}")
    cols[5].metric("Quant score", f"{float(latest['quant_score']):.2f}")

    st.markdown(f"**Strategy:** {badge}  -  reasoned by `{latest['ai_provider']}`")

    codes = _load_list(latest["reason_codes"])
    if codes:
        st.markdown("**Reason codes:** " + ", ".join(f"`{c}`" for c in codes))
    if latest["no_trade_reason"]:
        st.warning(f"**Why no trade:** {latest['no_trade_reason']}")

    bull = _load_json(latest["bull_argument"])
    bear = _load_json(latest["bear_argument"])
    verdict = _load_json(latest["judge_verdict"])
    if bull or bear:
        left, right = st.columns(2)
        with left:
            st.markdown("##### Bull agent")
            st.write(bull.get("thesis", "-"))
            for point in bull.get("key_points", []):
                st.markdown(f"- {point}")
        with right:
            st.markdown("##### Bear agent")
            st.write(bear.get("thesis", "-"))
            for point in bear.get("key_points", []):
                st.markdown(f"- {point}")
    if verdict:
        st.markdown("##### Judge")
        st.write(verdict.get("rationale", "-"))

    with st.expander("Feature vector behind this decision"):
        st.json(_load_json(latest["features"]))

# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
st.subheader("Current market regime")
seen: dict[str, dict[str, Any]] = {}
for row in decisions:
    seen.setdefault(row["symbol"], row)
if seen:
    cols = st.columns(len(seen))
    for col, (symbol, row) in zip(cols, seen.items(), strict=True):
        col.metric(symbol, row["regime"], delta=row["direction"])
else:
    st.write("No regime assessments recorded yet.")

# --------------------------------------------------------------------------- #
# Positions, trades, rejections
# --------------------------------------------------------------------------- #
st.subheader("Active positions")
if open_positions:
    st.dataframe(
        [
            {
                "symbol": p.symbol,
                "strategy": p.strategy.value,
                "qty": p.quantity,
                "long": p.long_symbol,
                "short": p.short_symbol,
                "entry debit": _fmt_money(to_dollars(p.entry_debit_cents)),
                "max loss": _fmt_money(to_dollars(p.max_loss_cents)),
                "max profit": _fmt_money(to_dollars(p.max_profit_cents)),
                "expires": p.expiration.isoformat(),
                "opened": p.opened_at.isoformat(timespec="minutes"),
            }
            for p in open_positions
        ],
        use_container_width=True,
    )
else:
    st.write("No open positions.")

st.subheader("Recent closed trades")
if outcomes:
    st.dataframe(
        [
            {
                "symbol": o["symbol"],
                "strategy": o["strategy"],
                "regime": o["regime"],
                "qty": o["quantity"],
                "P&L": _fmt_money(to_dollars(int(o["realized_pnl_cents"]))),
                "return on risk": f"{float(o['return_on_defined_risk']):.1%}",
                "held (min)": round(float(o["holding_minutes"]), 1),
                "exit": o["exit_reason"],
                "closed": o["closed_at"],
            }
            for o in outcomes[:20]
        ],
        use_container_width=True,
    )
else:
    st.write("No closed trades yet.")

st.subheader("Recent Risk Governor rejections")
st.caption(
    "Every refused trade, with the machine-readable code that refused it. "
    "This is the audit trail a judge should read first."
)
if rejections:
    st.dataframe(
        [
            {
                "symbol": r.get("symbol") or "-",
                "strategy": r.get("strategy") or "-",
                "reason codes": ", ".join(_load_list(r["reason_codes"])),
                "detail": (r["detail"] or "")[:220],
                "when": r["ts"],
            }
            for r in rejections
        ],
        use_container_width=True,
    )
else:
    st.write("No rejections recorded.")

# --------------------------------------------------------------------------- #
# Competition analytics
# --------------------------------------------------------------------------- #
st.subheader("Competition analytics")
report = build_report(outcomes)
report.pop("_unused", None)
tabs = st.tabs(["Overall", "By strategy", "By regime", "By symbol", "By confidence"])
with tabs[0]:
    st.json(report["overall"])
with tabs[1]:
    st.json(report["by_strategy"] or {"note": "no closed trades"})
with tabs[2]:
    st.json(report["by_regime"] or {"note": "no closed trades"})
with tabs[3]:
    st.json(report["by_symbol"] or {"note": "no closed trades"})
with tabs[4]:
    st.json(report["by_confidence"] or {"note": "no closed trades"})

# --------------------------------------------------------------------------- #
# Health and risk configuration
# --------------------------------------------------------------------------- #
st.subheader("Agent health")
last_cycle = next((e for e in events if e["event_type"] == "cycle_complete"), None)
cols = st.columns(4)
cols[0].metric("Last cycle", last_cycle["ts"][:19] if last_cycle else "never")
cols[1].metric("Journal", Path(config.settings.database_path).name)
cols[2].metric("Data source", config.settings.data_source)
cols[3].metric("Dry run", str(config.settings.dry_run))

with st.expander("Hard risk limits in force (config/risk.yaml)"):
    st.json(
        {
            "max_defined_loss_per_trade": config.risk.max_defined_loss_per_trade,
            "high_confidence_max_defined_loss": (
                config.risk.high_confidence_max_defined_loss
            ),
            "absolute_max_defined_loss": config.risk.absolute_max_defined_loss,
            "max_open_positions": config.risk.max_open_positions,
            "max_portfolio_defined_risk": config.risk.max_portfolio_defined_risk,
            "daily_loss_circuit_breaker": config.risk.daily_loss_circuit_breaker,
            "correlation_groups": config.risk.correlation_groups,
            "allowed_strategies": config.risk.allowed_strategies,
        }
    )

with st.expander("Recent journal events"):
    st.dataframe(
        [
            {"when": e["ts"], "event": e["event_type"], "symbol": e["symbol"] or "-"}
            for e in events
        ],
        use_container_width=True,
    )

journal.close()
st.caption(
    "AlphaMesh trades Alpaca PAPER only. Nothing here is investment advice. "
    f"Page rendered {datetime.now(UTC).isoformat(timespec='seconds')}."
)
