# AlphaMesh — competition write-up

**Autonomous AI options-trading agent · Alpaca paper trading · MIT licensed**

---

## What it is

AlphaMesh is a headless agent that trades defined-risk options spreads on SPY
and QQQ. One cycle runs the whole lifecycle with no human in the loop: retrieve
market data, compute features, score the opportunity, classify the regime,
convene an adversarial AI council, choose `NO_TRADE` / `BULL_CALL_SPREAD` /
`BEAR_PUT_SPREAD`, select real Alpaca contracts, size against hard risk limits,
submit a multi-leg order, monitor, manage and exit — journalling every step.

The governing principle is one line: **the AI advises; deterministic code
decides anything that touches money.**

## AI logic

Three roles debate, behind a cost gate.

A deterministic quantitative score (momentum, trend strength weighted by fit
quality, VWAP deviation, volume acceleration, opening-range position) runs every
cycle. **Below the configured threshold the AI is never invoked** — the council
is expensive and the market is not, and a test asserts zero provider calls when
the gate is closed.

Above it: a **bull agent** and a **bear agent** receive *identical* structured
market evidence and argue opposite sides. A **judge** receives both arguments
plus the quantitative and regime evidence and returns strict JSON.

The judge's authority is deliberately tiny. It may return exactly one of three
strategies and a confidence. It cannot pick contracts, strikes, expirations,
quantities, prices, or any amount of capital. Anything outside the allowed set
is **refused** with `AI_UNSUPPORTED_STRATEGY`, never coerced onto a strategy we
do allow. A directional verdict in an `UNSTABLE` or `UNKNOWN` regime is
overridden at the judge boundary.

The evidence packet is a whitelist. The council never sees account equity,
buying power, open positions, risk limits or position sizes — so it cannot
reason about, let alone argue for, a larger allocation. A test asserts this
against the real field names of `AccountState` and `RiskLimits`.

Without an Anthropic key the council falls back to deterministic heuristics and
the agent keeps trading. No test requires a paid LLM call.

## Risk gates

The Risk Governor is authoritative and non-bypassable. It loads `config/risk.yaml`
once and never mutates it. Thirteen gates run on every trade, each producing a
machine-readable reason code on failure:

paper mode · account tradeable · options level ≥ 3 · strategy allowlist ·
defined risk (debit strictly below strike width) · per-leg quote freshness,
spread width, book depth and greeks · duplicate order and duplicate position ·
daily loss circuit breaker ($2,000) · max open positions (3) · correlated
exposure (SPY and QQQ share one bucket: 2 positions, $1,500) · portfolio defined
risk ($3,000) · per-trade defined loss ($500, $750 at high confidence) ·
absolute per-trade ceiling ($1,000) · buying power (2× max loss).

Three properties make these gates hold rather than merely exist:

1. **Confidence cannot widen a limit.** It selects between two pre-configured
   caps, both bounded by an absolute ceiling. A parametrised test sweeps
   confidence from 0.0 to 1.0 and beyond and asserts the cap never exceeds it.
2. **Money is integer cents.** Binary floats cannot represent `0.05` exactly,
   and a cap off by a fraction of a cent is a cap that can be walked through.
   Conversion happens once, at the edge, with `Decimal` half-up rounding.
3. **Defined loss is arithmetic, not estimation.** For a vertical debit spread
   the maximum loss *is* the premium paid. `SpreadStructure` refuses to
   construct when the net debit is not strictly below the strike width, so an
   undefined-risk structure cannot exist as a value in this system.

Exits are hard rules the LLM has no input to: profit target, max spread loss,
max holding time, regime invalidation, end-of-day flatten, and an emergency
circuit-breaker exit that overrides everything.

Execution safety: the client order id is a pure function of the trade's
economics and is reserved in the journal *before* the network call, so a crash
mid-submit cannot produce a duplicate position. An ambiguous submission is never
retried — the agent reconciles by client order id first.

## Alpaca infrastructure

**Trading API** drives continuous autonomous execution. `AlpacaPaperBroker`
submits `order_class: "mleg"` multi-leg limit orders to
`paper-api.alpaca.markets`. Its constructor runs the paper endpoint guard and
every account read re-checks the `PA` paper prefix.

**Market data** supplies 1-minute bars and OPRA option chains with implied
volatility and greeks. Contract selection targets ≈0.55 delta long / ≈0.30 delta
short at 2–10 DTE, choosing the closest eligible pair with deterministic
tie-breaks. If no real contract passes the liquidity gates, the answer is
`NO_TRADE`; no option symbol is ever invented, and a test asserts every selected
symbol appears verbatim in the chain it came from.

**MCP** is used for read-only discovery, account verification and market
capture — `get_account_info`, `get_account_config`, `get_clock`,
`get_stock_snapshot`, `get_stock_bars`, `get_option_chain`, `get_orders`,
`get_all_positions`. Production code (`alphamesh/alpaca/mcp_adapter.py`) ingests
these payloads, and verbatim samples are committed under
`data/mcp_capture/raw/` so the ingest path is tested against real responses
rather than a guess at their shape. Order placement deliberately does **not**
go through MCP: the autonomous runtime must keep working with no MCP host
attached.

**CLI** is wrapped for operational inspection with an allowlist of four
read-only commands. When no CLI binary is installed, `alphamesh cli-info` says
so plainly. Nothing simulates CLI output.

**Paper enforcement** is three independent checks, all of which must pass:
`ALPACA_PAPER=true`; the trading host is a known Alpaca paper host (live hosts
rejected by name, **unrecognised hosts rejected too**); the account number
carries the `PA` prefix. Failure raises and blocks startup.

---

## Validation record

Stated precisely, because an unverified claim is worth less than an honest gap.

### Verified through the Alpaca MCP connector

| Item | Result |
|---|---|
| Account is paper | Yes — account number prefix `PA`, `get_account_info` |
| Account status | `ACTIVE`, not blocked, trading not suspended |
| Options trading level | 3 (spreads permitted) |
| Equity | $100,000.00 |
| Options buying power | $100,000.00 |
| Market clock | Retrieved; market closed at capture time |
| SPY / QQQ 1-minute bars | Retrieved, 180 bars each, SIP feed |
| SPY / QQQ snapshots | Retrieved |
| SPY call chain | Retrieved with quotes, IV and greeks |
| SPY put chain | Retrieved with quotes, IV and greeks |
| QQQ call chain | Retrieved with quotes, IV and greeks |
| Real candidate spreads built | Yes — three, from the captured chains (below) |

Real spreads constructed by production code from the captured Alpaca chains:

```
SPY BULL_CALL_SPREAD   LONG SPY260903C00769000 (δ 0.5391)
                       SHORT SPY260903C00774000 (δ 0.3037)
                       width $5.00  limit $2.38  max loss $238  max profit $262

SPY BEAR_PUT_SPREAD    LONG SPY260904P00771000 (δ -0.5408)
                       SHORT SPY260904P00764000 (δ -0.2954)
                       width $7.00  limit $2.61  max loss $261  max profit $439

QQQ BULL_CALL_SPREAD   LONG QQQ260903C00715000 (δ 0.5552)
                       SHORT QQQ260903C00723000 (δ 0.2983)
                       width $8.00  limit $3.86  max loss $386  max profit $414
```

### Not validated

| Item | Status |
|---|---|
| **Any order placed** | **No. Zero orders have been submitted to Alpaca.** |
| **P&L** | **None. No trade has been opened or closed, so there is no P&L to report.** |
| Alpaca REST from the Python runtime | Not exercised. The build container's network policy returns HTTP 403 at CONNECT for `paper-api.alpaca.markets` and `data.alpaca.markets`, so the REST client could not be run here. Its code path is unit-tested but not network-tested. |
| Alpaca CLI execution | No CLI binary is installed in this environment. The adapter reports unavailable rather than simulating output. |
| Live-market behaviour | The captured session is a quiet Friday afternoon. The agent correctly declined to trade it. |
| Railway deployment | `Dockerfile`, `railway.json` and `Procfile` are written and `alphamesh health` works locally, exiting non-zero when the paper guard fails. The image has **not** been built: no Docker daemon is available in this environment. No deploy has been performed. |

### What the captured session actually showed

Replaying the captured Alpaca session bar by bar through the full pipeline,
peak quantitative score reached **0.549** against a **0.55** gate. The agent
returned `NO_TRADE` on all 242 evaluated decisions, classifying the tape as
`RANGE_BOUND`.

That is the correct answer for that tape, and it is worth stating plainly: this
system's first demonstrated behaviour on real market data was declining to
trade. `NO_TRADE` is a first-class outcome, not a failure to act.

Lowering the gate to expose the entry path showed the liquidity gates doing
their job on real contracts — `STALE_QUOTES`, `WIDE_SPREAD` and
`ILLIQUID_CONTRACT` rejections, correctly refusing chain quotes timestamped
after the bar being evaluated.

The full entry→fill→manage→exit lifecycle is exercised end to end in
`tests/test_orchestrator.py` against the real captured option chain, including
duplicate protection, restart recovery, ambiguous-submission handling,
circuit-breaker exits and correlated-exposure gating.

### Quality gates

- **338 tests pass.** No network access, no LLM key, no Alpaca credentials
  required.
- **`ruff check` clean.**
- **`mypy` clean** across 49 source files with `disallow_untyped_defs`.

---

## Running it

```bash
alphamesh preflight   # safety + connectivity report; places no orders
alphamesh run         # the autonomous loop
streamlit run dashboard/app.py
```

Offline, against the committed real-Alpaca capture:

```bash
ALPHAMESH_DATA_SOURCE=mcp_capture alphamesh replay-session
```

---

*AlphaMesh trades Alpaca paper accounts only. This is not investment advice.*
