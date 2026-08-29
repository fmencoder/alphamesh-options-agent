# AlphaMesh

**Autonomous AI options-trading agent on Alpaca paper trading.**

AlphaMesh scans SPY and QQQ, scores the opportunity quantitatively, classifies
the market regime, convenes a three-role AI reasoning council, selects real
Alpaca option contracts deterministically, sizes the trade against
non-bypassable risk limits, submits a defined-risk multi-leg spread, manages the
position, exits it, and writes every step to an audit journal.

It runs headless. No human approves a trade, and no human can widen a risk limit
at runtime.

> **PAPER TRADING ONLY.** AlphaMesh refuses to start unless paper mode is
> positively proven. See [Paper-trading disclaimer](#paper-trading-disclaimer).

---

## What it does

One cycle, start to finish, with no human in the loop:

```
market data  →  features  →  quantitative opportunity gate
             →  regime classification
             →  AI reasoning council (bull / bear / judge)
             →  NO_TRADE | BULL_CALL_SPREAD | BEAR_PUT_SPREAD
             →  real Alpaca option chain
             →  deterministic contract selection
             →  defined maximum loss
             →  Risk Governor  (non-bypassable)
             →  Alpaca multi-leg order  (idempotent)
             →  fill monitoring  →  position management  →  exit
             →  P&L + complete decision audit trail
```

`NO_TRADE` is a first-class outcome at every stage, and it always carries a
machine-readable reason. Standing aside is a decision the system is designed to
make well, not a failure mode.

## Why options

Vertical debit spreads give the agent something a stock position cannot: a
**maximum loss that is known exactly before the order is sent**. For a bull call
spread bought at a net debit of $2.38 with $5.00 wide strikes, the loss is
capped at $238 per spread and the profit at $262 — arithmetic, not estimation.

That property is what makes a hard risk budget enforceable. The Risk Governor
divides a dollar cap by an exact per-contract loss and rounds down. There is no
stop-loss to gap through and no assumption about fill quality on the way out.

v1 trades **only** defined-risk verticals. No naked options. No undefined-risk
structure. `alphamesh/models/domain.py` enforces this in the type system:
`SpreadStructure` refuses to construct when the net debit is not strictly below
the strike width.

## Architecture

```
alphamesh/
  safety.py            Paper-mode guard. Fails closed.
  config.py            Environment settings + immutable YAML policy
  orchestrator.py      The autonomous lifecycle

  alpaca/
    client.py          Stack factory; runs the paper guard before anything else
    market_data.py     Alpaca REST bars  |  MCP-capture replay
    options.py         Alpaca REST option chain  |  MCP-capture replay
    execution.py       Alpaca paper broker  |  deterministic simulator
    mcp_adapter.py     Alpaca MCP response ingestion
    cli_adapter.py     Alpaca CLI operational path

  intelligence/
    features.py        Deterministic feature extraction from 1-minute bars
    scoring.py         Quantitative opportunity score and AI-invocation gate
    regime.py          Six-regime classifier
    reasoning.py       LLM provider abstraction (Anthropic / null / scripted)

  agents/
    market_scout.py    Scans and scores the universe (no LLM cost)
    regime_agent.py    Regime assessment
    bull_agent.py      Argues the bullish case
    bear_agent.py      Argues the bearish case
    judge_agent.py     Rules; constrained to three strategies
    strategy_agent.py  Runs the council behind the quant gate

  strategies/
    contracts.py       Deterministic vertical selection from a real chain
    bull_call.py       Bull call spread
    bear_put.py        Bear put spread

  risk/
    governor.py        THE authority. Nothing reaches the broker without it.
    sizing.py          Confidence → dollar cap → whole contracts, rounded down
    liquidity.py       Quote freshness, spread width, book depth, greeks
    portfolio.py       Aggregate and correlated exposure
    circuit_breaker.py Daily loss halt
    money.py           Integer-cent arithmetic

  execution/
    order_builder.py   Idempotent client order ids, Alpaca mleg payloads
    state_machine.py   Explicit, validated state transitions
    monitor.py         Broker read-back and position marking
    exits.py           Hard exit rules the LLM cannot override
    recovery.py        Restart reconciliation

  persistence/        SQLite audit journal with secret redaction
  analytics.py        Competition reporting + bounded adaptive weighting

dashboard/app.py      Streamlit judge-facing dashboard
```

## Autonomous decision lifecycle

| Stage | Owner | Can the AI influence it? |
|---|---|---|
| Market data retrieval | `MarketScout` | No |
| Feature computation | `intelligence/features.py` | No |
| Opportunity score + gate | `intelligence/scoring.py` | No |
| Regime classification | `intelligence/regime.py` | No |
| Bull / bear arguments | `agents/{bull,bear}_agent.py` | Yes — advisory text and a conviction |
| Strategy choice | `agents/judge_agent.py` | Yes — one of exactly three values |
| Contract selection | `strategies/contracts.py` | **No** |
| Position sizing | `risk/sizing.py` | Only via a confidence that picks between two pre-set caps |
| Risk approval | `risk/governor.py` | **No** |
| Order construction | `execution/order_builder.py` | **No** |
| Submission | `alpaca/execution.py` | **No** |
| Exits | `execution/exits.py` | **No** |

## AI reasoning council

Three roles, deliberately adversarial:

- **Bull agent** receives only structured market evidence and argues the
  strongest honest bullish case.
- **Bear agent** receives the *same* evidence and argues the bearish case.
- **Judge** receives both arguments plus the quantitative and regime evidence,
  and returns strict structured JSON.

The judge may return exactly one of `NO_TRADE`, `BULL_CALL_SPREAD`,
`BEAR_PUT_SPREAD`. Anything else is refused outright with reason code
`AI_UNSUPPORTED_STRATEGY` — never mapped onto a strategy we do allow.

The evidence packet is a **whitelist** (`agents/evidence.py`). The council never
sees account equity, buying power, open positions, risk limits or position
sizes, so it cannot reason about — let alone argue for — a larger allocation.
A test asserts this against the real field names of `AccountState` and
`RiskLimits`.

**Provider abstraction.** `intelligence/reasoning.py` defines a
`ReasoningProvider` protocol with three implementations: `AnthropicProvider`
(Anthropic Messages API), `NullProvider` (no key configured) and
`ScriptedProvider` (tests). Without `ANTHROPIC_API_KEY` the council falls back
to deterministic heuristics and the agent keeps trading — degraded, not stopped.
**No test requires a paid LLM call.**

## Quantitative opportunity gate

The AI is expensive and slow; the market is neither. `MarketScout` computes a
deterministic score every cycle from 1-minute bars:

1-minute / 5-minute / 15-minute returns, VWAP deviation, ATR, realised
volatility, volume acceleration, opening-range position, least-squares trend
strength weighted by fit quality, and distance from the window high and low.

Five weighted components produce a score in `[0, 1]`. **Below
`quant_score_threshold`, the council is never invoked** — a test asserts the
provider records zero calls. Every threshold lives in `config/strategies.yaml`.

## Risk Governor

Authoritative and non-bypassable. It loads `config/risk.yaml` once at
construction and never mutates it. Thirteen gates run on every trade:

| Gate | Default | Reason code on failure |
|---|---|---|
| Paper mode | required | `LIVE_TRADING_FORBIDDEN` |
| Account tradeable, options level ≥ 3 | required | `ACCOUNT_NOT_TRADEABLE`, `OPTIONS_LEVEL_INSUFFICIENT` |
| Strategy allowlist | 2 verticals | `UNSUPPORTED_STRATEGY` |
| Defined risk (debit < width) | required | `UNDEFINED_RISK` |
| Leg liquidity and quote freshness | 120 s, 25% spread | `STALE_QUOTES`, `WIDE_SPREAD`, `ILLIQUID_CONTRACT`, `NO_QUOTE`, `MISSING_GREEKS` |
| Duplicate order / position | required | `DUPLICATE_ORDER` |
| Daily loss circuit breaker | $2,000 | `DAILY_DRAWDOWN_LIMIT` |
| Max open positions | 3 | `MAX_OPEN_POSITIONS` |
| Correlated exposure (SPY≈QQQ) | 2 positions, $1,500 | `CORRELATED_EXPOSURE` |
| Portfolio defined risk | $3,000 | `MAX_PORTFOLIO_RISK` |
| Per-trade defined loss | $500 / $750 high-confidence | `MAX_POSITION_RISK` |
| Absolute per-trade ceiling | $1,000 | `MAX_POSITION_RISK` |
| Buying power | 2× max loss | `INSUFFICIENT_BUYING_POWER` |

**Confidence cannot widen a limit.** It selects between two pre-configured caps,
both bounded by `absolute_max_defined_loss`. A parametrised test walks
confidence from 0.0 to 1.0 (and outside it) and asserts the cap never exceeds
the ceiling.

Money is carried in **integer cents** throughout the risk boundary. Binary
floats cannot represent `0.05` exactly, and a cap off by a fraction of a cent is
a cap that can be walked through.

## Options construction

Bull call spread: buy ≈0.55-delta call, sell a higher-strike ≈0.30-delta call.
Bear put spread: buy ≈-0.55-delta put, sell a lower-strike ≈-0.30-delta put.
2–10 DTE. Both legs must share one expiration.

Selection is entirely mechanical — filter for liquidity, keep contracts whose
delta lands inside the configured band, pick the pair closest to target deltas,
break ties on fixed keys (nearest expiry, delta error, debit/width ratio,
strike). The same chain always produces the same spread; a test asserts it.

If no real contract satisfies the filters, the answer is `NO_TRADE`. **No option
symbol is ever invented** — a test asserts every selected symbol appears
verbatim in the Alpaca chain it came from.

Worked example, from the committed capture of a real Alpaca chain:

```
SPY BULL_CALL_SPREAD
  LONG  SPY260903C00769000  K=769  delta=0.5391  bid/ask 3.86/3.94
  SHORT SPY260903C00774000  K=774  delta=0.3037  bid/ask 1.50/1.60
  width $5.00   limit $2.38   max loss $238   max profit $262
```

## Alpaca integration

**Trading API** — continuous autonomous execution. `AlpacaPaperBroker` submits
`order_class: "mleg"` limit orders to `paper-api.alpaca.markets`. Its
constructor runs the paper endpoint guard, and every `account()` call re-checks
the `PA` account prefix, so a mid-session config change cannot move order flow
to a live account.

**Market data** — 1-minute bars and OPRA option chains with IV and greeks.

### MCP / CLI usage

**MCP** is used for read-only discovery, account verification and market
capture. `alphamesh/alpaca/mcp_adapter.py` ingests these tools:

| Tool | Purpose |
|---|---|
| `get_account_info` | Confirm the account is paper (`PA` prefix), read equity and options level |
| `get_account_config` | Verify trading is not suspended |
| `get_clock` | Session state before capture or a dry run |
| `get_stock_snapshot` | Reference last trade/quote per symbol |
| `get_stock_bars` | The 1-minute window the feature engine consumes |
| `get_option_chain` | Real OPRA contracts, quotes, IV and greeks |
| `get_orders`, `get_all_positions` | Operator read-back |

Every one is read-only. **Order placement deliberately does not go through
MCP**, because the autonomous runtime must keep working with no MCP host
attached. Run `alphamesh mcp-info` to print this.

The fixtures under `data/mcp_capture/` were produced from exactly these
responses. Verbatim samples are kept in `data/mcp_capture/raw/`, and
`tests/test_mcp_adapter.py` parses them with production code and cross-checks
the values against the committed CSVs — so the fixtures are provably tied back
to the MCP server they came from. `scripts/mcp_capture.py` runs the same path
for a fresh capture.

**CLI** — `alphamesh/alpaca/cli_adapter.py` wraps an installed Alpaca CLI for
operational inspection, with an allowlist of four read-only commands
(`account get`, `clock get`, `positions list`, `orders list`). If no CLI is
installed, `alphamesh cli-info` says so plainly; **nothing simulates CLI
output**.

## Execution safety

- **Idempotency.** The client order id is a pure function of the trade's
  economics: `alphamesh-{SYMBOL}-{BCS|BPS}-{signal_hash}`. Two runs that would
  place the same trade produce the same id, so Alpaca rejects the duplicate even
  if our journal were lost.
- **Reserve-before-send.** The id is written to the journal *before* the network
  call. A crash mid-submit leaves a `CONSTRUCTED` row that recovery reconciles,
  never a silent duplicate.
- **No blind retries.** A timeout raises `AmbiguousSubmissionError`. The agent
  does not resend; it reconciles by client order id first.
- **Explicit state machine.** `DISCOVERED → ANALYZED → AI_APPROVED →
  RISK_APPROVED → CONSTRUCTED → SUBMITTED → PARTIALLY_FILLED → FILLED →
  MONITORING → EXIT_REQUESTED → CLOSED`, plus `REJECTED` and `FAILED`. Illegal
  transitions raise. Every hop is persisted.
- **Restart recovery.** Every non-terminal order is re-read from the broker
  before any new order can be built.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Shows the PAPER MODE banner, equity against the $100,000 competition start,
total and daily P&L, win rate, open defined risk, the current regime, and the
latest decision broken out into bull thesis, bear thesis, judge rationale, all
three scores and the reason codes — plus active positions, closed trades and
**recent Risk Governor rejections with the code that refused each one**. It is
built so a judge can see *why* a trade happened or did not.

When there are no closed trades it says so and reports no P&L, rather than
estimating one.

## Competition metrics

`alphamesh report` slices closed trades by strategy, regime, symbol, confidence
bucket and exit reason, reporting trade count, win rate, total and average P&L,
return on defined risk, average holding time, largest win and largest loss.

`analytics.adaptive_weights` lays the groundwork for adaptive strategy
weighting: a pairing needs ≥5 closed trades to earn any adjustment, and the
adjustment is hard-clamped to ±25%. It is scoring guidance only — **the Risk
Governor never reads it**, and a test asserts the weight keys share no name with
any risk field.

## Local setup

```bash
git clone https://github.com/fmencoder/alphamesh-options-agent
cd alphamesh-options-agent
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,dashboard]"

cp .env.example .env      # then add your Alpaca PAPER keys
```

```bash
alphamesh preflight       # safety + connectivity report, places no orders
alphamesh once            # one full lifecycle cycle
alphamesh run             # the autonomous loop
alphamesh report          # competition analytics
alphamesh health          # JSON health record for a probe
alphamesh mcp-info        # exactly which MCP tools are used
alphamesh cli-info        # whether the Alpaca CLI path is available
```

Offline, against the committed real-Alpaca capture:

```bash
ALPHAMESH_DATA_SOURCE=mcp_capture alphamesh replay
ALPHAMESH_DATA_SOURCE=mcp_capture alphamesh replay-session
```

`replay-session` walks the captured session bar by bar. Fills come from the
in-process simulator, so its P&L is **simulated** and is labelled as such
on every run.

Checks:

```bash
pytest        # 338 tests, no network, no LLM key
ruff check .
mypy
```

## Railway deployment

Two services from one image:

```
agent      python -m alphamesh.main run
dashboard  streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
```

`Dockerfile`, `railway.json` and `Procfile` are included (the image has not yet
been built — see the validation record). Mount a volume at
`/app/data` so the journal survives redeploys. Set `ALPACA_PAPER`,
`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `ANTHROPIC_API_KEY` and
`ALPHAMESH_DRY_RUN=false` as Railway variables. **No secret is baked into the
image.** The container health check runs `alphamesh health`, which exits
non-zero when the paper guard does not pass — an unsafe container never reports
healthy.

The runtime is a plain Python process. It does not depend on Claude Code, an
MCP host, or any interactive session remaining open.

## Security

- No API key, token or account credential is committed. `.env` is gitignored;
  `.env.example` carries placeholders only.
- The journal runs every payload through `redact()`, which blanks any key whose
  name matches a credential marker (`api_key`, `secret`, `token`, `password`,
  `authorization`, `account_number`, …) recursively. Tests assert that a live
  key and a real account number never reach the database.
- `AccountState.account_number` is excluded from `repr`.
- `Settings.redacted()` renders `<set>` / `<unset>`, never a partial value.
- The Alpaca MCP security envelope is stripped as metadata and its text is never
  treated as instructions.

## Paper-trading disclaimer

AlphaMesh trades **Alpaca paper accounts only** and is built to make live
trading impossible rather than merely discouraged:

- `ALPACA_PAPER` must be `true`.
- The trading endpoint must be a known Alpaca **paper** host. Live hosts are
  rejected by name, and **unrecognised hosts are rejected too** — an unknown
  host is not proof of paper trading.
- The account number must carry the `PA` paper prefix, re-checked on every
  account read.

Any failure raises `LiveTradingForbiddenError` and blocks startup. This is not
investment advice, and nothing here should be pointed at real money.

## Hackathon requirements mapping

| Requirement | Where |
|---|---|
| **Autonomous AI trading agent** | `orchestrator.py` runs the entire lifecycle headless via `alphamesh run`; no human approves a trade |
| **Alpaca Trading API** | `alpaca/execution.py` — multi-leg (`mleg`) paper option orders, order read-back, positions |
| **Alpaca MCP server** | `alpaca/mcp_adapter.py` + `data/mcp_capture/` — eight read-only tools ingested by production code and tested against verbatim responses |
| **Alpaca CLI** | `alpaca/cli_adapter.py` — allowlisted read-only operational commands; honest when absent |
| **Options trading** | Defined-risk vertical debit spreads only, built from real OPRA chains with greeks |
| **Paper trading** | `safety.py` fails closed on three independent checks |
| **$100,000 competition account** | Dashboard tracks equity against the start; the Risk Governor budgets against it |
| **P&L as a judging criterion** | `analytics.py` + `alphamesh report`; **no P&L is reported when no trade has closed** |
| **Creativity** | Adversarial bull/bear/judge council behind a quantitative cost gate, with confidence that can pick between caps but never widen one |
| **Presentation** | `dashboard/app.py`, `docs/ARCHITECTURE.md`, `docs/COMPETITION_WRITEUP.md` |
| **MIT licence** | `LICENSE`, unchanged |

---

## Validation status

Read [`docs/COMPETITION_WRITEUP.md`](docs/COMPETITION_WRITEUP.md) for the
precise record of what has and has not been validated against Alpaca. In short:
account, market data and option chains were verified through the Alpaca MCP
connector; **no order has been placed**; and no P&L exists yet.

## Licence

MIT. See [`LICENSE`](LICENSE).
