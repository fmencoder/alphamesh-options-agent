# AlphaMesh architecture

## Design principle

One sentence governs the whole system:

> **The AI advises. Deterministic code decides anything that touches money.**

Everything below follows from that. The reasoning council can argue, weigh and
choose a direction. It cannot pick a contract, a quantity, a price, or a dollar
of risk.

---

## Layers

```
                        ┌──────────────────────────┐
                        │   safety.py (guard)      │  fails closed at startup
                        └────────────┬─────────────┘
                                     │
   ┌─────────────────────────────────▼─────────────────────────────────┐
   │                          orchestrator.py                          │
   └──┬────────────┬────────────┬────────────┬────────────┬────────────┘
      │            │            │            │            │
  ┌───▼───┐   ┌────▼─────┐  ┌───▼────┐  ┌────▼─────┐ ┌────▼──────┐
  │ scout │   │  regime  │  │council │  │ strategy │ │   risk    │
  │       │   │          │  │        │  │ contracts│ │ governor  │
  └───┬───┘   └────┬─────┘  └───┬────┘  └────┬─────┘ └────┬──────┘
      │            │            │            │            │
      └────────────┴────────────┴────────────┴────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │   execution + journal    │
                        └──────────────────────────┘
```

### 1. Safety (`safety.py`)

Runs before any client is constructed. Three independent checks, all of which
must pass:

1. `ALPACA_PAPER` is true.
2. The trading base URL resolves to a host in `PAPER_HOSTS`. Hosts in
   `LIVE_HOSTS` are rejected by name; **anything unrecognised is also rejected**,
   because an unknown host is not evidence of paper trading.
3. The account number begins with `PA`.

Check 3 runs on *every* account read, not only at startup, so a mid-session
configuration change cannot quietly move order flow.

Failure raises `LiveTradingForbiddenError`, which propagates to the CLI as exit
code 2 with a blocked-startup message.

### 2. Configuration (`config.py`)

Two sources, deliberately separated:

- **Environment** → `Settings`: credentials, endpoints, paths, runtime toggles.
  Secrets are `repr=False` and `redacted()` renders `<set>` / `<unset>` rather
  than a truncated value.
- **YAML** → `RiskLimits`, `StrategyConfig`, `UniverseConfig`: frozen Pydantic
  models loaded once at startup.

`RiskLimits` is the important one. It is immutable, and nothing in the AI path
is given a handle to it.

### 3. Market data (`alpaca/market_data.py`, `alpaca/options.py`)

Two implementations behind one protocol:

- `AlpacaRestMarketData` / `AlpacaRestOptionChain` — production, via
  `alpaca-py`. Imports are deferred so the SDK is not needed to run tests.
- `CaptureMarketData` / `CaptureOptionChain` — replay of real Alpaca payloads
  captured through the MCP server. The data is real; only the transport differs.

The capture path exists because some deployment environments block outbound
access to `data.alpaca.markets` — including the container this project was
developed in. It is also what makes the whole test suite hermetic.

### 4. Intelligence (`intelligence/`)

Pure functions over an immutable bar window. No I/O, no randomness, no clock
reads — the same bars always produce the same features, which is what makes a
journalled decision reconstructable months later.

**`features.py`** computes: 1m/5m/15m returns, rolling VWAP and deviation,
Wilder ATR, annualised realised volatility, volume acceleration, opening-range
position, trend strength, and distance from window extremes.

`trend_strength` is worth a note: it is a least-squares slope through recent
closes, normalised by mean price, then **multiplied by the fit's R²**. A noisy
drift therefore scores lower than a clean move of the same magnitude, which is
the behaviour a directional strategy wants.

**`scoring.py`** maps features to a `[0, 1]` opportunity score through five
weighted components (momentum, trend, VWAP, participation, range position).
Unbounded inputs pass through a smooth monotone squash, so no single feature can
dominate. Direction requires **agreement**: a mixed sign vote returns `NEUTRAL`,
which downstream is a reason to stand aside rather than to guess.

**`regime.py`** classifies into six regimes with explicit thresholds, checked
hardest-first: extreme realised volatility → `UNSTABLE`; volatility ratio above
baseline → `VOLATILITY_EXPANSION`; strong signed trend → `BULLISH_TREND` /
`BEARISH_TREND`; weak trend in a tight range → `RANGE_BOUND`; anything else →
`UNKNOWN`. `UNKNOWN` and `UNSTABLE` both set `favors_no_trade`.

### 5. The AI reasoning council (`agents/`)

**Evidence packet** (`evidence.py`) is a whitelist of eleven market features
plus the quant score and regime assessment. It is built additively, so a new
field on `AccountState` cannot leak into it by accident.

**Bull and bear agents** receive identical evidence and argue opposite sides.
Each returns a thesis, key points and a conviction in `[0, 1]`. Neither can
fail: malformed output, a timeout or a missing key all fall back to a
deterministic heuristic.

The heuristic conviction is built as
`quant_score × directional_support × regime_factor × gain`, clamped to `[0, 1]`.
Grounding it in the already-normalised quant score rather than in raw returns
matters: index-ETF intraday returns are far too small to use directly, and an
earlier formulation that did so produced convictions that never reached the
judge's confidence floor — meaning the agent would never have traded without an
LLM key.

**Judge** (`judge_agent.py`) is the only AI role whose output steers the
pipeline, and its authority is tiny. Every field is re-validated:

- Strategy must be one of three. Anything else → `NO_TRADE` +
  `AI_UNSUPPORTED_STRATEGY`. It is **refused, never coerced** onto a strategy
  we do allow.
- A directional verdict in an `UNSTABLE` or `UNKNOWN` regime is overridden here,
  not left for downstream code to notice.
- Confidence is clamped to `[0, 1]`.

**Strategy agent** (`strategy_agent.py`) orders the authority explicitly:

1. Quantitative gate — below threshold, the council is **never invoked**.
2. Regime veto — `UNKNOWN` / `UNSTABLE` force `NO_TRADE` before any model cost.
3. Judge, constrained to three strategies.
4. Configured minimum judge confidence.

Steps 1 and 2 are cost controls as much as risk controls. A test asserts the
provider records zero calls when the gate is closed.

### 6. Contract selection (`strategies/contracts.py`)

Mechanical, in this order:

1. Filter to the right option type.
2. Apply per-contract liquidity gates (`risk/liquidity.py`).
3. Group by expiration, keep only the configured DTE window.
4. Within each expiration, keep contracts whose delta lands in the configured
   band for the long and short legs.
5. For every eligible pair: check strike width, compute the limit price, reject
   if the debit meets or exceeds the width (no defined payoff) or if the
   debit/width ratio is worse than configured.
6. Sort by `(expiration, delta error, debit/width ratio, long strike, short
   strike)` and take the first.

Step 6's tie-breaks are all deterministic, so the same chain always yields the
same spread.

**Pricing.** `mid` is the fair value of the pair; `natural` is what it costs to
cross both spreads immediately. The limit sits a configurable fraction of the
way from mid toward natural, so the agent pays for fills without lifting the
whole offer.

### 7. Risk Governor (`risk/governor.py`)

The authority. Thirteen gates, each appending to `checks_run` so the journal
records what actually ran, and each failure appending a `ReasonCode` rather than
short-circuiting — the journal shows *all* the ways a trade was unsuitable.

Two details worth calling out:

**Headroom-aware sizing.** Rather than rejecting a trade that would breach an
aggregate cap, the governor computes remaining portfolio and correlated-group
headroom and passes the smaller to the sizer, which shrinks the trade to fit.
Only when nothing fits does it reject — and it then names the *binding*
constraint (`MAX_PORTFOLIO_RISK` or `CORRELATED_EXPOSURE`) rather than a generic
"could not size".

**Re-assertion after sizing.** The per-trade cap, absolute ceiling, portfolio
cap and group cap are all checked again against the final computed loss, not
just against the sizer's inputs.

**Integer cents.** `risk/money.py` converts once at the edge via
`Decimal.quantize(ROUND_HALF_UP)` and everything downstream is integer
arithmetic. `to_cents(1.005) == 101` and `to_cents(0.05) == 5` exactly — neither
is true of naive float rounding.

### 8. Execution (`execution/`, `alpaca/execution.py`)

**Idempotency** rests on two independent mechanisms:

1. The client order id is a SHA-256 digest of `(decision_id, symbol, strategy,
   both leg symbols, quantity, limit price)`. Alpaca rejects a duplicate id, so
   the same trade cannot be placed twice even if our journal were lost.
2. The id is written to the journal **before** the network call
   (`Journal.reserve_order`, which returns `False` on a primary-key collision).
   A crash between reservation and submission leaves a `CONSTRUCTED` row that
   recovery reconciles.

**Ambiguous submission** — a timeout or reset — raises
`AmbiguousSubmissionError`. The agent never resends. The reservation stays; the
next `startup()` reads the order back from the broker by client order id and
either promotes it (the order did land) or retires it as `FAILED` (it did not).

**State machine** (`state_machine.py`) is an explicit adjacency map. Illegal
transitions raise rather than being silently accepted, so an orchestrator bug
surfaces loudly instead of producing an order in an impossible state.

**Exits** (`exits.py`) run hardest-first: circuit breaker, expiration,
end-of-day flatten, max holding time, then mark-based profit target and stop,
then regime invalidation. A position that cannot be marked does **not** trigger
a price exit — a missing mark is not a zero value.

**Exit execution** is a broker round trip, not a journal write. `exits.py`
decides *whether* to close; the orchestrator then builds the mirror-image order
and sends it through `Broker.close_spread`, which flips both legs to
`SELL_TO_CLOSE` / `BUY_TO_CLOSE`. The lifecycle is:

```
MONITORING -> EXIT_REQUESTED -> closing order submitted -> broker confirms fill
           -> position CLOSED, realised P&L booked from that fill
```

Only the last hop retires a position. Four properties this holds:

- **A submitted close is not a close.** An order on the wire leaves the position
  in `EXIT_REQUESTED`; it is still real exposure and still marked and managed.
- **Realised P&L only ever comes from fills.** Entry fill against exit fill. A
  *mark* is never realised money — booking one was the defect that let nine live
  spreads look closed while the account still held them.
- **A partial fill never closes.** The unfilled balance is exposure, so the
  position stays open and the order is left alone; the sweep below skips it.
- **Closing is never gated on risk headroom.** The exit path does not consult
  the governor at all, so no cap can refuse a risk-reducing close. An over-cap
  portfolio blocks new exposure and lets exposure fall through exits.

Exits are only ever sent into an open session. Recovery and adoption run
overnight, but discovering that a position should be closed is not a licence to
queue an order against a dead book: the next open re-prices it against live
quotes. An unfilled close is retired after `ALPHAMESH_EXIT_ORDER_TTL_SECONDS`
(default 120), confirmed dead with the broker, and re-quoted next cycle — an
exit that never fills is the same failure as never sending one. The entry TTL
sweep and the exit sweep are kept apart by the `kind` column on `orders`;
cancelling a close because it looked like a stale entry would leave the position
open with nothing managing it.

**Adoption** (`adoption.py`) closes the gap in the other direction. A spread the
broker holds and the journal has forgotten — after a restart, a lost write, or
a bad close — is unmanaged risk *and* permanently blocks its underlying at the
broker-truth entry guard. Every cycle, before any risk number is computed, raw
option legs are grouped by (underlying, expiration, type) and resolved into
verticals: exactly one long against one short, equal size, strikes forming a
supported debit spread. The entry debit comes from the originating multi-leg
order where it can be found, otherwise from the per-leg cost basis; a spread
whose basis cannot be established is not adopted.

Ambiguity fails closed. A leg set that is not unmistakably one of the two
supported verticals — an odd leg, mismatched sizes, a credit structure — is
reported, never guessed at: a wrong pairing would produce a "closing" order that
flattens one leg and leaves the other naked. While any such position exists,
`PortfolioState.exposure_fully_accounted` is false and the governor refuses new
exposure, because the aggregate caps would be measured against an understated
portfolio. Exits are unaffected.

Adoption never submits anything, so it is safe outside market hours.

### 9. Persistence (`persistence/`)

Seven tables linked by `decision_id`: `events`, `decisions`, `risk_decisions`,
`orders`, `state_transitions`, `positions`, `outcomes`. Together they
reconstruct: what the agent observed, what it computed, the regime, the bull and
bear arguments, the judge verdict, the contracts chosen, the governor's verdict
and every check it ran, the order, the fills, the exit and the P&L.

Schema v2 adds `orders.kind` / `orders.position_id` (which separates a
closing order from an entry), `positions.origin` / `positions.entry_basis`
(whether a position was opened by the agent or adopted from the broker, and
where its cost basis came from), and `outcomes.reconciliation_note`.

A journal written by an older build already exists on the production volume,
so these are applied by `ALTER TABLE` at open, additively — no row is dropped
or rewritten, and indexes over the new columns are created only after the
columns are, since an index on a column that does not exist yet would abort
startup with the real volume attached.

`reconciliation_note` is how a phantom close is preserved rather than erased.
When adoption finds a live broker spread whose journal position reads CLOSED
with no filled closing order behind it, the outcome row is annotated
`PHANTOM_NO_BROKER_EXIT_ORDER` and a `phantom_close_reconciled` event is
written. The row stays exactly as it was — the competition record has to show
what actually happened — but annotated outcomes are excluded from every
realised-P&L total, because that money was never earned. A close backed by a
real filled exit order is never annotated, even if the same two contracts are
later traded again.

`redact()` recurses through every payload and blanks any key whose *name*
matches a credential marker by substring — so `apca_api_secret_key` is caught by
the `secret` marker without an exact-name allowlist that could drift.

---

## Failure modes and how they are handled

| Failure | Behaviour |
|---|---|
| No `ANTHROPIC_API_KEY` | Heuristic council; agent keeps trading |
| LLM times out / 500s | `LLMUnavailableError` → heuristic fallback + `AI_UNAVAILABLE` |
| LLM returns prose, not JSON | `MalformedAIOutputError` → heuristic fallback |
| LLM picks an unsupported strategy | `NO_TRADE` + `AI_UNSUPPORTED_STRATEGY`; refused, not coerced |
| Market data outage for one symbol | Logged; the scan continues with the others |
| Market data outage entirely | Cycle reports zero symbols; loop continues |
| Broker unreachable | Cycle records the error and returns; loop continues |
| Submission times out | `AmbiguousSubmissionError`; no resend; reconciled next startup |
| Process killed mid-submit | Reservation row found on restart and reconciled |
| Chain has no eligible contract | `NO_TRADE` with the specific liquidity codes |
| Quotes stale or from the future | `STALE_QUOTES`; both directions are refused |
| Daily loss limit breached | New entries refused; open positions flattened |

---

## What deliberately is *not* here

No Kubernetes, Kafka, Redis, Postgres or microservices. The system is one Python
process, one SQLite file and one Streamlit page. Every one of those would have
been architecture in place of a working vertical slice.

Adaptive strategy weighting is present but bounded (`analytics.adaptive_weights`,
clamped to ±25% after ≥5 closed trades) and is scoring guidance only. Wiring it
into the live score is left for a next pass, documented rather than half-built.
