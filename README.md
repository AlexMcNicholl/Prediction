# Prediction-market screener

A **read-only** analysis system for prediction-market contracts tradeable on
[Wealthsimple Predict](https://www.wealthsimple.com/), which is a retail wrapper over the
[Kalshi](https://kalshi.com) exchange. It pulls Kalshi's public market data, stores a
time series of it, computes analytical signals, and surfaces a ranked shortlist of
contracts worth your **manual** analysis — plus a full queryable dataset for open-ended
exploration.

## What this is not

- It **never places, routes, or automates a trade.** The API client physically refuses
  any non-market-data path, only issues `GET`, and sends no credentials. This is
  asserted by tests, not just by policy.
- It **never touches Wealthsimple** — no login, no scraping, no API. There is no
  official Wealthsimple API, and scraping one would be fragile and against their terms.
- It **never claims a contract is a good trade.** Every output is labelled a *candidate
  for manual analysis*.

## The framing that shapes the whole design

In a prediction market, **"high profit potential" and "high risk" are the same
variable**. A contract at $0.20 pays 5× precisely because the market thinks it probably
won't happen. There is no screen for "high profit, low risk" — that screen would just
return mispriced contracts, and if they were reliably identifiable they wouldn't be
mispriced.

The only real edge is **disagreeing with the market price for a defensible reason**.
So:

- `edge = model_prob − implied_price` is computed **only** where an independent
  fair-value model covers the contract. Everywhere else, edge is `NULL` and the
  contract is explicitly marked *no model*.
- The composite score gives zero weight to edge you cannot capture — if the best
  available side is negative-EV after fees, the disagreement isn't actionable.
- Contracts with no model are scored on structure alone and rank structurally lower,
  because you have no stated reason to disagree with their price.
- Ranking deliberately uses the **EV-weighted** annualised return, not
  "annualised if it wins". The latter would put every longshot on top for being
  unlikely — which is the exact error the system exists to avoid.

## Quick start

```bash
git clone <this repo> && cd Prediction
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Confirm the live Kalshi API still matches what this code assumes.
python -m screener.cli verify-api

# 2. Fetch data, compute signals, build the dashboard.
python -m screener.cli run --no-notify

# 3. Open the result.
open out/dashboard.html
```

No API key is required. Kalshi's market-data endpoints are public; only trading needs
signed auth, which this system does not use.

## Commands

| Command | What it does |
|---|---|
| `verify-api` | Probes the live API and reports any drift in base URL, endpoints, field names, pagination or the cents assumption. **Run this first.** |
| `ingest` | Fetches every open market (all pages), stores metadata + a price snapshot, classifies Predict availability, enriches the shortlist with orderbooks/candles/trades. |
| `signals` | Recomputes fair-value estimates and all signals over stored data. |
| `report` | Regenerates the dashboard and digest. `--notify` also sends it. |
| `run` | `ingest` → `signals` → `report` → notify. This is what the cron job runs. |
| `export --format csv\|parquet` | Dumps every table plus a joined signal view to `exports/`. |
| `info` | Row counts and last-run summary. |

Useful flags: `--config path.yaml`, `--log-level DEBUG`, `--max-pages N` (cap ingestion
while testing).

## How it works

```
ingest/     Kalshi client (cursor pagination, retry/backoff, rate limit) + normalisation
predict/    Predict availability: config allowlist + Canadian 30-day term-to-maturity rule
store/      SQLite: markets, snapshots (append-only), orderbooks, candles, trades, signals
models/     Pluggable fair-value estimators -> {prob, source, asof, confidence}
signals/    Fees, EV, Kelly, spread, liquidity, longshot, momentum, staleness, composite score
analysis/   pandas query layer over the FULL dataset + CSV/Parquet export
report/     Self-contained HTML dashboard + text digest (email / Telegram)
```

### Data model

`markets` holds slow-moving metadata and the **full resolution rules**. `snapshots` is
append-only — one row per market per run — which is what makes volatility, momentum and
staleness computable. Every payload keeps a `raw_json` copy so history can be
reprocessed when models change. Upserts never overwrite a known value with `NULL`.

### Price conventions

Kalshi prices are **integer cents**: a `yes_bid` of 62 means a 62% implied probability.
Two conventions matter when reading output:

- `implied_prob` uses the **bid/ask midpoint** — the fairest single number for "what the
  market thinks".
- `entry_price`, and everything derived from it (EV, Kelly, annualised returns), uses
  the **ask you would actually pay**, plus fees. Screening on the mid would overstate
  every edge by half the spread.

Contracts settle in **USD**.

## Configuration

Everything lives in `config.yaml`. The knobs you'll actually touch:

| Key | Purpose |
|---|---|
| `predict.series_prefix_allowlist` | Which Kalshi series exist on Predict (hand-curated — see below) |
| `predict.term_to_maturity_days` | The Canadian 30-day rule |
| `signals.*` | Thresholds for edge, spread, liquidity, longshot band, staleness, momentum |
| `signals.score_weights` | Composite score weights — every component is stored separately |
| `fees.*` | Fee assumptions, folded into all EV maths |
| `sizing.bankroll`, `sizing.kelly_fraction`, `sizing.max_stake_fraction` | Position sizing |
| `models.*` | Which fair-value models run, and their inputs |
| `notifications.*` | Email / Telegram delivery |

Secrets never go in `config.yaml`. Use `.env` locally (copy `.env.example`) and GitHub
Secrets in CI. **No secret is required** for the core pipeline.

### Curating the Predict availability list

There is no official Kalshi → Predict mapping, so the allowlist is hand-curated. The
shipped list covers economics, financial markets and climate/weather — the three
categories Wealthsimple is authorised to offer in Canada — but it is a starting point,
not gospel.

To tune it:

1. Run the pipeline, then open the notebook and run the **"Why contracts were
   excluded"** cells. They list every series excluded purely for not being on the
   allowlist, ranked by contract count.
2. Browse the Predict app. If you can trade a series that appears in that list, add its
   ticker prefix to `predict.series_prefix_allowlist`.
3. If something on the allowlist is *not* offered in Predict, add it to
   `predict.series_denylist`.

Matching is: exact series ticker, **or** any allowlisted prefix, **or** an allowlisted
category. The denylist always wins. Every excluded contract stores the reason, so
nothing disappears silently.

### Fees

The default model is Kalshi's published taker formula:

```
fee = ceil(coefficient × contracts × P × (1 − P))    rounded up to the next cent
```

with `coefficient = 0.07` and makers at ~25% of that. The curve peaks at a 50¢ contract
and falls toward zero at the extremes. Because manual app execution is effectively
always taker, `fees.assume_taker` defaults to `true`.

**Fee schedules change.** Check
[Kalshi's fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf) and update
`fees.taker_coefficient` (and `per_contract_cap_dollars`, if a cap applies to your
series) before trusting the EV numbers.

## Adding a fair-value model

Each estimator returns `{prob, source, asof, confidence}` so every edge is auditable.

1. Subclass `FairValueEstimator` in `screener/models/`:

```python
from .base import Estimate, FairValueEstimator, MarketContext

class MyEstimator(FairValueEstimator):
    name = "my-model"
    priority = 50            # higher wins when several estimators match

    def supports(self, market) -> bool:
        return self.series_of(market).startswith("KXTHING")

    def estimate(self, market, context) -> Estimate | None:
        # Return None rather than guessing when you cannot produce a number.
        prob, description = self.threshold_probability(market, mu=..., sigma=...)
        if prob is None:
            return None
        return Estimate(
            model=self.name, prob=prob, source="where this came from",
            asof="2026-09-05", confidence=0.6, notes=description,
        )
```

2. Register it in `screener/models/registry.py` under `BUILTIN_ESTIMATORS`.
3. Add it to `models.enabled` in `config.yaml`, with its own config section.

`threshold_probability` handles the greater / less / between strike shapes for you,
given a mean and standard deviation.

**Shipped models**

| Model | Status | Source |
|---|---|---|
| `cpi` | Working | Cleveland Fed inflation nowcast, or `manual_override` (the reliable path — the Fed publishes HTML, not a stable API) |
| `weather` | Working | US National Weather Service (free, no key) blended with climatology |
| `equity` | Working | Lognormal from spot + implied vol. **You supply spot/IV** via `manual_quotes`; no paid feed is wired in |
| `rates` | Stub | Manual probabilities only. Proper OIS / rate-futures inference needs a data source you'd have to choose — the extension point is `RatesEstimator._implied_from_market_data` |
| `manual` | Working | Per-ticker overrides in config; beats every automated model |
| `no-model` | Always | Explicit "no view" so a missing model is visible, not silent |

## Scheduling (GitHub Actions)

`.github/workflows/screener.yml` runs the pipeline three times daily, commits the
SQLite DB and dashboard back to the repo, uploads them as artifacts, and sends the
digest. Change the cadence by editing the `cron:` line — GitHub only reads the schedule
from the workflow file.

Optional secrets (`Settings → Secrets and variables → Actions`): `SMTP_USERNAME`,
`SMTP_PASSWORD`, `TELEGRAM_BOT_TOKEN`.

To publish the dashboard to GitHub Pages for a stable phone URL, set the repository
variable `PUBLISH_PAGES=true` and enable Pages with source "GitHub Actions".

`.github/workflows/tests.yml` runs the suite on every push. The tests are fully
offline — they run against fixtures and never contact Kalshi.

## Exploration

`analysis.ipynb` covers: dataset health, allowlist tuning, edge distribution,
spread-vs-liquidity, price history for any ticker, favorite–longshot **calibration**
(predicted vs realised once contracts settle), and a settled-contracts signal check.

```python
from screener.analysis.queries import Analysis

a = Analysis("data/screener.db")
a.markets(category="Economics", max_days_to_close=30)
a.signals(min_abs_edge=0.05, max_spread_cents=3)
a.price_history("KXCPI-26SEP-T2.9")
a.calibration(bins=10)
a.sql("SELECT ...")          # opened read-only; exploration cannot mutate data
```

The calibration and backtest views need **settled** contracts, so they stay empty until
the cron has been running for a while. That's expected.

Note on the backtest: it compares realised outcomes against fired signals. It is *not*
a strategy backtest — it ignores execution, slippage, and the fact that you trade
manually, hours later, at a different price.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers cursor pagination (including the repeated-cursor and empty-page traps), the fee
formula against exact rational arithmetic, EV and Kelly closed forms, the Predict
allowlist and 30-day boundary, storage idempotency, model dispatch, the full
ingest→dashboard pipeline, and structural read-only guarantees (no order-entry calls,
no non-`GET` verbs, no Wealthsimple host in any string literal).

## Known limitations

- **The Predict allowlist is a guess** until you check it against the app. It is the
  single biggest source of error in what gets shown.
- **`models.cpi` scrapes an HTML page.** The parser only accepts values in a plausible
  inflation range, so a layout change yields *no estimate* rather than a wrong one —
  but `manual_override` is the dependable path.
- **`models.equity` uses inputs you supply.** A stale spot price makes the estimate
  worse than useless. The notes field on every estimate says so.
- **`models.rates` is a stub.** Adding a real OIS/rate-futures source is a decision
  about a data provider, which is yours to make.
- **Model sigmas are assumptions, not measurements.** `models.cpi.sigma` and
  `models.weather.forecast_sigma_f` should be tuned against realised forecast errors
  once you have settled history.
- **Annualised returns are capped for display** (`signals.max_annualized_display`);
  a contract closing in hours produces an arithmetically true but meaningless figure.

## Licence / disclaimer

Personal research tooling. Nothing here is financial advice. Prediction-market
contracts can and do expire worthless. You are responsible for every trade you place
manually.
