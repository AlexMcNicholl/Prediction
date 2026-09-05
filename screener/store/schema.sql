-- Prediction-market screener schema.
--
-- Design rules:
--   * `markets` holds slow-moving metadata + resolution rules (upserted).
--   * `snapshots` is append-only: one row per market per poll. This is what
--     makes volatility, momentum and staleness computable. Never delete.
--   * every ingested payload keeps a `raw_json` copy so models can be changed
--     and the history reprocessed without re-fetching.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL DEFAULT 'running',  -- running|ok|partial|failed
    command           TEXT,
    markets_seen      INTEGER DEFAULT 0,
    markets_tradeable INTEGER DEFAULT 0,
    api_requests      INTEGER DEFAULT 0,
    error_count       INTEGER DEFAULT 0,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS series (
    series_ticker TEXT PRIMARY KEY,
    title         TEXT,
    category      TEXT,
    frequency     TEXT,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    raw_json      TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_ticker  TEXT PRIMARY KEY,
    series_ticker TEXT,
    title         TEXT,
    sub_title     TEXT,
    category      TEXT,
    mutually_exclusive INTEGER,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    raw_json      TEXT
);

CREATE TABLE IF NOT EXISTS markets (
    ticker            TEXT PRIMARY KEY,
    event_ticker      TEXT,
    series_ticker     TEXT,
    market_type       TEXT,
    title             TEXT,
    subtitle          TEXT,
    yes_sub_title     TEXT,
    no_sub_title      TEXT,
    category          TEXT,
    status            TEXT,
    open_time         TEXT,
    close_time        TEXT,
    expected_expiration_time TEXT,
    expiration_time   TEXT,
    latest_expiration_time   TEXT,
    -- Resolution detail. Surfacing these is a hard requirement: misreading
    -- settlement rules is the most common way retail loses money.
    rules_primary     TEXT,
    rules_secondary   TEXT,
    settlement_source TEXT,
    settlement_timer_seconds INTEGER,
    can_close_early   INTEGER,
    strike_type       TEXT,
    floor_strike      REAL,
    cap_strike        REAL,
    notional_value    INTEGER,
    tick_size         INTEGER,
    -- Outcome, populated once the market settles (drives backtests).
    result            TEXT,
    settled_at        TEXT,
    first_seen        TEXT NOT NULL,
    last_seen         TEXT NOT NULL,
    raw_json          TEXT
);

CREATE INDEX IF NOT EXISTS idx_markets_series ON markets(series_ticker);
CREATE INDEX IF NOT EXISTS idx_markets_close  ON markets(close_time);
CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT NOT NULL,
    ts            TEXT NOT NULL,          -- UTC ISO-8601, the poll time
    run_id        INTEGER,
    -- All prices are integer CENTS as returned by Kalshi (62 => 62% implied).
    yes_bid       INTEGER,
    yes_ask       INTEGER,
    no_bid        INTEGER,
    no_ask        INTEGER,
    last_price    INTEGER,
    previous_price INTEGER,
    spread        INTEGER,                -- yes_ask - yes_bid, cents
    mid_price     REAL,                   -- cents, (yes_bid+yes_ask)/2
    volume        INTEGER,
    volume_24h    INTEGER,
    open_interest INTEGER,
    liquidity     INTEGER,                -- Kalshi reports this in cents
    status        TEXT,
    raw_json      TEXT,
    UNIQUE(ticker, ts),
    FOREIGN KEY (ticker) REFERENCES markets(ticker) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_ts ON snapshots(ticker, ts);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts        ON snapshots(ts);

CREATE TABLE IF NOT EXISTS orderbooks (
    orderbook_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    ts           TEXT NOT NULL,
    run_id       INTEGER,
    raw_json     TEXT,
    UNIQUE(ticker, ts)
);

CREATE TABLE IF NOT EXISTS orderbook_levels (
    orderbook_id INTEGER NOT NULL,
    side         TEXT NOT NULL,           -- 'yes' | 'no'
    level        INTEGER NOT NULL,        -- 0 = best
    price        INTEGER NOT NULL,        -- cents
    quantity     INTEGER NOT NULL,
    PRIMARY KEY (orderbook_id, side, level),
    FOREIGN KEY (orderbook_id) REFERENCES orderbooks(orderbook_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS candles (
    ticker         TEXT NOT NULL,
    period_minutes INTEGER NOT NULL,
    end_ts         TEXT NOT NULL,
    open_price     INTEGER,
    high_price     INTEGER,
    low_price      INTEGER,
    close_price    INTEGER,
    yes_bid_close  INTEGER,
    yes_ask_close  INTEGER,
    volume         INTEGER,
    open_interest  INTEGER,
    raw_json       TEXT,
    PRIMARY KEY (ticker, period_minutes, end_ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_ticker ON candles(ticker, end_ts);

CREATE TABLE IF NOT EXISTS trades (
    trade_id     TEXT PRIMARY KEY,
    ticker       TEXT NOT NULL,
    ts           TEXT NOT NULL,
    yes_price    INTEGER,
    no_price     INTEGER,
    count        INTEGER,
    taker_side   TEXT,
    raw_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts ON trades(ticker, ts);

CREATE TABLE IF NOT EXISTS predict_availability (
    ticker            TEXT PRIMARY KEY,
    run_id            INTEGER,
    ts                TEXT NOT NULL,
    tradeable         INTEGER NOT NULL,   -- 0/1
    reason            TEXT,               -- why not, when tradeable = 0
    days_to_close     REAL,
    matched_rule      TEXT
);

CREATE TABLE IF NOT EXISTS model_estimates (
    ticker      TEXT NOT NULL,
    run_id      INTEGER NOT NULL,
    ts          TEXT NOT NULL,
    model       TEXT NOT NULL,
    prob        REAL,
    source      TEXT,
    asof        TEXT,
    confidence  REAL,
    notes       TEXT,
    PRIMARY KEY (ticker, run_id, model)
);

CREATE TABLE IF NOT EXISTS signals (
    ticker              TEXT NOT NULL,
    run_id              INTEGER NOT NULL,
    ts                  TEXT NOT NULL,
    implied_prob        REAL,
    model_prob          REAL,
    model_name          TEXT,
    model_confidence    REAL,
    edge                REAL,
    edge_flag           INTEGER,
    side                TEXT,             -- 'yes' | 'no' | NULL
    entry_price         REAL,             -- dollars, incl. side choice
    fee_per_contract    REAL,
    ev_per_contract     REAL,
    ev_pct_of_cost      REAL,
    kelly_fraction_full REAL,
    kelly_fraction_used REAL,
    stake_dollars       REAL,
    contracts           INTEGER,
    days_to_close       REAL,
    annualized_if_win   REAL,
    expected_annualized REAL,
    spread_cents        INTEGER,
    spread_flag         INTEGER,
    liquidity_flag      INTEGER,
    longshot_flag       INTEGER,
    momentum_24h        REAL,
    momentum_flag       INTEGER,
    stale_hours         REAL,
    stale_flag          INTEGER,
    score               REAL,
    score_components    TEXT,             -- JSON: every component, visible
    notes               TEXT,
    PRIMARY KEY (ticker, run_id)
);

CREATE INDEX IF NOT EXISTS idx_signals_run   ON signals(run_id);
CREATE INDEX IF NOT EXISTS idx_signals_score ON signals(score DESC);

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
