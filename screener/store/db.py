"""SQLite storage layer.

Everything here is idempotent: re-running ingestion for the same markets
updates slow-moving metadata in place and appends a new snapshot row. History
is never overwritten or deleted.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..logging_utils import get_logger

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "1"


def utc_now_iso() -> str:
    """Current UTC time as a second-resolution ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


class Database:
    """Thin wrapper over a SQLite connection with the screener's write paths."""

    def __init__(self, path: str | Path, store_raw_json: bool = True) -> None:
        self.path = Path(path)
        self.store_raw_json = store_raw_json
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._ensure_schema()

    # ------------------------------------------------------------------ setup

    def _ensure_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(sql)
        self.conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (SCHEMA_VERSION,),
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------------- runs

    def start_run(self, command: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(started_at, status, command) VALUES(?, 'running', ?)",
            (utc_now_iso(), command),
        )
        self.conn.commit()
        run_id = int(cur.lastrowid)
        log.info("run %s started (%s)", run_id, command)
        return run_id

    def finish_run(
        self,
        run_id: int,
        status: str = "ok",
        markets_seen: int = 0,
        markets_tradeable: int = 0,
        api_requests: int = 0,
        error_count: int = 0,
        notes: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE runs SET finished_at = ?, status = ?, markets_seen = ?,
                   markets_tradeable = ?, api_requests = ?, error_count = ?, notes = ?
             WHERE run_id = ?
            """,
            (
                utc_now_iso(),
                status,
                markets_seen,
                markets_tradeable,
                api_requests,
                error_count,
                notes,
                run_id,
            ),
        )
        self.conn.commit()
        log.info(
            "run %s finished status=%s seen=%s tradeable=%s errors=%s",
            run_id, status, markets_seen, markets_tradeable, error_count,
        )

    def latest_run_id(self) -> int | None:
        row = self.conn.execute(
            "SELECT run_id FROM runs WHERE status IN ('ok','partial') "
            "ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        return int(row["run_id"]) if row else None

    # --------------------------------------------------------------- upserts

    def upsert_series(self, records: Iterable[Mapping[str, Any]]) -> int:
        now = utc_now_iso()
        rows = [
            (
                r.get("ticker") or r.get("series_ticker"),
                r.get("title"),
                r.get("category"),
                r.get("frequency"),
                now,
                now,
                _json_or_none(r) if self.store_raw_json else None,
            )
            for r in records
            if (r.get("ticker") or r.get("series_ticker"))
        ]
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO series(series_ticker, title, category, frequency,
                                   first_seen, last_seen, raw_json)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(series_ticker) DO UPDATE SET
                    title = COALESCE(excluded.title, series.title),
                    category = COALESCE(excluded.category, series.category),
                    frequency = COALESCE(excluded.frequency, series.frequency),
                    last_seen = excluded.last_seen,
                    raw_json = COALESCE(excluded.raw_json, series.raw_json)
                """,
                rows,
            )
        return len(rows)

    def upsert_events(self, records: Iterable[Mapping[str, Any]]) -> int:
        now = utc_now_iso()
        rows = [
            (
                r.get("event_ticker"),
                r.get("series_ticker"),
                r.get("title"),
                r.get("sub_title") or r.get("subtitle"),
                r.get("category"),
                1 if r.get("mutually_exclusive") else 0,
                now,
                now,
                _json_or_none(r) if self.store_raw_json else None,
            )
            for r in records
            if r.get("event_ticker")
        ]
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO events(event_ticker, series_ticker, title, sub_title,
                                   category, mutually_exclusive, first_seen,
                                   last_seen, raw_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_ticker) DO UPDATE SET
                    series_ticker = COALESCE(excluded.series_ticker, events.series_ticker),
                    title = COALESCE(excluded.title, events.title),
                    sub_title = COALESCE(excluded.sub_title, events.sub_title),
                    category = COALESCE(excluded.category, events.category),
                    mutually_exclusive = excluded.mutually_exclusive,
                    last_seen = excluded.last_seen,
                    raw_json = COALESCE(excluded.raw_json, events.raw_json)
                """,
                rows,
            )
        return len(rows)

    MARKET_COLUMNS = (
        "ticker", "event_ticker", "series_ticker", "market_type", "title",
        "subtitle", "yes_sub_title", "no_sub_title", "category", "status",
        "open_time", "close_time", "expected_expiration_time", "expiration_time",
        "latest_expiration_time", "rules_primary", "rules_secondary",
        "settlement_source", "settlement_timer_seconds", "can_close_early",
        "strike_type", "floor_strike", "cap_strike", "notional_value",
        "tick_size", "result", "settled_at",
    )

    def upsert_markets(self, markets: Sequence[Mapping[str, Any]]) -> int:
        """Insert or update market metadata. Never clobbers a value with NULL."""
        if not markets:
            return 0
        now = utc_now_iso()
        rows = []
        for m in markets:
            values = [m.get(col) for col in self.MARKET_COLUMNS]
            # Normalise booleans SQLite would otherwise store as Python bools.
            # An ABSENT value must stay NULL: the COALESCE in the upsert below
            # would otherwise let a partial payload overwrite a known value.
            idx = self.MARKET_COLUMNS.index("can_close_early")
            if values[idx] is not None:
                values[idx] = 1 if values[idx] else 0
            rows.append(
                tuple(values)
                + (now, now, _json_or_none(m) if self.store_raw_json else None)
            )

        placeholders = ", ".join("?" * (len(self.MARKET_COLUMNS) + 3))
        updatable = [c for c in self.MARKET_COLUMNS if c != "ticker"]
        update_clause = ", ".join(
            f"{c} = COALESCE(excluded.{c}, markets.{c})" for c in updatable
        )
        sql = f"""
            INSERT INTO markets({', '.join(self.MARKET_COLUMNS)}, first_seen, last_seen, raw_json)
            VALUES({placeholders})
            ON CONFLICT(ticker) DO UPDATE SET
                {update_clause},
                last_seen = excluded.last_seen,
                raw_json = COALESCE(excluded.raw_json, markets.raw_json)
        """
        with self.transaction() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def insert_snapshots(
        self, snapshots: Sequence[Mapping[str, Any]], run_id: int | None = None
    ) -> int:
        """Append price snapshots. ``UNIQUE(ticker, ts)`` makes this idempotent."""
        if not snapshots:
            return 0
        rows = []
        for s in snapshots:
            yes_bid, yes_ask = s.get("yes_bid"), s.get("yes_ask")
            spread = (
                yes_ask - yes_bid
                if isinstance(yes_bid, int) and isinstance(yes_ask, int)
                else None
            )
            mid = (yes_bid + yes_ask) / 2.0 if spread is not None else None
            rows.append(
                (
                    s["ticker"], s.get("ts") or utc_now_iso(), run_id,
                    yes_bid, yes_ask, s.get("no_bid"), s.get("no_ask"),
                    s.get("last_price"), s.get("previous_price"), spread, mid,
                    s.get("volume"), s.get("volume_24h"), s.get("open_interest"),
                    s.get("liquidity"), s.get("status"),
                    _json_or_none(s.get("raw")) if self.store_raw_json else None,
                )
            )
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO snapshots(ticker, ts, run_id, yes_bid, yes_ask, no_bid,
                                      no_ask, last_price, previous_price, spread,
                                      mid_price, volume, volume_24h, open_interest,
                                      liquidity, status, raw_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, ts) DO NOTHING
                """,
                rows,
            )
        return len(rows)

    def insert_orderbook(
        self, ticker: str, payload: Mapping[str, Any], run_id: int | None = None,
        ts: str | None = None,
    ) -> None:
        """Store an orderbook snapshot plus its flattened levels."""
        ts = ts or utc_now_iso()
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO orderbooks(ticker, ts, run_id, raw_json) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(ticker, ts) DO NOTHING",
                (ticker, ts, run_id, _json_or_none(payload) if self.store_raw_json else None),
            )
            # With ON CONFLICT DO NOTHING, `lastrowid` can be stale when the
            # insert was a no-op; `rowcount` is the reliable signal.
            if cur.rowcount == 0:
                return
            book_id = int(cur.lastrowid)
            levels: list[tuple[int, str, int, int, int]] = []
            for side in ("yes", "no"):
                raw_levels = (payload or {}).get(side) or []
                for i, entry in enumerate(raw_levels):
                    # Kalshi returns [price_cents, quantity] pairs.
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        continue
                    try:
                        levels.append((book_id, side, i, int(entry[0]), int(entry[1])))
                    except (TypeError, ValueError):
                        continue
            if levels:
                conn.executemany(
                    "INSERT INTO orderbook_levels(orderbook_id, side, level, price, quantity) "
                    "VALUES(?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    levels,
                )

    def upsert_candles(
        self, ticker: str, period_minutes: int, candles: Sequence[Mapping[str, Any]]
    ) -> int:
        if not candles:
            return 0
        rows = []
        for c in candles:
            price = c.get("price") or {}
            yes_bid = c.get("yes_bid") or {}
            yes_ask = c.get("yes_ask") or {}
            end_ts = c.get("end_period_ts")
            if isinstance(end_ts, (int, float)):
                end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()
            else:
                end_iso = str(end_ts) if end_ts else utc_now_iso()
            rows.append(
                (
                    ticker, period_minutes, end_iso,
                    price.get("open"), price.get("high"), price.get("low"),
                    price.get("close"), yes_bid.get("close"), yes_ask.get("close"),
                    c.get("volume"), c.get("open_interest"),
                    _json_or_none(c) if self.store_raw_json else None,
                )
            )
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO candles(ticker, period_minutes, end_ts, open_price,
                                    high_price, low_price, close_price,
                                    yes_bid_close, yes_ask_close, volume,
                                    open_interest, raw_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, period_minutes, end_ts) DO UPDATE SET
                    open_price = excluded.open_price,
                    high_price = excluded.high_price,
                    low_price = excluded.low_price,
                    close_price = excluded.close_price,
                    yes_bid_close = excluded.yes_bid_close,
                    yes_ask_close = excluded.yes_ask_close,
                    volume = excluded.volume,
                    open_interest = excluded.open_interest
                """,
                rows,
            )
        return len(rows)

    def upsert_trades(self, ticker: str, trades: Sequence[Mapping[str, Any]]) -> int:
        if not trades:
            return 0
        rows = []
        for t in trades:
            trade_id = t.get("trade_id") or t.get("id")
            if not trade_id:
                continue
            ts = t.get("created_time") or t.get("ts")
            rows.append(
                (
                    str(trade_id), t.get("ticker") or ticker, str(ts),
                    t.get("yes_price"), t.get("no_price"), t.get("count"),
                    t.get("taker_side"),
                    _json_or_none(t) if self.store_raw_json else None,
                )
            )
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO trades(trade_id, ticker, ts, yes_price, no_price,
                                   count, taker_side, raw_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO NOTHING
                """,
                rows,
            )
        return len(rows)

    def replace_availability(
        self, records: Sequence[Mapping[str, Any]], run_id: int
    ) -> int:
        if not records:
            return 0
        now = utc_now_iso()
        rows = [
            (
                r["ticker"], run_id, now, 1 if r.get("tradeable") else 0,
                r.get("reason"), r.get("days_to_close"), r.get("matched_rule"),
            )
            for r in records
        ]
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO predict_availability(ticker, run_id, ts, tradeable,
                                                 reason, days_to_close, matched_rule)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    run_id = excluded.run_id, ts = excluded.ts,
                    tradeable = excluded.tradeable, reason = excluded.reason,
                    days_to_close = excluded.days_to_close,
                    matched_rule = excluded.matched_rule
                """,
                rows,
            )
        return len(rows)

    def insert_model_estimates(
        self, estimates: Sequence[Mapping[str, Any]], run_id: int
    ) -> int:
        if not estimates:
            return 0
        now = utc_now_iso()
        rows = [
            (
                e["ticker"], run_id, now, e["model"], e.get("prob"),
                e.get("source"), e.get("asof"), e.get("confidence"), e.get("notes"),
            )
            for e in estimates
        ]
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO model_estimates(ticker, run_id, ts, model, prob,
                                            source, asof, confidence, notes)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, run_id, model) DO UPDATE SET
                    prob = excluded.prob, source = excluded.source,
                    asof = excluded.asof, confidence = excluded.confidence,
                    notes = excluded.notes
                """,
                rows,
            )
        return len(rows)

    SIGNAL_COLUMNS = (
        "ticker", "run_id", "ts", "implied_prob", "model_prob", "model_name",
        "model_confidence", "edge", "edge_flag", "side", "entry_price",
        "fee_per_contract", "ev_per_contract", "ev_pct_of_cost",
        "kelly_fraction_full", "kelly_fraction_used", "stake_dollars",
        "contracts", "days_to_close", "annualized_if_win", "expected_annualized",
        "spread_cents", "spread_flag", "liquidity_flag", "longshot_flag",
        "momentum_24h", "momentum_flag", "stale_hours", "stale_flag", "score",
        "score_components", "notes",
    )

    def insert_signals(self, signals: Sequence[Mapping[str, Any]], run_id: int) -> int:
        if not signals:
            return 0
        now = utc_now_iso()
        rows = []
        for s in signals:
            record = dict(s)
            record["run_id"] = run_id
            record.setdefault("ts", now)
            if isinstance(record.get("score_components"), Mapping):
                record["score_components"] = _json_or_none(record["score_components"])
            rows.append(tuple(record.get(c) for c in self.SIGNAL_COLUMNS))
        placeholders = ", ".join("?" * len(self.SIGNAL_COLUMNS))
        updatable = [c for c in self.SIGNAL_COLUMNS if c not in ("ticker", "run_id")]
        update_clause = ", ".join(f"{c} = excluded.{c}" for c in updatable)
        with self.transaction() as conn:
            conn.executemany(
                f"""
                INSERT INTO signals({', '.join(self.SIGNAL_COLUMNS)})
                VALUES({placeholders})
                ON CONFLICT(ticker, run_id) DO UPDATE SET {update_clause}
                """,
                rows,
            )
        return len(rows)

    # -------------------------------------------------------------- read side

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None

    def latest_snapshot_map(self) -> dict[str, sqlite3.Row]:
        """Most recent snapshot per ticker."""
        rows = self.conn.execute(
            """
            SELECT s.* FROM snapshots s
            JOIN (SELECT ticker, MAX(ts) AS mts FROM snapshots GROUP BY ticker) m
              ON s.ticker = m.ticker AND s.ts = m.mts
            """
        ).fetchall()
        return {r["ticker"]: r for r in rows}

    def snapshot_history(self, ticker: str, limit: int = 500) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM snapshots WHERE ticker = ? ORDER BY ts DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()

    def table_counts(self) -> dict[str, int]:
        tables = [
            "runs", "series", "events", "markets", "snapshots", "orderbooks",
            "candles", "trades", "predict_availability", "model_estimates", "signals",
        ]
        return {t: int(self.scalar(f"SELECT COUNT(*) FROM {t}") or 0) for t in tables}
