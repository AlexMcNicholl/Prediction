"""Reusable query functions returning pandas DataFrames.

This is the open-ended exploration surface: every function works over the FULL
dataset (all markets, all snapshots, tradeable or not), not just the shortlist.
Import these in ``analysis.ipynb`` or any script.

    from screener.analysis.queries import Analysis
    a = Analysis("data/screener.db")
    df = a.markets(category="Economics", max_days_to_close=30)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


class Analysis:
    """Read-only pandas view over the screener database."""

    def __init__(self, db_path: str | Path = "data/screener.db") -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"No database at {self.db_path}. Run `python -m screener.cli ingest` first."
            )
        # Read-only URI connection: exploration can never mutate the dataset.
        self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Analysis":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ raw

    def sql(self, query: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        """Escape hatch: run any SELECT and get a DataFrame back."""
        return pd.read_sql_query(query, self.conn, params=list(params))

    def tables(self) -> pd.DataFrame:
        return self.sql(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )

    def runs(self, limit: int = 50) -> pd.DataFrame:
        return self.sql("SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (limit,))

    # -------------------------------------------------------------- markets

    def markets(
        self,
        category: str | None = None,
        series: str | None = None,
        tradeable_only: bool = False,
        status: str | None = None,
        max_days_to_close: float | None = None,
        min_days_to_close: float | None = None,
        search: str | None = None,
    ) -> pd.DataFrame:
        """Market metadata joined with the latest snapshot and availability.

        Every filter is optional; with no arguments you get everything.
        """
        query = """
            SELECT m.*,
                   a.tradeable, a.reason AS not_tradeable_reason,
                   a.days_to_close, a.matched_rule,
                   s.ts AS snapshot_ts, s.yes_bid, s.yes_ask, s.no_bid, s.no_ask,
                   s.last_price, s.spread, s.mid_price, s.volume, s.volume_24h,
                   s.open_interest, s.liquidity
              FROM markets m
              LEFT JOIN predict_availability a ON a.ticker = m.ticker
              LEFT JOIN (
                    SELECT s1.* FROM snapshots s1
                    JOIN (SELECT ticker, MAX(ts) AS mts FROM snapshots GROUP BY ticker) s2
                      ON s1.ticker = s2.ticker AND s1.ts = s2.mts
              ) s ON s.ticker = m.ticker
             WHERE 1=1
        """
        params: list[Any] = []
        if category:
            query += " AND LOWER(m.category) = LOWER(?)"
            params.append(category)
        if series:
            query += " AND m.series_ticker = ?"
            params.append(series)
        if status:
            query += " AND m.status = ?"
            params.append(status)
        if tradeable_only:
            query += " AND a.tradeable = 1"
        if max_days_to_close is not None:
            query += " AND a.days_to_close <= ?"
            params.append(max_days_to_close)
        if min_days_to_close is not None:
            query += " AND a.days_to_close >= ?"
            params.append(min_days_to_close)
        if search:
            query += " AND (m.title LIKE ? OR m.subtitle LIKE ? OR m.ticker LIKE ?)"
            params.extend([f"%{search}%"] * 3)

        df = self.sql(query, params)
        return self._add_derived(df)

    @staticmethod
    def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
        """Add the columns everyone immediately wants."""
        if df.empty:
            return df
        if "mid_price" in df:
            df["implied_prob"] = df["mid_price"] / 100.0
        if {"yes_ask", "yes_bid"}.issubset(df.columns):
            df["spread_pct"] = (df["yes_ask"] - df["yes_bid"]) / 100.0
        if "liquidity" in df:
            df["liquidity_dollars"] = df["liquidity"] / 100.0
        for col in ("close_time", "open_time", "snapshot_ts"):
            if col in df:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        return df

    # -------------------------------------------------------------- signals

    def signals(
        self,
        run_id: int | None = None,
        tradeable_only: bool = True,
        min_score: float | None = None,
        min_abs_edge: float | None = None,
        max_spread_cents: int | None = None,
        price_band: tuple[float, float] | None = None,
        max_days_to_close: float | None = None,
        has_model: bool | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Signal rows joined with the market metadata needed to read them."""
        if run_id is None:
            run_id = self.sql("SELECT MAX(run_id) AS r FROM signals")["r"].iloc[0]
        if run_id is None or (isinstance(run_id, float) and pd.isna(run_id)):
            return pd.DataFrame()

        query = """
            SELECT g.*, m.title, m.subtitle, m.category, m.series_ticker,
                   m.event_ticker, m.close_time, m.status, m.rules_primary,
                   m.rules_secondary, m.settlement_source, m.strike_type,
                   m.floor_strike, m.cap_strike,
                   a.tradeable, a.reason AS not_tradeable_reason,
                   s.yes_bid, s.yes_ask, s.no_bid, s.no_ask, s.last_price,
                   s.volume, s.open_interest, s.liquidity
              FROM signals g
              JOIN markets m ON m.ticker = g.ticker
              LEFT JOIN predict_availability a ON a.ticker = g.ticker
              LEFT JOIN (
                    SELECT s1.* FROM snapshots s1
                    JOIN (SELECT ticker, MAX(ts) AS mts FROM snapshots GROUP BY ticker) s2
                      ON s1.ticker = s2.ticker AND s1.ts = s2.mts
              ) s ON s.ticker = g.ticker
             WHERE g.run_id = ?
        """
        params: list[Any] = [int(run_id)]
        if tradeable_only:
            query += " AND a.tradeable = 1"
        if min_score is not None:
            query += " AND g.score >= ?"
            params.append(min_score)
        if min_abs_edge is not None:
            query += " AND ABS(COALESCE(g.edge, 0)) >= ?"
            params.append(min_abs_edge)
        if max_spread_cents is not None:
            query += " AND g.spread_cents <= ?"
            params.append(max_spread_cents)
        if price_band is not None:
            query += " AND g.implied_prob BETWEEN ? AND ?"
            params.extend(price_band)
        if max_days_to_close is not None:
            query += " AND g.days_to_close <= ?"
            params.append(max_days_to_close)
        if has_model is True:
            query += " AND g.model_prob IS NOT NULL"
        elif has_model is False:
            query += " AND g.model_prob IS NULL"

        query += " ORDER BY g.score DESC"
        if limit:
            query += f" LIMIT {int(limit)}"

        df = self.sql(query, params)
        if not df.empty and "score_components" in df:
            df["score_components"] = df["score_components"].apply(_safe_json)
            df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce", utc=True)
        return df

    # ------------------------------------------------------------ snapshots

    def snapshots(
        self, ticker: str | None = None, since: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Raw time-series snapshots - the basis for volatility and momentum."""
        query = "SELECT * FROM snapshots WHERE 1=1"
        params: list[Any] = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if since:
            query += " AND ts >= ?"
            params.append(since)
        query += " ORDER BY ts ASC"
        if limit:
            query += f" LIMIT {int(limit)}"
        df = self.sql(query, params)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
            df["implied_prob"] = df["mid_price"] / 100.0
        return df

    def price_history(self, ticker: str) -> pd.DataFrame:
        """Combined snapshot + candle history for one contract."""
        snaps = self.snapshots(ticker=ticker)
        candles = self.sql(
            "SELECT end_ts AS ts, close_price AS mid_price, volume "
            "FROM candles WHERE ticker = ? ORDER BY end_ts ASC",
            (ticker,),
        )
        if not candles.empty:
            candles["ts"] = pd.to_datetime(candles["ts"], errors="coerce", utc=True)
            candles["implied_prob"] = candles["mid_price"] / 100.0
            candles["source"] = "candle"
        if not snaps.empty:
            snaps = snaps[["ts", "mid_price", "volume", "implied_prob"]].copy()
            snaps["source"] = "snapshot"
        combined = pd.concat([c for c in (candles, snaps) if not c.empty], ignore_index=True)
        return combined.sort_values("ts").reset_index(drop=True) if not combined.empty else combined

    def trades(self, ticker: str | None = None, limit: int = 1000) -> pd.DataFrame:
        query = "SELECT * FROM trades"
        params: list[Any] = []
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker)
        query += f" ORDER BY ts DESC LIMIT {int(limit)}"
        df = self.sql(query, params)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        return df

    def orderbook(self, ticker: str) -> pd.DataFrame:
        return self.sql(
            """
            SELECT o.ts, l.side, l.level, l.price, l.quantity
              FROM orderbooks o JOIN orderbook_levels l ON l.orderbook_id = o.orderbook_id
             WHERE o.ticker = ?
             ORDER BY o.ts DESC, l.side, l.level
            """,
            (ticker,),
        )

    # ------------------------------------------------------- research views

    def settled_markets(self) -> pd.DataFrame:
        """Markets with a known outcome - the basis for calibration/backtests."""
        df = self.sql(
            """
            SELECT m.ticker, m.title, m.series_ticker, m.category, m.result,
                   m.close_time, m.settled_at
              FROM markets m
             WHERE m.result IS NOT NULL AND m.result != ''
            """
        )
        if df.empty:
            return df
        df["resolved_yes"] = (
            df["result"].astype(str).str.lower().isin(["yes", "y", "true", "1"]).astype(int)
        )
        return df

    def calibration(self, bins: int = 10, hours_before_close: float = 24.0) -> pd.DataFrame:
        """Predicted-vs-realised calibration, the favorite-longshot bias check.

        Takes each settled market's last snapshot at least ``hours_before_close``
        before it closed, buckets by implied probability, and compares the
        bucket's mean price against the realised YES rate. A well-calibrated
        market sits on the diagonal; the classic retail bias shows up as
        longshot buckets resolving YES less often than their price implies.
        """
        settled = self.settled_markets()
        if settled.empty:
            return pd.DataFrame()

        snaps = self.sql(
            """
            SELECT s.ticker, s.ts, s.mid_price, s.last_price, m.close_time
              FROM snapshots s
              JOIN markets m ON m.ticker = s.ticker
             WHERE m.result IS NOT NULL AND m.result != ''
            """
        )
        if snaps.empty:
            return pd.DataFrame()

        snaps["ts"] = pd.to_datetime(snaps["ts"], errors="coerce", utc=True)
        snaps["close_time"] = pd.to_datetime(snaps["close_time"], errors="coerce", utc=True)
        snaps["hours_before"] = (
            snaps["close_time"] - snaps["ts"]
        ).dt.total_seconds() / 3600.0
        eligible = snaps[snaps["hours_before"] >= hours_before_close].copy()
        if eligible.empty:
            return pd.DataFrame()

        eligible["price"] = eligible["mid_price"].fillna(eligible["last_price"])
        eligible = eligible.dropna(subset=["price"])
        # The snapshot closest to the cutoff, per ticker.
        latest = eligible.sort_values("hours_before").groupby("ticker", as_index=False).first()
        merged = latest.merge(
            settled[["ticker", "resolved_yes"]], on="ticker", how="inner"
        )
        if merged.empty:
            return pd.DataFrame()

        merged["implied_prob"] = merged["price"] / 100.0
        merged["bucket"] = pd.cut(
            merged["implied_prob"], bins=[i / bins for i in range(bins + 1)],
            include_lowest=True,
        )
        grouped = merged.groupby("bucket", observed=True).agg(
            n=("ticker", "count"),
            mean_implied=("implied_prob", "mean"),
            realized_yes_rate=("resolved_yes", "mean"),
        ).reset_index()
        grouped["bias"] = grouped["realized_yes_rate"] - grouped["mean_implied"]
        return grouped

    def signal_backtest(
        self, signal_column: str = "edge_flag", hours_before_close: float = 24.0
    ) -> pd.DataFrame:
        """Realised YES rate and mean EV, split by whether a signal fired.

        This is a coarse check that a signal separates outcomes at all. It is
        NOT a strategy backtest: it ignores execution, slippage, and the fact
        that you trade manually, hours later, at a different price.
        """
        signals = self.sql(
            f"""
            SELECT g.ticker, g.{signal_column} AS flag, g.implied_prob,
                   g.model_prob, g.edge, g.ev_per_contract, g.score, m.result
              FROM signals g
              JOIN markets m ON m.ticker = g.ticker
             WHERE m.result IS NOT NULL AND m.result != ''
            """
        )
        if signals.empty:
            return pd.DataFrame()
        signals["resolved_yes"] = (
            signals["result"].astype(str).str.lower().isin(["yes", "y", "true", "1"]).astype(int)
        )
        return signals.groupby("flag").agg(
            n=("ticker", "count"),
            realized_yes_rate=("resolved_yes", "mean"),
            mean_implied=("implied_prob", "mean"),
            mean_edge=("edge", "mean"),
            mean_ev=("ev_per_contract", "mean"),
            mean_score=("score", "mean"),
        ).reset_index()

    def spread_vs_liquidity(self, tradeable_only: bool = True) -> pd.DataFrame:
        """Scatter-ready frame: spread against volume / open interest."""
        df = self.markets(tradeable_only=tradeable_only)
        if df.empty:
            return df
        cols = [
            "ticker", "title", "series_ticker", "category", "spread",
            "volume", "open_interest", "liquidity_dollars", "implied_prob",
            "days_to_close",
        ]
        return df[[c for c in cols if c in df.columns]].dropna(subset=["spread"])

    def edge_distribution(self, run_id: int | None = None) -> pd.DataFrame:
        df = self.signals(run_id=run_id, tradeable_only=False, has_model=True)
        if df.empty:
            return df
        return df[
            ["ticker", "title", "series_ticker", "implied_prob", "model_prob",
             "edge", "model_name", "model_confidence", "score"]
        ].sort_values("edge")

    def summary(self) -> dict[str, Any]:
        """One-glance dataset health check."""
        counts = {
            row[0]: self.sql(f"SELECT COUNT(*) AS n FROM {row[0]}")["n"].iloc[0]
            for row in self.sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).values
        }
        latest_run = self.sql("SELECT * FROM runs ORDER BY run_id DESC LIMIT 1")
        return {
            "db_path": str(self.db_path),
            "row_counts": counts,
            "latest_run": latest_run.to_dict("records")[0] if not latest_run.empty else None,
        }


def _safe_json(value: Any) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}
