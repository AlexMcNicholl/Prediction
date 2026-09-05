"""Shared fixtures: a synthetic Kalshi dataset and a fake API client.

The development sandbox for this repo has no network access to Kalshi, so the
whole pipeline is exercised against these fixtures. They mirror the field
names and value ranges the live API returns (prices in integer cents,
RFC3339 timestamps, cursor pagination).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.config import Config  # noqa: E402
from screener.store.db import Database  # noqa: E402

NOW = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_market(
    ticker: str,
    series: str,
    days_to_close: float,
    yes_bid: int,
    yes_ask: int,
    *,
    category: str = "Economics",
    status: str = "active",
    volume: int = 5000,
    open_interest: int = 3000,
    liquidity: int = 250000,
    floor_strike: float | None = None,
    cap_strike: float | None = None,
    strike_type: str | None = None,
    result: str | None = None,
) -> dict:
    """One market payload shaped like Kalshi's /markets response."""
    return {
        "ticker": ticker,
        "event_ticker": ticker.rsplit("-", 1)[0],
        "series_ticker": series,
        "market_type": "binary",
        "title": f"{series} contract {ticker}",
        "subtitle": "test subtitle",
        "yes_sub_title": "Yes",
        "no_sub_title": "No",
        "category": category,
        "status": status,
        "open_time": iso(NOW - timedelta(days=20)),
        "close_time": iso(NOW + timedelta(days=days_to_close)),
        "expiration_time": iso(NOW + timedelta(days=days_to_close + 1)),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": 100 - yes_ask,
        "no_ask": 100 - yes_bid,
        "last_price": (yes_bid + yes_ask) // 2,
        "previous_price": (yes_bid + yes_ask) // 2 - 2,
        "volume": volume,
        "volume_24h": volume // 10,
        "open_interest": open_interest,
        "liquidity": liquidity,
        "rules_primary": f"Settles YES if the {series} measure exceeds the strike.",
        "rules_secondary": "Sourced from the official release.",
        "settlement_sources": [{"name": "BLS", "url": "https://www.bls.gov"}],
        "can_close_early": True,
        "strike_type": strike_type,
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "notional_value": 100,
        "tick_size": 1,
        "result": result,
    }


@pytest.fixture
def sample_markets() -> list[dict]:
    """A spread of markets covering every filter and model branch."""
    return [
        # Tradeable: on the allowlist, closes inside 30 days, CPI model applies.
        make_market("KXCPI-26SEP-T2.9", "KXCPI", 15, 60, 64,
                    floor_strike=2.9, strike_type="greater"),
        # Priced at a 0.08 mid: unambiguously inside the longshot band.
        make_market("KXCPI-26SEP-T3.5", "KXCPI", 15, 6, 10,
                    floor_strike=3.5, strike_type="greater"),
        # Tradeable but no model covers it.
        make_market("KXU3-26SEP-T4.2", "KXU3", 10, 45, 48),
        # Thin book + wide spread.
        make_market("KXGDP-26SEP-T2", "KXGDP", 7, 30, 42,
                    volume=10, open_interest=5, liquidity=200),
        # Weather, tradeable.
        make_market("KXHIGHNY-26SEP06-B80", "KXHIGHNY", 1, 55, 58,
                    category="Climate and Weather", floor_strike=75, cap_strike=80),
        # Excluded: beyond the 30-day term-to-maturity rule.
        make_market("KXCPI-26DEC-T3.0", "KXCPI", 90, 50, 54,
                    floor_strike=3.0, strike_type="greater"),
        # Excluded: series not on the Predict allowlist.
        make_market("KXOSCAR-26-BP", "KXOSCAR", 10, 20, 25, category="Entertainment"),
        # Excluded: explicitly denylisted.
        make_market("KXPRES-28-DEM", "KXPRES", 10, 50, 52, category="Politics"),
        # Excluded: already closed.
        make_market("KXCPI-26AUG-T2.5", "KXCPI", -3, 90, 95,
                    floor_strike=2.5, strike_type="greater", status="closed",
                    result="yes"),
    ]


@pytest.fixture
def config() -> Config:
    """Config matching the shipped config.yaml, with models pinned for tests."""
    return Config(
        {
            "api": {"base_url": "https://example.invalid/trade-api/v2", "page_limit": 100},
            "ingest": {
                "fetch_orderbooks": True, "fetch_candles": True, "fetch_trades": True,
                "orderbook_depth": 5, "candles_period_minutes": 60,
                "candles_lookback_hours": 168, "trades_limit": 10,
                "max_enriched_markets": 50, "store_raw_json": False,
            },
            "predict": {
                "term_to_maturity_days": 30,
                "require_open_status": True,
                "category_allowlist": ["Economics", "Financials", "Climate and Weather"],
                "series_prefix_allowlist": ["KXCPI", "KXU3", "KXGDP", "KXHIGH", "KXNASDAQ100"],
                "series_allowlist": [],
                "series_denylist": ["KXPRES", "KXELECTION"],
            },
            "signals": {
                "edge_threshold": 0.05, "spread_threshold_cents": 3,
                "min_volume": 100, "min_open_interest": 100,
                "min_liquidity_dollars": 500, "longshot_low": 0.10,
                "longshot_high": 0.90, "staleness_hours": 24,
                "momentum_lookback_hours": 24, "momentum_threshold": 0.05,
                "max_annualized_display": 100.0,
                "score_weights": {
                    "edge": 0.40, "liquidity": 0.20, "spread": 0.15,
                    "annualized": 0.15, "momentum": 0.10,
                },
            },
            "fees": {
                "taker_coefficient": 0.07, "maker_coefficient": 0.0175,
                "per_contract_cap_dollars": None,
                "settlement_fee_per_contract": 0.0, "assume_taker": True,
            },
            "sizing": {"bankroll": 1000.0, "kelly_fraction": 0.25, "max_stake_fraction": 0.05},
            "models": {
                "enabled": ["cpi", "weather"],
                "cpi": {
                    "sigma": 0.15, "nowcast_url": None,
                    "manual_override": {"cpi_yoy": 2.9, "asof": "2026-09-01"},
                },
                "weather": {
                    "provider": "manual", "manual_forecasts": {"NY": 78},
                    "climatology_weight": 0.30, "forecast_sigma_f": 3.0,
                },
            },
            "report": {
                "output_dir": "out", "dashboard_filename": "dashboard.html",
                "top_n_digest": 5, "max_table_rows": 100,
                "timezone": "America/Toronto", "sparkline_points": 48,
            },
            "notifications": {"enabled": False, "channels": []},
            "storage": {"db_path": "data/screener.db", "raw_dir": "data/raw",
                        "export_dir": "exports"},
            "logging": {"level": "WARNING"},
        }
    )


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db", store_raw_json=False)
    yield database
    database.close()


class FakeKalshiClient:
    """Stands in for KalshiClient, serving the fixture dataset."""

    def __init__(self, markets: list[dict]) -> None:
        self.markets = markets
        self.calls: list[str] = []

        class _Stats:
            requests = 0
            retries = 0
            errors = 0
            pages = 0
            rate_limited = 0
        self.stats = _Stats()

        # Mirrors the real client's limiter so the ingest summary can report it.
        class _Limiter:
            current_rps = float("inf")
            throttle_events = 0
        self._limiter = _Limiter()

    def get_exchange_status(self) -> dict:
        self.calls.append("status")
        return {"trading_active": True, "exchange_active": True}

    def get_markets(self, status=None, max_pages=None, **kwargs) -> list[dict]:
        self.calls.append("markets")
        if status == "open":
            return [m for m in self.markets if m["status"] != "closed"]
        return list(self.markets)

    def get_events(self, status=None, max_pages=None, **kwargs) -> list[dict]:
        self.calls.append("events")
        seen: dict[str, dict] = {}
        for m in self.markets:
            seen[m["event_ticker"]] = {
                "event_ticker": m["event_ticker"],
                "series_ticker": m["series_ticker"],
                "title": m["title"],
                "category": m["category"],
            }
        return list(seen.values())

    def get_series_list(self, category=None) -> list[dict]:
        self.calls.append("series")
        return [
            {"ticker": s, "title": s, "category": "Economics"}
            for s in {m["series_ticker"] for m in self.markets}
        ]

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        self.calls.append(f"orderbook:{ticker}")
        return {"yes": [[60, 500], [59, 800]], "no": [[36, 400]]}

    def get_candlesticks(self, series_ticker, ticker, period_minutes=60, lookback_hours=336):
        self.calls.append(f"candles:{ticker}")
        base = int(NOW.timestamp())
        return [
            {
                "end_period_ts": base - i * 3600,
                "price": {"open": 60, "high": 65, "low": 58, "close": 62 - i},
                "volume": 40,
                "open_interest": 3000,
            }
            for i in range(5)
        ]

    def get_trades(self, ticker: str, limit: int = 100) -> list[dict]:
        self.calls.append(f"trades:{ticker}")
        return [
            {
                "trade_id": f"{ticker}-t{i}",
                "ticker": ticker,
                "created_time": iso(NOW - timedelta(hours=i)),
                "yes_price": 62, "no_price": 38, "count": 10, "taker_side": "yes",
            }
            for i in range(3)
        ]


@pytest.fixture
def fake_client(sample_markets) -> FakeKalshiClient:
    return FakeKalshiClient(sample_markets)
