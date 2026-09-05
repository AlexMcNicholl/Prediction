"""Ingestion orchestration: Kalshi -> normalise -> SQLite.

One pass fetches every open market (following all pages), records a
point-in-time snapshot per market, classifies Predict availability, then
enriches only the contracts that survive the filter with orderbooks,
candlesticks and trades. Enrichment is the expensive part, so it is bounded by
``ingest.max_enriched_markets``.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import Config
from ..logging_utils import get_logger
from ..predict.availability import PredictFilter
from ..store.db import Database, utc_now_iso
from .kalshi import KalshiAPIError, KalshiClient
from . import normalize

log = get_logger(__name__)


@dataclass
class IngestResult:
    run_id: int
    markets_seen: int = 0
    markets_tradeable: int = 0
    snapshots_written: int = 0
    orderbooks: int = 0
    candles: int = 0
    trades: int = 0
    api_requests: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.errors:
            return "ok"
        return "partial" if self.markets_seen else "failed"


def build_client(config: Config) -> KalshiClient:
    api = config.section("api")
    return KalshiClient(
        base_url=api.get("base_url", "https://api.elections.kalshi.com/trade-api/v2"),
        fallback_base_urls=tuple(api.get("fallback_base_urls") or ()),
        timeout=float(api.get("timeout_seconds", 30)),
        max_retries=int(api.get("max_retries", 5)),
        backoff_base=float(api.get("backoff_base_seconds", 1.0)),
        backoff_max=float(api.get("backoff_max_seconds", 60.0)),
        requests_per_second=float(api.get("requests_per_second", 4.0)),
        floor_requests_per_second=float(api.get("floor_requests_per_second", 0.5)),
        page_limit=int(api.get("page_limit", 200)),
        user_agent=api.get("user_agent", "prediction-screener/1.0"),
    )


def _dump_raw(raw_dir: Path, name: str, payload: Any) -> None:
    """Persist a raw payload so history can be reprocessed if models change."""
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = raw_dir / f"{name}-{stamp}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, default=str)
        log.debug("wrote raw snapshot %s", path)
    except OSError as exc:
        log.warning("could not write raw snapshot %s: %s", name, exc)


def ingest(
    config: Config, db: Database, client: KalshiClient | None = None,
    run_id: int | None = None, max_pages: int | None = None,
) -> IngestResult:
    """Run one full ingestion pass."""
    client = client or build_client(config)
    run_id = run_id if run_id is not None else db.start_run("ingest")
    result = IngestResult(run_id=run_id)
    ingest_cfg = config.section("ingest")
    now_iso = utc_now_iso()

    # --- exchange status (non-fatal) -------------------------------------
    try:
        status = client.get_exchange_status()
        log.info(
            "exchange status: trading_active=%s exchange_active=%s",
            status.get("trading_active"), status.get("exchange_active"),
        )
    except KalshiAPIError as exc:
        log.warning("exchange status unavailable: %s", exc)
        result.errors.append(f"exchange_status: {exc}")

    # --- all open markets, every page ------------------------------------
    try:
        raw_markets = client.get_markets(status="open", max_pages=max_pages)
    except KalshiAPIError as exc:
        log.error("market ingestion failed: %s", exc)
        result.errors.append(f"markets: {exc}")
        result.api_requests = client.stats.requests
        return result

    if ingest_cfg.get("store_raw_json", True):
        _dump_raw(Path(config.get("storage.raw_dir", "data/raw")), "markets", raw_markets)

    markets: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for raw in raw_markets:
        try:
            markets.append(normalize.normalize_market(raw))
            snapshots.append(normalize.normalize_snapshot(raw, now_iso))
        except (ValueError, KeyError) as exc:
            log.warning("skipping unparseable market payload: %s", exc)
            result.errors.append(f"normalize: {exc}")

    result.markets_seen = len(markets)
    db.upsert_markets(markets)
    result.snapshots_written = db.insert_snapshots(snapshots, run_id=run_id)
    log.info("stored %d markets and %d snapshots", len(markets), result.snapshots_written)

    # --- events and series metadata (best-effort) ------------------------
    try:
        events = [normalize.normalize_event(e) for e in client.get_events(status="open")]
        db.upsert_events(events)
        log.info("stored %d events", len(events))
    except KalshiAPIError as exc:
        log.warning("events unavailable: %s", exc)
        result.errors.append(f"events: {exc}")

    try:
        series = [normalize.normalize_series(s) for s in client.get_series_list()]
        db.upsert_series(series)
        log.info("stored %d series", len(series))
    except KalshiAPIError as exc:
        log.debug("series listing unavailable: %s", exc)

    # --- Predict availability --------------------------------------------
    predict_filter = PredictFilter.from_config(config.section("predict"))
    availability = predict_filter.classify_all(markets)
    db.replace_availability([a.as_record() for a in availability], run_id)
    tradeable = [a for a in availability if a.tradeable]
    result.markets_tradeable = len(tradeable)

    for reason, count in predict_filter.summarize_exclusions(availability):
        log.info("  excluded %5d: %s", count, reason)

    # --- enrich only the tradeable shortlist ------------------------------
    market_index = {m["ticker"]: m for m in markets}
    limit = int(ingest_cfg.get("max_enriched_markets", 250))
    to_enrich = [a.ticker for a in tradeable][:limit]
    if len(tradeable) > limit:
        log.warning(
            "enriching only %d of %d tradeable markets (ingest.max_enriched_markets)",
            limit, len(tradeable),
        )

    _enrich(client, db, market_index, to_enrich, ingest_cfg, run_id, result)

    result.api_requests = client.stats.requests
    log.info(
        "ingest complete: %d markets, %d tradeable, %d API requests, %d errors, "
        "%d rate-limit responses (final rate %.1f req/s)",
        result.markets_seen, result.markets_tradeable,
        result.api_requests, len(result.errors),
        client.stats.rate_limited, client._limiter.current_rps,
    )
    return result


def _enrich(
    client: KalshiClient,
    db: Database,
    market_index: Mapping[str, Mapping[str, Any]],
    tickers: Sequence[str],
    ingest_cfg: Mapping[str, Any],
    run_id: int,
    result: IngestResult,
) -> None:
    """Fetch orderbooks / candles / trades for the shortlist.

    Every call is individually non-fatal: one bad market must not lose the run.

    Enrichment is also bounded by wall clock. It is the expensive phase (three
    calls per market), and when the API rate-limits us the client slows itself
    down - so an unbounded loop can outlast the CI job timeout and produce
    nothing at all. A partial enrichment plus a dashboard beats a timeout.
    """
    want_books = ingest_cfg.get("fetch_orderbooks", True)
    want_candles = ingest_cfg.get("fetch_candles", True)
    want_trades = ingest_cfg.get("fetch_trades", True)
    budget = float(ingest_cfg.get("enrichment_budget_seconds", 420))
    started = time.monotonic()

    for index, ticker in enumerate(tickers):
        if budget > 0 and time.monotonic() - started > budget:
            skipped = len(tickers) - index
            log.warning(
                "enrichment budget of %.0fs spent after %d markets; skipping the "
                "remaining %d. Signals still compute from the market snapshot; "
                "raise ingest.enrichment_budget_seconds or lower "
                "ingest.max_enriched_markets.",
                budget, index, skipped,
            )
            result.errors.append(
                f"enrichment budget exhausted, {skipped} markets not enriched"
            )
            break
        market = market_index.get(ticker, {})

        if want_books:
            try:
                book = client.get_orderbook(ticker, int(ingest_cfg.get("orderbook_depth", 10)))
                db.insert_orderbook(ticker, book, run_id=run_id)
                result.orderbooks += 1
            except KalshiAPIError as exc:
                log.debug("orderbook failed for %s: %s", ticker, exc)
                result.errors.append(f"orderbook {ticker}: {exc}")

        if want_candles:
            series_ticker = market.get("series_ticker")
            if series_ticker:
                period = int(ingest_cfg.get("candles_period_minutes", 60))
                try:
                    candles = client.get_candlesticks(
                        str(series_ticker), ticker, period,
                        int(ingest_cfg.get("candles_lookback_hours", 336)),
                    )
                    result.candles += db.upsert_candles(ticker, period, candles)
                except KalshiAPIError as exc:
                    log.debug("candles failed for %s: %s", ticker, exc)
                    result.errors.append(f"candles {ticker}: {exc}")

        if want_trades:
            try:
                trades = client.get_trades(ticker, int(ingest_cfg.get("trades_limit", 100)))
                result.trades += db.upsert_trades(ticker, trades)
            except KalshiAPIError as exc:
                log.debug("trades failed for %s: %s", ticker, exc)
                result.errors.append(f"trades {ticker}: {exc}")

    log.info(
        "enrichment: %d orderbooks, %d candles, %d trades",
        result.orderbooks, result.candles, result.trades,
    )
