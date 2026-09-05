"""Command-line entry point.

    python -m screener.cli verify-api      # confirm the live Kalshi contract
    python -m screener.cli ingest          # fetch + store market data
    python -m screener.cli signals         # recompute signals over stored data
    python -m screener.cli report          # regenerate dashboard + digest
    python -m screener.cli run             # ingest -> signals -> report -> notify
    python -m screener.cli export --format parquet
    python -m screener.cli info            # dataset health

This tool is READ-ONLY. It has no order-placement path and never contacts
Wealthsimple.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import Config, load_config, load_dotenv_if_present
from .logging_utils import get_logger, setup_logging

log = get_logger(__name__)


# --------------------------------------------------------------- primitives

def _open_db(config: Config):
    from .store.db import Database

    return Database(
        config.get("storage.db_path", "data/screener.db"),
        store_raw_json=bool(config.get("ingest.store_raw_json", True)),
    )


def _records_without_nan(df) -> list[dict[str, Any]]:
    """DataFrame rows as dicts with NaN replaced by None.

    `df.where(df.notna(), None)` does NOT do this: a float64 column coerces the
    None straight back to NaN, so `value is None` stays False downstream and
    the template would render "nan%" where it should say "no model".
    """
    if df is None or df.empty:
        return []
    records: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if value is None:
                clean[key] = None
            elif isinstance(value, float) and value != value:  # NaN
                clean[key] = None
            elif value != value:  # NaT and other self-unequal sentinels
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


def _build_stats(config: Config, db, run_id: int, signals: Sequence[dict]) -> dict[str, Any]:
    total = int(db.scalar("SELECT COUNT(*) FROM markets") or 0)
    tradeable = int(
        db.scalar("SELECT COUNT(*) FROM predict_availability WHERE tradeable = 1") or 0
    )
    fee_taker = config.get("fees.taker_coefficient", 0.07)
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_markets": total,
        "tradeable": tradeable,
        "with_model": sum(1 for s in signals if s.get("model_prob") is not None),
        "edge_flagged": sum(1 for s in signals if s.get("edge_flag")),
        "term_days": int(config.get("predict.term_to_maturity_days", 30)),
        "bankroll": f"${float(config.get('sizing.bankroll', 1000)):,.0f}",
        "fee_note": f"taker coefficient {fee_taker}",
    }


def _compute_signals(config: Config, db, run_id: int) -> list[dict[str, Any]]:
    """Recompute signals for every tradeable contract in the database."""
    from .models.registry import ModelRegistry
    from .signals.engine import SignalConfig, SignalEngine
    from .signals.fees import FeeModel

    rows = db.query(
        """
        SELECT m.*, a.tradeable, a.days_to_close
          FROM markets m
          JOIN predict_availability a ON a.ticker = m.ticker
         WHERE a.tradeable = 1
        """
    )
    markets = [dict(r) for r in rows]
    if not markets:
        log.warning(
            "no tradeable markets found - check predict.series_prefix_allowlist "
            "in config.yaml, or run `ingest` first"
        )
        return []

    snapshots = {t: dict(r) for t, r in db.latest_snapshot_map().items()}
    histories = {
        m["ticker"]: [dict(r) for r in db.snapshot_history(m["ticker"], limit=200)]
        for m in markets
    }

    registry = ModelRegistry.from_config(config.section("models"))
    now = datetime.now(timezone.utc)
    estimates = registry.estimate_all(markets, now=now, config=config.data)
    db.insert_model_estimates(
        [e.as_record(t) for t, e in estimates.items() if e.prob is not None], run_id
    )

    engine = SignalEngine(
        SignalConfig.from_config(config.section("signals")),
        FeeModel.from_config(config.section("fees")),
        bankroll=float(config.get("sizing.bankroll", 1000)),
        kelly_fraction=float(config.get("sizing.kelly_fraction", 0.25)),
        max_stake_fraction=float(config.get("sizing.max_stake_fraction", 0.05)),
    )
    signals = engine.compute_all(
        markets,
        snapshots,
        estimates,
        histories,
        {m["ticker"]: m.get("days_to_close") for m in markets},
        now,
    )
    db.insert_signals(signals, run_id)
    return signals


def _render_outputs(
    config: Config, db, run_id: int, signals: Sequence[dict], notify: bool
) -> Path:
    """Build the dashboard and digest from the freshest signal rows."""
    from .analysis.queries import Analysis
    from .report.dashboard import render_dashboard
    from .report.digest import build_digest, deliver

    # Re-read through the analysis layer so the dashboard sees exactly the
    # joined view (titles, rules, close times) that exploration does.
    db.conn.commit()
    with Analysis(config.get("storage.db_path", "data/screener.db")) as analysis:
        df = analysis.signals(run_id=run_id, tradeable_only=True)
        rows: list[dict[str, Any]] = _records_without_nan(df)
        histories = {}
        for row in rows[: int(config.get("report.max_table_rows", 400))]:
            ticker = str(row.get("ticker"))
            points = analysis.snapshots(ticker=ticker, limit=None)
            if not points.empty:
                tail = points.tail(int(config.get("report.sparkline_points", 48)))
                histories[ticker] = [
                    (ts, prob)
                    for ts, prob in zip(tail["ts"], tail["implied_prob"])
                    if prob == prob  # drop NaN
                ]

    if not rows:
        rows = list(signals)

    stats = _build_stats(config, db, run_id, rows)
    out_dir = Path(config.get("report.output_dir", "out"))
    path = render_dashboard(
        rows,
        stats,
        histories=histories,
        weights=config.get("signals.score_weights", None),
        output_path=out_dir / config.get("report.dashboard_filename", "dashboard.html"),
        tz_name=config.get("report.timezone", "UTC"),
        max_rows=int(config.get("report.max_table_rows", 400)),
        max_annualized=float(config.get("signals.max_annualized_display", 100.0)),
        kelly_setting=float(config.get("sizing.kelly_fraction", 0.25)),
    )

    subject, body = build_digest(
        rows, stats, top_n=int(config.get("report.top_n_digest", 10))
    )
    digest_path = out_dir / "digest.txt"
    digest_path.write_text(body, encoding="utf-8")
    log.info("digest written: %s", digest_path)

    if notify:
        results = deliver(subject, body, config.section("notifications"))
        for channel, ok in results.items():
            log.info("digest via %s: %s", channel, "sent" if ok else "FAILED")
    return path


# ---------------------------------------------------------------- commands

def cmd_verify_api(args: argparse.Namespace, config: Config) -> int:
    from .ingest.runner import build_client
    from .ingest.verify import verify

    client = build_client(config)
    # Diagnostics should fail fast. Ingestion keeps the configured retry budget,
    # but a verification run against an unreachable host should say so in
    # seconds rather than grinding through minutes of backoff.
    client.max_retries = min(client.max_retries, args.retries)
    client.backoff_max = min(client.backoff_max, 2.0)
    client.timeout = min(client.timeout, float(args.timeout))
    if args.no_fallback:
        client.fallback_base_urls = ()

    report = verify(client)
    print(report.render())
    if report.unreachable:
        print(
            "Next steps: check outbound network access to the API host. Market data\n"
            "is public, so a 401/403 from the API itself (rather than a proxy) would\n"
            "instead mean the authentication model changed.\n"
        )
    return 1 if report.failed else 0


def cmd_ingest(args: argparse.Namespace, config: Config) -> int:
    from .ingest.runner import ingest

    with _open_db(config) as db:
        run_id = db.start_run("ingest")
        result = ingest(config, db, run_id=run_id, max_pages=args.max_pages)
        db.finish_run(
            run_id, result.status, result.markets_seen, result.markets_tradeable,
            result.api_requests, len(result.errors),
            notes="; ".join(result.errors[:5]) or None,
        )
        print(
            f"Ingested {result.markets_seen} markets, "
            f"{result.markets_tradeable} tradeable on Predict, "
            f"{result.api_requests} API requests, {len(result.errors)} errors."
        )
    return 0 if result.status != "failed" else 1


def cmd_signals(args: argparse.Namespace, config: Config) -> int:
    with _open_db(config) as db:
        run_id = args.run_id or db.latest_run_id() or db.start_run("signals")
        signals = _compute_signals(config, db, run_id)
        flagged = sum(1 for s in signals if s.get("edge_flag"))
        print(f"Computed {len(signals)} signal rows ({flagged} edge-flagged) for run {run_id}.")
    return 0


def cmd_report(args: argparse.Namespace, config: Config) -> int:
    with _open_db(config) as db:
        run_id = args.run_id or db.latest_run_id()
        if run_id is None:
            print("No completed runs yet. Run `ingest` first.", file=sys.stderr)
            return 1
        rows = [dict(r) for r in db.query("SELECT * FROM signals WHERE run_id = ?", (run_id,))]
        path = _render_outputs(config, db, run_id, rows, notify=args.notify)
        print(f"Dashboard: {path}")
    return 0


def cmd_run(args: argparse.Namespace, config: Config) -> int:
    """The full recurring pipeline, as GitHub Actions invokes it."""
    from .ingest.runner import ingest

    with _open_db(config) as db:
        run_id = db.start_run("run")
        result = ingest(config, db, run_id=run_id, max_pages=args.max_pages)
        if result.status == "failed":
            db.finish_run(run_id, "failed", error_count=len(result.errors),
                          notes="; ".join(result.errors[:5]) or None)
            print("Ingestion failed; see the log above.", file=sys.stderr)
            return 1

        signals = _compute_signals(config, db, run_id)
        path = _render_outputs(config, db, run_id, signals, notify=not args.no_notify)
        db.finish_run(
            run_id, result.status, result.markets_seen, result.markets_tradeable,
            result.api_requests, len(result.errors),
            notes="; ".join(result.errors[:5]) or None,
        )
        print(
            f"Run {run_id} complete: {result.markets_seen} markets, "
            f"{result.markets_tradeable} tradeable, {len(signals)} signals.\n"
            f"Dashboard: {path}"
        )
    return 0


def cmd_export(args: argparse.Namespace, config: Config) -> int:
    from .analysis.export import export_dataset

    paths = export_dataset(
        config.get("storage.db_path", "data/screener.db"),
        args.out or config.get("storage.export_dir", "exports"),
        args.format,
    )
    print(f"Exported {len(paths)} files to {args.out or config.get('storage.export_dir')}")
    return 0


def cmd_info(args: argparse.Namespace, config: Config) -> int:
    from .analysis.queries import Analysis

    db_path = config.get("storage.db_path", "data/screener.db")
    if not Path(db_path).exists():
        print(f"No database at {db_path}. Run `ingest` first.", file=sys.stderr)
        return 1
    with Analysis(db_path) as analysis:
        summary = analysis.summary()
    print(json.dumps(summary, indent=2, default=str))
    return 0


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="screener",
        description=(
            "Read-only prediction-market screener for contracts available on "
            "Wealthsimple Predict. Surfaces candidates for MANUAL analysis; it "
            "never places trades and never contacts Wealthsimple."
        ),
    )
    # Global flags are also attached to every subcommand, so they work in
    # either position: `cli --log-level DEBUG ingest` and `cli ingest --log-level DEBUG`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="path to config.yaml")
    common.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")

    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-api", parents=[common],
                       help="confirm the live Kalshi API contract")
    p.add_argument("--retries", type=int, default=0,
                   help="retry budget per check (default 0: diagnostics fail fast)")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="per-request timeout in seconds (default 15)")
    p.add_argument("--no-fallback", action="store_true",
                   help="probe only api.base_url, skipping fallback_base_urls")
    p.set_defaults(func=cmd_verify_api)

    p = sub.add_parser("ingest", parents=[common],
                       help="fetch market data and store snapshots")
    p.add_argument("--max-pages", type=int, default=None, help="cap pages (for testing)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("signals", parents=[common], help="recompute signals over stored data")
    p.add_argument("--run-id", type=int, default=None)
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("report", parents=[common], help="regenerate the dashboard and digest")
    p.add_argument("--run-id", type=int, default=None)
    p.add_argument("--notify", action="store_true", help="also send the digest")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run", parents=[common], help="ingest, compute signals, report, notify")
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--no-notify", action="store_true", help="build the digest but do not send")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("export", parents=[common], help="export tables to CSV or Parquet")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("info", parents=[common], help="dataset health summary")
    p.set_defaults(func=cmd_info)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_present()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1
    setup_logging(args.log_level or config.get("logging.level", "INFO"))
    try:
        return int(args.func(args, config))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        log.exception("command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
