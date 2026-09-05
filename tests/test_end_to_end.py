"""Full pipeline: ingest -> filter -> models -> signals -> dashboard.

Runs against the fixture dataset via a fake client, so it exercises the same
code path GitHub Actions runs without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from screener.analysis.export import export_dataset
from screener.analysis.queries import Analysis
from screener.ingest.runner import ingest
from screener.report.dashboard import render_dashboard
from screener.report.digest import build_digest


@pytest.fixture
def ingested(config, db, fake_client, tmp_path):
    """Run one ingestion pass and return the run id."""
    config.data["storage"]["db_path"] = str(db.path)
    config.data["storage"]["raw_dir"] = str(tmp_path / "raw")
    run_id = db.start_run("test")
    result = ingest(config, db, client=fake_client, run_id=run_id)
    db.finish_run(run_id, result.status, result.markets_seen, result.markets_tradeable)
    return run_id, result


class TestIngestion:
    def test_stores_open_markets(self, ingested, db):
        _, result = ingested
        assert result.markets_seen == 8  # 9 fixtures minus the closed one
        assert db.scalar("SELECT COUNT(*) FROM markets") == 8

    def test_writes_a_snapshot_per_market(self, ingested, db):
        assert db.scalar("SELECT COUNT(*) FROM snapshots") == 8

    def test_applies_the_predict_filter(self, ingested, db):
        _, result = ingested
        # KXCPI x2, KXU3, KXGDP, KXHIGHNY are tradeable;
        # the 90-day CPI, KXOSCAR and KXPRES are not.
        assert result.markets_tradeable == 5
        rows = db.query("SELECT ticker FROM predict_availability WHERE tradeable = 1")
        assert {r["ticker"] for r in rows} == {
            "KXCPI-26SEP-T2.9", "KXCPI-26SEP-T3.5", "KXU3-26SEP-T4.2",
            "KXGDP-26SEP-T2", "KXHIGHNY-26SEP06-B80",
        }

    def test_records_why_each_exclusion_happened(self, ingested, db):
        rows = {
            r["ticker"]: r["reason"]
            for r in db.query("SELECT ticker, reason FROM predict_availability "
                              "WHERE tradeable = 0")
        }
        assert "30-day" in rows["KXCPI-26DEC-T3.0"]
        assert "allowlist" in rows["KXOSCAR-26-BP"]
        assert "denylist" in rows["KXPRES-28-DEM"]

    def test_enriches_only_the_tradeable_shortlist(self, ingested, db, fake_client):
        enriched = {c.split(":", 1)[1] for c in fake_client.calls if c.startswith("orderbook:")}
        assert len(enriched) == 5
        assert "KXOSCAR-26-BP" not in enriched
        assert db.scalar("SELECT COUNT(*) FROM orderbooks") == 5
        assert db.scalar("SELECT COUNT(*) FROM trades") > 0
        assert db.scalar("SELECT COUNT(*) FROM candles") > 0

    def test_resolution_rules_are_stored(self, ingested, db):
        row = db.query("SELECT rules_primary, settlement_source FROM markets "
                       "WHERE ticker = 'KXCPI-26SEP-T2.9'")[0]
        assert row["rules_primary"]
        assert "BLS" in row["settlement_source"]

    def test_rerunning_does_not_duplicate(self, config, db, fake_client, ingested):
        before = db.scalar("SELECT COUNT(*) FROM markets")
        run_id = db.start_run("again")
        ingest(config, db, client=fake_client, run_id=run_id)
        assert db.scalar("SELECT COUNT(*) FROM markets") == before


class TestSignals:
    @pytest.fixture
    def signals(self, config, db, ingested):
        from screener.cli import _compute_signals

        run_id, _ = ingested
        return run_id, _compute_signals(config, db, run_id)

    def test_one_signal_per_tradeable_market(self, signals):
        _, rows = signals
        assert len(rows) == 5

    def test_models_are_applied_where_they_fit(self, signals):
        _, rows = signals
        by_ticker = {r["ticker"]: r for r in rows}
        assert by_ticker["KXCPI-26SEP-T2.9"]["model_name"] == "cpi"
        assert by_ticker["KXHIGHNY-26SEP06-B80"]["model_name"] == "weather"
        # No model covers unemployment or GDP in this config.
        assert by_ticker["KXU3-26SEP-T4.2"]["model_prob"] is None

    def test_edge_is_computed_only_with_a_model(self, signals):
        _, rows = signals
        for row in rows:
            assert (row["edge"] is not None) == (row["model_prob"] is not None)

    def test_thin_book_is_flagged(self, signals):
        _, rows = signals
        gdp = next(r for r in rows if r["ticker"] == "KXGDP-26SEP-T2")
        assert gdp["liquidity_flag"] == 1
        assert gdp["spread_flag"] == 1

    def test_longshot_is_flagged(self, signals):
        _, rows = signals
        longshot = next(r for r in rows if r["ticker"] == "KXCPI-26SEP-T3.5")
        assert longshot["longshot_flag"] == 1

    def test_signals_are_persisted_with_components(self, signals, db):
        run_id, _ = signals
        rows = db.query("SELECT score_components FROM signals WHERE run_id = ?", (run_id,))
        assert len(rows) == 5
        assert "edge" in json.loads(rows[0]["score_components"])

    def test_model_estimates_are_recorded(self, signals, db):
        run_id, _ = signals
        rows = db.query("SELECT model, prob, source FROM model_estimates WHERE run_id = ?",
                        (run_id,))
        assert {r["model"] for r in rows} == {"cpi", "weather"}
        assert all(r["source"] for r in rows)


class TestAnalysisLayer:
    def test_queries_see_all_markets_not_just_the_shortlist(self, config, db, ingested):
        with Analysis(db.path) as analysis:
            assert len(analysis.markets()) == 8
            assert len(analysis.markets(tradeable_only=True)) == 5
            assert len(analysis.markets(category="Economics")) > 0
            assert len(analysis.markets(search="KXCPI")) == 3

    def test_snapshots_and_history(self, config, db, ingested):
        with Analysis(db.path) as analysis:
            assert not analysis.snapshots().empty
            assert not analysis.price_history("KXCPI-26SEP-T2.9").empty

    def test_spread_vs_liquidity_frame(self, config, db, ingested):
        with Analysis(db.path) as analysis:
            df = analysis.spread_vs_liquidity()
            assert {"spread", "volume", "open_interest"}.issubset(df.columns)

    def test_analysis_connection_is_read_only(self, config, db, ingested):
        import sqlite3

        with Analysis(db.path) as analysis:
            with pytest.raises(sqlite3.OperationalError):
                analysis.conn.execute("DELETE FROM markets")

    def test_export_writes_files(self, config, db, ingested, tmp_path):
        from screener.cli import _compute_signals

        run_id, _ = ingested
        _compute_signals(config, db, run_id)
        db.conn.commit()
        paths = export_dataset(db.path, tmp_path / "exports", "csv")
        assert paths
        assert (tmp_path / "exports" / "markets.csv").exists()
        assert (tmp_path / "exports" / "signals_joined.csv").exists()


class TestReporting:
    @pytest.fixture
    def rendered(self, config, db, ingested, tmp_path):
        from screener.cli import _compute_signals, _render_outputs

        run_id, _ = ingested
        signals = _compute_signals(config, db, run_id)
        config.data["report"]["output_dir"] = str(tmp_path / "out")
        path = _render_outputs(config, db, run_id, signals, notify=False)
        return path, signals

    def test_dashboard_is_a_single_self_contained_file(self, rendered):
        path, _ = rendered
        html = Path(path).read_text(encoding="utf-8")
        assert path.exists()
        assert "<style>" in html and "<script>" in html
        # No external resources: it must open on a phone with no network.
        assert "src=\"http" not in html
        assert "cdn." not in html
        assert "<link" not in html

    def test_dashboard_states_the_framing(self, rendered):
        html = Path(rendered[0]).read_text(encoding="utf-8")
        assert "not recommendations" in html.lower()
        assert "manual" in html.lower()
        assert "USD" in html

    def test_dashboard_shows_resolution_rules_and_close_time(self, rendered):
        html = Path(rendered[0]).read_text(encoding="utf-8")
        assert "Settles YES if" in html
        assert "Resolution source" in html
        assert "Settlement" in html

    def test_dashboard_exposes_every_score_component(self, rendered):
        html = Path(rendered[0]).read_text(encoding="utf-8")
        for component in ("Edge", "Liquidity", "Spread", "Annualized", "Momentum"):
            assert component in html

    def test_dashboard_marks_unmodelled_contracts(self, rendered):
        html = Path(rendered[0]).read_text(encoding="utf-8")
        assert "no model" in html.lower()

    def test_digest_lists_candidates_with_reasons(self, config, db, rendered):
        _, signals = rendered
        stats = {"run_id": 1, "total_markets": 8, "tradeable": 5,
                 "with_model": 3, "edge_flagged": 1, "term_days": 30}
        subject, body = build_digest(signals, stats, top_n=5)
        assert "candidates" in subject.lower()
        assert "CANDIDATES FOR MANUAL ANALYSIS" in body
        assert "settles per" in body.lower()
        assert "never places trades" in body.lower()

    def test_empty_signal_set_still_renders(self, tmp_path):
        path = render_dashboard(
            [], {"run_id": 1, "total_markets": 0, "tradeable": 0, "with_model": 0,
                 "edge_flagged": 0, "term_days": 30, "bankroll": "$1,000",
                 "fee_note": "taker 0.07"},
            output_path=tmp_path / "empty.html",
        )
        html = path.read_text(encoding="utf-8")
        assert "No contracts passed" in html


class TestSafety:
    """Structural guarantees, checked against the AST rather than raw text.

    The read-only promise is the whole basis of this system, so it is asserted
    in code rather than left to review: no order-entry call may exist, and no
    Wealthsimple host may appear in any string the code could request.
    """

    def source_files(self):
        root = Path(__file__).resolve().parents[1] / "screener"
        return sorted(root.rglob("*.py"))

    def test_no_order_entry_calls_exist(self):
        import ast

        banned = {"place_order", "create_order", "submit_order", "cancel_order",
                  "batch_create_orders"}
        offenders = []
        for path in self.source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                name = None
                if isinstance(node, ast.Call):
                    func = node.func
                    name = getattr(func, "attr", None) or getattr(func, "id", None)
                elif isinstance(node, ast.FunctionDef):
                    name = node.name
                if name in banned:
                    offenders.append(f"{path.name}:{node.lineno} {name}")
        assert not offenders, f"order-entry surface found: {offenders}"

    def test_no_wealthsimple_host_in_any_string_literal(self):
        """Prose may mention Wealthsimple; a reachable host or URL may not."""
        import ast
        import re

        # A real domain or URL, not the word in a sentence.
        host = re.compile(r"(?://|@|\bhttps?://)?[\w.-]*wealthsimple\.(com|ca)\b", re.I)
        offenders = []
        for path in self.source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if host.search(node.value):
                        offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"Wealthsimple host referenced in: {offenders}"

    def test_client_only_issues_get_requests(self):
        import ast

        from screener.ingest import kalshi

        tree = ast.parse(Path(kalshi.__file__).read_text(encoding="utf-8"))
        verbs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attr = getattr(node.func, "attr", None)
                if attr in {"get", "post", "put", "patch", "delete", "request"}:
                    verbs.add(attr)
        assert verbs <= {"get"}, f"non-GET HTTP verbs in the client: {verbs}"

    def test_forbidden_paths_are_refused_at_runtime(self):
        from screener.ingest.kalshi import KalshiAPIError, KalshiClient

        client = KalshiClient(requests_per_second=0)
        for path in ("/portfolio/orders", "/orders", "/portfolio/positions"):
            with pytest.raises(KalshiAPIError, match="read-only"):
                client._request(path)
