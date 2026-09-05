"""Storage: idempotent upserts that never lose history."""

from __future__ import annotations

import pytest

from screener.store.db import Database


class TestMarkets:
    def test_upsert_is_idempotent(self, db):
        market = {"ticker": "A", "title": "T", "status": "active"}
        db.upsert_markets([market])
        db.upsert_markets([market])
        assert db.scalar("SELECT COUNT(*) FROM markets") == 1

    def test_partial_update_does_not_clobber_known_values(self, db):
        db.upsert_markets([{
            "ticker": "A", "title": "Full title", "rules_primary": "Rules here",
            "can_close_early": True, "floor_strike": 2.9,
        }])
        db.upsert_markets([{"ticker": "A", "status": "active"}])
        row = db.query("SELECT * FROM markets WHERE ticker='A'")[0]
        assert row["title"] == "Full title"
        assert row["rules_primary"] == "Rules here"
        assert row["can_close_early"] == 1
        assert row["floor_strike"] == 2.9
        assert row["status"] == "active"

    def test_changed_values_are_updated(self, db):
        db.upsert_markets([{"ticker": "A", "status": "active"}])
        db.upsert_markets([{"ticker": "A", "status": "settled", "result": "yes"}])
        row = db.query("SELECT * FROM markets WHERE ticker='A'")[0]
        assert row["status"] == "settled"
        assert row["result"] == "yes"

    def test_first_seen_is_preserved(self, db):
        db.upsert_markets([{"ticker": "A", "title": "T"}])
        first = db.query("SELECT first_seen FROM markets")[0]["first_seen"]
        db.upsert_markets([{"ticker": "A", "title": "T2"}])
        assert db.query("SELECT first_seen FROM markets")[0]["first_seen"] == first


class TestSnapshots:
    def test_snapshots_accumulate_over_time(self, db):
        db.upsert_markets([{"ticker": "A"}])
        for i, ts in enumerate(["2026-09-01T00:00:00+00:00", "2026-09-02T00:00:00+00:00"]):
            db.insert_snapshots([{"ticker": "A", "ts": ts, "yes_bid": 60 + i, "yes_ask": 64 + i}])
        assert db.scalar("SELECT COUNT(*) FROM snapshots") == 2

    def test_same_timestamp_is_not_duplicated(self, db):
        db.upsert_markets([{"ticker": "A"}])
        row = {"ticker": "A", "ts": "2026-09-01T00:00:00+00:00", "yes_bid": 60, "yes_ask": 64}
        db.insert_snapshots([row])
        db.insert_snapshots([row])
        assert db.scalar("SELECT COUNT(*) FROM snapshots") == 1

    def test_spread_and_mid_are_derived(self, db):
        db.upsert_markets([{"ticker": "A"}])
        db.insert_snapshots([{"ticker": "A", "ts": "2026-09-01T00:00:00+00:00",
                              "yes_bid": 60, "yes_ask": 64}])
        row = db.query("SELECT spread, mid_price FROM snapshots")[0]
        assert row["spread"] == 4
        assert row["mid_price"] == pytest.approx(62.0)

    def test_missing_prices_leave_derived_null(self, db):
        db.upsert_markets([{"ticker": "A"}])
        db.insert_snapshots([{"ticker": "A", "ts": "2026-09-01T00:00:00+00:00"}])
        row = db.query("SELECT spread, mid_price FROM snapshots")[0]
        assert row["spread"] is None and row["mid_price"] is None

    def test_latest_snapshot_map_returns_the_newest(self, db):
        db.upsert_markets([{"ticker": "A"}])
        db.insert_snapshots([
            {"ticker": "A", "ts": "2026-09-01T00:00:00+00:00", "yes_bid": 50, "yes_ask": 54},
            {"ticker": "A", "ts": "2026-09-03T00:00:00+00:00", "yes_bid": 70, "yes_ask": 74},
        ])
        assert db.latest_snapshot_map()["A"]["yes_bid"] == 70


class TestOtherTables:
    def test_orderbook_levels_are_flattened(self, db):
        db.upsert_markets([{"ticker": "A"}])
        db.insert_orderbook("A", {"yes": [[60, 100], [59, 250]], "no": [[36, 90]]},
                            ts="2026-09-01T00:00:00+00:00")
        levels = db.query("SELECT side, level, price, quantity FROM orderbook_levels "
                          "ORDER BY side, level")
        assert [tuple(r) for r in levels] == [
            ("no", 0, 36, 90), ("yes", 0, 60, 100), ("yes", 1, 59, 250)
        ]

    def test_duplicate_orderbook_is_ignored(self, db):
        db.upsert_markets([{"ticker": "A"}])
        for _ in range(2):
            db.insert_orderbook("A", {"yes": [[60, 100]]}, ts="2026-09-01T00:00:00+00:00")
        assert db.scalar("SELECT COUNT(*) FROM orderbooks") == 1
        assert db.scalar("SELECT COUNT(*) FROM orderbook_levels") == 1

    def test_malformed_orderbook_levels_are_skipped(self, db):
        db.upsert_markets([{"ticker": "A"}])
        db.insert_orderbook("A", {"yes": [[60, 100], "garbage", [None, 5]]},
                            ts="2026-09-01T00:00:00+00:00")
        assert db.scalar("SELECT COUNT(*) FROM orderbook_levels") == 1

    def test_candles_upsert_on_conflict(self, db):
        db.upsert_markets([{"ticker": "A"}])
        candle = {"end_period_ts": 1757030400, "price": {"open": 60, "high": 65,
                  "low": 58, "close": 62}, "volume": 40}
        db.upsert_candles("A", 60, [candle])
        candle["price"]["close"] = 70
        db.upsert_candles("A", 60, [candle])
        rows = db.query("SELECT close_price FROM candles")
        assert len(rows) == 1 and rows[0]["close_price"] == 70

    def test_trades_are_deduplicated_by_id(self, db):
        db.upsert_markets([{"ticker": "A"}])
        trade = {"trade_id": "t1", "created_time": "2026-09-01T00:00:00Z",
                 "yes_price": 62, "count": 10}
        db.upsert_trades("A", [trade])
        db.upsert_trades("A", [trade])
        assert db.scalar("SELECT COUNT(*) FROM trades") == 1

    def test_trades_without_an_id_are_skipped(self, db):
        db.upsert_markets([{"ticker": "A"}])
        assert db.upsert_trades("A", [{"created_time": "x"}]) == 0

    def test_availability_replaces_per_ticker(self, db):
        db.upsert_markets([{"ticker": "A"}])
        db.replace_availability([{"ticker": "A", "tradeable": True, "days_to_close": 5}], 1)
        db.replace_availability([{"ticker": "A", "tradeable": False,
                                  "reason": "closed", "days_to_close": -1}], 2)
        rows = db.query("SELECT tradeable, reason FROM predict_availability")
        assert len(rows) == 1
        assert rows[0]["tradeable"] == 0 and rows[0]["reason"] == "closed"

    def test_signals_upsert_per_run(self, db):
        db.upsert_markets([{"ticker": "A"}])
        db.insert_signals([{"ticker": "A", "score": 0.5,
                            "score_components": {"edge": 0.1}}], run_id=1)
        db.insert_signals([{"ticker": "A", "score": 0.9,
                            "score_components": {"edge": 0.2}}], run_id=1)
        db.insert_signals([{"ticker": "A", "score": 0.7}], run_id=2)
        rows = db.query("SELECT run_id, score, score_components FROM signals ORDER BY run_id")
        assert len(rows) == 2
        assert rows[0]["score"] == 0.9
        assert '"edge": 0.2' in rows[0]["score_components"].replace('"edge":0.2', '"edge": 0.2')


class TestRuns:
    def test_run_lifecycle(self, db):
        run_id = db.start_run("test")
        assert db.query("SELECT status FROM runs")[0]["status"] == "running"
        db.finish_run(run_id, "ok", markets_seen=10, markets_tradeable=3)
        row = db.query("SELECT * FROM runs")[0]
        assert row["status"] == "ok"
        assert row["markets_seen"] == 10
        assert row["finished_at"] is not None

    def test_latest_run_id_ignores_unfinished_runs(self, db):
        first = db.start_run("a")
        db.finish_run(first, "ok")
        db.start_run("b")  # still running
        assert db.latest_run_id() == first


def test_schema_is_created_and_reopenable(tmp_path):
    path = tmp_path / "x.db"
    db = Database(path)
    db.upsert_markets([{"ticker": "A", "title": "T"}])
    db.close()

    reopened = Database(path)
    assert reopened.scalar("SELECT COUNT(*) FROM markets") == 1
    assert reopened.scalar("SELECT value FROM schema_meta WHERE key='version'") is not None
    reopened.close()
