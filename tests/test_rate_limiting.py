"""Adaptive rate limiting and the enrichment time budget.

Both were added after the first live run: Kalshi 429'd an unauthenticated
reader repeatedly at 8 req/s, and an unbounded enrichment loop under those
conditions would outlast the CI job timeout.
"""

from __future__ import annotations

import pytest
import responses

from screener.ingest.kalshi import KalshiClient, RateLimiter

BASE = "https://api.test.invalid/trade-api/v2"


class TestRateLimiter:
    def test_starts_at_the_configured_rate(self):
        assert RateLimiter(4.0).current_rps == pytest.approx(4.0)

    def test_a_429_halves_the_rate(self):
        limiter = RateLimiter(4.0, floor_rps=0.5)
        assert limiter.penalize() is True
        assert limiter.current_rps == pytest.approx(2.0)
        limiter.penalize()
        assert limiter.current_rps == pytest.approx(1.0)

    def test_the_rate_never_drops_below_the_floor(self):
        limiter = RateLimiter(4.0, floor_rps=0.5)
        for _ in range(20):
            limiter.penalize()
        assert limiter.current_rps == pytest.approx(0.5)

    def test_penalize_reports_when_it_stops_changing(self):
        limiter = RateLimiter(1.0, floor_rps=0.5)
        assert limiter.penalize() is True   # 1.0 -> 0.5
        assert limiter.penalize() is False  # already at the floor

    def test_events_are_counted(self):
        limiter = RateLimiter(4.0)
        limiter.penalize()
        limiter.penalize()
        assert limiter.throttle_events == 2

    def test_an_unthrottled_limiter_is_a_no_op(self):
        limiter = RateLimiter(0)
        assert limiter.penalize() is False
        limiter.wait()  # must not raise or sleep


class TestClientBackoff:
    @responses.activate
    def test_a_429_slows_the_client_for_the_rest_of_the_run(self):
        responses.add(responses.GET, f"{BASE}/markets", json={"e": 1}, status=429)
        responses.add(responses.GET, f"{BASE}/markets",
                      json={"markets": [{"ticker": "A"}], "cursor": ""}, status=200)

        client = KalshiClient(base_url=BASE, requests_per_second=100.0,
                              floor_requests_per_second=1.0, max_retries=2,
                              backoff_base=0.0, backoff_max=0.0)
        before = client._limiter.current_rps
        client.get_markets()
        assert client._limiter.current_rps < before
        assert client.stats.rate_limited == 1

    @responses.activate
    def test_repeated_429s_compound_the_slowdown(self):
        for _ in range(3):
            responses.add(responses.GET, f"{BASE}/markets", json={"e": 1}, status=429)
        responses.add(responses.GET, f"{BASE}/markets",
                      json={"markets": [], "cursor": ""}, status=200)

        client = KalshiClient(base_url=BASE, requests_per_second=100.0,
                              floor_requests_per_second=1.0, max_retries=5,
                              backoff_base=0.0, backoff_max=0.0)
        client.get_markets()
        assert client.stats.rate_limited == 3
        assert client._limiter.current_rps == pytest.approx(12.5)

    @responses.activate
    def test_server_errors_do_not_trigger_the_rate_penalty(self):
        """A 500 is not backpressure - only 429 should slow us down."""
        responses.add(responses.GET, f"{BASE}/markets", json={"e": 1}, status=503)
        responses.add(responses.GET, f"{BASE}/markets",
                      json={"markets": [], "cursor": ""}, status=200)

        client = KalshiClient(base_url=BASE, requests_per_second=10.0,
                              max_retries=2, backoff_base=0.0, backoff_max=0.0)
        client.get_markets()
        assert client.stats.rate_limited == 0
        assert client._limiter.current_rps == pytest.approx(10.0)

    def test_default_rate_is_below_the_level_that_got_throttled(self):
        """8 req/s produced sustained 429s live; the default must be lower."""
        assert KalshiClient().requests_per_second < 8.0


class TestEnrichmentBudget:
    def test_budget_stops_enrichment_and_records_why(self, config, db, fake_client,
                                                     sample_markets, monkeypatch):
        from screener.ingest import runner

        config.data["ingest"]["enrichment_budget_seconds"] = 0.0001
        run_id = db.start_run("t")
        result = runner.ingest(config, db, client=fake_client, run_id=run_id)

        assert any("budget exhausted" in e for e in result.errors)
        # The run still completed and stored markets - a partial enrichment
        # beats a timeout that produces nothing.
        assert result.markets_seen > 0
        assert db.scalar("SELECT COUNT(*) FROM markets") > 0

    def test_a_generous_budget_enriches_everything(self, config, db, fake_client):
        from screener.ingest import runner

        config.data["ingest"]["enrichment_budget_seconds"] = 600
        run_id = db.start_run("t")
        result = runner.ingest(config, db, client=fake_client, run_id=run_id)

        assert not any("budget exhausted" in e for e in result.errors)
        assert result.orderbooks == result.markets_tradeable

    def test_budget_disabled_when_zero(self, config, db, fake_client):
        from screener.ingest import runner

        config.data["ingest"]["enrichment_budget_seconds"] = 0
        run_id = db.start_run("t")
        result = runner.ingest(config, db, client=fake_client, run_id=run_id)
        assert not any("budget exhausted" in e for e in result.errors)
