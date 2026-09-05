"""The API verification report must distinguish drift from unreachability."""

from __future__ import annotations

import responses

from screener.ingest.kalshi import KalshiClient
from screener.ingest.verify import PREREQ, VerifyReport, verify

BASE = "https://api.test.invalid/trade-api/v2"


class TestReport:
    def test_network_failures_read_as_unreachable(self):
        report = VerifyReport()
        report.add("a", False, "API error: ProxyError('Unable to connect to proxy')")
        report.add("b", False, f"{PREREQ}: /markets returned no sample")
        assert report.unreachable
        assert "UNREACHABLE" in report.render()
        assert "drifted" not in report.render()

    def test_field_drift_is_not_reported_as_unreachable(self):
        report = VerifyReport()
        report.add("market field names", False, "MISSING required-ish fields: yes_bid")
        assert not report.unreachable
        assert "drifted" in report.render()

    def test_a_clean_report_says_so(self):
        report = VerifyReport()
        report.add("a", True, "fine")
        assert not report.failed and not report.unreachable
        assert "All checks passed" in report.render()

    def test_warnings_do_not_count_as_failures(self):
        report = VerifyReport()
        report.add("a", True, "fine")
        report.add("b", False, "empty book", warning=True)
        assert not report.failed
        assert "1 warning(s)" in report.render()


@responses.activate
def test_verify_passes_against_a_conforming_api():
    market = {
        "ticker": "KXCPI-26SEP-T2.9", "event_ticker": "KXCPI-26SEP",
        "series_ticker": "KXCPI", "title": "CPI above 2.9%?", "status": "active",
        "close_time": "2026-09-20T12:00:00Z", "yes_bid": 60, "yes_ask": 64,
        "no_bid": 36, "no_ask": 40, "last_price": 62, "volume": 1200,
        "open_interest": 800, "liquidity": 50000, "rules_primary": "Settles per BLS.",
    }
    responses.add(responses.GET, f"{BASE}/exchange/status",
                  json={"trading_active": True, "exchange_active": True})
    responses.add(responses.GET, f"{BASE}/markets",
                  json={"markets": [market], "cursor": ""})
    responses.add(responses.GET, f"{BASE}/markets/{market['ticker']}",
                  json={"market": market})
    responses.add(responses.GET, f"{BASE}/markets/{market['ticker']}/orderbook",
                  json={"orderbook": {"yes": [[60, 100]], "no": [[36, 90]]}})
    responses.add(responses.GET, f"{BASE}/markets/trades",
                  json={"trades": [{"trade_id": "t1", "yes_price": 62}]})
    responses.add(
        responses.GET,
        f"{BASE}/series/KXCPI/markets/{market['ticker']}/candlesticks",
        json={"candlesticks": [{"end_period_ts": 1757030400}]},
    )
    responses.add(responses.GET, f"{BASE}/events",
                  json={"events": [{"event_ticker": "KXCPI-26SEP"}], "cursor": ""})
    responses.add(responses.GET, f"{BASE}/series", json={"series": [{"ticker": "KXCPI"}]})

    client = KalshiClient(base_url=BASE, requests_per_second=0, max_retries=0,
                          backoff_base=0.0)
    report = verify(client)
    assert not report.failed, [(c.name, c.detail) for c in report.failed]
    assert not report.unreachable


@responses.activate
def test_verify_flags_a_renamed_price_field():
    market = {
        "ticker": "T", "event_ticker": "E", "title": "x", "status": "active",
        "close_time": "2026-09-20T12:00:00Z", "no_bid": 36, "no_ask": 40,
        "last_price": 62, "volume": 1, "open_interest": 1, "liquidity": 1,
        "rules_primary": "r",
        # yes_bid / yes_ask renamed upstream:
        "bid_yes": 60, "ask_yes": 64,
    }
    responses.add(responses.GET, f"{BASE}/exchange/status", json={})
    responses.add(responses.GET, f"{BASE}/markets", json={"markets": [market], "cursor": ""})
    responses.add(responses.GET, f"{BASE}/markets/T", json={"market": market})
    responses.add(responses.GET, f"{BASE}/markets/T/orderbook", json={"orderbook": {}})
    responses.add(responses.GET, f"{BASE}/markets/trades", json={"trades": []})
    responses.add(responses.GET, f"{BASE}/series/T/markets/T/candlesticks",
                  json={"candlesticks": []})
    responses.add(responses.GET, f"{BASE}/events", json={"events": [], "cursor": ""})
    responses.add(responses.GET, f"{BASE}/series", json={"series": []})

    client = KalshiClient(base_url=BASE, requests_per_second=0, max_retries=0,
                          backoff_base=0.0)
    report = verify(client)
    failures = {c.name: c.detail for c in report.failed}
    assert "market field names" in failures
    assert "yes_bid" in failures["market field names"]
    # A renamed field is drift, not a connectivity problem.
    assert not report.unreachable
    assert "new/unmapped fields" in failures["market field names"]
