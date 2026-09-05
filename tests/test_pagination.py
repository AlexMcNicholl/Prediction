"""Cursor pagination must fetch EVERY page, not just page one."""

from __future__ import annotations

import pytest
import responses

from screener.ingest.kalshi import KalshiAPIError, KalshiClient

BASE = "https://api.test.invalid/trade-api/v2"


def make_client(**kwargs) -> KalshiClient:
    defaults = {
        "base_url": BASE, "requests_per_second": 0, "max_retries": 1,
        "backoff_base": 0.0, "backoff_max": 0.0,
    }
    return KalshiClient(**{**defaults, **kwargs})


@responses.activate
def test_paginate_follows_every_page():
    pages = [
        ({"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "c1"}),
        ({"markets": [{"ticker": "C"}, {"ticker": "D"}], "cursor": "c2"}),
        ({"markets": [{"ticker": "E"}], "cursor": ""}),
    ]
    for page in pages:
        responses.add(responses.GET, f"{BASE}/markets", json=page, status=200)

    tickers = [m["ticker"] for m in make_client().get_markets(status="open")]
    assert tickers == ["A", "B", "C", "D", "E"]
    assert len(responses.calls) == 3


@responses.activate
def test_paginate_passes_cursor_through():
    responses.add(responses.GET, f"{BASE}/markets",
                  json={"markets": [{"ticker": "A"}], "cursor": "NEXT"}, status=200)
    responses.add(responses.GET, f"{BASE}/markets",
                  json={"markets": [{"ticker": "B"}], "cursor": None}, status=200)

    make_client().get_markets(status="open")
    assert "cursor" not in responses.calls[0].request.url
    assert "cursor=NEXT" in responses.calls[1].request.url


@responses.activate
def test_missing_cursor_key_ends_pagination():
    responses.add(responses.GET, f"{BASE}/markets",
                  json={"markets": [{"ticker": "A"}]}, status=200)
    assert len(make_client().get_markets()) == 1


@responses.activate
def test_repeated_cursor_does_not_loop_forever():
    """A server echoing the same cursor must not hang the run."""
    for _ in range(5):
        responses.add(responses.GET, f"{BASE}/markets",
                      json={"markets": [{"ticker": "A"}], "cursor": "SAME"}, status=200)

    markets = make_client().get_markets()
    assert len(markets) == 2  # first page, then the repeat is detected
    assert len(responses.calls) == 2


@responses.activate
def test_empty_page_with_cursor_stops():
    responses.add(responses.GET, f"{BASE}/markets",
                  json={"markets": [], "cursor": "c1"}, status=200)
    assert make_client().get_markets() == []
    assert len(responses.calls) == 1


@responses.activate
def test_max_pages_truncates():
    for i in range(10):
        responses.add(responses.GET, f"{BASE}/markets",
                      json={"markets": [{"ticker": f"T{i}"}], "cursor": f"c{i}"}, status=200)
    assert len(make_client().get_markets(max_pages=3)) == 3


@responses.activate
def test_limit_defaults_to_configured_page_size():
    responses.add(responses.GET, f"{BASE}/markets", json={"markets": [], "cursor": ""}, status=200)
    make_client(page_limit=200).get_markets()
    assert "limit=200" in responses.calls[0].request.url


@responses.activate
def test_retries_on_429_then_succeeds():
    responses.add(responses.GET, f"{BASE}/markets", json={"error": "rate limited"}, status=429)
    responses.add(responses.GET, f"{BASE}/markets",
                  json={"markets": [{"ticker": "A"}], "cursor": ""}, status=200)

    assert len(make_client().get_markets()) == 1
    assert len(responses.calls) == 2


@responses.activate
def test_retries_on_500_then_gives_up():
    for _ in range(6):
        responses.add(responses.GET, f"{BASE}/markets", json={"error": "boom"}, status=500)
    with pytest.raises(KalshiAPIError):
        make_client(max_retries=2).get_markets()


@responses.activate
def test_client_error_is_not_retried():
    responses.add(responses.GET, f"{BASE}/markets", json={"error": "bad request"}, status=400)
    with pytest.raises(KalshiAPIError) as exc:
        make_client().get_markets()
    assert exc.value.status_code == 400
    assert len(responses.calls) == 1  # 4xx must not burn retries


@responses.activate
def test_falls_back_to_secondary_base_url():
    alt = "https://alt.test.invalid/trade-api/v2"
    responses.add(responses.GET, f"{BASE}/markets", json={"e": 1}, status=503)
    responses.add(responses.GET, f"{alt}/markets",
                  json={"markets": [{"ticker": "A"}], "cursor": ""}, status=200)

    client = KalshiClient(base_url=BASE, fallback_base_urls=(alt,), requests_per_second=0,
                          max_retries=0, backoff_base=0.0)
    assert len(client.get_markets()) == 1


@responses.activate
def test_non_list_payload_raises():
    responses.add(responses.GET, f"{BASE}/markets", json={"markets": {"oops": 1}}, status=200)
    with pytest.raises(KalshiAPIError, match="Expected list"):
        make_client().get_markets()


def test_trading_endpoints_are_refused():
    """The client must be structurally incapable of reaching order entry."""
    client = make_client()
    for path in ("/portfolio/orders", "/portfolio/positions", "/orders", "/portfolio/fills"):
        with pytest.raises(KalshiAPIError, match="read-only"):
            client._request(path)


@responses.activate
def test_no_auth_header_is_sent():
    responses.add(responses.GET, f"{BASE}/markets", json={"markets": [], "cursor": ""}, status=200)
    make_client().get_markets()
    assert "Authorization" not in responses.calls[0].request.headers
