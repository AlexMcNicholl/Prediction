"""Read-only client for Kalshi's public market-data API.

Scope note: this client wraps MARKET-DATA endpoints only. It contains no
authentication, no order entry, and no portfolio access by design - the
screener must never be able to place a trade. Kalshi's market-data endpoints
are publicly readable; only trading requires signed auth, which we do not use.

Verified 2026-09 against the public API reference. Run
``python -m screener.cli verify-api`` to re-confirm the live contract.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Mapping, Sequence

import requests

from ..logging_utils import get_logger

log = get_logger(__name__)

# Endpoints that must never be reachable from this client. Guarded in _request
# so a future edit cannot accidentally introduce a trading call path.
_FORBIDDEN_PATH_FRAGMENTS = ("/portfolio", "/orders", "/positions", "/fills")


class KalshiAPIError(RuntimeError):
    """Raised when the API returns an unrecoverable error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimiter:
    """Adaptive throttle: enforce a minimum gap, and back off on 429s.

    A fixed rate is a guess, and guessing high gets you rate-limited: the first
    live run of this screener hit repeated 429s at 8 req/s unauthenticated. So
    the limiter halves its own rate every time the server pushes back, down to
    ``floor_rps``, and holds that slower rate for the rest of the run rather
    than immediately creeping back up and getting throttled again.
    """

    def __init__(self, requests_per_second: float, floor_rps: float = 0.5) -> None:
        self.base_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self.max_interval = 1.0 / floor_rps if floor_rps > 0 else 0.0
        self.min_interval = self.base_interval
        self.throttle_events = 0
        self._last: float = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()

    def penalize(self) -> bool:
        """Halve the request rate after a 429. True if the rate actually changed."""
        if self.base_interval <= 0:
            return False
        previous = self.min_interval
        self.min_interval = min(self.min_interval * 2.0, self.max_interval)
        self.throttle_events += 1
        return self.min_interval > previous

    @property
    def current_rps(self) -> float:
        return 1.0 / self.min_interval if self.min_interval > 0 else float("inf")


@dataclass
class ClientStats:
    requests: int = 0
    retries: int = 0
    errors: int = 0
    pages: int = 0
    rate_limited: int = 0


@dataclass
class KalshiClient:
    """HTTP client for Kalshi public market data."""

    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    fallback_base_urls: Sequence[str] = field(default_factory=tuple)
    timeout: float = 30.0
    max_retries: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    # Kalshi 429s unauthenticated readers well below 8 req/s (observed live).
    requests_per_second: float = 4.0
    floor_requests_per_second: float = 0.5
    page_limit: int = 200
    user_agent: str = "prediction-screener/1.0 (read-only market-data research)"
    session: requests.Session = field(default_factory=requests.Session)
    stats: ClientStats = field(default_factory=ClientStats)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.fallback_base_urls = tuple(u.rstrip("/") for u in self.fallback_base_urls)
        self._limiter = RateLimiter(
            self.requests_per_second, self.floor_requests_per_second
        )
        self.session.headers.update(
            {"User-Agent": self.user_agent, "Accept": "application/json"}
        )

    # ------------------------------------------------------------ transport

    def _request(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """GET ``path`` with retry/backoff. Only GET is ever issued."""
        lowered = path.lower()
        for fragment in _FORBIDDEN_PATH_FRAGMENTS:
            if fragment in lowered:
                raise KalshiAPIError(
                    f"Refusing to call non-market-data endpoint {path!r}. "
                    "This screener is read-only by design."
                )

        clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        bases = (self.base_url,) + tuple(self.fallback_base_urls)
        last_error: Exception | None = None

        for base_index, base in enumerate(bases):
            url = f"{base}{path}"
            for attempt in range(self.max_retries + 1):
                self._limiter.wait()
                self.stats.requests += 1
                try:
                    resp = self.session.get(url, params=clean, timeout=self.timeout)
                except requests.RequestException as exc:
                    last_error = exc
                    self.stats.errors += 1
                    if attempt >= self.max_retries:
                        break
                    self._sleep_backoff(attempt, reason=f"network error: {exc}")
                    self.stats.retries += 1
                    continue

                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise KalshiAPIError(
                            f"Non-JSON response from {url}: {resp.text[:200]!r}"
                        ) from exc

                # 429 and 5xx are retryable; 4xx (other than 429) are not.
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = KalshiAPIError(
                        f"HTTP {resp.status_code} from {url}: {resp.text[:200]}",
                        resp.status_code,
                    )
                    self.stats.errors += 1
                    if resp.status_code == 429:
                        self.stats.rate_limited += 1
                    if resp.status_code == 429 and self._limiter.penalize():
                        log.warning(
                            "rate limited; slowing to %.1f req/s for the rest of "
                            "this run", self._limiter.current_rps,
                        )
                    if attempt >= self.max_retries:
                        break
                    retry_after = self._retry_after_seconds(resp)
                    self._sleep_backoff(
                        attempt, reason=f"HTTP {resp.status_code}", override=retry_after
                    )
                    self.stats.retries += 1
                    continue

                raise KalshiAPIError(
                    f"HTTP {resp.status_code} from {url}: {resp.text[:200]}",
                    resp.status_code,
                )

            if base_index + 1 < len(bases):
                log.warning("base %s exhausted retries, trying fallback base", base)

        raise KalshiAPIError(f"Request to {path} failed after retries: {last_error}")

    @staticmethod
    def _retry_after_seconds(resp: requests.Response) -> float | None:
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    def _sleep_backoff(
        self, attempt: int, reason: str, override: float | None = None
    ) -> None:
        if override is not None:
            delay = min(override, self.backoff_max)
        else:
            # Exponential backoff with full jitter.
            ceiling = min(self.backoff_base * (2 ** attempt), self.backoff_max)
            delay = random.uniform(0, ceiling)
        log.warning("retrying in %.2fs (attempt %d): %s", delay, attempt + 1, reason)
        time.sleep(delay)

    # ------------------------------------------------------------ pagination

    def paginate(
        self,
        path: str,
        item_key: str,
        params: Mapping[str, Any] | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield every item across all pages of a cursor-paginated endpoint.

        Kalshi returns a ``cursor`` field; an empty or missing cursor means the
        last page. We also guard against a server echoing the same cursor back,
        which would otherwise loop forever.
        """
        cursor: str | None = None
        seen_cursors: set[str] = set()
        pages = 0

        while True:
            call_params = dict(params or {})
            call_params.setdefault("limit", self.page_limit)
            if cursor:
                call_params["cursor"] = cursor

            payload = self._request(path, call_params)
            pages += 1
            self.stats.pages += 1

            items = payload.get(item_key) or []
            if not isinstance(items, list):
                raise KalshiAPIError(
                    f"Expected list under {item_key!r} from {path}, got {type(items).__name__}"
                )
            for item in items:
                yield item

            cursor = (payload.get("cursor") or "").strip() or None
            log.debug("%s page %d: %d items, next_cursor=%s", path, pages, len(items), cursor)

            if not cursor:
                return
            if cursor in seen_cursors:
                log.warning("cursor repeated on %s; stopping to avoid an infinite loop", path)
                return
            seen_cursors.add(cursor)
            if not items:
                # A cursor with no items would otherwise spin.
                log.warning("empty page with a cursor on %s; stopping", path)
                return
            if max_pages is not None and pages >= max_pages:
                log.warning("hit max_pages=%d on %s; results truncated", max_pages, path)
                return

    # --------------------------------------------------------- market data

    def get_exchange_status(self) -> dict[str, Any]:
        return self._request("/exchange/status")

    def get_markets(
        self,
        status: str | None = "open",
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        """All markets matching the filters, following every page."""
        params = {
            "status": status,
            "series_ticker": series_ticker,
            "event_ticker": event_ticker,
            "min_close_ts": min_close_ts,
            "max_close_ts": max_close_ts,
        }
        markets = list(self.paginate("/markets", "markets", params, max_pages=max_pages))
        log.info("fetched %d markets (status=%s)", len(markets), status)
        return markets

    def get_market(self, ticker: str) -> dict[str, Any]:
        payload = self._request(f"/markets/{ticker}")
        return payload.get("market", payload)

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        payload = self._request(f"/markets/{ticker}/orderbook", {"depth": depth})
        return payload.get("orderbook") or {}

    def get_trades(self, ticker: str, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._request("/markets/trades", {"ticker": ticker, "limit": limit})
        return payload.get("trades") or []

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        period_minutes: int = 60,
        lookback_hours: int = 336,
    ) -> list[dict[str, Any]]:
        """OHLC history. Kalshi nests this under the series in the path."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        payload = self._request(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            {
                "start_ts": int(start.timestamp()),
                "end_ts": int(end.timestamp()),
                "period_interval": period_minutes,
            },
        )
        return payload.get("candlesticks") or []

    def get_events(
        self, status: str | None = "open", with_nested_markets: bool = False,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        params = {"status": status, "with_nested_markets": with_nested_markets or None}
        return list(self.paginate("/events", "events", params, max_pages=max_pages))

    def get_series_list(self, category: str | None = None) -> list[dict[str, Any]]:
        payload = self._request("/series", {"category": category})
        return payload.get("series") or []

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        payload = self._request(f"/series/{series_ticker}")
        return payload.get("series", payload)
