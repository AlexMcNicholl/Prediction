"""Live API verification.

The build was written against the public Kalshi reference, but APIs change and
this repo's development sandbox could not reach Kalshi directly. Run this
first - and after any Kalshi change - to confirm the contract this code
assumes: base URL, endpoint paths, cursor pagination, field names, and that no
authentication is required for market data.

    python -m screener.cli verify-api

Every check is independent; one failure does not stop the rest, so a single
run tells you everything that drifted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..logging_utils import get_logger
from .kalshi import KalshiAPIError, KalshiClient

log = get_logger(__name__)

#: Fields the screener reads off a market payload. Missing ones degrade to
#: NULL rather than crashing, but you want to know.
EXPECTED_MARKET_FIELDS = [
    "ticker", "event_ticker", "title", "status", "close_time",
    "yes_bid", "yes_ask", "no_bid", "no_ask", "last_price",
    "volume", "open_interest", "liquidity", "rules_primary",
]

#: Marks a check that could not run because an earlier fetch produced nothing.
#: These are cascade failures, not independent evidence that the API changed.
PREREQ = "prerequisite check failed"

OPTIONAL_MARKET_FIELDS = [
    "subtitle", "yes_sub_title", "no_sub_title", "category", "open_time",
    "expiration_time", "expected_expiration_time", "rules_secondary",
    "strike_type", "floor_strike", "cap_strike", "volume_24h",
    "previous_price", "can_close_early", "market_type", "result",
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    warning: bool = False


@dataclass
class VerifyReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str, warning: bool = False) -> None:
        self.checks.append(Check(name, ok, detail, warning))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.warning]

    @property
    def unreachable(self) -> bool:
        """True when the failures look like a blocked host, not API drift."""
        failures = self.failed
        if not failures:
            return False
        # "sample" catches the checks that only failed because an earlier
        # fetch produced nothing to inspect - they are cascade failures, not
        # independent evidence of drift.
        markers = ("proxyerror", "connection", "timed out", "timeout",
                   "max retries exceeded", "name or service not known",
                   PREREQ)
        return all(
            any(m in str(c.detail).lower() for m in markers) for c in failures
        )

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.warning]

    def render(self) -> str:
        lines = ["", "Kalshi API verification", "=" * 60]
        for check in self.checks:
            mark = "PASS" if check.ok else ("WARN" if check.warning else "FAIL")
            lines.append(f"[{mark}] {check.name}")
            for piece in str(check.detail).splitlines():
                lines.append(f"       {piece}")
        lines.append("=" * 60)
        if self.failed and self.unreachable:
            lines.append(
                f"{len(self.failed)} check(s) could not run: the API was UNREACHABLE "
                "from this machine."
            )
            lines.append(
                "This is a connectivity problem, not API drift - nothing was verified "
                "either way."
            )
        elif self.failed:
            lines.append(
                f"{len(self.failed)} check(s) FAILED - the code's assumptions have drifted."
            )
        elif self.warnings:
            lines.append(f"All required checks passed, with {len(self.warnings)} warning(s).")
        else:
            lines.append("All checks passed.")
        lines.append("")
        return "\n".join(lines)


def _run(report: VerifyReport, name: str, fn: Callable[[], tuple[bool, str, bool]]) -> Any:
    try:
        ok, detail, warning = fn()
        report.add(name, ok, detail, warning)
    except KalshiAPIError as exc:
        report.add(name, False, f"API error: {exc}")
    except Exception as exc:
        report.add(name, False, f"unexpected error: {type(exc).__name__}: {exc}")


def verify(client: KalshiClient) -> VerifyReport:
    """Probe the live API and report what matches and what drifted."""
    report = VerifyReport()
    state: dict[str, Any] = {}

    report.add(
        "no authentication configured",
        "Authorization" not in client.session.headers,
        "The client sends no auth header. Market data must be publicly readable; "
        "if these checks 401, Kalshi has changed that and the design needs revisiting.",
    )

    def check_status() -> tuple[bool, str, bool]:
        payload = client.get_exchange_status()
        keys = ", ".join(sorted(payload)[:8])
        return True, f"{client.base_url}/exchange/status -> keys: {keys}", False

    _run(report, "GET /exchange/status", check_status)

    def check_markets() -> tuple[bool, str, bool]:
        payload = client._request("/markets", {"status": "open", "limit": 5})
        markets = payload.get("markets")
        if not isinstance(markets, list) or not markets:
            return False, f"expected a non-empty 'markets' list, got {type(markets).__name__}", False
        state["sample"] = markets[0]
        state["cursor"] = payload.get("cursor")
        has_cursor = "cursor" in payload
        return (
            True,
            f"returned {len(markets)} markets; 'cursor' key present: {has_cursor}",
            False,
        )

    _run(report, "GET /markets (status=open)", check_markets)

    def check_fields() -> tuple[bool, str, bool]:
        sample = state.get("sample")
        if not sample:
            return False, f"{PREREQ}: /markets returned no sample to inspect", False
        missing = [f for f in EXPECTED_MARKET_FIELDS if f not in sample]
        missing_opt = [f for f in OPTIONAL_MARKET_FIELDS if f not in sample]
        unknown = sorted(
            set(sample) - set(EXPECTED_MARKET_FIELDS) - set(OPTIONAL_MARKET_FIELDS)
        )
        detail = [f"sample ticker: {sample.get('ticker')}"]
        if missing:
            detail.append(f"MISSING required-ish fields: {', '.join(missing)}")
        if missing_opt:
            detail.append(f"absent optional fields: {', '.join(missing_opt)}")
        if unknown:
            detail.append(f"new/unmapped fields (kept in raw_json): {', '.join(unknown)}")
        if not missing:
            detail.append("all fields the screener reads are present.")
        return (not missing), "\n".join(detail), False

    _run(report, "market field names", check_fields)

    def check_prices() -> tuple[bool, str, bool]:
        sample = state.get("sample")
        if not sample:
            return False, f"{PREREQ}: no sample market to price-check", False
        prices = {k: sample.get(k) for k in ("yes_bid", "yes_ask", "no_bid", "no_ask", "last_price")}
        numeric = [v for v in prices.values() if isinstance(v, (int, float))]
        if not numeric:
            return False, f"no numeric price fields on the sample market: {prices}", False
        in_cents = all(0 <= v <= 100 for v in numeric)
        detail = f"{prices}"
        if in_cents:
            return True, f"all prices within 0-100, consistent with cents. {detail}", False
        return False, f"prices outside 0-100 - the cents assumption may be wrong. {detail}", False

    _run(report, "prices are integer cents", check_prices)

    def check_pagination() -> tuple[bool, str, bool]:
        first = client._request("/markets", {"status": "open", "limit": 2})
        cursor = (first.get("cursor") or "").strip()
        first_tickers = [m.get("ticker") for m in first.get("markets", [])]
        if not cursor:
            return (
                True,
                "only one page available right now, so paging could not be exercised. "
                f"first page: {first_tickers}",
                True,
            )
        second = client._request("/markets", {"status": "open", "limit": 2, "cursor": cursor})
        second_tickers = [m.get("ticker") for m in second.get("markets", [])]
        overlap = set(first_tickers) & set(second_tickers)
        if overlap:
            return False, f"page 2 repeated tickers from page 1: {overlap}", False
        return (
            True,
            f"cursor paging advances correctly.\npage 1: {first_tickers}\npage 2: {second_tickers}",
            False,
        )

    _run(report, "cursor pagination", check_pagination)

    def check_single_market() -> tuple[bool, str, bool]:
        sample = state.get("sample") or {}
        ticker = sample.get("ticker")
        if not ticker:
            return False, f"{PREREQ}: no sample ticker available", False
        market = client.get_market(ticker)
        return bool(market.get("ticker")), f"GET /markets/{ticker} -> {market.get('ticker')}", False

    _run(report, "GET /markets/{ticker}", check_single_market)

    def check_orderbook() -> tuple[bool, str, bool]:
        ticker = (state.get("sample") or {}).get("ticker")
        if not ticker:
            return False, f"{PREREQ}: no sample ticker available", False
        book = client.get_orderbook(ticker)
        sides = sorted(k for k in book if k in ("yes", "no"))
        shape = ""
        for side in sides:
            levels = book.get(side) or []
            if levels:
                shape = f" e.g. {side}[0] = {levels[0]}"
                break
        return (
            bool(sides),
            f"sides returned: {sides or 'none (book may be empty)'}{shape}",
            not sides,
        )

    _run(report, "GET /markets/{ticker}/orderbook", check_orderbook)

    def check_trades() -> tuple[bool, str, bool]:
        ticker = (state.get("sample") or {}).get("ticker")
        if not ticker:
            return False, f"{PREREQ}: no sample ticker available", False
        trades = client.get_trades(ticker, limit=5)
        if not trades:
            return True, "endpoint reachable but returned no trades for this market", True
        keys = ", ".join(sorted(trades[0])[:10])
        return True, f"{len(trades)} trades; fields: {keys}", False

    _run(report, "GET /markets/trades", check_trades)

    def check_candles() -> tuple[bool, str, bool]:
        sample = state.get("sample") or {}
        ticker = sample.get("ticker")
        series = sample.get("series_ticker") or (
            str(ticker).split("-", 1)[0] if ticker else None
        )
        if not (ticker and series):
            return False, f"{PREREQ}: no sample ticker/series available", False
        candles = client.get_candlesticks(str(series), str(ticker), 60, 168)
        if not candles:
            return True, (
                f"path /series/{series}/markets/{ticker}/candlesticks reachable "
                "but returned no candles for this window"
            ), True
        keys = ", ".join(sorted(candles[0])[:10])
        return True, f"{len(candles)} candles; fields: {keys}", False

    _run(report, "GET /series/{s}/markets/{t}/candlesticks", check_candles)

    def check_events() -> tuple[bool, str, bool]:
        events = client.get_events(status="open", max_pages=1)
        if not events:
            return True, "endpoint reachable, no open events returned", True
        keys = ", ".join(sorted(events[0])[:10])
        return True, f"{len(events)} events on page 1; fields: {keys}", False

    _run(report, "GET /events", check_events)

    def check_series() -> tuple[bool, str, bool]:
        series = client.get_series_list()
        if not series:
            return True, "endpoint reachable, no series returned (may need a category)", True
        return True, f"{len(series)} series returned", False

    _run(report, "GET /series", check_series)

    report.add(
        "client request stats",
        True,
        f"{client.stats.requests} requests, {client.stats.retries} retries, "
        f"{client.stats.errors} error responses. If retries are high, lower "
        "api.requests_per_second in config.yaml.",
    )

    return report
