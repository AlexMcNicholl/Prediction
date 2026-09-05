"""Wealthsimple Predict availability filter.

There is no official Kalshi -> Predict mapping and no Wealthsimple API. This
module applies a hand-curated, config-driven allowlist plus the Canadian
30-day term-to-maturity rule, and records WHY each contract was excluded so the
allowlist can be tuned against what the Predict app actually shows.

Nothing here talks to Wealthsimple. It only classifies Kalshi markets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from ..logging_utils import get_logger

log = get_logger(__name__)

TRADEABLE_STATUSES = {"active", "open", "initialized"}


@dataclass(frozen=True)
class AvailabilityResult:
    ticker: str
    tradeable: bool
    reason: str | None
    days_to_close: float | None
    matched_rule: str | None

    def as_record(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "tradeable": self.tradeable,
            "reason": self.reason,
            "days_to_close": self.days_to_close,
            "matched_rule": self.matched_rule,
        }


class PredictFilter:
    """Classifies Kalshi markets as tradeable-on-Predict or not."""

    def __init__(
        self,
        category_allowlist: Iterable[str] = (),
        series_prefix_allowlist: Iterable[str] = (),
        series_allowlist: Iterable[str] = (),
        series_denylist: Iterable[str] = (),
        term_to_maturity_days: int = 30,
        require_open_status: bool = True,
    ) -> None:
        self.categories = {c.strip().lower() for c in category_allowlist if c}
        self.prefixes = tuple(
            sorted((p.strip().upper() for p in series_prefix_allowlist if p), key=len, reverse=True)
        )
        self.series = {s.strip().upper() for s in series_allowlist if s}
        self.denylist = tuple(sorted({d.strip().upper() for d in series_denylist if d}))
        self.term_days = int(term_to_maturity_days)
        self.require_open_status = require_open_status

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PredictFilter":
        return cls(
            category_allowlist=config.get("category_allowlist", []) or [],
            series_prefix_allowlist=config.get("series_prefix_allowlist", []) or [],
            series_allowlist=config.get("series_allowlist", []) or [],
            series_denylist=config.get("series_denylist", []) or [],
            term_to_maturity_days=config.get("term_to_maturity_days", 30),
            require_open_status=config.get("require_open_status", True),
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def days_until(close_time: Any, now: datetime | None = None) -> float | None:
        """Days from ``now`` until ``close_time``. Negative if already past."""
        if not close_time:
            return None
        now = now or datetime.now(timezone.utc)
        if isinstance(close_time, datetime):
            parsed = close_time
        else:
            try:
                parsed = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (parsed - now).total_seconds() / 86400.0

    def _match_allowlist(self, market: Mapping[str, Any]) -> str | None:
        """Return the rule that admitted this market, or None."""
        series = (market.get("series_ticker") or "").strip().upper()
        if series and series in self.series:
            return f"series:{series}"
        if series:
            for prefix in self.prefixes:
                if series.startswith(prefix):
                    return f"prefix:{prefix}"
        category = (market.get("category") or "").strip().lower()
        if category and category in self.categories:
            return f"category:{market.get('category')}"
        return None

    def _denied(self, market: Mapping[str, Any]) -> str | None:
        series = (market.get("series_ticker") or "").strip().upper()
        ticker = (market.get("ticker") or "").strip().upper()
        for entry in self.denylist:
            if series.startswith(entry) or ticker.startswith(entry):
                return entry
        return None

    # ----------------------------------------------------------------- public

    def classify(
        self, market: Mapping[str, Any], now: datetime | None = None
    ) -> AvailabilityResult:
        """Classify one market. Reasons are ordered most-decisive first."""
        ticker = str(market.get("ticker") or "")
        days = self.days_until(market.get("close_time"), now=now)

        denied = self._denied(market)
        if denied:
            return AvailabilityResult(
                ticker, False, f"series on Predict denylist ({denied})", days, None
            )

        status = (market.get("status") or "").strip().lower()
        if self.require_open_status and status and status not in TRADEABLE_STATUSES:
            return AvailabilityResult(ticker, False, f"market status is {status!r}", days, None)

        matched = self._match_allowlist(market)
        if not matched:
            series = market.get("series_ticker") or "?"
            return AvailabilityResult(
                ticker, False,
                f"series {series} not on the Predict allowlist "
                "(econ / financial / climate only)",
                days, None,
            )

        if days is None:
            return AvailabilityResult(
                ticker, False, "no close_time on the market payload", None, matched
            )
        if days < 0:
            return AvailabilityResult(
                ticker, False, f"already closed ({abs(days):.1f} days ago)", days, matched
            )
        if days > self.term_days:
            return AvailabilityResult(
                ticker, False,
                f"closes in {days:.1f} days, beyond the {self.term_days}-day "
                "Canadian term-to-maturity limit",
                days, matched,
            )

        return AvailabilityResult(ticker, True, None, days, matched)

    def classify_all(
        self, markets: Sequence[Mapping[str, Any]], now: datetime | None = None
    ) -> list[AvailabilityResult]:
        now = now or datetime.now(timezone.utc)
        results = [self.classify(m, now=now) for m in markets]
        tradeable = sum(1 for r in results if r.tradeable)
        log.info(
            "Predict filter: %d/%d markets tradeable (<=%d days to close)",
            tradeable, len(results), self.term_days,
        )
        return results

    def summarize_exclusions(
        self, results: Sequence[AvailabilityResult], top_n: int = 8
    ) -> list[tuple[str, int]]:
        """Most common exclusion reasons - useful for tuning the allowlist."""
        counts: dict[str, int] = {}
        for r in results:
            if r.tradeable or not r.reason:
                continue
            # Collapse the variable numbers out so reasons group together.
            key = r.reason.split("(")[0].strip()
            if "beyond the" in key:
                key = f"beyond the {self.term_days}-day term-to-maturity limit"
            elif key.startswith("series ") and "not on the Predict allowlist" in key:
                key = "series not on the Predict allowlist"
            elif key.startswith("market status is"):
                key = "market status not open"
            counts[key] = counts.get(key, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
