"""Equity-index / asset-price fair-value estimator.

Converts a price-threshold contract into a probability with a lognormal
(Black-Scholes style) distribution built from spot and implied volatility.

No paid market-data source is wired in. Supply spot/IV through
``models.equity.manual_quotes``, or point ``models.equity.quote_url`` at a free
JSON endpoint you trust that returns ``{"SYMBOL": {"spot": x, "iv": y}}``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import requests

from ..logging_utils import get_logger
from .base import Estimate, FairValueEstimator, MarketContext, norm_cdf

log = get_logger(__name__)

EQUITY_SERIES_PREFIXES = (
    "KXNASDAQ100", "KXINX", "KXDJIA", "KXSPX", "KXBTC", "KXETH",
    "KXGOLD", "KXOIL", "KXTSX",
)


class EquityEstimator(FairValueEstimator):
    """Lognormal threshold probability from spot + implied vol."""

    name = "equity"
    priority = 40

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.quote_url = self.config.get("quote_url")
        self.default_iv = float(self.config.get("default_iv", 0.20))
        self.risk_free_rate = float(self.config.get("risk_free_rate", 0.04))
        self.manual_quotes = {
            str(k).upper(): v for k, v in (self.config.get("manual_quotes") or {}).items()
        }
        self._remote_quotes: dict[str, Any] | None = None

    def supports(self, market: Mapping[str, Any]) -> bool:
        series = self.series_of(market)
        if not series or not any(series.startswith(p) for p in EQUITY_SERIES_PREFIXES):
            return False
        _, floor_strike, cap_strike = self.strike_bounds(market)
        if floor_strike is None and cap_strike is None:
            return False
        return self._quote_for(series) is not None

    # --------------------------------------------------------------- quotes

    def _load_remote_quotes(self) -> dict[str, Any]:
        if self._remote_quotes is not None:
            return self._remote_quotes
        self._remote_quotes = {}
        if not self.quote_url:
            return self._remote_quotes
        try:
            resp = requests.get(self.quote_url, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, Mapping):
                self._remote_quotes = {str(k).upper(): v for k, v in payload.items()}
                log.info("loaded %d remote quotes", len(self._remote_quotes))
        except Exception as exc:
            log.warning("quote_url fetch failed (%s); using manual quotes only", exc)
        return self._remote_quotes

    def _quote_for(self, series: str) -> Mapping[str, Any] | None:
        """Longest-prefix quote lookup, manual taking precedence."""
        for source in (self.manual_quotes, self._load_remote_quotes()):
            if not source:
                continue
            if series in source:
                candidate = source[series]
                if isinstance(candidate, Mapping) and candidate.get("spot") is not None:
                    return candidate
            for key in sorted(source, key=len, reverse=True):
                if series.startswith(key) or key.startswith(series):
                    candidate = source[key]
                    if isinstance(candidate, Mapping) and candidate.get("spot") is not None:
                        return candidate
        return None

    # ------------------------------------------------------------- estimate

    def estimate(
        self, market: Mapping[str, Any], context: MarketContext
    ) -> Estimate | None:
        series = self.series_of(market)
        quote = self._quote_for(series)
        if not quote:
            return None
        try:
            spot = float(quote["spot"])
        except (KeyError, TypeError, ValueError):
            return None
        iv = float(quote.get("iv", self.default_iv))
        if spot <= 0 or iv <= 0:
            return None

        years = self._years_to_close(market, context)
        if years is None or years <= 0:
            return None

        _, floor_strike, cap_strike = self.strike_bounds(market)
        prob = self._lognormal_probability(spot, iv, years, floor_strike, cap_strike)
        if prob is None:
            return None

        return Estimate(
            model=self.name,
            prob=prob,
            source=f"lognormal from spot={spot:g}, iv={iv:.1%} ({series})",
            asof=quote.get("asof"),
            confidence=0.45,
            notes=(
                f"Risk-neutral lognormal over {years * 365:.1f} days at "
                f"r={self.risk_free_rate:.2%}. Spot and IV are configured inputs, "
                "not live market data - a stale spot makes this estimate worse "
                "than useless, so check models.equity.manual_quotes before trusting it."
            ),
        )

    def _lognormal_probability(
        self, spot: float, iv: float, years: float,
        floor_strike: float | None, cap_strike: float | None,
    ) -> float | None:
        """P(S_T in the contract's range) under a risk-neutral lognormal."""

        def prob_above(strike: float) -> float:
            # d2 from Black-Scholes: P(S_T > K) = N(d2).
            d2 = (
                math.log(spot / strike)
                + (self.risk_free_rate - 0.5 * iv * iv) * years
            ) / (iv * math.sqrt(years))
            return norm_cdf(d2)

        try:
            if floor_strike is not None and cap_strike is not None:
                if floor_strike <= 0 or cap_strike <= 0:
                    return None
                return max(0.0, prob_above(floor_strike) - prob_above(cap_strike))
            if floor_strike is not None and floor_strike > 0:
                return prob_above(floor_strike)
            if cap_strike is not None and cap_strike > 0:
                return 1.0 - prob_above(cap_strike)
        except (ValueError, ZeroDivisionError):
            return None
        return None

    @staticmethod
    def _years_to_close(
        market: Mapping[str, Any], context: MarketContext
    ) -> float | None:
        from datetime import datetime, timezone

        close_time = market.get("close_time")
        if not close_time:
            return None
        try:
            close = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
        except ValueError:
            return None
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        now = context.now or datetime.now(timezone.utc)
        if getattr(now, "tzinfo", None) is None:
            now = now.replace(tzinfo=timezone.utc)
        seconds = (close - now).total_seconds()
        return seconds / (365.0 * 86400.0) if seconds > 0 else None
