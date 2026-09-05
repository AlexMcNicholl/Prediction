"""Central-bank rate-decision estimator (stub).

Deriving implied probabilities properly needs OIS or rate-futures data
(BoC OIS, CME FedWatch-style term SOFR). No free, licence-clean source is
wired in, and adding a paid one is your call - so this estimator only reads
hand-entered probabilities from ``models.rates.manual_probabilities``.

The interface is complete: drop a real source into ``_implied_from_market_data``
and every downstream signal picks it up with no other change.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..logging_utils import get_logger
from .base import Estimate, FairValueEstimator, MarketContext

log = get_logger(__name__)

RATE_SERIES_PREFIXES = ("KXFED", "KXFEDDECISION", "KXBOC", "KXRATE", "KXECB")


class RatesEstimator(FairValueEstimator):
    """Manual-only rates estimator with a clean extension point."""

    name = "rates"
    priority = 45

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.manual = {
            str(k).upper(): v
            for k, v in (self.config.get("manual_probabilities") or {}).items()
        }

    def supports(self, market: Mapping[str, Any]) -> bool:
        series = self.series_of(market)
        if not series or not any(series.startswith(p) for p in RATE_SERIES_PREFIXES):
            return False
        return str(market.get("ticker", "")).upper() in self.manual

    def estimate(
        self, market: Mapping[str, Any], context: MarketContext
    ) -> Estimate | None:
        ticker = str(market.get("ticker", "")).upper()
        raw = self.manual.get(ticker)
        if raw is None:
            return self._implied_from_market_data(market, context)
        try:
            prob = float(raw["prob"] if isinstance(raw, Mapping) else raw)
        except (TypeError, ValueError, KeyError):
            return None
        return Estimate(
            model=self.name,
            prob=prob,
            source="manual rate-decision probability (config.yaml)",
            asof=raw.get("asof") if isinstance(raw, Mapping) else None,
            confidence=0.6,
            notes=(
                "Hand-entered from an OIS / rate-futures reading. Wire a live "
                "source into RatesEstimator._implied_from_market_data to automate."
            ),
        )

    def _implied_from_market_data(
        self, market: Mapping[str, Any], context: MarketContext
    ) -> Estimate | None:
        """Extension point for OIS / rate-futures derived probabilities."""
        return None
