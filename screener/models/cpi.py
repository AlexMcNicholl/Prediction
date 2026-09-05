"""Inflation (CPI / PCE) fair-value estimator.

Uses the Cleveland Fed inflation nowcast as the central estimate and a normal
error band around it to convert a threshold contract into a probability.

The Cleveland Fed publishes the nowcast as an HTML page rather than a stable
JSON API, so automated scraping is deliberately best-effort: if the fetch or
parse fails, the estimator returns None (no model) instead of guessing. The
reliable path is ``models.cpi.manual_override`` in config.yaml, which you
update when you check the nowcast yourself.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import requests

from ..logging_utils import get_logger
from .base import Estimate, FairValueEstimator, MarketContext

log = get_logger(__name__)

CPI_SERIES_PREFIXES = ("KXCPI", "KXCPICORE", "KXPCE", "KXINFLATION", "KXCPIYOY")


class CPIEstimator(FairValueEstimator):
    """Threshold probability for inflation-print contracts."""

    name = "cpi"
    priority = 50

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.sigma = float(self.config.get("sigma", 0.15))
        self.nowcast_url = self.config.get("nowcast_url")
        self.manual_override = self.config.get("manual_override")
        self._cached: tuple[float, str] | None = None
        self._fetch_attempted = False

    def supports(self, market: Mapping[str, Any]) -> bool:
        series = self.series_of(market)
        if not series or not any(series.startswith(p) for p in CPI_SERIES_PREFIXES):
            return False
        _, floor_strike, cap_strike = self.strike_bounds(market)
        return floor_strike is not None or cap_strike is not None

    # ------------------------------------------------------------ nowcast

    def _nowcast(self) -> tuple[float, str] | None:
        """``(cpi_yoy_percent, asof)`` from config override, else the web page."""
        override = self.manual_override
        if isinstance(override, Mapping) and override.get("cpi_yoy") is not None:
            return float(override["cpi_yoy"]), str(override.get("asof", "manual"))

        if self._cached is not None or self._fetch_attempted:
            return self._cached
        self._fetch_attempted = True
        if not self.nowcast_url:
            return None

        try:
            resp = requests.get(
                self.nowcast_url,
                timeout=20,
                headers={"User-Agent": "prediction-screener/1.0 (research)"},
            )
            resp.raise_for_status()
            value = self._parse_nowcast(resp.text)
            if value is None:
                log.warning(
                    "Cleveland Fed page fetched but the nowcast could not be parsed; "
                    "set models.cpi.manual_override in config.yaml"
                )
                return None
            self._cached = (value, "fetched from clevelandfed.org")
            log.info("CPI nowcast parsed: %.2f%% YoY", value)
            return self._cached
        except Exception as exc:  # network, TLS, parse - all non-fatal
            log.warning(
                "CPI nowcast unavailable (%s); falling back to no-model. "
                "Set models.cpi.manual_override to supply it by hand.", exc,
            )
            return None

    @staticmethod
    def _parse_nowcast(html: str) -> float | None:
        """Pull a plausible CPI YoY figure out of the nowcast page.

        Intentionally conservative: only accepts a value in a sane inflation
        range so a layout change yields None rather than a wrong number.
        """
        patterns = [
            r"CPI[^%]{0,120}?(-?\d{1,2}\.\d{1,2})\s*%",
            r"year[- ]over[- ]year[^%]{0,120}?(-?\d{1,2}\.\d{1,2})\s*%",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
                try:
                    value = float(match.group(1))
                except ValueError:
                    continue
                if -5.0 <= value <= 20.0:
                    return value
        return None

    # ----------------------------------------------------------- estimate

    def estimate(
        self, market: Mapping[str, Any], context: MarketContext
    ) -> Estimate | None:
        nowcast = self._nowcast()
        if nowcast is None:
            return None
        mu, asof = nowcast
        prob, description = self.threshold_probability(market, mu, self.sigma)
        if prob is None:
            return None
        return Estimate(
            model=self.name,
            prob=prob,
            source=f"Cleveland Fed inflation nowcast ({asof})",
            asof=asof,
            confidence=0.6,
            notes=(
                f"{description}. Nowcast {mu:.2f}% YoY, assumed forecast error "
                f"sd={self.sigma:.2f}pp. Sigma is a configured assumption, not "
                "an observed quantity - tune models.cpi.sigma against realised "
                "nowcast errors."
            ),
        )
