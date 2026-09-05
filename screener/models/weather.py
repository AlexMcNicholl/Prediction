"""Weather / temperature fair-value estimator.

Blends a point forecast with climatology and converts a temperature-threshold
contract into a probability via a normal error band.

Default provider is the US National Weather Service (api.weather.gov): free,
no API key, no paid data source. Set ``models.weather.provider: manual`` to
supply forecasts by hand instead.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import requests

from ..logging_utils import get_logger
from .base import Estimate, FairValueEstimator, MarketContext

log = get_logger(__name__)

WEATHER_SERIES_PREFIXES = ("KXHIGH", "KXLOW", "KXTEMP", "KXRAIN", "KXSNOW")

# Kalshi weather markets encode the station in the ticker, e.g. KXHIGHNY-...
# Map the station code to the NWS gridpoint we ask for a forecast.
STATION_COORDS: dict[str, tuple[float, float]] = {
    "NY": (40.7789, -73.9692),   # Central Park
    "CHI": (41.9860, -87.9336),  # O'Hare
    "MIA": (25.7906, -80.3164),
    "AUS": (30.1830, -97.6799),
    "DEN": (39.8467, -104.6562),
    "LAX": (33.9382, -118.3866),
    "PHIL": (39.8683, -75.2311),
    "SEA": (47.4444, -122.3138),
    "HOU": (29.9902, -95.3368),
    "BOS": (42.3606, -71.0097),
}

# Rough monthly climatological daily-high normals (deg F) for blending. These
# are coarse on purpose - they are a weak prior, not a forecast.
DEFAULT_CLIMATOLOGY_F = {
    1: 39, 2: 42, 3: 50, 4: 61, 5: 71, 6: 80,
    7: 85, 8: 83, 9: 76, 10: 64, 11: 54, 12: 44,
}


class WeatherEstimator(FairValueEstimator):
    """Temperature-threshold probability from forecast + climatology."""

    name = "weather"
    priority = 50

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.provider = str(self.config.get("provider", "nws")).lower()
        self.base_url = str(self.config.get("nws_base_url", "https://api.weather.gov"))
        self.climatology_weight = float(self.config.get("climatology_weight", 0.30))
        self.sigma = float(self.config.get("forecast_sigma_f", 3.0))
        self.manual_forecasts = {
            str(k).upper(): v
            for k, v in (self.config.get("manual_forecasts") or {}).items()
        }
        self._forecast_cache: dict[str, float | None] = {}

    def supports(self, market: Mapping[str, Any]) -> bool:
        if self.provider == "none":
            return False
        series = self.series_of(market)
        if not series or not any(series.startswith(p) for p in WEATHER_SERIES_PREFIXES):
            return False
        _, floor_strike, cap_strike = self.strike_bounds(market)
        return floor_strike is not None or cap_strike is not None

    # ------------------------------------------------------------- forecast

    @staticmethod
    def station_code(market: Mapping[str, Any]) -> str | None:
        """Extract the station suffix from a series like ``KXHIGHNY``."""
        series = str(market.get("series_ticker") or "").upper()
        match = re.match(r"^KX(?:HIGH|LOW|TEMP|RAIN|SNOW)([A-Z]+)$", series)
        return match.group(1) if match else None

    def _nws_forecast_high(self, station: str) -> float | None:
        """Next-period forecast temperature (deg F) from api.weather.gov."""
        if station in self._forecast_cache:
            return self._forecast_cache[station]
        coords = STATION_COORDS.get(station)
        if not coords:
            log.debug("no coordinates configured for weather station %s", station)
            self._forecast_cache[station] = None
            return None

        lat, lon = coords
        headers = {
            "User-Agent": "prediction-screener/1.0 (research; contact via repo)",
            "Accept": "application/geo+json",
        }
        try:
            points = requests.get(
                f"{self.base_url}/points/{lat:.4f},{lon:.4f}", timeout=20, headers=headers
            )
            points.raise_for_status()
            forecast_url = points.json().get("properties", {}).get("forecast")
            if not forecast_url:
                raise ValueError("no forecast URL in the NWS points response")

            forecast = requests.get(forecast_url, timeout=20, headers=headers)
            forecast.raise_for_status()
            periods = forecast.json().get("properties", {}).get("periods") or []
            for period in periods:
                if period.get("isDaytime") and period.get("temperature") is not None:
                    value = float(period["temperature"])
                    self._forecast_cache[station] = value
                    log.info("NWS forecast for %s: %.0fF", station, value)
                    return value
            raise ValueError("no daytime period with a temperature")
        except Exception as exc:
            log.warning("NWS forecast unavailable for %s (%s); using climatology only", station, exc)
            self._forecast_cache[station] = None
            return None

    def _blended_mu(
        self, market: Mapping[str, Any], context: MarketContext
    ) -> tuple[float, str, str] | None:
        """``(mu, source, detail)`` blending forecast with climatology."""
        station = self.station_code(market) or "?"
        manual = self.manual_forecasts.get(station)
        forecast: float | None
        if manual is not None:
            forecast = float(manual)
            source = f"manual forecast for {station}"
        elif self.provider == "manual":
            return None
        else:
            forecast = self._nws_forecast_high(station)
            source = f"NWS forecast for {station}"

        month = getattr(context.now, "month", None) or 1
        climate = float(DEFAULT_CLIMATOLOGY_F.get(month, 60))

        if forecast is None:
            # Climatology alone is a very weak signal; say so loudly.
            return climate, f"climatology only ({station})", (
                f"No forecast available; using the monthly climatological normal "
                f"{climate:.0f}F alone. Treat this as a weak prior, not a forecast."
            )

        weight = max(0.0, min(1.0, self.climatology_weight))
        mu = (1.0 - weight) * forecast + weight * climate
        detail = (
            f"Blend of forecast {forecast:.0f}F and climatology {climate:.0f}F "
            f"at climatology weight {weight:.2f} -> mu {mu:.1f}F."
        )
        return mu, source, detail

    # ------------------------------------------------------------- estimate

    def estimate(
        self, market: Mapping[str, Any], context: MarketContext
    ) -> Estimate | None:
        blended = self._blended_mu(market, context)
        if blended is None:
            return None
        mu, source, detail = blended
        # Climatology-only estimates carry much wider uncertainty.
        sigma = self.sigma if "climatology only" not in source else self.sigma * 2.5
        prob, description = self.threshold_probability(market, mu, sigma)
        if prob is None:
            return None
        return Estimate(
            model=self.name,
            prob=prob,
            source=source,
            asof=str(context.now) if context.now else None,
            confidence=0.35 if "climatology only" in source else 0.55,
            notes=f"{description}. {detail} Assumed forecast error sd={sigma:.1f}F.",
        )
