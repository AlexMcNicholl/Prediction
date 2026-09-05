"""Fair-value estimators and registry dispatch."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from screener.models.base import Estimate, MarketContext, NoModelEstimator, norm_cdf
from screener.models.cpi import CPIEstimator
from screener.models.equity import EquityEstimator
from screener.models.manual import ManualEstimator
from screener.models.rates import RatesEstimator
from screener.models.registry import ModelRegistry
from screener.models.weather import WeatherEstimator

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
CTX = MarketContext(now=NOW)


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


class TestEstimate:
    def test_probability_is_clamped_away_from_zero_and_one(self):
        assert Estimate("m", 0.0, "s").prob == 0.001
        assert Estimate("m", 1.0, "s").prob == 0.999
        assert Estimate("m", -5, "s").prob == 0.001

    def test_none_probability_survives(self):
        assert Estimate("m", None, "s").prob is None

    def test_record_shape_is_auditable(self):
        record = Estimate("m", 0.6, "src", asof="2026-09-01", confidence=0.7).as_record("T")
        assert set(record) == {"ticker", "model", "prob", "source", "asof",
                               "confidence", "notes"}


class TestCPI:
    def estimator(self):
        return CPIEstimator({"sigma": 0.15, "nowcast_url": None,
                             "manual_override": {"cpi_yoy": 2.9, "asof": "2026-09-01"}})

    def test_at_the_strike_is_a_coin_flip(self):
        market = {"series_ticker": "KXCPI", "floor_strike": 2.9, "strike_type": "greater"}
        assert self.estimator().estimate(market, CTX).prob == pytest.approx(0.5, abs=1e-6)

    def test_far_above_the_nowcast_is_unlikely(self):
        market = {"series_ticker": "KXCPI", "floor_strike": 3.5, "strike_type": "greater"}
        assert self.estimator().estimate(market, CTX).prob < 0.01

    def test_less_than_inverts(self):
        above = {"series_ticker": "KXCPI", "floor_strike": 3.1, "strike_type": "greater"}
        below = {"series_ticker": "KXCPI", "cap_strike": 3.1, "strike_type": "less"}
        est = self.estimator()
        assert est.estimate(above, CTX).prob + est.estimate(below, CTX).prob == pytest.approx(1.0, abs=1e-6)

    def test_range_market(self):
        market = {"series_ticker": "KXCPI", "floor_strike": 2.8, "cap_strike": 3.0}
        assert 0.3 < self.estimator().estimate(market, CTX).prob < 0.6

    def test_only_supports_inflation_series_with_strikes(self):
        est = self.estimator()
        assert est.supports({"series_ticker": "KXCPI", "floor_strike": 2.9})
        assert not est.supports({"series_ticker": "KXCPI"})  # no strike
        assert not est.supports({"series_ticker": "KXHIGHNY", "floor_strike": 80})

    def test_no_nowcast_means_no_estimate(self):
        est = CPIEstimator({"sigma": 0.15, "nowcast_url": None, "manual_override": None})
        assert est.estimate({"series_ticker": "KXCPI", "floor_strike": 2.9}, CTX) is None

    def test_parser_rejects_implausible_values(self):
        assert CPIEstimator._parse_nowcast("CPI is 95.5%") is None
        assert CPIEstimator._parse_nowcast("no numbers here") is None
        assert CPIEstimator._parse_nowcast("CPI year-over-year 2.85%") == pytest.approx(2.85)

    def test_estimate_is_auditable(self):
        market = {"series_ticker": "KXCPI", "floor_strike": 2.9, "strike_type": "greater"}
        estimate = self.estimator().estimate(market, CTX)
        assert "Cleveland Fed" in estimate.source
        assert "sd=" in estimate.notes


class TestWeather:
    def estimator(self):
        return WeatherEstimator({"provider": "manual", "manual_forecasts": {"NY": 78},
                                 "climatology_weight": 0.30, "forecast_sigma_f": 3.0})

    def test_station_code_extraction(self):
        assert WeatherEstimator.station_code({"series_ticker": "KXHIGHNY"}) == "NY"
        assert WeatherEstimator.station_code({"series_ticker": "KXLOWCHI"}) == "CHI"
        assert WeatherEstimator.station_code({"series_ticker": "KXCPI"}) is None

    def test_range_probability_is_sane(self):
        market = {"series_ticker": "KXHIGHNY", "floor_strike": 75, "cap_strike": 80}
        prob = self.estimator().estimate(market, CTX).prob
        assert 0.3 < prob < 0.9

    def test_forecast_blends_with_climatology(self):
        market = {"series_ticker": "KXHIGHNY", "floor_strike": 77, "strike_type": "greater"}
        estimate = self.estimator().estimate(market, CTX)
        assert "Blend of forecast" in estimate.notes

    def test_provider_none_disables(self):
        est = WeatherEstimator({"provider": "none"})
        assert not est.supports({"series_ticker": "KXHIGHNY", "floor_strike": 75})

    def test_manual_provider_without_a_forecast_returns_none(self):
        est = WeatherEstimator({"provider": "manual", "manual_forecasts": {}})
        assert est.estimate({"series_ticker": "KXHIGHNY", "floor_strike": 75}, CTX) is None


class TestEquity:
    def estimator(self):
        return EquityEstimator({
            "manual_quotes": {"KXNASDAQ100": {"spot": 20000, "iv": 0.18}},
            "risk_free_rate": 0.04, "default_iv": 0.2,
        })

    def market(self, strike=21000):
        return {"series_ticker": "KXNASDAQ100", "floor_strike": strike,
                "strike_type": "greater", "close_time": "2026-09-30T20:00:00+00:00"}

    def test_higher_strike_is_less_likely(self):
        est = self.estimator()
        near = est.estimate(self.market(20100), CTX).prob
        far = est.estimate(self.market(23000), CTX).prob
        assert near > far

    def test_at_the_money_is_near_a_coin_flip(self):
        prob = self.estimator().estimate(self.market(20000), CTX).prob
        assert 0.4 < prob < 0.6

    def test_no_quote_means_unsupported(self):
        est = EquityEstimator({"manual_quotes": {}})
        assert not est.supports(self.market())

    def test_expired_market_gives_no_estimate(self):
        market = self.market()
        market["close_time"] = "2020-01-01T00:00:00+00:00"
        assert self.estimator().estimate(market, CTX) is None

    def test_notes_warn_about_stale_inputs(self):
        assert "stale spot" in self.estimator().estimate(self.market(), CTX).notes


class TestManualAndRates:
    def test_manual_reads_scalar_and_mapping_forms(self):
        est = ManualEstimator({"rates": {"manual_probabilities": {
            "A": 0.8, "B": {"prob": 0.6, "asof": "2026-09-01", "notes": "read"}
        }}})
        assert est.estimate({"ticker": "A"}, CTX).prob == 0.8
        estimate_b = est.estimate({"ticker": "B"}, CTX)
        assert estimate_b.prob == 0.6 and estimate_b.notes == "read"

    def test_manual_is_case_insensitive(self):
        est = ManualEstimator({"x": {"manual_probabilities": {"abc": 0.5}}})
        assert est.supports({"ticker": "ABC"})

    def test_rates_without_manual_input_yields_nothing(self):
        est = RatesEstimator({"manual_probabilities": {}})
        assert not est.supports({"series_ticker": "KXFED", "ticker": "KXFED-1"})

    def test_rates_manual_probability(self):
        est = RatesEstimator({"manual_probabilities": {"KXFED-1": 0.82}})
        assert est.estimate({"series_ticker": "KXFED", "ticker": "KXFED-1"}, CTX).prob == 0.82


class TestRegistry:
    def registry(self):
        return ModelRegistry.from_config({
            "enabled": ["cpi", "weather", "equity", "rates"],
            "cpi": {"sigma": 0.15, "nowcast_url": None,
                    "manual_override": {"cpi_yoy": 2.9}},
            "weather": {"provider": "manual", "manual_forecasts": {"NY": 78}},
            "equity": {"manual_quotes": {"KXNASDAQ100": {"spot": 20000, "iv": 0.18}}},
            "rates": {"manual_probabilities": {"KXFED-1": 0.82}},
        })

    def test_dispatches_to_the_right_model(self):
        reg = self.registry()
        cases = [
            ({"ticker": "a", "series_ticker": "KXCPI", "floor_strike": 2.9}, "cpi"),
            ({"ticker": "b", "series_ticker": "KXHIGHNY", "floor_strike": 75,
              "cap_strike": 80}, "weather"),
            ({"ticker": "c", "series_ticker": "KXNASDAQ100", "floor_strike": 21000,
              "close_time": "2026-09-30T20:00:00+00:00"}, "equity"),
            ({"ticker": "KXFED-1", "series_ticker": "KXFED"}, "manual"),
            ({"ticker": "z", "series_ticker": "KXUNKNOWN"}, "no-model"),
        ]
        for market, expected in cases:
            assert reg.estimate(market, CTX).model == expected

    def test_manual_override_beats_an_automated_model(self):
        reg = ModelRegistry.from_config({
            "enabled": ["cpi"],
            "cpi": {"sigma": 0.15, "manual_override": {"cpi_yoy": 2.9},
                    "nowcast_url": None,
                    "manual_probabilities": {"KXCPI-X": 0.99}},
        })
        market = {"ticker": "KXCPI-X", "series_ticker": "KXCPI", "floor_strike": 2.9}
        estimate = reg.estimate(market, CTX)
        assert estimate.model == "manual" and estimate.prob == 0.99

    def test_no_model_fallback_is_explicit(self):
        estimate = self.registry().estimate({"ticker": "z", "series_ticker": "KXZZZ"}, CTX)
        assert estimate.prob is None
        assert estimate.confidence == 0.0
        assert "No fair-value model" in estimate.notes

    def test_a_raising_estimator_does_not_break_dispatch(self):
        class Exploding(NoModelEstimator):
            name = "boom"
            priority = 999

            def supports(self, market):
                return True

            def estimate(self, market, context):
                raise RuntimeError("kaboom")

        reg = ModelRegistry([Exploding()])
        assert reg.estimate({"ticker": "x"}, CTX).model == "no-model"

    def test_unknown_model_name_is_skipped(self):
        reg = ModelRegistry.from_config({"enabled": ["does-not-exist"]})
        assert reg.estimate({"ticker": "x"}, CTX).model == "no-model"

    def test_estimate_all_returns_one_per_ticker(self):
        markets = [{"ticker": "a", "series_ticker": "KXCPI", "floor_strike": 2.9},
                   {"ticker": "z", "series_ticker": "KXZZZ"}]
        results = self.registry().estimate_all(markets, now=NOW)
        assert set(results) == {"a", "z"}
