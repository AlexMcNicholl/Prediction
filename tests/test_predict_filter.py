"""Predict availability: the allowlist and the Canadian 30-day rule."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from screener.predict.availability import PredictFilter

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


def market(ticker="KXCPI-26SEP-T2.9", series="KXCPI", days=10,
           category="Economics", status="active"):
    return {
        "ticker": ticker,
        "series_ticker": series,
        "category": category,
        "status": status,
        "close_time": (NOW + timedelta(days=days)).isoformat(),
    }


@pytest.fixture
def flt() -> PredictFilter:
    return PredictFilter(
        category_allowlist=["Economics", "Financials", "Climate and Weather"],
        series_prefix_allowlist=["KXCPI", "KXU3", "KXHIGH"],
        series_allowlist=["KXSPECIAL"],
        series_denylist=["KXPRES", "KXELECTION"],
        term_to_maturity_days=30,
    )


class TestTermToMaturity:
    def test_inside_the_window_is_tradeable(self, flt):
        assert flt.classify(market(days=10), now=NOW).tradeable

    def test_beyond_thirty_days_is_excluded(self, flt):
        result = flt.classify(market(days=45), now=NOW)
        assert not result.tradeable
        assert "30-day" in result.reason

    def test_boundary_at_exactly_thirty_days_is_allowed(self, flt):
        assert flt.classify(market(days=30), now=NOW).tradeable

    def test_just_past_the_boundary_is_excluded(self, flt):
        assert not flt.classify(market(days=30.001), now=NOW).tradeable

    def test_already_closed_is_excluded(self, flt):
        result = flt.classify(market(days=-2), now=NOW)
        assert not result.tradeable
        assert "already closed" in result.reason

    def test_configurable_window(self):
        seven_day = PredictFilter(
            series_prefix_allowlist=["KXCPI"], term_to_maturity_days=7
        )
        assert seven_day.classify(market(days=5), now=NOW).tradeable
        assert not seven_day.classify(market(days=10), now=NOW).tradeable

    def test_missing_close_time_is_excluded(self, flt):
        m = market()
        m["close_time"] = None
        result = flt.classify(m, now=NOW)
        assert not result.tradeable
        assert "no close_time" in result.reason

    def test_days_to_close_is_reported(self, flt):
        assert flt.classify(market(days=12), now=NOW).days_to_close == pytest.approx(12.0)


class TestAllowlist:
    def test_prefix_match_admits(self, flt):
        result = flt.classify(market(series="KXCPICORE"), now=NOW)
        assert result.tradeable
        assert result.matched_rule == "prefix:KXCPI"

    def test_exact_series_match_admits(self, flt):
        result = flt.classify(market(series="KXSPECIAL", category="Other"), now=NOW)
        assert result.tradeable
        assert result.matched_rule == "series:KXSPECIAL"

    def test_category_match_admits_unlisted_series(self, flt):
        result = flt.classify(market(series="KXNEWECON", category="Financials"), now=NOW)
        assert result.tradeable
        assert result.matched_rule.startswith("category:")

    def test_unlisted_series_and_category_excluded(self, flt):
        result = flt.classify(
            market(series="KXOSCAR", category="Entertainment"), now=NOW
        )
        assert not result.tradeable
        assert "not on the Predict allowlist" in result.reason

    def test_denylist_beats_everything(self, flt):
        result = flt.classify(
            market(ticker="KXPRES-28-DEM", series="KXPRES", category="Economics"), now=NOW
        )
        assert not result.tradeable
        assert "denylist" in result.reason

    def test_denylist_matches_on_ticker_prefix(self, flt):
        result = flt.classify(
            market(ticker="KXELECTION-26-X", series="UNKNOWN"), now=NOW
        )
        assert not result.tradeable
        assert "denylist" in result.reason

    def test_matching_is_case_insensitive(self, flt):
        assert flt.classify(market(series="kxcpi"), now=NOW).tradeable
        assert flt.classify(market(series="KXNEW", category="economics"), now=NOW).tradeable


class TestStatus:
    def test_non_open_status_excluded(self, flt):
        result = flt.classify(market(status="settled"), now=NOW)
        assert not result.tradeable
        assert "status" in result.reason

    def test_status_check_can_be_disabled(self):
        lenient = PredictFilter(
            series_prefix_allowlist=["KXCPI"], require_open_status=False
        )
        assert lenient.classify(market(status="settled"), now=NOW).tradeable

    def test_open_and_initialized_are_accepted(self, flt):
        for status in ("active", "open", "initialized"):
            assert flt.classify(market(status=status), now=NOW).tradeable


class TestBatch:
    def test_classify_all_and_summary(self, flt):
        markets = [
            market(series="KXCPI", days=10),
            market(series="KXCPI", days=90),
            market(series="KXOSCAR", category="Entertainment"),
            market(ticker="KXPRES-1", series="KXPRES"),
        ]
        results = flt.classify_all(markets, now=NOW)
        assert sum(r.tradeable for r in results) == 1

        reasons = dict(flt.summarize_exclusions(results))
        assert sum(reasons.values()) == 3
        assert any("30-day" in key for key in reasons)

    def test_records_round_trip(self, flt):
        record = flt.classify(market(), now=NOW).as_record()
        assert set(record) == {
            "ticker", "tradeable", "reason", "days_to_close", "matched_rule"
        }


def test_days_until_handles_naive_and_z_suffix():
    assert PredictFilter.days_until("2026-09-15T12:00:00Z", now=NOW) == pytest.approx(10.0)
    assert PredictFilter.days_until("2026-09-15T12:00:00", now=NOW) == pytest.approx(10.0)
    assert PredictFilter.days_until("not a date", now=NOW) is None
    assert PredictFilter.days_until(None, now=NOW) is None
