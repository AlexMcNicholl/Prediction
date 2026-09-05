"""Payload normalisation must be tolerant of API drift."""

from __future__ import annotations

import pytest

from screener.ingest import normalize


class TestSeriesDerivation:
    def test_explicit_field_wins(self):
        assert normalize.derive_series_ticker(
            {"series_ticker": "KXCPI", "ticker": "OTHER-X"}
        ) == "KXCPI"

    def test_derived_from_the_event_ticker(self):
        assert normalize.derive_series_ticker(
            {"event_ticker": "KXCPI-26SEP", "ticker": "KXCPI-26SEP-T2.9"}
        ) == "KXCPI"

    def test_derived_from_the_ticker_alone(self):
        assert normalize.derive_series_ticker({"ticker": "KXCPI-26SEP-T2.9"}) == "KXCPI"

    def test_no_hint_yields_none(self):
        assert normalize.derive_series_ticker({}) is None


class TestMarket:
    def test_maps_the_fields_the_screener_reads(self):
        raw = {
            "ticker": "KXCPI-26SEP-T2.9", "event_ticker": "KXCPI-26SEP",
            "title": "CPI above 2.9%?", "status": "active",
            "close_time": "2026-09-20T12:00:00Z", "rules_primary": "Settles per BLS.",
            "floor_strike": 2.9, "strike_type": "greater", "can_close_early": True,
        }
        market = normalize.normalize_market(raw)
        assert market["ticker"] == "KXCPI-26SEP-T2.9"
        assert market["series_ticker"] == "KXCPI"
        assert market["rules_primary"] == "Settles per BLS."
        assert market["floor_strike"] == 2.9
        assert market["close_time"].startswith("2026-09-20T12:00:00")

    def test_missing_fields_become_none_not_errors(self):
        market = normalize.normalize_market({"ticker": "A"})
        assert market["title"] is None and market["rules_primary"] is None

    def test_a_ticker_is_required(self):
        with pytest.raises(ValueError, match="no ticker"):
            normalize.normalize_market({"title": "orphan"})

    def test_settlement_source_accepts_a_list_of_objects(self):
        market = normalize.normalize_market({
            "ticker": "A",
            "settlement_sources": [{"name": "BLS", "url": "https://bls.gov"}],
        })
        assert "BLS" in market["settlement_source"]

    def test_settlement_source_accepts_a_plain_string(self):
        market = normalize.normalize_market({"ticker": "A", "settlement_source": "BLS"})
        assert market["settlement_source"] == "BLS"

    def test_unparseable_numbers_degrade_to_none(self):
        market = normalize.normalize_market({"ticker": "A", "floor_strike": "n/a",
                                             "tick_size": "x"})
        assert market["floor_strike"] is None and market["tick_size"] is None

    def test_epoch_timestamps_are_converted(self):
        market = normalize.normalize_market({"ticker": "A", "close_time": 1789000000})
        assert market["close_time"].startswith("20")

    def test_unparseable_timestamp_is_kept_verbatim(self):
        market = normalize.normalize_market({"ticker": "A", "close_time": "soon"})
        assert market["close_time"] == "soon"


class TestSnapshot:
    def test_prices_stay_in_cents(self):
        snapshot = normalize.normalize_snapshot(
            {"ticker": "A", "yes_bid": 60, "yes_ask": 64, "volume": 100}, "T"
        )
        assert snapshot["yes_bid"] == 60 and snapshot["yes_ask"] == 64
        assert snapshot["ts"] == "T"

    def test_raw_payload_is_retained_for_reprocessing(self):
        raw = {"ticker": "A", "yes_bid": 60, "some_new_field": "keep me"}
        assert normalize.normalize_snapshot(raw, "T")["raw"]["some_new_field"] == "keep me"

    def test_booleans_are_not_treated_as_numbers(self):
        snapshot = normalize.normalize_snapshot({"ticker": "A", "volume": True}, "T")
        assert snapshot["volume"] is None
