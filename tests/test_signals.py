"""Signal engine: edge, flags, momentum, staleness, scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from screener.models.base import Estimate
from screener.signals import metrics
from screener.signals.engine import SignalConfig, SignalEngine
from screener.signals.fees import FeeModel

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def engine() -> SignalEngine:
    return SignalEngine(SignalConfig(), FeeModel(), bankroll=1000,
                        kelly_fraction=0.25, max_stake_fraction=0.05)


def snap(yes_bid=60, yes_ask=64, volume=5000, open_interest=3000, liquidity=250000):
    return {
        "yes_bid": yes_bid, "yes_ask": yes_ask,
        "no_bid": 100 - yes_ask, "no_ask": 100 - yes_bid,
        "last_price": (yes_bid + yes_ask) // 2,
        "volume": volume, "open_interest": open_interest, "liquidity": liquidity,
    }


def mkt(days=15):
    return {"ticker": "T", "close_time": (NOW + timedelta(days=days)).isoformat()}


class TestBasics:
    def test_implied_prob_from_cents(self):
        assert metrics.implied_prob(62) == pytest.approx(0.62)
        assert metrics.implied_prob(0) == 0.0
        assert metrics.implied_prob(100) == 1.0
        assert metrics.implied_prob(None) is None

    def test_implied_uses_the_midpoint(self, engine):
        row = engine.compute(mkt(), snap(60, 64), None, [], now=NOW)
        assert row["implied_prob"] == pytest.approx(0.62)

    def test_spread(self, engine):
        assert engine.compute(mkt(), snap(60, 64), None, [], now=NOW)["spread_cents"] == 4
        assert metrics.spread_cents(None, 64) is None

    def test_edge_is_model_minus_market(self, engine):
        est = Estimate(model="t", prob=0.75, source="t")
        row = engine.compute(mkt(), snap(60, 64), est, [], now=NOW)
        assert row["edge"] == pytest.approx(0.13)
        assert row["edge_flag"] == 1

    def test_small_edge_is_not_flagged(self, engine):
        est = Estimate(model="t", prob=0.64, source="t")
        row = engine.compute(mkt(), snap(60, 64), est, [], now=NOW)
        assert row["edge_flag"] == 0

    def test_no_model_gives_no_edge(self, engine):
        row = engine.compute(mkt(), snap(), None, [], now=NOW)
        assert row["edge"] is None
        assert row["edge_flag"] == 0
        assert row["ev_per_contract"] is None


class TestSideSelection:
    def test_picks_yes_when_model_is_bullish(self, engine):
        est = Estimate(model="t", prob=0.85, source="t")
        row = engine.compute(mkt(), snap(60, 64), est, [], now=NOW)
        assert row["side"] == "yes"
        assert row["entry_price"] == pytest.approx(0.64)

    def test_picks_no_when_model_is_bearish(self, engine):
        est = Estimate(model="t", prob=0.20, source="t")
        row = engine.compute(mkt(), snap(60, 64), est, [], now=NOW)
        assert row["side"] == "no"
        assert row["entry_price"] == pytest.approx(0.40)

    def test_ev_uses_the_ask_not_the_mid(self, engine):
        """Screening on the mid would overstate every edge by half the spread."""
        est = Estimate(model="t", prob=0.75, source="t")
        row = engine.compute(mkt(), snap(60, 64), est, [], now=NOW)
        fee = FeeModel().fee_per_contract(0.64)
        assert row["ev_per_contract"] == pytest.approx(0.75 - 0.64 - fee)

    def test_negative_ev_is_reported_not_hidden(self, engine):
        """When BOTH sides lose to the spread, say so rather than hiding it.

        A wide book (36/64) means you pay 0.64 whichever way you lean, so a
        model at 0.50 is under water on both sides.
        """
        est = Estimate(model="t", prob=0.50, source="t")
        row = engine.compute(mkt(), snap(36, 64), est, [], now=NOW)
        assert row["ev_per_contract"] < 0
        assert row["kelly_fraction_used"] == 0.0
        assert row["contracts"] == 0

    def test_picks_the_better_side_even_when_the_mid_looks_wrong(self, engine):
        """A model below the mid can still make the NO side positive-EV."""
        est = Estimate(model="t", prob=0.50, source="t")
        row = engine.compute(mkt(), snap(60, 64), est, [], now=NOW)
        assert row["side"] == "no"
        assert row["ev_per_contract"] > 0

    def test_defaults_to_yes_without_a_model(self, engine):
        row = engine.compute(mkt(), snap(), None, [], now=NOW)
        assert row["side"] == "yes"

    def test_missing_no_ask_is_derived_from_the_yes_bid(self, engine):
        """Buying NO at p is selling YES at 100-p; a partial book must not
        force every contract onto the YES side."""
        partial = {"yes_bid": 6, "yes_ask": 10, "volume": 5000,
                   "open_interest": 3000, "liquidity": 250000}
        est = Estimate(model="t", prob=0.01, source="t")
        row = engine.compute(mkt(), partial, est, [], now=NOW)
        assert row["side"] == "no"
        assert row["entry_price"] == pytest.approx(0.94)
        assert row["ev_per_contract"] > 0

    def test_missing_yes_ask_is_derived_from_the_no_bid(self, engine):
        partial = {"no_bid": 90, "no_ask": 94, "volume": 5000,
                   "open_interest": 3000, "liquidity": 250000}
        est = Estimate(model="t", prob=0.90, source="t")
        row = engine.compute(mkt(), partial, est, [], now=NOW)
        assert row["side"] == "yes"
        assert row["entry_price"] == pytest.approx(0.10)


class TestFlags:
    def test_longshot_band(self, engine):
        assert engine.compute(mkt(), snap(3, 7), None, [], now=NOW)["longshot_flag"] == 1
        assert engine.compute(mkt(), snap(93, 97), None, [], now=NOW)["longshot_flag"] == 1
        assert engine.compute(mkt(), snap(45, 55), None, [], now=NOW)["longshot_flag"] == 0

    def test_wide_spread_flag(self, engine):
        assert engine.compute(mkt(), snap(50, 62), None, [], now=NOW)["spread_flag"] == 1
        assert engine.compute(mkt(), snap(60, 61), None, [], now=NOW)["spread_flag"] == 0

    def test_thin_book_flag(self, engine):
        thin = snap(volume=5, open_interest=2, liquidity=100)
        assert engine.compute(mkt(), thin, None, [], now=NOW)["liquidity_flag"] == 1
        assert engine.compute(mkt(), snap(), None, [], now=NOW)["liquidity_flag"] == 0

    def test_liquidity_is_converted_from_cents(self, engine):
        """Kalshi reports liquidity in cents; the threshold is in dollars."""
        # 60000 cents = $600, above the $500 default threshold.
        row = engine.compute(mkt(), snap(liquidity=60000), None, [], now=NOW)
        assert row["liquidity_flag"] == 0
        row = engine.compute(mkt(), snap(liquidity=40000), None, [], now=NOW)
        assert row["liquidity_flag"] == 1


class TestMomentumAndStaleness:
    def history(self, prices, hours_apart=6):
        return [
            {"ts": (NOW - timedelta(hours=i * hours_apart)).isoformat(), "mid_price": p}
            for i, p in enumerate(prices)
        ]

    def test_momentum_measures_the_move(self):
        # Newest first: 70 now, 50 twenty-four hours ago.
        hist = self.history([70, 65, 60, 55, 50], hours_apart=6)
        assert metrics.momentum(hist, 24) == pytest.approx(0.20)

    def test_momentum_needs_two_points(self):
        assert metrics.momentum([], 24) is None
        assert metrics.momentum(self.history([60]), 24) is None

    def test_momentum_flag_fires(self, engine):
        hist = self.history([70, 65, 60, 55, 50], hours_apart=6)
        row = engine.compute(mkt(), snap(), None, hist, now=NOW)
        assert row["momentum_flag"] == 1

    def test_flat_price_is_stale(self):
        hist = self.history([60, 60, 60, 60], hours_apart=12)
        assert metrics.staleness_hours(hist, NOW) == pytest.approx(36.0)

    def test_recent_move_is_not_stale(self, engine):
        hist = self.history([70, 60, 60], hours_apart=2)
        row = engine.compute(mkt(), snap(), None, hist, now=NOW)
        assert row["stale_flag"] == 0

    def test_staleness_without_history(self):
        assert metrics.staleness_hours([], NOW) is None


class TestScoring:
    def test_components_are_all_exposed(self, engine):
        est = Estimate(model="t", prob=0.75, source="t")
        row = engine.compute(mkt(), snap(), est, [], now=NOW)
        assert set(row["score_components"]) == {
            "edge", "liquidity", "spread", "annualized", "momentum",
            "has_model", "actionable",
        }

    def test_score_is_bounded(self, engine):
        est = Estimate(model="t", prob=0.99, source="t")
        row = engine.compute(mkt(), snap(1, 2), est, [], now=NOW)
        assert 0.0 <= row["score"] <= 1.0

    def test_no_model_scores_below_a_modelled_equivalent(self, engine):
        est = Estimate(model="t", prob=0.85, source="t")
        modelled = engine.compute(mkt(), snap(), est, [], now=NOW)
        unmodelled = engine.compute(mkt(), snap(), None, [], now=NOW)
        assert unmodelled["score"] < modelled["score"]
        assert unmodelled["score_components"]["has_model"] is False

    def test_tighter_spread_scores_higher(self, engine):
        tight = engine.compute(mkt(), snap(61, 62), None, [], now=NOW)
        wide = engine.compute(mkt(), snap(50, 70), None, [], now=NOW)
        assert tight["score_components"]["spread"] > wide["score_components"]["spread"]

    def test_score_ranking_does_not_reward_longshots_for_being_longshots(self, engine):
        """Ranking on annualized-if-win would put every longshot on top."""
        est_long = Estimate(model="t", prob=0.05, source="t")
        est_fav = Estimate(model="t", prob=0.60, source="t")
        longshot = engine.compute(mkt(), snap(3, 5), est_long, [], now=NOW)
        favorite = engine.compute(mkt(), snap(55, 57), est_fav, [], now=NOW)
        assert longshot["annualized_if_win"] >= favorite["annualized_if_win"]
        assert longshot["score_components"]["annualized"] == 0.0

    def test_weights_are_configurable(self):
        cfg = SignalConfig(score_weights={"edge": 1.0})
        engine = SignalEngine(cfg, FeeModel())
        est = Estimate(model="t", prob=0.85, source="t")
        row = engine.compute(mkt(), snap(), est, [], now=NOW)
        assert row["score"] == pytest.approx(row["score_components"]["edge"])

    def test_uncapturable_edge_scores_zero(self):
        """A disagreement the spread swallows is not a reason to look."""
        engine = SignalEngine(SignalConfig(), FeeModel())
        est = Estimate(model="t", prob=0.50, source="t")
        # 36/64 book: you pay 0.64 either way, so a 0.50 model loses on both.
        row = engine.compute(mkt(), snap(36, 64), est, [], now=NOW)
        assert row["ev_per_contract"] < 0
        assert row["score_components"]["actionable"] is False
        assert row["score_components"]["edge"] == 0.0

    def test_capturable_edge_scores_above_zero(self):
        engine = SignalEngine(SignalConfig(), FeeModel())
        est = Estimate(model="t", prob=0.85, source="t")
        row = engine.compute(mkt(), snap(60, 64), est, [], now=NOW)
        assert row["ev_per_contract"] > 0
        assert row["score_components"]["actionable"] is True
        assert row["score_components"]["edge"] > 0


class TestRobustness:
    def test_empty_snapshot_does_not_crash(self, engine):
        row = engine.compute(mkt(), {}, None, [], now=NOW)
        assert row["score"] == 0.0
        assert row["implied_prob"] is None

    def test_missing_close_time(self, engine):
        row = engine.compute({"ticker": "T"}, snap(), None, [], now=NOW)
        assert row["days_to_close"] is None
        assert row["annualized_if_win"] is None

    def test_compute_all_sorts_by_score_descending(self, engine):
        markets = [mkt(), {"ticker": "U", "close_time": mkt()["close_time"]}]
        rows = engine.compute_all(
            markets,
            {"T": snap(), "U": snap(volume=1, open_interest=1, liquidity=1)},
            {"T": Estimate(model="t", prob=0.85, source="t")},
            now=NOW,
        )
        assert [r["ticker"] for r in rows] == ["T", "U"]

    def test_compute_all_survives_a_bad_row(self, engine):
        rows = engine.compute_all(
            [mkt(), {"ticker": "BAD", "close_time": "not-a-date"}],
            {"T": snap()}, {}, now=NOW,
        )
        assert len(rows) == 2  # bad close_time degrades to None, does not drop the row
