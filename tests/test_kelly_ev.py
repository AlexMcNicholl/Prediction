"""Fee, expected-value and Kelly math."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from screener.signals import metrics
from screener.signals.fees import FeeModel


class TestFees:
    def test_matches_published_formula(self):
        """fee = ceil(0.07 * C * P * (1-P)), rounded up to the cent.

        Computed with exact rational arithmetic so the assertion tests the
        published formula itself, not another float expression carrying the
        same rounding noise.
        """
        fees = FeeModel(taker_coefficient=0.07)
        for price, contracts in ((0.50, 100), (0.62, 100), (0.20, 250), (0.05, 1000)):
            p = Fraction(str(price))
            exact_cents = Fraction(7, 100) * contracts * p * (1 - p) * 100
            expected = math.ceil(exact_cents) / 100
            assert fees.total_fee(price, contracts) == pytest.approx(expected)

    def test_curve_peaks_at_fifty_cents(self):
        fees = FeeModel()
        at_50 = fees.total_fee(0.50, 1000)
        assert at_50 > fees.total_fee(0.20, 1000)
        assert at_50 > fees.total_fee(0.80, 1000)
        assert fees.total_fee(0.99, 1000) < fees.total_fee(0.90, 1000)

    def test_symmetric_about_fifty(self):
        fees = FeeModel()
        assert fees.total_fee(0.30, 500) == pytest.approx(fees.total_fee(0.70, 500))

    def test_maker_is_cheaper_than_taker(self):
        fees = FeeModel()
        assert fees.total_fee(0.5, 1000, taker=False) < fees.total_fee(0.5, 1000, taker=True)

    def test_per_contract_cap_applies(self):
        capped = FeeModel(per_contract_cap_dollars=0.005)
        assert capped.total_fee(0.50, 100) == pytest.approx(0.50)

    def test_settlement_fee_added(self):
        fees = FeeModel(taker_coefficient=0.0, settlement_fee_per_contract=0.01)
        assert fees.total_fee(0.5, 10) == pytest.approx(0.10)

    def test_zero_and_one_are_free(self):
        fees = FeeModel()
        assert fees.total_fee(0.0, 100) == 0.0
        assert fees.total_fee(1.0, 100) == 0.0

    def test_price_is_clamped(self):
        fees = FeeModel()
        assert fees.total_fee(1.5, 10) == 0.0
        assert fees.total_fee(-0.5, 10) == 0.0

    def test_zero_contracts(self):
        assert FeeModel().total_fee(0.5, 0) == 0.0

    def test_per_contract_defaults_to_conservative_single_contract(self):
        fees = FeeModel()
        assert fees.fee_per_contract(0.62) >= fees.fee_per_contract(0.62, contracts=1000)

    def test_from_config(self):
        fees = FeeModel.from_config(
            {"taker_coefficient": 0.1, "maker_coefficient": 0.02,
             "per_contract_cap_dollars": 0.035, "settlement_fee_per_contract": 0.001,
             "assume_taker": False}
        )
        assert fees.taker_coefficient == 0.1
        assert fees.per_contract_cap_dollars == 0.035
        assert fees.assume_taker is False


class TestExpectedValue:
    def test_ev_is_prob_minus_cost(self):
        assert metrics.expected_value_per_contract(0.70, 0.62, 0.02) == pytest.approx(0.06)

    def test_ev_negative_when_overpaying(self):
        assert metrics.expected_value_per_contract(0.50, 0.62, 0.02) == pytest.approx(-0.14)

    def test_fair_price_with_no_fee_is_zero_ev(self):
        assert metrics.expected_value_per_contract(0.62, 0.62, 0.0) == pytest.approx(0.0)

    def test_fees_always_reduce_ev(self):
        without = metrics.expected_value_per_contract(0.70, 0.62, 0.0)
        with_fee = metrics.expected_value_per_contract(0.70, 0.62, 0.02)
        assert with_fee < without


class TestKelly:
    def test_matches_closed_form(self):
        """f* = (p - cost) / (1 - cost)."""
        p, price, fee = 0.70, 0.62, 0.02
        expected = (p - price - fee) / (1 - price - fee)
        assert metrics.kelly_fraction(p, price, fee) == pytest.approx(expected)

    def test_zero_when_no_edge(self):
        assert metrics.kelly_fraction(0.62, 0.62, 0.0) == 0.0

    def test_zero_when_negative_edge(self):
        assert metrics.kelly_fraction(0.30, 0.62, 0.02) == 0.0

    def test_zero_when_cost_exceeds_payout(self):
        assert metrics.kelly_fraction(0.99, 0.99, 0.05) == 0.0

    def test_certainty_approaches_full_bankroll(self):
        assert metrics.kelly_fraction(1.0, 0.50, 0.0) == pytest.approx(1.0)

    def test_bounded_to_unit_interval(self):
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            for price in (0.01, 0.5, 0.99):
                assert 0.0 <= metrics.kelly_fraction(p, price, 0.01) <= 1.0

    def test_larger_edge_gives_larger_stake(self):
        small = metrics.kelly_fraction(0.65, 0.62, 0.0)
        large = metrics.kelly_fraction(0.85, 0.62, 0.0)
        assert large > small


class TestSizing:
    def test_fractional_kelly_scales_the_stake(self):
        full = metrics.size_position(0.70, 0.62, 0.02, 1000, 1.0, 1.0)
        quarter = metrics.size_position(0.70, 0.62, 0.02, 1000, 0.25, 1.0)
        assert quarter.kelly_used == pytest.approx(full.kelly_used * 0.25)
        assert quarter.stake_dollars == pytest.approx(full.stake_dollars * 0.25, rel=1e-3)

    def test_max_stake_fraction_caps_the_position(self):
        plan = metrics.size_position(0.95, 0.10, 0.01, 1000, 1.0, 0.05)
        assert plan.kelly_used == pytest.approx(0.05)
        assert plan.stake_dollars == pytest.approx(50.0)

    def test_contracts_fit_within_the_stake(self):
        plan = metrics.size_position(0.70, 0.62, 0.02, 1000, 0.25, 0.05)
        assert plan.contracts * plan.cost_per_contract <= plan.stake_dollars + 1e-9

    def test_no_edge_means_no_position(self):
        plan = metrics.size_position(0.50, 0.62, 0.02, 1000, 0.25, 0.05)
        assert plan.kelly_used == 0.0 and plan.contracts == 0


class TestAnnualized:
    def test_longer_horizon_lowers_the_annualized_figure(self):
        near = metrics.annualized_if_win(0.90, 0.01, 5, cap=1e9)
        far = metrics.annualized_if_win(0.90, 0.01, 100, cap=1e9)
        assert near > far

    def test_cheaper_contract_annualizes_higher(self):
        """The core framing: the longshot's return IS its risk."""
        longshot = metrics.annualized_if_win(0.20, 0.02, 30, cap=1e12)
        favorite = metrics.annualized_if_win(0.97, 0.01, 30, cap=1e12)
        assert longshot > favorite

    def test_display_cap_is_applied(self):
        assert metrics.annualized_if_win(0.01, 0.0, 1, cap=100.0) == 100.0

    def test_expected_annualized_uses_model_probability(self):
        good = metrics.expected_annualized(0.80, 0.62, 0.02, 20, cap=1e9)
        bad = metrics.expected_annualized(0.40, 0.62, 0.02, 20, cap=1e9)
        assert good > 0 > bad

    def test_zero_probability_is_total_loss(self):
        assert metrics.expected_annualized(0.0, 0.5, 0.02, 10) == -1.0

    def test_sub_hour_horizon_does_not_explode(self):
        value = metrics.annualized_if_win(0.5, 0.02, 0.0, cap=100.0)
        assert value is not None and math.isfinite(value)

    def test_none_days_returns_none(self):
        assert metrics.annualized_if_win(0.5, 0.02, None) is None
