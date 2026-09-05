"""Trading-fee model.

Kalshi's published taker fee is::

    fee = ceil(coefficient * C * P * (1 - P))     rounded up to the next cent

where ``C`` is the contract count and ``P`` the price in dollars. The curve is
parabolic: it peaks at a 50c contract and falls toward zero at the extremes.
Maker fees run at roughly a quarter of taker.

Every coefficient is configurable because fee schedules change - verify yours
against https://kalshi.com/docs/kalshi-fee-schedule.pdf before trusting the EV
numbers this produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FeeModel:
    """Configurable Kalshi-style fee model. All amounts in USD."""

    taker_coefficient: float = 0.07
    maker_coefficient: float = 0.0175
    per_contract_cap_dollars: float | None = None
    settlement_fee_per_contract: float = 0.0
    assume_taker: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FeeModel":
        return cls(
            taker_coefficient=float(config.get("taker_coefficient", 0.07)),
            maker_coefficient=float(config.get("maker_coefficient", 0.0175)),
            per_contract_cap_dollars=(
                float(config["per_contract_cap_dollars"])
                if config.get("per_contract_cap_dollars") is not None
                else None
            ),
            settlement_fee_per_contract=float(
                config.get("settlement_fee_per_contract", 0.0)
            ),
            assume_taker=bool(config.get("assume_taker", True)),
        )

    def coefficient(self, taker: bool | None = None) -> float:
        is_taker = self.assume_taker if taker is None else taker
        return self.taker_coefficient if is_taker else self.maker_coefficient

    def total_fee(
        self, price_dollars: float, contracts: int = 1, taker: bool | None = None
    ) -> float:
        """Total fee for ``contracts`` bought at ``price_dollars``."""
        if contracts <= 0:
            return 0.0
        price = min(max(float(price_dollars), 0.0), 1.0)
        raw_cents = self.coefficient(taker) * contracts * price * (1.0 - price) * 100.0
        # Round away binary-float noise before the ceiling. Without this,
        # `1 - 0.70` (= 0.2999...93) rounds a cent differently from `1 - 0.30`,
        # so the fee curve would not be symmetric about 50c as the formula is.
        fee = math.ceil(round(raw_cents, 9)) / 100.0
        if self.per_contract_cap_dollars is not None:
            fee = min(fee, self.per_contract_cap_dollars * contracts)
        return fee + self.settlement_fee_per_contract * contracts

    def fee_per_contract(
        self, price_dollars: float, contracts: int = 1, taker: bool | None = None
    ) -> float:
        """Per-contract fee.

        Defaults to a single contract, which is the conservative case: the
        cent-rounding is amortised across a larger order, so screening on the
        one-contract fee never understates cost.
        """
        contracts = max(1, int(contracts))
        return self.total_fee(price_dollars, contracts, taker) / contracts

    def round_trip_fee_per_contract(
        self, entry_price: float, exit_price: float | None = None,
        taker: bool | None = None,
    ) -> float:
        """Fee to enter and later exit, for contracts sold before resolution.

        Held to settlement there is no exit fee, so this is only relevant when
        you plan to trade out early.
        """
        exit_price = entry_price if exit_price is None else exit_price
        return self.fee_per_contract(entry_price, taker=taker) + self.fee_per_contract(
            exit_price, taker=taker
        )
