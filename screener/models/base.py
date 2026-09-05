"""Fair-value model interface.

Each estimator returns an INDEPENDENT probability estimate for a contract,
which the signal engine compares against the market price to compute edge.
That comparison is the only real source of edge in a prediction market, so
every estimate must be auditable: ``{prob, source, asof, confidence}`` plus a
free-text note explaining how the number was derived.

An estimator that cannot speak to a market returns ``None`` rather than a
guess. A contract with no model gets no edge signal at all - it is still
listed, but explicitly marked "no model".
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..logging_utils import get_logger

log = get_logger(__name__)


def norm_cdf(x: float) -> float:
    """Standard normal CDF (via erf, so we avoid a scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class Estimate:
    """An auditable fair-probability estimate."""

    model: str
    prob: float | None
    source: str
    asof: str | None = None
    confidence: float = 0.5
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.prob is not None:
            # Clamp rather than reject: a model at exactly 0 or 1 is a modelling
            # artefact, and a 0/1 probability makes Kelly degenerate.
            object.__setattr__(self, "prob", max(0.001, min(0.999, float(self.prob))))

    def as_record(self, ticker: str) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "model": self.model,
            "prob": self.prob,
            "source": self.source,
            "asof": self.asof,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class MarketContext:
    """Everything an estimator may need beyond the market row itself."""

    config: Mapping[str, Any] = field(default_factory=dict)
    now: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


class FairValueEstimator(ABC):
    """Base class for fair-probability estimators."""

    #: Stable identifier, stored with every estimate.
    name: str = "base"
    #: Higher wins when several estimators support the same market.
    priority: int = 0

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def supports(self, market: Mapping[str, Any]) -> bool:
        """Whether this estimator can produce a number for ``market``."""

    @abstractmethod
    def estimate(
        self, market: Mapping[str, Any], context: MarketContext
    ) -> Estimate | None:
        """Return an estimate, or None if it cannot be produced right now."""

    # -------------------------------------------------------------- helpers

    @staticmethod
    def series_of(market: Mapping[str, Any]) -> str:
        return str(market.get("series_ticker") or "").upper()

    @staticmethod
    def strike_bounds(
        market: Mapping[str, Any]
    ) -> tuple[str | None, float | None, float | None]:
        """``(strike_type, floor, cap)`` for threshold-style markets."""
        strike_type = market.get("strike_type")
        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")
        floor_val = float(floor_strike) if floor_strike is not None else None
        cap_val = float(cap_strike) if cap_strike is not None else None
        return (
            str(strike_type).lower() if strike_type else None,
            floor_val,
            cap_val,
        )

    @classmethod
    def threshold_probability(
        cls, market: Mapping[str, Any], mu: float, sigma: float
    ) -> tuple[float | None, str | None]:
        """P(outcome resolves YES) under Normal(mu, sigma) for a strike market.

        Returns ``(probability, description)``. Handles the greater / less /
        between strike shapes Kalshi uses; anything else yields ``None`` so the
        caller can fall back to "no model" rather than invent a number.
        """
        if sigma <= 0:
            return None, None
        strike_type, floor_strike, cap_strike = cls.strike_bounds(market)

        if floor_strike is not None and cap_strike is not None:
            prob = norm_cdf((cap_strike - mu) / sigma) - norm_cdf(
                (floor_strike - mu) / sigma
            )
            return prob, f"P({floor_strike} < X <= {cap_strike} | mu={mu:.4g}, sd={sigma:.4g})"

        if strike_type in {"greater", "greater_or_equal"} or (
            strike_type is None and floor_strike is not None
        ):
            if floor_strike is None:
                return None, None
            prob = 1.0 - norm_cdf((floor_strike - mu) / sigma)
            return prob, f"P(X > {floor_strike} | mu={mu:.4g}, sd={sigma:.4g})"

        if strike_type in {"less", "less_or_equal"}:
            bound = cap_strike if cap_strike is not None else floor_strike
            if bound is None:
                return None, None
            prob = norm_cdf((bound - mu) / sigma)
            return prob, f"P(X < {bound} | mu={mu:.4g}, sd={sigma:.4g})"

        if cap_strike is not None:
            prob = norm_cdf((cap_strike - mu) / sigma)
            return prob, f"P(X < {cap_strike} | mu={mu:.4g}, sd={sigma:.4g})"

        return None, None


class NoModelEstimator(FairValueEstimator):
    """Terminal fallback: explicitly declares that no model applies.

    This exists so "we have no view" is a first-class, visible state rather
    than a silently missing row.
    """

    name = "no-model"
    priority = -100

    def supports(self, market: Mapping[str, Any]) -> bool:
        return True

    def estimate(
        self, market: Mapping[str, Any], context: MarketContext
    ) -> Estimate | None:
        return Estimate(
            model=self.name,
            prob=None,
            source="none",
            confidence=0.0,
            notes=(
                "No fair-value model covers this contract. Edge is undefined - "
                "the market price is the only estimate available."
            ),
        )
