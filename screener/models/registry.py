"""Estimator registry and dispatch.

Adding a new fair-value model is two steps: subclass ``FairValueEstimator``,
then register it here (and switch it on in ``config.yaml`` under
``models.enabled``). Nothing else in the pipeline changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from ..logging_utils import get_logger
from .base import Estimate, FairValueEstimator, MarketContext, NoModelEstimator
from .cpi import CPIEstimator
from .equity import EquityEstimator
from .manual import ManualEstimator
from .rates import RatesEstimator
from .weather import WeatherEstimator

log = get_logger(__name__)

#: name -> factory. A factory takes the model's own config sub-section.
BUILTIN_ESTIMATORS: dict[str, Callable[[Mapping[str, Any]], FairValueEstimator]] = {
    "cpi": CPIEstimator,
    "weather": WeatherEstimator,
    "equity": EquityEstimator,
    "rates": RatesEstimator,
}


class ModelRegistry:
    """Holds the active estimators and picks the best one per market."""

    def __init__(self, estimators: Sequence[FairValueEstimator]) -> None:
        # Highest priority first, so the first supporting estimator wins.
        self.estimators = sorted(estimators, key=lambda e: e.priority, reverse=True)
        self.fallback = NoModelEstimator()

    @classmethod
    def from_config(cls, models_config: Mapping[str, Any]) -> "ModelRegistry":
        enabled = list(models_config.get("enabled") or [])
        estimators: list[FairValueEstimator] = []

        # The manual override sees the whole models section so it can pick up
        # manual_probabilities from any sub-model.
        estimators.append(ManualEstimator(models_config))

        for name in enabled:
            factory = BUILTIN_ESTIMATORS.get(str(name).lower())
            if factory is None:
                log.warning("unknown model %r in models.enabled; skipping", name)
                continue
            section = models_config.get(name) or {}
            try:
                estimators.append(factory(section))
            except Exception as exc:
                log.error("failed to construct model %r: %s", name, exc)

        log.info(
            "model registry active: %s",
            ", ".join(e.name for e in estimators) or "(manual only)",
        )
        return cls(estimators)

    # ------------------------------------------------------------- dispatch

    def estimate(
        self, market: Mapping[str, Any], context: MarketContext | None = None
    ) -> Estimate:
        """Best available estimate for ``market``, never None.

        Falls through to the explicit ``no-model`` estimate so downstream code
        always has something to record.
        """
        ctx = context or MarketContext(now=datetime.now(timezone.utc))
        for estimator in self.estimators:
            try:
                if not estimator.supports(market):
                    continue
                result = estimator.estimate(market, ctx)
            except Exception as exc:
                log.error(
                    "estimator %s raised on %s: %s",
                    estimator.name, market.get("ticker"), exc,
                )
                continue
            if result is not None and result.prob is not None:
                return result
        return self.fallback.estimate(market, ctx)  # type: ignore[return-value]

    def estimate_all(
        self, markets: Sequence[Mapping[str, Any]], now: datetime | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Estimate]:
        ctx = MarketContext(config=config or {}, now=now or datetime.now(timezone.utc))
        results: dict[str, Estimate] = {}
        for market in markets:
            ticker = market.get("ticker")
            if not ticker:
                continue
            results[str(ticker)] = self.estimate(market, ctx)
        modelled = sum(1 for e in results.values() if e.prob is not None)
        log.info(
            "fair-value models produced %d estimates over %d markets "
            "(%d have no model and therefore no edge signal)",
            modelled, len(results), len(results) - modelled,
        )
        return results
