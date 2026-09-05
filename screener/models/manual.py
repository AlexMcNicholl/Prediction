"""Manual-override estimator.

Highest-priority estimator: whatever you type in ``config.yaml`` wins over any
automated model. This is the escape hatch for contracts where you have done
the reading and the machine has not.
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import Estimate, FairValueEstimator, MarketContext


class ManualEstimator(FairValueEstimator):
    """Reads per-ticker probabilities from config.

    Config shape (any of ``models.<name>.manual_probabilities``)::

        models:
          rates:
            manual_probabilities:
              KXFED-26SEP-C25: 0.82
              KXCPI-26SEP-T2.9: {prob: 0.61, asof: "2026-09-04", notes: "own read"}
    """

    name = "manual"
    priority = 1000

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.overrides: dict[str, Any] = {}
        for section in (self.config or {}).values():
            if isinstance(section, Mapping):
                manual = section.get("manual_probabilities")
                if isinstance(manual, Mapping):
                    self.overrides.update({str(k).upper(): v for k, v in manual.items()})

    def supports(self, market: Mapping[str, Any]) -> bool:
        return str(market.get("ticker", "")).upper() in self.overrides

    def estimate(
        self, market: Mapping[str, Any], context: MarketContext
    ) -> Estimate | None:
        raw = self.overrides.get(str(market.get("ticker", "")).upper())
        if raw is None:
            return None
        if isinstance(raw, Mapping):
            prob = raw.get("prob")
            asof = raw.get("asof")
            notes = raw.get("notes")
            confidence = float(raw.get("confidence", 0.7))
        else:
            prob, asof, notes, confidence = raw, None, None, 0.7
        try:
            prob = float(prob)
        except (TypeError, ValueError):
            return None
        return Estimate(
            model=self.name,
            prob=prob,
            source="manual override (config.yaml)",
            asof=asof,
            confidence=confidence,
            notes=notes or "Hand-entered estimate; overrides all automated models.",
        )
