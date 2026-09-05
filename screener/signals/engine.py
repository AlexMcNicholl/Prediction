"""Signal / screening engine.

Produces one signal record per contract. Every number it emits is a
DESCRIPTION of the contract, not a recommendation, and the composite score is
a transparent weighted sum whose components are all stored alongside it.

Two conventions matter for reading the output:

* ``implied_prob`` uses the BID/ASK MIDPOINT - the fairest single number for
  "what the market thinks".
* ``entry_price`` and everything derived from it (EV, Kelly, annualised
  returns) uses the ASK you would actually pay, plus fees. Screening on the
  mid would systematically overstate every edge by half the spread.

Contracts with no fair-value model are scored on structure alone and are
capped well below modelled contracts by construction: with no independent
estimate you have no stated reason to disagree with the price, and
disagreeing for a defensible reason is the only real edge available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..logging_utils import get_logger
from ..models.base import Estimate
from . import metrics
from .fees import FeeModel

log = get_logger(__name__)

DEFAULT_WEIGHTS = {
    "edge": 0.40,
    "liquidity": 0.20,
    "spread": 0.15,
    "annualized": 0.15,
    "momentum": 0.10,
}


@dataclass
class SignalConfig:
    edge_threshold: float = 0.05
    spread_threshold_cents: int = 3
    min_volume: int = 100
    min_open_interest: int = 100
    min_liquidity_dollars: float = 500.0
    longshot_low: float = 0.10
    longshot_high: float = 0.90
    staleness_hours: float = 24.0
    momentum_lookback_hours: float = 24.0
    momentum_threshold: float = 0.05
    max_annualized_display: float = 100.0
    score_weights: Mapping[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.score_weights:
            self.score_weights = dict(DEFAULT_WEIGHTS)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SignalConfig":
        return cls(
            edge_threshold=float(config.get("edge_threshold", 0.05)),
            spread_threshold_cents=int(config.get("spread_threshold_cents", 3)),
            min_volume=int(config.get("min_volume", 100)),
            min_open_interest=int(config.get("min_open_interest", 100)),
            min_liquidity_dollars=float(config.get("min_liquidity_dollars", 500)),
            longshot_low=float(config.get("longshot_low", 0.10)),
            longshot_high=float(config.get("longshot_high", 0.90)),
            staleness_hours=float(config.get("staleness_hours", 24)),
            momentum_lookback_hours=float(config.get("momentum_lookback_hours", 24)),
            momentum_threshold=float(config.get("momentum_threshold", 0.05)),
            max_annualized_display=float(config.get("max_annualized_display", 100.0)),
            score_weights=dict(config.get("score_weights") or DEFAULT_WEIGHTS),
        )


class SignalEngine:
    """Computes the full signal set for a batch of contracts."""

    def __init__(
        self,
        signal_config: SignalConfig,
        fee_model: FeeModel,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        max_stake_fraction: float = 0.05,
    ) -> None:
        self.cfg = signal_config
        self.fees = fee_model
        self.bankroll = float(bankroll)
        self.kelly_fraction = float(kelly_fraction)
        self.max_stake_fraction = float(max_stake_fraction)

    # ------------------------------------------------------------- per row

    def compute(
        self,
        market: Mapping[str, Any],
        snapshot: Mapping[str, Any] | None,
        estimate: Estimate | None = None,
        history: Sequence[Mapping[str, Any]] = (),
        days_to_close: float | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        ticker = str(market.get("ticker"))
        snapshot = snapshot or {}

        yes_bid = snapshot.get("yes_bid")
        yes_ask = snapshot.get("yes_ask")
        no_ask = snapshot.get("no_ask")
        last_price = snapshot.get("last_price")

        mid = metrics.mid_price_cents(yes_bid, yes_ask)
        implied = metrics.implied_prob(mid if mid is not None else last_price)
        spread = metrics.spread_cents(yes_bid, yes_ask)

        if days_to_close is None:
            days_to_close = _days_until(market.get("close_time"), now)

        model_prob = estimate.prob if estimate else None
        model_name = estimate.model if estimate else None
        model_conf = estimate.confidence if estimate else None
        edge_value = metrics.edge(model_prob, implied)

        # Buying NO at p is the same trade as selling YES at 100-p, so a
        # missing side can be recovered from the other one. Without this, a
        # payload carrying only YES prices would force every contract to the
        # YES side even when NO is the obviously better trade.
        yes_ask_eff = yes_ask if yes_ask is not None else (
            100 - snapshot["no_bid"] if snapshot.get("no_bid") is not None else None
        )
        no_ask_eff = no_ask if no_ask is not None else (
            100 - yes_bid if yes_bid is not None else None
        )

        side, entry_price, win_prob = self._choose_side(
            model_prob, yes_ask_eff, no_ask_eff, implied
        )

        fee = ev = ev_pct = None
        kelly_full = kelly_used = stake = None
        contracts = None
        exp_annualized = None

        if entry_price is not None:
            fee = self.fees.fee_per_contract(entry_price)
            annualized_win = metrics.annualized_if_win(
                entry_price, fee, days_to_close, self.cfg.max_annualized_display
            )
            if win_prob is not None:
                ev = metrics.expected_value_per_contract(win_prob, entry_price, fee)
                cost = entry_price + fee
                ev_pct = ev / cost if cost > 0 else None
                plan = metrics.size_position(
                    win_prob, entry_price, fee, self.bankroll,
                    self.kelly_fraction, self.max_stake_fraction,
                )
                kelly_full = plan.kelly_full
                kelly_used = plan.kelly_used
                stake = plan.stake_dollars
                contracts = plan.contracts
                exp_annualized = metrics.expected_annualized(
                    win_prob, entry_price, fee, days_to_close,
                    self.cfg.max_annualized_display,
                )
        else:
            annualized_win = None

        momentum_value = metrics.momentum(history, self.cfg.momentum_lookback_hours)
        stale = metrics.staleness_hours(history, now)

        volume = snapshot.get("volume") or 0
        open_interest = snapshot.get("open_interest") or 0
        # Kalshi reports `liquidity` in cents.
        liquidity_dollars = (snapshot.get("liquidity") or 0) / 100.0

        liquidity_thin = (
            volume < self.cfg.min_volume
            or open_interest < self.cfg.min_open_interest
            or liquidity_dollars < self.cfg.min_liquidity_dollars
        )

        components = self._score_components(
            edge_value, volume, open_interest, spread, exp_annualized,
            momentum_value, actionable=(ev is not None and ev > 0),
        )
        score = self._composite(components)

        return {
            "ticker": ticker,
            "implied_prob": implied,
            "model_prob": model_prob,
            "model_name": model_name,
            "model_confidence": model_conf,
            "edge": edge_value,
            "edge_flag": int(
                edge_value is not None and abs(edge_value) >= self.cfg.edge_threshold
            ),
            "side": side,
            "entry_price": entry_price,
            "fee_per_contract": fee,
            "ev_per_contract": ev,
            "ev_pct_of_cost": ev_pct,
            "kelly_fraction_full": kelly_full,
            "kelly_fraction_used": kelly_used,
            "stake_dollars": stake,
            "contracts": contracts,
            "days_to_close": days_to_close,
            "annualized_if_win": annualized_win,
            "expected_annualized": exp_annualized,
            "spread_cents": spread,
            "spread_flag": int(
                spread is not None and spread > self.cfg.spread_threshold_cents
            ),
            "liquidity_flag": int(liquidity_thin),
            "longshot_flag": int(
                metrics.longshot_flag(implied, self.cfg.longshot_low, self.cfg.longshot_high)
            ),
            "momentum_24h": momentum_value,
            "momentum_flag": int(
                momentum_value is not None
                and abs(momentum_value) >= self.cfg.momentum_threshold
            ),
            "stale_hours": stale,
            "stale_flag": int(stale is not None and stale >= self.cfg.staleness_hours),
            "score": score,
            "score_components": components,
            "notes": estimate.notes if estimate else None,
        }

    # ------------------------------------------------------------- helpers

    def _choose_side(
        self,
        model_prob: float | None,
        yes_ask: int | None,
        no_ask: int | None,
        implied: float | None,
    ) -> tuple[str | None, float | None, float | None]:
        """Pick the side to price, and the price actually payable for it.

        With a model, choose whichever side has the higher EV (which may still
        be negative - that is reported, not hidden). Without a model, default
        to the YES ask so structural metrics are still computable.
        """
        yes_price = yes_ask / 100.0 if yes_ask else None
        no_price = no_ask / 100.0 if no_ask else None

        if model_prob is None:
            if yes_price:
                return "yes", yes_price, None
            if no_price:
                return "no", no_price, None
            return None, None, None

        candidates: list[tuple[float, str, float, float]] = []
        if yes_price:
            fee = self.fees.fee_per_contract(yes_price)
            candidates.append(
                (metrics.expected_value_per_contract(model_prob, yes_price, fee),
                 "yes", yes_price, model_prob)
            )
        if no_price:
            fee = self.fees.fee_per_contract(no_price)
            candidates.append(
                (metrics.expected_value_per_contract(1.0 - model_prob, no_price, fee),
                 "no", no_price, 1.0 - model_prob)
            )
        if not candidates:
            return None, None, None
        best = max(candidates, key=lambda c: c[0])
        return best[1], best[2], best[3]

    def _score_components(
        self,
        edge_value: float | None,
        volume: float,
        open_interest: float,
        spread: int | None,
        exp_annualized: float | None,
        momentum_value: float | None,
        actionable: bool = False,
    ) -> dict[str, Any]:
        """Each component on [0, 1]. Absent inputs contribute 0, never a guess.

        ``actionable`` is whether the best available side is positive-EV after
        fees. Edge you cannot capture is not a reason to look at a contract, so
        it scores zero - otherwise a contract whose spread swallows the whole
        disagreement would rank top purely for disagreeing.
        """
        edge_component = (
            metrics.normalized(edge_value, 2.0 * self.cfg.edge_threshold)
            if actionable else 0.0
        )

        volume_part = min(1.0, volume / max(1.0, 4.0 * self.cfg.min_volume))
        oi_part = min(1.0, open_interest / max(1.0, 4.0 * self.cfg.min_open_interest))
        liquidity_component = 0.5 * volume_part + 0.5 * oi_part

        if spread is None:
            spread_component = 0.0
        else:
            spread_component = 1.0 - min(
                1.0, max(0, spread) / max(1.0, 2.0 * self.cfg.spread_threshold_cents)
            )

        # Deliberately uses the MODEL-based expected return, not the
        # annualised-if-win figure. Ranking on "if it wins" would put every
        # longshot at the top purely for being unlikely.
        annualized_component = (
            min(1.0, max(0.0, exp_annualized)) if exp_annualized is not None else 0.0
        )
        momentum_component = metrics.normalized(
            momentum_value, 2.0 * self.cfg.momentum_threshold
        )

        return {
            "edge": round(edge_component, 4),
            "liquidity": round(liquidity_component, 4),
            "spread": round(spread_component, 4),
            "annualized": round(annualized_component, 4),
            "momentum": round(momentum_component, 4),
            "has_model": edge_value is not None,
            "actionable": actionable,
        }

    def _composite(self, components: Mapping[str, Any]) -> float:
        """Weighted sum over the FULL weight denominator.

        Missing components contribute zero rather than being excluded from the
        denominator, so a contract with no model cannot rank alongside one
        where you actually have an independent view.
        """
        weights = self.cfg.score_weights or DEFAULT_WEIGHTS
        total_weight = sum(float(w) for w in weights.values()) or 1.0
        score = sum(
            float(weights.get(key, 0.0)) * float(components.get(key, 0.0) or 0.0)
            for key in weights
        )
        return round(score / total_weight, 4)

    # --------------------------------------------------------------- batch

    def compute_all(
        self,
        markets: Sequence[Mapping[str, Any]],
        snapshots: Mapping[str, Mapping[str, Any]],
        estimates: Mapping[str, Estimate] | None = None,
        histories: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        availability: Mapping[str, float | None] | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        estimates = estimates or {}
        histories = histories or {}
        availability = availability or {}

        results: list[dict[str, Any]] = []
        for market in markets:
            ticker = str(market.get("ticker"))
            try:
                results.append(
                    self.compute(
                        market,
                        snapshots.get(ticker),
                        estimates.get(ticker),
                        histories.get(ticker, ()),
                        availability.get(ticker),
                        now,
                    )
                )
            except Exception as exc:
                log.error("signal computation failed for %s: %s", ticker, exc)
        results.sort(key=lambda r: (r.get("score") or 0.0), reverse=True)
        log.info(
            "computed %d signal rows (%d flagged on edge)",
            len(results), sum(r["edge_flag"] for r in results),
        )
        return results


def _days_until(close_time: Any, now: datetime) -> float | None:
    if not close_time:
        return None
    try:
        close = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
    except ValueError:
        return None
    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    return (close - now).total_seconds() / 86400.0
