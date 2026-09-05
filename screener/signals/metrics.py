"""Per-contract signal math.

Everything here is deliberately small and pure so it can be unit-tested and
audited. No function makes a recommendation; they compute quantities that a
human then judges.

Framing that the whole module assumes: in a prediction market, profit
potential and risk are the SAME variable. A contract at $0.20 pays 5x
precisely because the market thinks it probably will not happen. The only
edge is disagreeing with the price for a defensible reason, which is what
``edge`` measures - and only where a fair-value model exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Below this, annualisation is meaningless and explodes numerically.
MIN_DAYS_FOR_ANNUALISATION = 1.0 / 24.0  # one hour
DAYS_PER_YEAR = 365.0


def implied_prob(price_cents: float | None) -> float | None:
    """Convert a Kalshi cent price to an implied probability."""
    if price_cents is None:
        return None
    return max(0.0, min(1.0, float(price_cents) / 100.0))


def mid_price_cents(yes_bid: int | None, yes_ask: int | None) -> float | None:
    if yes_bid is None or yes_ask is None:
        return None
    return (float(yes_bid) + float(yes_ask)) / 2.0


def spread_cents(yes_bid: int | None, yes_ask: int | None) -> int | None:
    if yes_bid is None or yes_ask is None:
        return None
    return int(yes_ask) - int(yes_bid)


def edge(model_prob: float | None, market_prob: float | None) -> float | None:
    """``model_prob - market_prob``. Positive favours YES, negative favours NO."""
    if model_prob is None or market_prob is None:
        return None
    return float(model_prob) - float(market_prob)


def expected_value_per_contract(
    win_prob: float, entry_price: float, fee_per_contract: float
) -> float:
    """EV in dollars for one contract held to resolution.

    A contract pays $1 if it resolves in your favour and $0 otherwise, so::

        EV = win_prob * (1 - entry) - (1 - win_prob) * entry - fee
           = win_prob - entry - fee
    """
    return float(win_prob) - float(entry_price) - float(fee_per_contract)


def kelly_fraction(
    win_prob: float, entry_price: float, fee_per_contract: float = 0.0
) -> float:
    """Full-Kelly stake as a fraction of bankroll.

    For a binary contract costing ``cost`` and paying $1::

        f* = EV / (amount won if it resolves in your favour)
           = (p - cost) / (1 - cost)

    Returns 0.0 when there is no positive edge or the position cannot profit.
    """
    cost = float(entry_price) + float(fee_per_contract)
    net_win = 1.0 - cost
    if net_win <= 0 or cost <= 0:
        return 0.0
    ev = expected_value_per_contract(win_prob, entry_price, fee_per_contract)
    if ev <= 0:
        return 0.0
    return max(0.0, min(1.0, ev / net_win))


@dataclass(frozen=True)
class StakePlan:
    """Sizing output. Informational only - execution is always manual."""

    kelly_full: float
    kelly_used: float
    stake_dollars: float
    contracts: int
    cost_per_contract: float


def size_position(
    win_prob: float,
    entry_price: float,
    fee_per_contract: float,
    bankroll: float,
    kelly_fraction_setting: float,
    max_stake_fraction: float = 1.0,
) -> StakePlan:
    """Fractional-Kelly sizing, capped by ``max_stake_fraction`` of bankroll."""
    full = kelly_fraction(win_prob, entry_price, fee_per_contract)
    used = min(full * float(kelly_fraction_setting), float(max_stake_fraction))
    used = max(0.0, used)
    stake = float(bankroll) * used
    cost = float(entry_price) + float(fee_per_contract)
    contracts = int(stake // cost) if cost > 0 else 0
    return StakePlan(
        kelly_full=full,
        kelly_used=used,
        stake_dollars=round(stake, 2),
        contracts=contracts,
        cost_per_contract=round(cost, 4),
    )


def _annualise(growth_multiple: float, days: float, cap: float | None) -> float | None:
    """Annualise a total growth multiple over ``days``, with a display cap."""
    if growth_multiple < 0 or days is None:
        return None
    days = max(float(days), MIN_DAYS_FOR_ANNUALISATION)
    if growth_multiple == 0:
        return -1.0
    try:
        value = growth_multiple ** (DAYS_PER_YEAR / days) - 1.0
    except (OverflowError, ValueError):
        return cap
    if math.isnan(value) or math.isinf(value):
        return cap
    if cap is not None:
        value = max(-1.0, min(value, cap))
    return value


def annualized_if_win(
    entry_price: float, fee_per_contract: float, days_to_close: float | None,
    cap: float | None = 100.0,
) -> float | None:
    """Annualised return assuming the contract resolves in your favour.

    Model-free, so it is comparable across every contract. This is the number
    that lets a near-certain $0.97 contract be compared against a $0.20
    longshot on a capital-efficiency basis - it says nothing about how LIKELY
    either outcome is.
    """
    cost = float(entry_price) + float(fee_per_contract)
    if cost <= 0 or days_to_close is None:
        return None
    return _annualise(1.0 / cost, days_to_close, cap)


def expected_annualized(
    win_prob: float, entry_price: float, fee_per_contract: float,
    days_to_close: float | None, cap: float | None = 100.0,
) -> float | None:
    """Annualised EV-weighted return. Requires a fair-value model.

    Since ``EV = p - cost``, the expected growth multiple is ``p / cost``.
    """
    cost = float(entry_price) + float(fee_per_contract)
    if cost <= 0 or days_to_close is None:
        return None
    return _annualise(float(win_prob) / cost, days_to_close, cap)


def longshot_flag(prob: float | None, low: float = 0.10, high: float = 0.90) -> bool:
    """Favorite-longshot bias band.

    Retail systematically overpays for longshots and underpays for favorites,
    so prices in the tails deserve extra scrutiny in BOTH directions.
    """
    if prob is None:
        return False
    return prob < low or prob > high


def momentum(
    history: Sequence[Mapping[str, Any]], lookback_hours: float = 24.0
) -> float | None:
    """Change in implied probability over ``lookback_hours``.

    ``history`` is snapshot rows ordered newest-first, each with ``ts`` and a
    usable price. Returns None when there is not enough history yet.
    """
    from datetime import datetime, timezone

    points: list[tuple[datetime, float]] = []
    for row in history:
        price = _snapshot_price(row)
        ts = _parse_ts(row.get("ts"))
        if price is None or ts is None:
            continue
        points.append((ts, price / 100.0))
    if len(points) < 2:
        return None

    points.sort(key=lambda p: p[0], reverse=True)
    latest_ts, latest_prob = points[0]
    cutoff = latest_ts.timestamp() - lookback_hours * 3600.0

    # The oldest point still inside the window; else the oldest we have.
    reference = points[-1]
    for ts, prob in points:
        if ts.timestamp() <= cutoff:
            reference = (ts, prob)
            break
    if reference[0] == latest_ts:
        return None
    return latest_prob - reference[1]


def staleness_hours(
    history: Sequence[Mapping[str, Any]], now: Any = None
) -> float | None:
    """Hours since the last observed CHANGE in price.

    A market whose price has not moved may simply be efficiently priced - or
    it may be untraded and the quote meaningless. Cross-check against volume.
    """
    from datetime import datetime, timezone

    now_dt = _parse_ts(now) or datetime.now(timezone.utc)
    points: list[tuple[datetime, float]] = []
    for row in history:
        price = _snapshot_price(row)
        ts = _parse_ts(row.get("ts"))
        if price is None or ts is None:
            continue
        points.append((ts, price))
    if not points:
        return None
    points.sort(key=lambda p: p[0], reverse=True)

    latest_price = points[0][1]
    for ts, price in points:
        if price != latest_price:
            # `ts` is the last snapshot at a DIFFERENT price, so the change
            # happened at the next snapshot after it.
            return max(0.0, (now_dt - ts).total_seconds() / 3600.0)
    # Price never changed across the whole history we hold.
    return max(0.0, (now_dt - points[-1][0]).total_seconds() / 3600.0)


def _snapshot_price(row: Mapping[str, Any]) -> float | None:
    """Best available price from a snapshot row: mid, then last, then bid."""
    for key in ("mid_price", "last_price", "yes_bid"):
        value = row.get(key) if hasattr(row, "get") else None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    yes_bid, yes_ask = row.get("yes_bid"), row.get("yes_ask")
    if yes_bid is not None and yes_ask is not None:
        return (float(yes_bid) + float(yes_ask)) / 2.0
    return None


def _parse_ts(value: Any):
    from datetime import datetime, timezone

    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def normalized(value: float | None, scale: float, floor: float = 0.0) -> float:
    """Map a magnitude onto [0, 1] against ``scale``, for score components."""
    if value is None or scale <= 0:
        return floor
    return max(0.0, min(1.0, abs(float(value)) / float(scale)))
