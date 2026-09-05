"""Normalise raw Kalshi payloads into the screener's storage shape.

Field mapping is deliberately tolerant: the API evolves, and a renamed or
newly-absent field should degrade one column to NULL rather than crash a run.
Anything unmapped survives in the ``raw_json`` blob for reprocessing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from ..logging_utils import get_logger

log = get_logger(__name__)


def _first(payload: Mapping[str, Any], *names: str) -> Any:
    """Return the first present, non-None value among ``names``."""
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_iso(value: Any) -> str | None:
    """Normalise a timestamp (ISO string or epoch seconds) to UTC ISO-8601."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text  # keep the raw string rather than lose the value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def derive_series_ticker(market: Mapping[str, Any]) -> str | None:
    """Series ticker, explicit if present, else derived from the ticker.

    Kalshi tickers nest as ``SERIES-EVENT-STRIKE`` (e.g. ``KXCPI-26SEP-T2.9``
    belongs to event ``KXCPI-26SEP`` in series ``KXCPI``).
    """
    explicit = _first(market, "series_ticker", "seriesTicker")
    if explicit:
        return str(explicit)
    for key in ("event_ticker", "ticker"):
        value = market.get(key)
        if value and isinstance(value, str) and "-" in value:
            return value.split("-", 1)[0]
        if value and isinstance(value, str):
            return value
    return None


def normalize_market(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Map a raw market payload to the ``markets`` table shape."""
    ticker = _first(raw, "ticker", "market_ticker")
    if not ticker:
        raise ValueError(f"Market payload has no ticker: {dict(list(raw.items())[:5])}")

    return {
        "ticker": str(ticker),
        "event_ticker": _first(raw, "event_ticker"),
        "series_ticker": derive_series_ticker(raw),
        "market_type": _first(raw, "market_type", "type"),
        "title": _first(raw, "title"),
        "subtitle": _first(raw, "subtitle", "sub_title"),
        "yes_sub_title": _first(raw, "yes_sub_title"),
        "no_sub_title": _first(raw, "no_sub_title"),
        "category": _first(raw, "category"),
        "status": _first(raw, "status"),
        "open_time": _as_iso(_first(raw, "open_time", "open_ts")),
        "close_time": _as_iso(_first(raw, "close_time", "close_ts")),
        "expected_expiration_time": _as_iso(_first(raw, "expected_expiration_time")),
        "expiration_time": _as_iso(_first(raw, "expiration_time")),
        "latest_expiration_time": _as_iso(_first(raw, "latest_expiration_time")),
        # Resolution detail - always surfaced to the user, never summarised away.
        "rules_primary": _first(raw, "rules_primary"),
        "rules_secondary": _first(raw, "rules_secondary"),
        "settlement_source": _extract_settlement_source(raw),
        "settlement_timer_seconds": _as_int(_first(raw, "settlement_timer_seconds")),
        "can_close_early": _first(raw, "can_close_early"),
        "strike_type": _first(raw, "strike_type"),
        "floor_strike": _as_float(_first(raw, "floor_strike")),
        "cap_strike": _as_float(_first(raw, "cap_strike")),
        "notional_value": _as_int(_first(raw, "notional_value")),
        "tick_size": _as_int(_first(raw, "tick_size")),
        "result": _first(raw, "result") or None,
        "settled_at": _as_iso(_first(raw, "settlement_time", "settled_time")),
    }


def _extract_settlement_source(raw: Mapping[str, Any]) -> str | None:
    """Best-effort settlement-source string.

    Kalshi has used both a plain ``settlement_source`` string and a list of
    ``{name, url}`` objects; handle either, and fall back to None.
    """
    value = _first(raw, "settlement_sources", "settlement_source")
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return value.get("name") or value.get("url") or None
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for entry in value:
            if isinstance(entry, Mapping):
                name = entry.get("name") or ""
                url = entry.get("url") or ""
                parts.append(f"{name} ({url})".strip() if url else str(name))
            elif entry:
                parts.append(str(entry))
        return "; ".join(p for p in parts if p) or None
    return str(value)


def normalize_snapshot(raw: Mapping[str, Any], ts: str) -> dict[str, Any]:
    """Map a raw market payload to a point-in-time ``snapshots`` row.

    All prices stay in integer cents, exactly as Kalshi reports them
    (a ``yes_bid`` of 62 means a 62% implied probability).
    """
    return {
        "ticker": str(_first(raw, "ticker", "market_ticker")),
        "ts": ts,
        "yes_bid": _as_int(_first(raw, "yes_bid")),
        "yes_ask": _as_int(_first(raw, "yes_ask")),
        "no_bid": _as_int(_first(raw, "no_bid")),
        "no_ask": _as_int(_first(raw, "no_ask")),
        "last_price": _as_int(_first(raw, "last_price")),
        "previous_price": _as_int(_first(raw, "previous_price")),
        "volume": _as_int(_first(raw, "volume")),
        "volume_24h": _as_int(_first(raw, "volume_24h")),
        "open_interest": _as_int(_first(raw, "open_interest")),
        "liquidity": _as_int(_first(raw, "liquidity")),
        "status": _first(raw, "status"),
        "raw": dict(raw),
    }


def normalize_event(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_ticker": _first(raw, "event_ticker", "ticker"),
        "series_ticker": _first(raw, "series_ticker"),
        "title": _first(raw, "title"),
        "sub_title": _first(raw, "sub_title", "subtitle"),
        "category": _first(raw, "category"),
        "mutually_exclusive": _first(raw, "mutually_exclusive"),
    }


def normalize_series(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ticker": _first(raw, "ticker", "series_ticker"),
        "title": _first(raw, "title"),
        "category": _first(raw, "category"),
        "frequency": _first(raw, "frequency"),
    }
