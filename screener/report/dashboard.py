"""Self-contained HTML dashboard generation.

Everything - CSS, JS, charts - is inlined into a single file so it opens on a
phone with no network, no CDN and no build step. Charts are hand-rolled inline
SVG for the same reason.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from ..logging_utils import get_logger

log = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


# --------------------------------------------------------------- formatting

def _pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value) * 100.0
    except (TypeError, ValueError):
        return "—"
    return f"{number:+.{digits}f}%" if signed else f"{number:.{digits}f}%"


def _money(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"${float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _days(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number < 1:
        return f"{number * 24:.0f}h"
    return f"{number:.1f}d"


def _annualized(value: Any, cap: float) -> str:
    """Annualised returns are capped for display; say so when we hit the cap."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number >= cap:
        return f">{cap * 100:,.0f}%"
    return f"{number * 100:,.0f}%"


def _local_time(value: Any, tz_name: str) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    utc_text = parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        from zoneinfo import ZoneInfo

        local = parsed.astimezone(ZoneInfo(tz_name))
        return f"{local.strftime('%a %d %b %Y, %H:%M')} {tz_name} ({utc_text})"
    except Exception:
        return utc_text


# -------------------------------------------------------------- SVG charts

def _svg_histogram(
    values: Sequence[float], bins: int = 20, width: int = 1000, height: int = 130,
    zero_centre: bool = False, fmt: str = "{:.2f}",
) -> Markup | None:
    """Inline SVG histogram. Returns None when there is nothing to draw."""
    numbers = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(numbers) < 2:
        return None

    low, high = min(numbers), max(numbers)
    if zero_centre:
        bound = max(abs(low), abs(high)) or 1.0
        low, high = -bound, bound
    if high <= low:
        high = low + 1e-6

    counts = [0] * bins
    step = (high - low) / bins
    for value in numbers:
        index = min(bins - 1, max(0, int((value - low) / step)))
        counts[index] += 1
    peak = max(counts) or 1

    pad_x, pad_top, pad_bottom = 4, 8, 20
    plot_h = height - pad_top - pad_bottom
    bar_w = (width - 2 * pad_x) / bins

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" preserveAspectRatio="none" style="display:block">'
    ]
    for i, count in enumerate(counts):
        bar_h = (count / peak) * plot_h
        x = pad_x + i * bar_w
        y = pad_top + (plot_h - bar_h)
        centre = low + (i + 0.5) * step
        colour = "var(--accent)"
        if zero_centre:
            colour = "var(--pos)" if centre > 0 else "var(--neg)"
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0.0, bar_w - 1.5):.2f}" '
            f'height="{max(0.0, bar_h):.2f}" fill="{colour}" opacity="0.78" rx="1.5">'
            f'<title>{fmt.format(centre)}: {count}</title></rect>'
        )
    baseline = pad_top + plot_h
    parts.append(
        f'<line x1="{pad_x}" y1="{baseline}" x2="{width - pad_x}" y2="{baseline}" '
        f'stroke="var(--border)" stroke-width="1"/>'
    )
    for frac, anchor in ((0.0, "start"), (0.5, "middle"), (1.0, "end")):
        label = fmt.format(low + (high - low) * frac)
        x = pad_x + (width - 2 * pad_x) * frac
        parts.append(
            f'<text x="{x:.1f}" y="{height - 6}" font-size="11" fill="var(--muted)" '
            f'text-anchor="{anchor}">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def _svg_sparkline(
    points: Sequence[tuple[Any, float]], width: int = 560, height: int = 60
) -> Markup | None:
    """Implied-probability sparkline from snapshot history."""
    values = [float(p[1]) for p in points if p[1] is not None]
    if len(values) < 2:
        return None

    low, high = min(values), max(values)
    if high - low < 0.02:  # keep a flat line visually flat, not noise-amplified
        mid = (high + low) / 2
        low, high = mid - 0.01, mid + 0.01
    span = high - low

    pad = 5
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    step = plot_w / (len(values) - 1)
    coords = [
        (pad + i * step, pad + plot_h - ((v - low) / span) * plot_h)
        for i, v in enumerate(values)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = f"{coords[0][0]:.1f},{height - pad:.1f} {line} {coords[-1][0]:.1f},{height - pad:.1f}"
    rising = values[-1] >= values[0]
    colour = "var(--pos)" if rising else "var(--neg)"

    return Markup(
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" role="img" style="display:block">'
        f'<polygon points="{area}" fill="{colour}" opacity="0.12"/>'
        f'<polyline points="{line}" fill="none" stroke="{colour}" stroke-width="1.8" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{coords[-1][0]:.1f}" cy="{coords[-1][1]:.1f}" r="2.6" fill="{colour}"/>'
        f'<text x="{pad}" y="{height - 1}" font-size="10" fill="var(--muted)">'
        f'{low * 100:.0f}%</text>'
        f'<text x="{width - pad}" y="{height - 1}" font-size="10" fill="var(--muted)" '
        f'text-anchor="end">now {values[-1] * 100:.0f}%</text>'
        f"</svg>"
    )


# ------------------------------------------------------------------ builder

def _rules_text(row: Mapping[str, Any]) -> str:
    """Full settlement text, never truncated."""
    parts = []
    for key, label in (("rules_primary", ""), ("rules_secondary", "Additional terms:\n")):
        value = row.get(key)
        if value:
            parts.append(f"{label}{str(value).strip()}")
    if not parts:
        return (
            "No settlement rules were returned by the API for this contract. "
            "Do not trade it until you have read the rules in the Predict app."
        )
    return "\n\n".join(parts)


def _strike_display(row: Mapping[str, Any]) -> str | None:
    floor_strike, cap_strike = row.get("floor_strike"), row.get("cap_strike")
    strike_type = row.get("strike_type")
    if floor_strike is not None and cap_strike is not None:
        return f"between {floor_strike:g} and {cap_strike:g}"
    if floor_strike is not None:
        return f"{strike_type or 'above'} {floor_strike:g}"
    if cap_strike is not None:
        return f"{strike_type or 'below'} {cap_strike:g}"
    return None


def build_candidate(
    row: Mapping[str, Any],
    weights: Mapping[str, float],
    history: Sequence[tuple[Any, float]] = (),
    tz_name: str = "UTC",
    max_annualized: float = 100.0,
    kelly_setting: float = 0.25,
) -> dict[str, Any]:
    """Shape one signal row for the template."""
    components = row.get("score_components") or {}
    if isinstance(components, str):
        try:
            components = json.loads(components)
        except ValueError:
            components = {}

    bars = [
        (name.replace("_", " ").title(), float(components.get(name, 0.0) or 0.0), weight)
        for name, weight in weights.items()
    ]

    edge_value = row.get("edge")
    candidate = dict(row)
    candidate.update(
        {
            "abs_edge": abs(float(edge_value)) if edge_value is not None else -99,
            "implied_pct": _pct(row.get("implied_prob")),
            "model_pct": _pct(row.get("model_prob")),
            "edge_pct": _pct(edge_value, signed=True),
            "entry_display": _money(row.get("entry_price"), 2),
            "ev_display": _money(row.get("ev_per_contract")),
            "ev_pct_display": _pct(row.get("ev_pct_of_cost")),
            "fee_display": _money(row.get("fee_per_contract")),
            "ann_display": _annualized(row.get("annualized_if_win"), max_annualized),
            "exp_ann_display": _annualized(row.get("expected_annualized"), max_annualized),
            "kelly_full_display": _pct(row.get("kelly_fraction_full")),
            "kelly_used_display": _pct(row.get("kelly_fraction_used")),
            "kelly_setting_display": f"{kelly_setting:g}×",
            "stake_display": _money(row.get("stake_dollars"), 2),
            "confidence_display": _pct(row.get("model_confidence"), 0),
            "days_display": _days(row.get("days_to_close")),
            "close_display": _local_time(row.get("close_time"), tz_name),
            "momentum_display": _pct(row.get("momentum_24h"), 1, signed=True),
            "stale_display": _days((row.get("stale_hours") or 0) / 24.0),
            "rules_display": _rules_text(row),
            "strike_display": _strike_display(row),
            "component_bars": bars,
            "sparkline": _svg_sparkline(history),
        }
    )
    return candidate


def render_dashboard(
    signals: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    histories: Mapping[str, Sequence[tuple[Any, float]]] | None = None,
    weights: Mapping[str, float] | None = None,
    output_path: str | Path = "out/dashboard.html",
    tz_name: str = "UTC",
    max_rows: int = 400,
    max_annualized: float = 100.0,
    kelly_setting: float = 0.25,
    title: str = "Prediction-market screener",
) -> Path:
    """Render the dashboard to a single self-contained HTML file."""
    histories = histories or {}
    weights = weights or {
        "edge": 0.40, "liquidity": 0.20, "spread": 0.15,
        "annualized": 0.15, "momentum": 0.10,
    }

    rows = list(signals)[:max_rows]
    candidates = [
        build_candidate(
            row, weights, histories.get(str(row.get("ticker")), ()),
            tz_name, max_annualized, kelly_setting,
        )
        for row in rows
    ]

    categories = sorted({c.get("category") for c in candidates if c.get("category")})
    charts = {
        "score_hist": _svg_histogram(
            [c.get("score") for c in candidates], bins=20, fmt="{:.2f}"
        ),
        "edge_hist": _svg_histogram(
            [c.get("edge") for c in candidates if c.get("edge") is not None],
            bins=21, zero_centre=True, fmt="{:+.0%}",
        ),
    }

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("dashboard.html.j2")
    rendered = template.render(
        title=title,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        run_id=stats.get("run_id", "—"),
        stats=stats,
        candidates=candidates,
        categories=categories,
        charts=charts,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    size_kb = path.stat().st_size / 1024
    log.info("dashboard written: %s (%d candidates, %.0f KB)", path, len(candidates), size_kb)
    return path
