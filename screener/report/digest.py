"""Concise digest: the top N candidates and why each is flagged.

Delivery channels are configurable (email via SMTP, Telegram via bot API).
Both are optional; with notifications disabled the digest is still built and
returned so it can be printed or written to a file.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any, Mapping, Sequence

import requests

from ..logging_utils import get_logger

log = get_logger(__name__)

DISCLAIMER = (
    "These are CANDIDATES FOR MANUAL ANALYSIS, not recommendations. "
    "In a prediction market, profit potential and risk are the same variable: "
    "a $0.20 contract pays 5x precisely because it probably won't happen. "
    "Read each contract's full settlement rules and close time before acting, "
    "and execute manually in the Wealthsimple Predict app. Contracts settle in USD."
)


def _reasons(row: Mapping[str, Any]) -> list[str]:
    """Plain-language explanation of why a contract surfaced."""
    reasons: list[str] = []
    edge = row.get("edge")
    if row.get("edge_flag") and edge is not None:
        direction = "above" if edge > 0 else "below"
        reasons.append(
            f"model {abs(edge) * 100:.0f}pp {direction} market "
            f"({(row.get('model_prob') or 0) * 100:.0f}% vs {(row.get('implied_prob') or 0) * 100:.0f}%)"
        )
    if row.get("model_prob") is None:
        reasons.append("no fair-value model — structural signals only")
    if row.get("longshot_flag"):
        reasons.append("tail price (favorite–longshot bias zone)")
    if row.get("spread_flag"):
        reasons.append(f"wide spread ({row.get('spread_cents')}c)")
    if row.get("liquidity_flag"):
        reasons.append("thin book — hard to size or exit")
    if row.get("momentum_flag") and row.get("momentum_24h") is not None:
        reasons.append(f"moved {row['momentum_24h'] * 100:+.0f}pp in 24h")
    if row.get("stale_flag") and row.get("stale_hours") is not None:
        reasons.append(f"price unchanged for {row['stale_hours']:.0f}h")
    return reasons


def build_digest(
    signals: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    top_n: int = 10,
    dashboard_url: str | None = None,
) -> tuple[str, str]:
    """Return ``(subject, body)`` as plain text."""
    top = list(signals)[:top_n]
    subject = (
        f"Predict screener: {len(top)} candidates "
        f"({stats.get('tradeable', 0)} tradeable of {stats.get('total_markets', 0)})"
    )

    lines = [
        "PREDICTION-MARKET SCREENER",
        f"Run #{stats.get('run_id', '?')} · {stats.get('generated_at', '')}",
        "",
        f"{stats.get('total_markets', 0)} markets ingested · "
        f"{stats.get('tradeable', 0)} tradeable on Predict "
        f"(<= {stats.get('term_days', 30)} days) · "
        f"{stats.get('with_model', 0)} with a fair-value model · "
        f"{stats.get('edge_flagged', 0)} edge-flagged",
        "",
        "-" * 60,
        "",
    ]

    if not top:
        lines.append("No contracts passed the Predict availability filter this run.")
        lines.append("Check predict.series_prefix_allowlist in config.yaml.")
    for i, row in enumerate(top, start=1):
        title = row.get("title") or row.get("ticker")
        if row.get("subtitle"):
            title = f"{title} — {row['subtitle']}"
        lines.append(f"{i}. {title}")
        lines.append(f"   {row.get('ticker')}  ·  score {row.get('score') or 0:.2f}")

        market_pct = (row.get("implied_prob") or 0) * 100
        pricing = f"   market {market_pct:.0f}%"
        if row.get("model_prob") is not None:
            pricing += f" · model {row['model_prob'] * 100:.0f}%"
        if row.get("side") and row.get("entry_price") is not None:
            pricing += f" · buy {str(row['side']).upper()} @ ${row['entry_price']:.2f}"
        if row.get("ev_per_contract") is not None:
            pricing += f" · EV ${row['ev_per_contract']:+.3f}/ct"
        lines.append(pricing)

        if row.get("days_to_close") is not None:
            close_line = f"   closes in {row['days_to_close']:.1f}d"
            if row.get("close_time"):
                close_line += f" ({row['close_time']})"
            lines.append(close_line)

        if row.get("stake_dollars"):
            lines.append(
                f"   sizing: {(row.get('kelly_fraction_used') or 0) * 100:.1f}% of bankroll "
                f"= ${row['stake_dollars']:.2f} ≈ {row.get('contracts') or 0} contracts"
            )

        for reason in _reasons(row):
            lines.append(f"   - {reason}")

        source = row.get("settlement_source")
        lines.append(f"   settles per: {source or 'see full rules in the Predict app'}")
        lines.append("")

    lines.extend(["-" * 60, ""])
    if dashboard_url:
        lines.append(f"Full dashboard: {dashboard_url}")
        lines.append("")
    lines.append(DISCLAIMER)
    lines.append("")
    lines.append("Read-only analysis. This system never places trades and never touches Wealthsimple.")

    return subject, "\n".join(lines)


# ------------------------------------------------------------------ senders

def send_email(
    subject: str, body: str, config: Mapping[str, Any]
) -> bool:
    """Send the digest over SMTP. Credentials come from the environment."""
    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    to_addresses = list(config.get("to_addresses") or [])
    from_address = config.get("from_address") or username

    if not (username and password and to_addresses and from_address):
        log.warning(
            "email digest skipped: need SMTP_USERNAME, SMTP_PASSWORD, "
            "notifications.email.from_address and to_addresses"
        )
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_address
    message["To"] = ", ".join(to_addresses)
    message.set_content(body)

    host = config.get("smtp_host", "smtp.gmail.com")
    port = int(config.get("smtp_port", 587))
    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(username, password)
            server.send_message(message)
        log.info("email digest sent to %d recipient(s)", len(to_addresses))
        return True
    except Exception as exc:
        log.error("email digest failed: %s", exc)
        return False


def send_telegram(body: str, config: Mapping[str, Any]) -> bool:
    """Send the digest via the Telegram bot API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = config.get("chat_id")
    if not (token and chat_id):
        log.warning(
            "telegram digest skipped: need TELEGRAM_BOT_TOKEN and "
            "notifications.telegram.chat_id"
        )
        return False

    # Telegram caps messages at 4096 characters.
    text = body if len(body) <= 4000 else body[:3960] + "\n\n[truncated — see dashboard]"
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=30,
        )
        resp.raise_for_status()
        log.info("telegram digest sent")
        return True
    except Exception as exc:
        log.error("telegram digest failed: %s", exc)
        return False


def deliver(
    subject: str, body: str, notifications: Mapping[str, Any]
) -> dict[str, bool]:
    """Send to every configured channel. Returns per-channel success."""
    if not notifications.get("enabled"):
        log.info("notifications disabled; digest built but not sent")
        return {}
    results: dict[str, bool] = {}
    for channel in notifications.get("channels") or []:
        name = str(channel).lower()
        if name == "email":
            results["email"] = send_email(subject, body, notifications.get("email") or {})
        elif name == "telegram":
            results["telegram"] = send_telegram(body, notifications.get("telegram") or {})
        else:
            log.warning("unknown notification channel %r", channel)
    return results
