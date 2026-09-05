"""CSV / Parquet export for external tools."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..logging_utils import get_logger
from .queries import Analysis

log = get_logger(__name__)

DEFAULT_TABLES = ("markets", "snapshots", "signals", "trades", "candles",
                  "predict_availability", "model_estimates", "runs")


def export_dataset(
    db_path: str | Path,
    out_dir: str | Path = "exports",
    fmt: str = "csv",
    tables: Iterable[str] = DEFAULT_TABLES,
) -> list[Path]:
    """Dump tables to ``out_dir``. Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with Analysis(db_path) as analysis:
        available = set(analysis.tables()["name"])
        for table in tables:
            if table not in available:
                log.warning("skipping unknown table %r", table)
                continue
            df = analysis.sql(f"SELECT * FROM {table}")
            path = _write(df, out / f"{table}.{_suffix(fmt)}", fmt)
            written.append(path)
            log.info("exported %-22s %7d rows -> %s", table, len(df), path)

        # The joined view is what most external analysis actually wants.
        joined = analysis.signals(tradeable_only=False)
        if not joined.empty:
            joined = joined.copy()
            joined["score_components"] = joined["score_components"].astype(str)
            path = _write(joined, out / f"signals_joined.{_suffix(fmt)}", fmt)
            written.append(path)
            log.info("exported %-22s %7d rows -> %s", "signals_joined", len(joined), path)

    return written


def _suffix(fmt: str) -> str:
    return "parquet" if fmt.lower() == "parquet" else "csv"


def _write(df: pd.DataFrame, path: Path, fmt: str) -> Path:
    if fmt.lower() == "parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            log.warning("pyarrow not installed; falling back to CSV for %s", path.name)
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path
