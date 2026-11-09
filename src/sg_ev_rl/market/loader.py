"""Load the merged half-hourly market table and check its time grid.

Timestamp convention (verified in reports/c/2026-11-09_price_diagnosis.md): EMC period p of
trading date D covers [D + (p-1)*30 min, D + p*30 min) SGT. The table's `timestamp` column is
the END of that interval. Use `interval_start` to align with a simulator step that starts at t.
"""
from pathlib import Path

import pandas as pd

from .paths import market_table_path
from .splits import assign_split

TZ = "Asia/Singapore"
STEP = pd.Timedelta(minutes=30)


def load_market(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or market_table_path())
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["interval_end"] = df["timestamp"]
    df["interval_start"] = df["timestamp"] - STEP
    df["split"] = assign_split(df["date"])
    check_grid(df)
    return df


def check_grid(df: pd.DataFrame) -> None:
    """Raise if the table is not a complete, ordered, tz-aware half-hourly grid."""
    ts = df["timestamp"]
    if ts.dt.tz is None or str(ts.dt.tz) != TZ:
        raise ValueError(f"timestamp must be tz-aware {TZ}, got {ts.dt.tz}")
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("timestamps must be strictly increasing")
    gaps = ts.diff().dropna()
    if not (gaps == STEP).all():
        raise ValueError(f"{int((gaps != STEP).sum())} gaps are not 30 minutes")
    counts = df.groupby("date")["period"].agg(["count", "min", "max"])
    if not ((counts["count"] == 48) & (counts["min"] == 1) & (counts["max"] == 48)).all():
        raise ValueError("every trading date must have periods 1..48")
    expected_end = pd.to_datetime(df["date"].astype(str)).dt.tz_localize(TZ) + df["period"] * STEP
    if not (expected_end.reset_index(drop=True) == ts.reset_index(drop=True)).all():
        raise ValueError("timestamp is not the end of period p (date + p*30 min)")
