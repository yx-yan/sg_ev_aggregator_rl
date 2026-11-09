"""Date splits from the 2026-11-02 roadmap, plus the 2026-11-09 real-scenario validation period.

All dates are SGT calendar dates of the trading day (EMC DATE column); a trading day
holds periods 1..48, so period 48 ends at 00:00 of the next calendar day.
"""
from datetime import date

import pandas as pd

SPLITS = {
    "train": (date(2025, 1, 1), date(2025, 12, 31)),
    "validation": (date(2026, 1, 1), date(2026, 3, 31)),
    "test": (date(2026, 4, 1), date(2026, 9, 3)),
    # Overlaps the charger-occupancy logs; used only to check trained policies with real sessions.
    "real_validation": (date(2026, 9, 4), date(2026, 10, 25)),
}

# Data on or after this date may not be used for tuning, model choice or service choice.
TUNING_CUTOFF = date(2026, 4, 1)


def split_of(d: date) -> str | None:
    for name, (start, end) in SPLITS.items():
        if start <= d <= end:
            return name
    return None


def assign_split(dates: pd.Series) -> pd.Series:
    """Split label for each trading date (None outside every split)."""
    return pd.Series([split_of(d) for d in dates], index=dates.index, dtype="object")
