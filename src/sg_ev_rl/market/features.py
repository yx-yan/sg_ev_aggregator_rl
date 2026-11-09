"""Relative price features that use only the current and past half-hours (no look-ahead).

All windows are trailing and include the current step t. Steps with fewer than `min_periods`
past values return NaN, so an episode should start after the warm-up.
"""
import numpy as np
import pandas as pd

WEEK = 336  # half-hours in 7 days


def rolling_percentile(price: pd.Series, window: int = WEEK, min_periods: int = 48) -> pd.Series:
    """Share of the trailing window (including t) with price <= price_t, in [0, 1]."""
    return price.rolling(window, min_periods=min_periods).apply(lambda w: (w <= w[-1]).mean(), raw=True)


def rolling_zscore(price: pd.Series, window: int = WEEK, min_periods: int = 48) -> pd.Series:
    roll = price.rolling(window, min_periods=min_periods)
    return (price - roll.mean()) / roll.std(ddof=0).replace(0, np.nan)


def ratio_to_trailing_median(price: pd.Series, window: int = WEEK, min_periods: int = 48) -> pd.Series:
    return price / price.rolling(window, min_periods=min_periods).median()


def add_relative_price_features(df: pd.DataFrame, col: str = "usep_price", window: int = WEEK) -> pd.DataFrame:
    """Expects rows ordered by time on a complete half-hourly grid (see loader.check_grid)."""
    p = df[col]
    return df.assign(**{
        f"{col}_pct_{window}": rolling_percentile(p, window),
        f"{col}_z_{window}": rolling_zscore(p, window),
        f"{col}_ratio_med_{window}": ratio_to_trailing_median(p, window),
    })
