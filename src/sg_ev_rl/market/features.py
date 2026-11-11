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


def profile_forecast(price: pd.Series, horizon: int = 24, days: int = 7) -> pd.Series:
    """Mean price over t+1..t+horizon, forecast as the average of the same slots on the past `days` days.
    Uses prices up to t - 48 + horizon only, so it needs horizon <= 48 to stay causal."""
    assert horizon <= 48
    ahead = price.rolling(horizon).mean()  # at j: mean of j-horizon+1..j
    return pd.concat([ahead.shift(48 * d - horizon) for d in range(1, days + 1)], axis=1).mean(axis=1)


def add_regime_features(df: pd.DataFrame, col: str, window: int = WEEK, horizon: int = 24) -> pd.DataFrame:
    """Current price and the profile forecast of the next `horizon` steps, both / trailing mean (for RL)."""
    p = df[col]
    mean = p.rolling(window, min_periods=48).mean().replace(0, np.nan)
    return df.assign(**{
        f"{col}_pct_{window}": rolling_percentile(p, window),
        f"{col}_ratio_mean_{window}": p / mean,
        f"{col}_fc{horizon}_ratio": profile_forecast(p, horizon) / mean,
    })
