"""Price forecasts for the rolling MPC, made at a decision time `now` from prices up to `now` only.

Methods (each forecasts every half-hour s >= now; the price at s = now is taken as known, as in the
RL observation and the B2 rule, which both see the current period's price):

    perfect    the realised price (look-ahead; an upper bound, not a baseline)
    last_day   price at the same slot on the most recent day with that slot at or before now
    last_week  price at the same slot one week back (or more, for horizons beyond a week)
    profile7   mean of the same slot over the 7 most recent days at or before now

None of the non-perfect methods reads a price after `now`; tests/sim/test_mpc.py checks this.
Where the market table has too little history (the first days of 2025), profile7 averages the days
it has, last_week falls back to last_day, and last_day falls back to the price at `now`.
"""
import warnings

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=30)
DAY, WEEK = 48, 336
METHODS = ("perfect", "last_day", "last_week", "profile7")


class PriceForecast:
    def __init__(self, market: pd.DataFrame, method: str = "profile7", cols=("usep_price", "reserve_price_CONRESA")):
        if method not in METHODS:
            raise ValueError(f"unknown forecast method {method!r}, expected one of {METHODS}")
        self.method = method
        idx = pd.DatetimeIndex(market["interval_start"])
        self.series = {c: pd.Series(market[c].to_numpy(float), index=idx) for c in cols}

    def predict(self, col: str, now: pd.Timestamp, steps: pd.DatetimeIndex) -> np.ndarray:
        """Forecast of `col` at each timestamp in `steps` (all >= now), made at `now`."""
        h = np.asarray((steps - now) / STEP).round().astype(int)
        if (h < 0).any():
            raise ValueError("forecast steps must not be before now")
        if self.method == "perfect":
            return self._lookup(col, [steps])
        k_day = -(-h // DAY)  # ceil(h / 48): smallest whole-day shift that lands at or before now
        if self.method == "profile7":
            out = self._lookup(col, [steps - (k_day + j) * DAY * STEP for j in range(7)])
        elif self.method == "last_week":
            out = self._lookup(col, [steps - (-(-h // WEEK)) * WEEK * STEP])
        else:
            out = np.full(len(steps), np.nan)
        for shifted in (steps - k_day * DAY * STEP, steps - h * STEP):  # fallbacks: last_day, then now
            out = np.where(np.isnan(out), self._lookup(col, [shifted]), out)
        now_ = h == 0
        out[now_] = self._lookup(col, [steps[now_]])
        if np.isnan(out).any():
            raise ValueError(f"market history does not cover the {self.method} forecast of {col}")
        return out

    def _lookup(self, col: str, shifted: list) -> np.ndarray:
        s = self.series[col]
        vals = np.stack([s.reindex(ts).to_numpy(float) for ts in shifted])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns stay NaN
            return np.nanmean(vals, axis=0)
