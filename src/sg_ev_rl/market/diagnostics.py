"""Descriptive diagnostics of USEP and its candidate drivers.

Descriptive statistics may cover every split, including test (allowed for window C); nothing here
fits parameters.
"""
import numpy as np
import pandas as pd

SPIKE_LEVELS = (500.0, 1000.0)
GEN_COLS = ["gen_ccgt_cogen_trigen_mwh", "gen_st_mwh", "gen_gt_mwh", "gen_igs_mwh",
            "gen_import_mwh", "gen_ess_mwh", "gen_others_mwh"]


def lag_correlation(df: pd.DataFrame, a: str, b: str, lags=range(-2, 3)) -> pd.Series:
    """corr(a_t, b_{t+lag}); the best lag shows how b is aligned relative to a."""
    return pd.Series({lag: df[a].corr(df[b].shift(-lag)) for lag in lags}, name=f"corr({a},{b}[t+lag])")


def solar_peak(df: pd.DataFrame) -> pd.Series:
    """Mean solar by period, plus the solar-weighted centre of the profile in hours after midnight
    assuming period p covers [(p-1)*30, p*30) minutes."""
    prof = df.groupby("period")["solar_mw"].mean()
    mid_hours = (prof.index - 0.5) * 0.5
    centre = float((prof * mid_hours).sum() / prof.sum())
    return pd.Series({"peak_period": int(prof.idxmax()), "centre_hours_if_period_start_based": centre})


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = pd.to_datetime(out["date"].astype(str)).dt.to_period("M")
    out["start_hour"] = out["interval_start"].dt.hour + out["interval_start"].dt.minute / 60
    out["day_type"] = np.where(out["is_weekend"], "weekend", "weekday")
    return out


def price_summary(df: pd.DataFrame, by: str | list[str], col: str = "usep_price") -> pd.DataFrame:
    g = df.groupby(by)[col]
    out = pd.DataFrame({
        "n": g.size(),
        "mean": g.mean(),
        "median": g.median(),
        "p10": g.quantile(0.10),
        "p90": g.quantile(0.90),
        "p99": g.quantile(0.99),
        "max": g.max(),
    })
    for lvl in SPIKE_LEVELS:
        out[f"share_gt_{int(lvl)}"] = g.apply(lambda s, lvl=lvl: float((s > lvl).mean()))
    # How much of the mean comes from periods above 500: mean minus the mean of the capped series.
    out["spike_contrib_gt_500"] = g.mean() - g.apply(lambda s: s.clip(upper=SPIKE_LEVELS[0]).mean())
    return out


def driver_summary(df: pd.DataFrame, by: str = "month") -> pd.DataFrame:
    gen = df[GEN_COLS].clip(lower=0)
    total = gen.sum(axis=1).replace(0, np.nan)
    tmp = df.assign(
        import_share=df["gen_import_mwh"].clip(lower=0) / total,
        ccgt_share=df["gen_ccgt_cogen_trigen_mwh"].clip(lower=0) / total,
        solar_share_of_gross=df["solar_mw"] / df["demand_gross_mw"],
    )
    g = tmp.groupby(by)
    return pd.DataFrame({
        "usep_mean": g["usep_price"].mean(),
        "usep_median": g["usep_price"].median(),
        "demand_gross_mean_mw": g["demand_gross_mw"].mean(),
        "demand_gross_max_mw": g["demand_gross_mw"].max(),
        "demand_net_mean_mw": g["demand_mw"].mean(),
        "solar_mean_mw": g["solar_mw"].mean(),
        "solar_share_of_gross": g["solar_share_of_gross"].mean(),
        "ccgt_share": g["ccgt_share"].mean(),
        "import_share": g["import_share"].mean(),
        "regulation_median": g["regulation_price"].median(),
        "reserve_conresa_median": g["reserve_price_CONRESA"].median(),
        "wep_minus_usep_mean": (g["wep_price"].mean() - g["usep_price"].mean()),
    })


def daily_spread(df: pd.DataFrame, hours: float = 4.0, col: str = "usep_price") -> pd.Series:
    """Per trading date: mean of the top `hours` minus mean of the bottom `hours` of half-hours."""
    k = int(hours * 2)

    def one(s):
        v = np.sort(s.to_numpy())
        return v[-k:].mean() - v[:k].mean()

    return df.groupby("date")[col].apply(one).rename(f"spread_top{k}_bottom{k}")
