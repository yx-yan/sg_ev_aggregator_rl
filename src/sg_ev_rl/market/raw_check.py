"""Independent re-read of raw EMC monthly files, to catch merge or unit errors in the merged table.

This does not reuse the build script: each file is parsed here on its own and compared cell by
cell with the merged table on (date, period).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import raw_market_dir

# merged column -> (series dir, file prefix, raw column, reduction over extra key or None)
CHECKS = {
    "usep_price": ("emc_usep_demand_forecast", "USEP", "USEP ($/MWh)", None),
    "demand_mw": ("emc_usep_demand_forecast", "USEP", "DEMAND (MW)", None),
    "solar_mw": ("emc_usep_demand_forecast", "USEP", "SOLAR(MW)", None),
    "regulation_price": ("emc_regulation_prices", "MFP", "PRICE ($/MWh)", None),
    "wep_price": ("emc_wholesale_electricity_prices", "WEP", "WEP ($/MWh)", None),
    "reserve_price_CONRESA": ("emc_reserve_prices", "MRP", "PRICE ($/MWh)", ("RESERVE GROUP", "CONRESA")),
    "reserve_price_PRIRESA": ("emc_reserve_prices", "MRP", "PRICE ($/MWh)", ("RESERVE GROUP", "PRIRESA")),
    "nodal_price_mean": ("emc_nodal_energy_prices", "MEP", "PRICE ($/MWh)", "mean"),
    "nodal_price_max": ("emc_nodal_energy_prices", "MEP", "PRICE ($/MWh)", "max"),
}


def raw_month_file(series_dir: str, prefix: str, month: pd.Period, root: Path | None = None) -> Path:
    root = root or raw_market_dir()
    return root / series_dir / "yearly" / str(month.year) / f"{prefix}_{month.strftime('%b')}-{month.year}.csv"


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["DATE"].str.strip(), format="%d-%b-%Y").dt.date
    df["period"] = df["PERIOD"].astype(int)
    return df


def raw_series(column: str, month: pd.Period, root: Path | None = None) -> pd.Series:
    series_dir, prefix, raw_col, how = CHECKS[column]
    df = read_raw(raw_month_file(series_dir, prefix, month, root))
    df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
    if isinstance(how, tuple):
        df = df[df[how[0]].str.strip() == how[1]]
    if how in ("mean", "max"):
        return df.groupby(["date", "period"])[raw_col].agg(how)
    return df.set_index(["date", "period"])[raw_col]


def compare_months(market: pd.DataFrame, months: list[pd.Period], tol: float = 1e-9,
                   root: Path | None = None) -> pd.DataFrame:
    """One row per (column, month): rows compared, mismatches, max abs difference, raw-only keys."""
    table = market.set_index(["date", "period"])
    rows = []
    for column in CHECKS:
        for month in months:
            raw = raw_series(column, month, root)
            joined = pd.concat([raw.rename("raw"), table[column].rename("table")], axis=1, join="inner")
            both_nan = joined["raw"].isna() & joined["table"].isna()
            diff = (joined["raw"] - joined["table"]).abs()
            bad = ~both_nan & ~(diff <= tol)
            rows.append({
                "column": column, "month": str(month), "rows": len(joined),
                "raw_rows": len(raw), "mismatches": int(bad.sum()),
                "max_abs_diff": float(np.nanmax(diff)) if diff.notna().any() else 0.0,
                "raw_keys_missing_in_table": int(len(raw.index.difference(table.index))),
            })
    return pd.DataFrame(rows)


def tpc_flags(months: list[pd.Period]) -> pd.DataFrame:
    """USEP report fields not carried into the merged table: temporary price cap flag, MAP, MAPT."""
    frames = []
    for month in months:
        df = read_raw(raw_month_file("emc_usep_demand_forecast", "USEP", month))
        keep = ["date", "period", "USEP ($/MWh)", "MAP ($/MWh)", "MAPT ($/MWh)", "TPC Applied"]
        frames.append(df[[c for c in keep if c in df.columns]])
    out = pd.concat(frames, ignore_index=True)
    out.columns = ["date", "period", "usep_price", "map_price", "mapt_price", "tpc_applied"][: len(out.columns)]
    for c in ("usep_price", "map_price", "mapt_price"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
