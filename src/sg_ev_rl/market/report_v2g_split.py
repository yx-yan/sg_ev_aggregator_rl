"""Tables and figures for reports/c/YYYY-MM-DD_v2g_and_split.md.

Run:  PYTHONPATH=src python -m sg_ev_rl.market.report_v2g_split --out reports/c
The V2G grid (about 30k small LPs, ~30 s) is cached in the runs directory.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .features import add_relative_price_features
from .loader import load_market
from .paths import runs_dir
from .report_price_diagnosis import INK, INK2, SERIES, SPLIT_ORDER, SURFACE, _style
from .splits import assign_split
from .v2g_roughcalc import WINDOWS, regulation_capacity_value, run_grid

EFFS = [0.90, 0.85, 0.80]
DEGS = [0.0, 0.03, 0.06, 0.10, 0.15]
BASE_EFF, BASE_DEG = 0.85, 0.06
BATTERY_KWH = 60.0
EV_KW = 7.0
DEV_SPLITS = ["train", "validation"]       # the only splits used for any keep/drop judgement


def top_share(s: pd.Series, q: float = 0.1) -> float:
    v = s.clip(lower=0).sort_values(ascending=False)
    k = max(1, int(round(len(v) * q)))
    return float(v.iloc[:k].sum() / v.sum()) if v.sum() > 0 else float("nan")


def v2g_tables(grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid.assign(positive=grid["profit_per_kwh_batt"] > 1e-4)
    g = grid.groupby(["window", "round_trip_eff", "deg_cost", "split"])
    out = g.agg(days=("date", "size"),
                mean_s_per_kwh_day=("profit_per_kwh_batt", "mean"),
                median_s_per_kwh_day=("profit_per_kwh_batt", "median"),
                share_days_positive=("positive", "mean"),
                mean_kwh_sold_per_kwh=("discharged_kwh", "mean"))
    out["top10pct_days_share_of_profit"] = g["profit_per_kwh_batt"].apply(top_share)
    out["s_per_ev_year_60kwh"] = out["mean_s_per_kwh_day"] * BATTERY_KWH * 365
    return out


SPIKE_PRICE = 500.0       # fixed round number, not fitted on any split
SPIKE_WEEK_SHARE = 0.02   # a week is a spike week if >= 2% of its half-hours exceed SPIKE_PRICE


def weekly_regimes(m: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Two week labels, both with thresholds fixed without looking at test data:
    level_regime: weekly median above the 90th percentile of TRAINING weekly medians;
    spike_regime: share of half-hours above 500 S$/MWh is at least 2%."""
    wk = m.assign(week=m["interval_start"].dt.tz_localize(None).dt.to_period("W-SUN"))
    w = wk.groupby("week").agg(median=("usep_price", "median"), split=("split", "first"),
                               n=("usep_price", "size"),
                               share_gt_500=("usep_price", lambda s: float((s > SPIKE_PRICE).mean())))
    w = w[w["n"] == 336]
    threshold = float(w.loc[w["split"] == "train", "median"].quantile(0.9))
    w["level_regime"] = np.where(w["median"] > threshold, "high", "normal")
    w["spike_regime"] = np.where(w["share_gt_500"] >= SPIKE_WEEK_SHARE, "spike", "normal")
    return w, threshold


def feature_shift(m: pd.DataFrame) -> pd.DataFrame:
    f = add_relative_price_features(m)
    rows = []
    for col in ["usep_price", "usep_price_pct_336", "usep_price_z_336", "usep_price_ratio_med_336"]:
        tr = f.loc[f["split"] == "train", col].dropna()
        for sp in ["validation", "test", "real_validation"]:
            other = f.loc[f["split"] == sp, col].dropna()
            rows.append({"feature": col, "vs_split": sp, "ks_stat_vs_train": float(ks_2samp(tr, other).statistic),
                         "train_median": float(tr.median()), "split_median": float(other.median())})
    return pd.DataFrame(rows)


def fig_v2g(tab: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True, facecolor=SURFACE)
    for ax, w in zip(axes, WINDOWS):
        _style(ax)
        for i, sp in enumerate(SPLIT_ORDER):
            s = tab.loc[(w, BASE_EFF, slice(None), sp), "s_per_ev_year_60kwh"]
            ax.plot(DEGS, s.values, color=SERIES[i], lw=2, marker="o", ms=4, label=sp.replace("_", " "))
        start_h, length_h = WINDOWS[w]
        end_h = (start_h + length_h) % 24
        ax.set_title(f"{w.split('_')[0]} {start_h:02.0f}:00–{end_h:02.0f}:00", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("degradation cost (S$/kWh sold)", color=INK2, fontsize=8)
    axes[0].set_ylabel("S$ per EV per year (60 kWh)", color=INK2)
    axes[0].set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.suptitle("V2G arbitrage value, round-trip efficiency 0.85 (perfect foresight, upper bound)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_weeks(w: pd.DataFrame, threshold: float, path: Path):
    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=SURFACE)
    _style(ax)
    x = w.index.to_timestamp()
    for i, sp in enumerate(SPLIT_ORDER):
        sel = w["split"] == sp
        ax.bar(x[sel], w.loc[sel, "median"], width=5, color=SERIES[i], label=sp.replace("_", " "))
    ax.axhline(threshold, color=INK2, lw=1, ls="--")
    ax.text(x[0], threshold + 6, f"level threshold {threshold:.0f} (train P90)", color=INK2, fontsize=8,
            va="bottom", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    ax.set_ylabel("weekly median USEP (S$/MWh)", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.95))
    fig.suptitle("Weekly median USEP and the high-price threshold", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/c"))
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    fig_dir, tab_dir = args.out / "figures", args.out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    m = load_market()
    cache = runs_dir() / "v2g_grid.parquet"
    if args.recompute or not cache.exists():
        grid = run_grid(m, EFFS, DEGS)
        grid["split"] = assign_split(grid["date"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(cache)
    grid = pd.read_parquet(cache)
    tab = v2g_tables(grid)
    tab.round(5).to_csv(tab_dir / "v2g_arbitrage_grid.csv")

    reg = {}
    for w, (start_h, length_h) in WINDOWS.items():
        r = regulation_capacity_value(m, start_h, length_h)
        sp = assign_split(pd.Series(list(r.index)))
        reg[w] = (pd.Series(r.values, index=sp.values).groupby(level=0).mean() * EV_KW).reindex(SPLIT_ORDER)
    reg_tab = pd.DataFrame(reg).T
    reg_tab.round(4).to_csv(tab_dir / "regulation_capacity_value_per_7kw_day.csv")

    weeks, threshold = weekly_regimes(m)
    regime_counts = pd.concat(
        {r: weeks.groupby(["split", r]).size().unstack(fill_value=0).reindex(SPLIT_ORDER).fillna(0).astype(int)
         for r in ("level_regime", "spike_regime")}, axis=1)
    regime_counts.to_csv(tab_dir / "weeks_by_regime.csv")
    weeks.to_csv(tab_dir / "weeks_labelled.csv")
    shift = feature_shift(m)
    shift.round(4).to_csv(tab_dir / "feature_shift_ks.csv", index=False)
    (tab_dir / "high_price_threshold.json").write_text(json.dumps(
        {"level_regime": "weekly median USEP above P90 of training-split weekly medians",
         "level_threshold_s_per_mwh": round(threshold, 2),
         "spike_regime": f"share of half-hours above {SPIKE_PRICE:.0f} S$/MWh >= {SPIKE_WEEK_SHARE:.0%}"}, indent=2))

    fig_v2g(tab, fig_dir / "v2g_value_by_degradation.png")
    fig_weeks(weeks, threshold, fig_dir / "weekly_median_regimes.png")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.loc[(slice(None), BASE_EFF, BASE_DEG, slice(None))].round(4))
        print(reg_tab.round(3))
        print(regime_counts, "threshold", round(threshold, 1))
        print(shift.round(3))


if __name__ == "__main__":
    main()
