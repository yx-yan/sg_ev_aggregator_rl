"""Build the figures and tables for reports/c/YYYY-MM-DD_price_diagnosis.md.

Run:  PYTHONPATH=src python -m sg_ev_rl.market.report_price_diagnosis --out reports/c
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import diagnostics as d
from .loader import load_market
from .raw_check import compare_months, tpc_flags
from .splits import SPLITS

INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
SPLIT_ORDER = ["train", "validation", "test", "real_validation"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _split_bands(ax, x_of_date, labels=True):
    for i, name in enumerate(SPLIT_ORDER):
        start, end = SPLITS[name]
        x0, x1 = x_of_date(pd.Timestamp(start)), x_of_date(pd.Timestamp(end))
        if i % 2:
            ax.axvspan(x0, x1, color="#f0efec", zorder=0)
        if not labels:
            continue
        ax.text(x0 + (x1 - x0) / 2, 1.0, name.replace("_", " "), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK2)


def fig_monthly(month_tab: pd.DataFrame, path: Path):
    x = month_tab.index.to_timestamp() + pd.Timedelta(days=14)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[2, 1],
                                 facecolor=SURFACE)
    for ax, labels in ((a1, True), (a2, False)):
        _style(ax)
        _split_bands(ax, lambda t: t, labels)
    a1.plot(x, month_tab["median"], color=SERIES[0], lw=2, marker="o", ms=4, label="median")
    a1.plot(x, month_tab["mean"], color=SERIES[1], lw=2, marker="o", ms=4, label="mean")
    a1.text(x[-1], month_tab["median"].iloc[-1], "  median", color=INK2, fontsize=9, va="center")
    a1.text(x[-1], month_tab["mean"].iloc[-1], "  mean", color=INK2, fontsize=9, va="center")
    a1.set_ylabel("USEP (S$/MWh)", color=INK2)
    a1.set_ylim(0, None)
    a1.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 0.93))
    a2.bar(x, month_tab["share_gt_500"] * 100, width=20, color=SERIES[0])
    a2.set_ylabel("% periods > 500", color=INK2)
    fig.suptitle("USEP by month: level step from Mar 2026, spikes from Jul 2026", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_hourly(m: pd.DataFrame, path: Path):
    prof = m.groupby(["day_type", "split", "start_hour"])["usep_price"].median()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, facecolor=SURFACE)
    for ax, day_type in zip(axes, ["weekday", "weekend"], strict=True):
        _style(ax)
        for i, name in enumerate(SPLIT_ORDER):
            s = prof.loc[(day_type, name)]
            ax.plot(s.index, s.values, color=SERIES[i], lw=2, label=name.replace("_", " "))
        ax.set_title(day_type, color=INK, fontsize=10, loc="left")
        ax.set_xticks(range(0, 25, 6))
        ax.set_xlabel("start of half-hour (SGT)", color=INK2, fontsize=9)
    axes[0].set_ylabel("median USEP (S$/MWh)", color=INK2)
    axes[0].set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.suptitle("Median USEP by time of day and split", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_drivers(drv: pd.DataFrame, path: Path):
    base = drv.iloc[0]
    x = drv.index.to_timestamp() + pd.Timedelta(days=14)
    lines = [("usep_median", "USEP median"), ("demand_gross_mean_mw", "gross demand (mean)"),
             ("regulation_median", "regulation price median"), ("solar_mean_mw", "solar (mean)")]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    _style(ax)
    _split_bands(ax, lambda t: t)
    for i, (col, label) in enumerate(lines):
        y = drv[col] / base[col] * 100
        ax.plot(x, y, color=SERIES[i], lw=2, marker="o", ms=3, label=label)
        ax.text(x[-1], y.iloc[-1], f"  {label}", color=INK2, fontsize=8, va="center")
    ax.axhline(100, color=MUTED, lw=1)
    ax.set_ylabel("index, Jan 2025 = 100", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.suptitle("USEP median vs candidate drivers (indexed)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/c"))
    args = ap.parse_args()
    fig_dir = args.out / "figures"
    tab_dir = args.out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    m = d.add_calendar(load_market())
    check = compare_months(m, list(pd.period_range("2026-03", "2026-08", freq="M")))
    check.to_csv(tab_dir / "raw_spot_check_2026-03_to_08.csv", index=False)

    month_tab = d.price_summary(m, "month")
    split_tab = d.price_summary(m, "split").reindex(SPLIT_ORDER)
    daytype_tab = d.price_summary(m, ["split", "day_type"]).reindex(SPLIT_ORDER, level=0)
    drv = d.driver_summary(m)
    tpc = tpc_flags(list(pd.period_range("2025-01", "2026-10", freq="M")))
    tpc["month"] = pd.to_datetime(tpc["date"].astype(str)).dt.to_period("M")
    tpc_tab = tpc.groupby("month").agg(
        tpc_applied_periods=("tpc_applied", lambda s: int((s.str.strip() == "Yes").sum())),
        mapt_mean=("mapt_price", "mean"))
    for name, tab in [("usep_by_month", month_tab), ("usep_by_split", split_tab),
                      ("usep_by_split_daytype", daytype_tab), ("drivers_by_month", drv),
                      ("tpc_by_month", tpc_tab)]:
        tab.round(4).to_csv(tab_dir / f"{name}.csv")

    x = m.assign(dg=m["demand_gross_mw"].diff(), dn=m["nem_demand_actual_mw"].diff())
    timing = {
        "corr_level_demand_gross_vs_ema_nem_by_lag": d.lag_correlation(m, "demand_gross_mw",
                                                                       "nem_demand_actual_mw").round(4).to_dict(),
        "corr_diff_demand_gross_vs_ema_nem_by_lag": d.lag_correlation(x, "dg", "dn").round(4).to_dict(),
        "solar": d.solar_peak(m).to_dict(),
        "solar_centre_hours_by_calendar_month": {
            int(k): round(float(v), 3) for k, v in m.groupby(pd.to_datetime(m["date"].astype(str)).dt.month)
            .apply(lambda g: d.solar_peak(g)["centre_hours_if_period_start_based"]).items()},
    }
    (tab_dir / "timestamp_checks.json").write_text(json.dumps(timing, indent=2, default=str))

    fig_monthly(month_tab, fig_dir / "usep_by_month.png")
    fig_hourly(m, fig_dir / "usep_hourly_by_split.png")
    fig_drivers(drv, fig_dir / "usep_vs_drivers_indexed.png")
    print("raw mismatches:", int(check["mismatches"].sum()), "of", int(check["rows"].sum()), "cells")
    print(split_tab.round(2).to_string())
    print(daytype_tab.round(2).to_string())


if __name__ == "__main__":
    main()
